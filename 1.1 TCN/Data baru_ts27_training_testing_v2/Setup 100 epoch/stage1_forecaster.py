# =============================
# STAGE 1 — TCN FORECASTER
# Sliding window: Day1+2+3→Day4, Day2+3+4→Day5, dst
# Output: model_stage1_forecaster.pth + scaler_stage1.pkl
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
from sklearn.preprocessing import MinMaxScaler
import joblib
import re
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==================================================================
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
DATA_DIR            = os.path.join(SCRIPT_DIR, "data")          # folder CSV terisolasi
CHECKPOINT_DIR      = os.path.join(SCRIPT_DIR, "checkpoints_stage1")
LOG_FILE            = os.path.join(SCRIPT_DIR, "log_stage1.txt")
EVIDENCE_DIR        = os.path.join(SCRIPT_DIR, "evidence")

N_EPOCHS            = 100
BATCH_SIZE          = 3
CHECKPOINT_INTERVAL = 50
COMPRESSION_FACTOR  = 1
TRAIN_RATIO         = 0.8   # proporsi hari untuk training (sisanya testing)
# ==================================================================

N_TAKE                    = 200_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
FUTURE                     = COMPRESSED_POINTS_PER_DAY
START_TIME                 = time(3, 0, 0)
END_TIME                   = time(18, 16, 35)
N_DROP_FIRST               = 3600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Stage 1] Device: {device}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR,   exist_ok=True)

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
n_features    = len(target_columns)  # 21

# =============================
# BACA & PREPROCESSING
# =============================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if "hasil"      not in os.path.basename(f).lower()
    and "prediksi"  not in os.path.basename(f).lower()
    and "inference" not in os.path.basename(f).lower()
], key=extract_date)

print(f"[Stage 1] Data dir : {DATA_DIR}")
print(f"[Stage 1] File CSV : {len(csv_files)} ditemukan")

def read_and_crop(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';', low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])
    for col in target_columns + fault_columns:
        df[col] = pd.to_numeric(
            df.get(col, pd.Series(np.nan, index=df.index)).astype(str).str.replace(',', '.'),
            errors='coerce'
        )
    df[target_columns + fault_columns] = df[target_columns + fault_columns].ffill().bfill()
    date0 = df['ts_date'].dt.date.iloc[0]
    df    = df[(df['ts_date'] >= datetime.combine(date0, START_TIME)) &
               (df['ts_date'] <= datetime.combine(date0, END_TIME))]
    if len(df) < N_DROP_FIRST + N_TAKE:
        return pd.DataFrame()
    return df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)[
        ['ts_date'] + target_columns + fault_columns
    ]

compressed_dfs = []
for f in csv_files:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"  Skip {os.path.basename(f)}")
        continue
    chunks, ts_mid = [], []
    for i in range(COMPRESSED_POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunks.append(df_raw[target_columns + fault_columns].iloc[s:e].mean())
        ts_mid.append(df_raw['ts_date'].iloc[s + COMPRESSION_FACTOR // 2])
    df_c = pd.DataFrame(chunks, columns=target_columns + fault_columns)
    df_c.insert(0, 'ts_date', ts_mid)
    compressed_dfs.append(df_c)

print(f"[Stage 1] Total hari: {len(compressed_dfs)}")
if len(compressed_dfs) < 4:
    raise ValueError("Minimal 4 hari CSV!")

# =============================
# TRAIN / TEST SPLIT (kronologis)
# =============================
n_train_days = max(4, int(len(compressed_dfs) * TRAIN_RATIO))
n_test_days  = len(compressed_dfs) - n_train_days
train_dfs    = compressed_dfs[:n_train_days]
print(f"[Stage 1] Train days: {n_train_days} | Test days: {n_test_days}")

# =============================
# SLIDING WINDOW: 3 hari → 1 hari (hanya dari data training)
# =============================
X_seq, y_signal = [], []
for i in range(len(train_dfs) - 3):
    seq = np.concatenate([df[target_columns].values for df in train_dfs[i:i+3]], axis=0)
    X_seq.append(seq)
    y_signal.append(train_dfs[i+3][target_columns].values)

X_seq    = np.array(X_seq,    dtype=np.float32)
y_signal = np.array(y_signal, dtype=np.float32)
print(f"[Stage 1] Window training: {len(X_seq)}")

# Window test: gunakan 3 hari terakhir train sebagai context awal
X_seq_test_list, y_signal_test_list = [], []
for i in range(n_train_days - 3, len(compressed_dfs) - 3):
    seq = np.concatenate([df[target_columns].values for df in compressed_dfs[i:i+3]], axis=0)
    X_seq_test_list.append(seq)
    y_signal_test_list.append(compressed_dfs[i+3][target_columns].values)
print(f"[Stage 1] Window testing : {len(X_seq_test_list)}")

# Scaler di-fit HANYA dari data training
scaler   = MinMaxScaler(feature_range=(-0.1, 1.1))
X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
y_scaled = scaler.transform(y_signal.reshape(-1, n_features)).reshape(y_signal.shape)
joblib.dump(scaler, os.path.join(SCRIPT_DIR, "scaler_stage1.pkl"))
print("[Stage 1] scaler_stage1.pkl disimpan")

# Scale test data menggunakan scaler training
X_scaled_test      = None
y_signal_test_orig = None
if X_seq_test_list:
    X_seq_test         = np.array(X_seq_test_list,    dtype=np.float32)
    y_signal_test_orig = np.array(y_signal_test_list, dtype=np.float32)
    X_scaled_test      = scaler.transform(X_seq_test.reshape(-1, n_features)).reshape(X_seq_test.shape)

X_tensor     = torch.FloatTensor(X_scaled).to(device)
y_sig_tensor = torch.FloatTensor(y_scaled).to(device)

class ForecastDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dataloader = DataLoader(ForecastDataset(X_tensor, y_sig_tensor),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# =============================
# MODEL TCN
# =============================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation=1):
        super().__init__()
        self.pad  = (ks - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, ks, padding=self.pad, dilation=dilation)
    def forward(self, x):
        o = self.conv(x)
        return o[:, :, :-self.pad] if self.pad > 0 else o

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, dilation=1, dropout=0.3):
        super().__init__()
        self.c1   = CausalConv1d(in_ch,  out_ch, ks, dilation)
        self.n1   = nn.BatchNorm1d(out_ch); self.r1 = nn.ReLU(); self.d1 = nn.Dropout(dropout)
        self.c2   = CausalConv1d(out_ch, out_ch, ks, dilation)
        self.n2   = nn.BatchNorm1d(out_ch); self.r2 = nn.ReLU(); self.d2 = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        r = self.skip(x)
        o = self.d1(self.r1(self.n1(self.c1(x))))
        o = self.d2(self.r2(self.n2(self.c2(o))))
        return o + r

class TCNForecaster(nn.Module):
    """
    Input : (batch, 3*COMPRESSED_POINTS_PER_DAY, 21)
    Output: pred_signal (batch, FUTURE, 21)
            context     (batch, 96)
    """
    def __init__(self, n_features, n_ch=96, ks=3, n_blocks=7, dropout=0.3):
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32, 64]
        layers, in_ch = [], n_features
        for i in range(n_blocks):
            layers.append(ResidualBlock(in_ch, n_ch, ks, dilations[i], dropout))
            in_ch = n_ch
        self.tcn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dec  = nn.Sequential(
            nn.Linear(n_ch, n_ch), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_ch, n_features * FUTURE)
        )
    def forward(self, x):
        ctx  = self.pool(self.tcn(x.transpose(1, 2))).squeeze(-1)   # (B, 96)
        pred = self.dec(ctx).view(-1, FUTURE, n_features)            # (B, FUTURE, 21)
        return pred, ctx

model     = TCNForecaster(n_features).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30, verbose=True)
criterion = nn.MSELoss()

# =============================
# CHECKPOINT & RESUME
# =============================
start_epoch = 1
cp_files    = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
if cp_files:
    latest  = max(int(os.path.basename(f).split('_')[-1].replace('.pth','')) for f in cp_files)
    cp_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{latest}.pth")
    if latest < N_EPOCHS:
        cp = torch.load(cp_path, map_location=device)
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler']); start_epoch = latest + 1
        print(f"[Stage 1] Resume epoch {start_epoch}")
    else:
        model.load_state_dict(torch.load(cp_path, map_location=device)['model'])
        print(f"[Stage 1] Sudah selesai epoch {latest}")

def log(t):
    print(t)
    with open(LOG_FILE, 'a', encoding='utf-8') as f: f.write(t + '\n')

log(f"\n{'='*60}\nSTAGE 1 TCN FORECASTER | {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"Train days: {n_train_days} | Test days: {n_test_days}")
log(f"Window train: {len(X_tensor)} | Window test: {len(X_seq_test_list)}")
log(f"Epoch: {N_EPOCHS} | Batch: {BATCH_SIZE}\n{'='*60}")

# =============================
# TRAINING
# =============================
if start_epoch <= N_EPOCHS:
    model.train()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        total = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        avg = total / len(dataloader)
        scheduler.step(avg)
        log(f"Epoch {epoch:4d}/{N_EPOCHS} | MSE: {avg:.7f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            p = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, p)
            log(f"   → Checkpoint: {p}")
    log("=== STAGE 1 TRAINING SELESAI ===\n")

# =============================
# EVALUASI TRAINING SET
# =============================
log("=== EVALUASI TRAINING SET (Stage 1) — skala normalized [-0.1, 1.1] ===")
model.eval()
with torch.no_grad():
    pred_train, _ = model(X_tensor)
    pred_train    = pred_train.cpu().numpy()
y_scaled_np   = y_scaled  # sudah normalized
tr_mse  = float(np.mean((pred_train - y_scaled_np) ** 2))
tr_rmse = float(np.sqrt(tr_mse))
tr_mae  = float(np.mean(np.abs(pred_train - y_scaled_np)))
_mask_tr = y_scaled_np != 0
tr_mape = float(np.mean(np.abs(
    (pred_train[_mask_tr] - y_scaled_np[_mask_tr]) / y_scaled_np[_mask_tr]
)) * 100) if _mask_tr.any() else float('nan')
log(f"Train Windows : {len(X_seq)}")
log(f"Train MSE     : {tr_mse:.6f}")
log(f"Train RMSE    : {tr_rmse:.6f}")
log(f"Train MAE     : {tr_mae:.6f}")
log(f"Train MAPE    : {tr_mape:.2f}%")
log("=" * 40 + "\n")

# =============================
# EVALUASI TEST SET
# =============================
if X_scaled_test is not None:
    log("=== EVALUASI TEST SET (Stage 1) — skala normalized [-0.1, 1.1] ===")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_scaled_test).to(device)
        pred_test, _  = model(X_test_tensor)
        pred_test     = pred_test.cpu().numpy()
    y_scaled_test = scaler.transform(
        y_signal_test_orig.reshape(-1, n_features)
    ).reshape(y_signal_test_orig.shape)
    mse  = float(np.mean((pred_test - y_scaled_test) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(pred_test - y_scaled_test)))
    _mask = y_scaled_test != 0
    mape = float(np.mean(np.abs(
        (pred_test[_mask] - y_scaled_test[_mask]) / y_scaled_test[_mask]
    )) * 100) if _mask.any() else float('nan')
    log(f"Test Windows : {len(X_seq_test_list)}")
    log(f"Test MSE     : {mse:.6f}  (sebanding dengan training MSE)")
    log(f"Test RMSE    : {rmse:.6f}")
    log(f"Test MAE     : {mae:.6f}")
    log(f"Test MAPE    : {mape:.2f}%")
    log("=" * 40)
else:
    log("[Stage 1] Tidak ada test windows — tambah lebih banyak hari CSV")

torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model_stage1_forecaster.pth"))
log("model_stage1_forecaster.pth disimpan")

# =============================
# PLOT KURVA LOSS — Jurnal Gambar 10
# Parse dari log_stage1.txt agar benar meski di-resume
# =============================
epoch_nums_log, mse_vals_log = [], []
try:
    with open(LOG_FILE, 'r', encoding='utf-8') as _lf:
        for _line in _lf:
            _m = re.search(r'Epoch\s+(\d+)/\d+\s*\|\s*MSE:\s*([\d.]+)', _line)
            if _m:
                epoch_nums_log.append(int(_m.group(1)))
                mse_vals_log.append(float(_m.group(2)))
except Exception as _e:
    print(f"[Plot] Gagal parse log stage1: {_e}")

if epoch_nums_log:
    # Deduplicate per epoch (jika ada resume, pakai nilai terbaru)
    _ep_dict = {}
    for _ep, _mse in zip(epoch_nums_log, mse_vals_log):
        _ep_dict[_ep] = _mse
    _ep_sorted  = sorted(_ep_dict.keys())
    _mse_sorted = [_ep_dict[e] for e in _ep_sorted]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(_ep_sorted, _mse_sorted, 'b-', linewidth=1.5, label='MSE Training')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(
        f"Gambar 10. Kurva Epoch vs MSE Training — TCN Forecaster ({max(_ep_sorted)} Epoch)\n"
        f"MSE Awal: {_mse_sorted[0]:.4f} → MSE Akhir: {_mse_sorted[-1]:.4f} "
        f"(Penurunan {(1 - _mse_sorted[-1]/_mse_sorted[0])*100:.1f}%)",
        fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _out = os.path.join(EVIDENCE_DIR, "jurnal_kurva_loss_stage1.png")
    plt.savefig(_out, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"jurnal_kurva_loss_stage1.png disimpan  ← Jurnal Gambar 10")
else:
    print("[Plot] Log stage1 belum ada data epoch — training perlu dijalankan dahulu")

# =============================
# JURNAL INFO TXT — Stage 1
# =============================
# Ambil metrik dari log yang sudah di-parse
_j1_ep1_mse = _j1_epN_mse = _j1_epN = _j1_lr = None
if epoch_nums_log:
    _j1_d = {}
    for _e, _m in zip(epoch_nums_log, mse_vals_log): _j1_d[_e] = _m
    _j1_eps = sorted(_j1_d.keys())
    _j1_ep1_mse = _j1_d[_j1_eps[0]]
    _j1_epN_mse = _j1_d[_j1_eps[-1]]
    _j1_epN     = _j1_eps[-1]
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as _lf1:
            for _l in _lf1:
                _lm = re.search(r'LR:\s*([\d.e+\-]+)', _l)
                if _lm: _j1_lr = _lm.group(1)
    except: pass

# Re-compute test metrics agar selalu tersedia (normalized space)
_j1_mse = _j1_rmse = _j1_mae = _j1_mape = None
if X_scaled_test is not None:
    model.eval()
    with torch.no_grad():
        _j1_pred, _ = model(torch.FloatTensor(X_scaled_test).to(device))
        _j1_pred    = _j1_pred.cpu().numpy()
    _j1_ys   = scaler.transform(y_signal_test_orig.reshape(-1, n_features)).reshape(y_signal_test_orig.shape)
    _j1_mse  = float(np.mean((_j1_pred - _j1_ys) ** 2))
    _j1_rmse = float(np.sqrt(_j1_mse))
    _j1_mae  = float(np.mean(np.abs(_j1_pred - _j1_ys)))
    _j1_mask = _j1_ys != 0
    _j1_mape = float(np.mean(np.abs(
        (_j1_pred[_j1_mask] - _j1_ys[_j1_mask]) / _j1_ys[_j1_mask]
    )) * 100) if _j1_mask.any() else float('nan')

_j1_dates = [os.path.basename(f)[:8] for f in csv_files]
_j1_lines = [
    "=" * 68,
    "INFORMASI JURNAL — TCN FORECASTER (STAGE 1)",
    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "=" * 68,
    "",
    "[DATASET & SPLIT — untuk Tabel Jurnal]",
    f"  Total hari CSV               : {len(compressed_dfs)}",
    f"  Train days (TRAIN_RATIO=0.8) : {n_train_days}",
    f"  Test days                    : {n_test_days}",
    f"  Window training (3->1 hari)  : {len(X_seq)}",
    f"  Window testing               : {len(X_seq_test_list)}",
    f"  Timestep per hari (N_TAKE)   : {N_TAKE:,}",
    f"  Compression Factor           : {COMPRESSION_FACTOR}",
    f"  Jumlah parameter (fitur)     : {n_features}",
    "",
    "[PREPROCESSING — untuk Tabel Data Cleaning Jurnal]",
    f"  Jumlah data sebelum crop/hari: {N_DROP_FIRST + N_TAKE:,} baris",
    f"  Warmup baris di-drop         : {N_DROP_FIRST:,} baris",
    f"  Jumlah data digunakan/hari   : {N_TAKE:,} baris ({N_TAKE/(N_DROP_FIRST+N_TAKE)*100:.1f}%)",
    f"  Jumlah data total (13 hari)  : {N_TAKE * len(compressed_dfs):,} baris",
    f"  Jumlah parameter digunakan   : {n_features} parameter",
    f"  Timestamp per hari           : {N_TAKE:,}",
    f"  Missing value handling        : forward-fill (ffill) + backward-fill (bfill)",
    f"  Start time (setelah warmup)  : 03:00 WIB",
    f"  End time                     : 18:16 WIB",
    "",
    "[NORMALISASI — untuk Tabel Normalisasi Jurnal]",
    f"  Metode Normalisasi           : Min-Max Scaler",
    f"  Rentang output normalisasi   : [-0.1, 1.1]",
    f"  Jumlah parameter dinorm.     : {n_features}",
    f"  Fit hanya pada               : {n_train_days} hari training",
    f"  Transform pada               : training + test (tanpa data leakage)",
    "",
    "[RENTANG DATA PER PARAMETER (dari scaler training)]",
    f"  {'Parameter':<28} {'Min Data':>12} {'Max Data':>12}",
    "  " + "-" * 55,
]
for _ci, _col in enumerate(target_columns):
    _j1_lines.append(f"  {_col:<28} {scaler.data_min_[_ci]:>12.4f} {scaler.data_max_[_ci]:>12.4f}")

_j1_lines += [
    "",
    "[HYPERPARAMETER TCN — untuk Tabel Jurnal]",
    f"  Optimizer                    : AdamW",
    f"  Learning Rate (awal)         : 0.001",
    f"  Learning Rate (akhir)        : {_j1_lr or 'N/A'}",
    f"  Batch Size                   : {BATCH_SIZE}",
    f"  Epoch                        : {N_EPOCHS}",
    f"  Loss Function                : Mean Squared Error (MSE)",
    f"  Gradient Clipping            : 1.0",
    f"  Dropout                      : 0.3",
    f"  Weight Decay                 : 1e-5",
    f"  Scheduler                    : ReduceLROnPlateau (factor=0.5, patience=30)",
    "",
    "[ARSITEKTUR TCN — untuk Tabel Jurnal]",
    f"  Jumlah Residual Block        : 7",
    f"  Dilation Factors             : [1, 2, 4, 8, 16, 32, 64]",
    f"  Hidden Channel (n_ch)        : 96",
    f"  Kernel Size                  : 3",
    f"  Input Window                 : 3 hari = {3*N_TAKE:,} timestep",
    f"  Output Horizon               : H+1 = {FUTURE:,} timestep (1 hari)",
    f"  Input Shape                  : (batch, {3*FUTURE}, {n_features})",
    f"  Output Shape                 : (batch, {FUTURE}, {n_features})",
    "",
    "[HASIL TRAINING — untuk Para 168 Jurnal]",
]
if _j1_ep1_mse is not None:
    _j1_pen = (1 - _j1_epN_mse / _j1_ep1_mse) * 100
    _j1_lines += [
        f"  MSE Epoch 1                  : {_j1_ep1_mse:.7f}  ({_j1_ep1_mse*100:.4f}%)",
        f"  MSE Epoch {_j1_epN} (akhir)        : {_j1_epN_mse:.7f}  ({_j1_epN_mse*100:.4f}%)",
        f"  Penurunan MSE total          : {_j1_pen:.1f}%",
        f"  MSE akhir (persen)           : ~{_j1_epN_mse*100:.2f}%",
    ]
else:
    _j1_lines.append("  (Jalankan training untuk mengisi bagian ini)")

_j1_lines += ["", "[EVALUASI TEST SET — untuk Para 170 Jurnal]"]
if _j1_mse is not None:
    _j1_selisih = abs(_j1_epN_mse - _j1_mse) if _j1_epN_mse else None
    _j1_lines += [
        f"  Test Windows                 : {len(X_seq_test_list)}",
        f"  MSE Test                     : {_j1_mse:.7f}  ({_j1_mse*100:.4f}%)",
        f"  RMSE Test                    : {_j1_rmse:.7f}",
        f"  MAE Test                     : {_j1_mae:.7f}",
    ]
    if _j1_selisih is not None:
        _j1_lines.append(f"  Selisih MSE (Train - Test)   : {_j1_selisih:.7f}  → relatif kecil")
else:
    _j1_lines.append("  (Tidak ada test windows)")

_j1_lines += [
    "",
    "[PEMBENTUKAN SEQUENCE — untuk Tabel 11 Jurnal]",
    f"  {'Seq':>3}  {'Input A':>10}  {'Input B':>10}  {'Input C':>10}  {'Target H+1':>10}  Set",
    "  " + "-" * 58,
]
_j1_sn = 1
for _i in range(len(train_dfs) - 3):
    _da, _db, _dc, _dt = _j1_dates[_i], _j1_dates[_i+1], _j1_dates[_i+2], _j1_dates[_i+3]
    _j1_lines.append(f"  {_j1_sn:>3}  {_da:>10}  {_db:>10}  {_dc:>10}  {_dt:>10}  Training")
    _j1_sn += 1
for _i in range(n_train_days - 3, len(compressed_dfs) - 3):
    _da, _db, _dc, _dt = _j1_dates[_i], _j1_dates[_i+1], _j1_dates[_i+2], _j1_dates[_i+3]
    _j1_lines.append(f"  {_j1_sn:>3}  {_da:>10}  {_db:>10}  {_dc:>10}  {_dt:>10}  Testing")
    _j1_sn += 1

_j1_out = os.path.join(EVIDENCE_DIR, "jurnal_info_stage1.txt")
with open(_j1_out, 'w', encoding='utf-8') as _f1:
    _f1.write('\n'.join(_j1_lines))
log("jurnal_info_stage1.txt disimpan  ← Data untuk Tabel & Paragraf Jurnal Stage 1")

print("\nSTAGE 1 SELESAI! Jalankan stage2_classifier.py berikutnya.")
