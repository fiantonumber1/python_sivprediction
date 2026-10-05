# =============================
# STAGE 1 — TCN FORECASTER (IMPROVED v2)
# Sliding window: Day1+2+3→Day4, Day2+3+4→Day5, dst
# Output: model_stage1_forecaster.pth + scaler_stage1.pkl
#
# PERUBAHAN vs versi asli (Setup 100 epoch):
#   1) DECODER DIPERBAIKI — masalah utama versi lama: seluruh 600.000 timestep
#      input di-AdaptiveAvgPool1d(1) jadi SATU vektor (96 dim), lalu direkonstruksi
#      lewat SATU nn.Linear(96, 21*200000) ≈ 403 JUTA parameter di satu layer.
#      Ini membuang semua informasi "jam berapa nilainya berapa" dan model cuma
#      bisa menebak rata-rata kasar. Versi baru pakai DECODER POINT-WISE per-timestep
#      (Conv1d kernel=1) yang diterapkan ke fitur hari terakhir (Day 3) yang sudah
#      di-encode TCN — jadi prediksi tiap timestep tetap "selaras waktu" dengan
#      pola harian (mis. jam 10:15 hari ini <-> jam 10:15 besok).
#   2) RESIDUAL/PERSISTENCE CONNECTION — model memprediksi SELISIH (delta) dari
#      sinyal Day 3, bukan sinyal absolut dari nol. Karena operasi SIV berpola
#      harian berulang, "besok ~= hari ini + koreksi kecil" adalah prior yang kuat
#      dan biasanya jauh lebih mudah dipelajari daripada meniru sinyal dari nol.
#   3) FiLM-style GLOBAL CONTEXT FUSION — context vector (rata-rata 3 hari, isinya
#      trend/fault jangka pendek) tetap dihitung dan dijumlahkan ke fitur per-timestep
#      sebelum decoder, supaya info "3 hari terakhir ada Warning" tetap kebawa.
#   4) VALIDATION SPLIT + EARLY STOPPING — versi lama scheduler ReduceLROnPlateau
#      memonitor TRAIN loss (yang nyaris selalu turun), jadi LR nyaris tidak pernah
#      diturunkan (log asli: LR tetap 1.00e-03 sepanjang 100 epoch). Versi baru
#      menyisihkan 4 window TERAKHIR dari training (kronologis) sebagai validation
#      internal → dipakai scheduler + early stopping (patience 15) + restore best
#      weights. Test set (8 window) TETAP tidak disentuh sampai evaluasi akhir.
#
# Arsitektur TCN encoder (residual block, dilation, n_ch) TIDAK diubah — tetap
# TCN sesuai permintaan. Yang diubah cuma DECODER-nya.
# =============================

import copy
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
# Referensi data ASLI (read-only) dari folder "Setup 100 epoch" di sebelahnya —
# supaya tidak duplikasi ~5GB CSV dan tidak pernah menyentuh/mengubah folder lama.
DATA_DIR             = os.path.join(SCRIPT_DIR, "..", "Setup 100 epoch", "data")
CHECKPOINT_DIR      = os.path.join(SCRIPT_DIR, "checkpoints_stage1")
LOG_FILE            = os.path.join(SCRIPT_DIR, "log_stage1.txt")
EVIDENCE_DIR        = os.path.join(SCRIPT_DIR, "evidence")

N_EPOCHS            = 100
BATCH_SIZE          = 3
CHECKPOINT_INTERVAL = 50
TRAIN_RATIO         = 0.8   # proporsi hari untuk training (sisanya testing)
N_VAL_WINDOWS       = 4     # window training TERAKHIR (kronologis) dipakai validation internal
EARLY_STOP_PATIENCE = 100   # dinonaktifkan efektif (>= N_EPOCHS) -- val set internal cuma 4
                            # window, terlalu noisy utk keputusan berhenti dini yang adil.
                            # Tetap jalan penuh 100 epoch (apple-to-apple dgn model lama),
                            # tapi tetap simpan & restore BEST checkpoint di akhir (bukan cuma
                            # ambil bobot epoch terakhir).
# ==================================================================

N_TAKE     = 200_000
FUTURE     = N_TAKE
START_TIME = time(3, 0, 0)
END_TIME                   = time(18, 16, 35)
N_DROP_FIRST               = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(os.cpu_count() or 1)
print(f"[Stage 1] Device: {device} | CPU threads: {torch.get_num_threads()}")
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
# BACA & PREPROCESSING (identik versi asli)
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

_needed_cols = set(['ts_date'] + target_columns + fault_columns)

def read_and_crop(filepath):
    df = pd.read_csv(
        filepath,
        encoding='utf-8-sig', sep=';', decimal=',',
        usecols=lambda c: c.strip() in _needed_cols,
        low_memory=False, on_bad_lines='skip'
    )
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])
    for col in target_columns + fault_columns:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[target_columns + fault_columns] = df[target_columns + fault_columns].ffill().bfill()
    date0 = df['ts_date'].dt.date.iloc[0]
    df    = df[(df['ts_date'] >= datetime.combine(date0, START_TIME)) &
               (df['ts_date'] <= datetime.combine(date0, END_TIME))]
    if len(df) < N_TAKE:
        return pd.DataFrame()
    return df.iloc[:N_TAKE].reset_index(drop=True)[['ts_date'] + target_columns + fault_columns]

compressed_dfs = []
for f in csv_files:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"  Skip {os.path.basename(f)}")
        continue
    compressed_dfs.append(df_raw[['ts_date'] + target_columns + fault_columns].copy())

print(f"[Stage 1] Total hari: {len(compressed_dfs)}")
if len(compressed_dfs) < 4:
    raise ValueError("Minimal 4 hari CSV!")

# =============================
# TRAIN / TEST SPLIT (kronologis) — identik versi asli
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
print(f"[Stage 1] Window training (total): {len(X_seq)}")

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

# =============================
# SPLIT INTERNAL: TRAIN (fit) vs VALIDATION (early stopping)
# Kronologis — N_VAL_WINDOWS terakhir dari window training dipakai validation.
# Test set (8 window) TETAP terpisah total, tidak disentuh di sini.
# =============================
n_val    = min(N_VAL_WINDOWS, max(1, len(X_scaled) // 5))
X_fit,  y_fit  = X_scaled[:-n_val], y_scaled[:-n_val]
X_val,  y_val  = X_scaled[-n_val:], y_scaled[-n_val:]
print(f"[Stage 1] Window fit(train): {len(X_fit)} | Window validation(internal): {len(X_val)}")

X_tensor     = torch.FloatTensor(X_fit).to(device)
y_sig_tensor = torch.FloatTensor(y_fit).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)

class ForecastDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dataloader = DataLoader(ForecastDataset(X_tensor, y_sig_tensor),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# =============================
# MODEL TCN — encoder sama seperti versi asli, DECODER DIPERBAIKI
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
    Input : (batch, 3*N_TAKE, 21)
    Output: pred_signal (batch, FUTURE, 21)
            ctx         (batch, 96)  -- global context (dipertahankan demi kompatibilitas
                                        signature dengan stage3_inference.py)

    DECODER BARU (point-wise + persistence residual + FiLM context):
      1. TCN encoder menghasilkan fitur per-timestep (B, n_ch, 3*FUTURE) — TIDAK
         langsung di-pool habis seperti versi lama.
      2. Ambil fitur hari TERAKHIR (Day 3): last_day_feat = feat[:, :, -FUTURE:]
         → tetap selaras posisi waktu dengan target (Day 4).
      3. Fusi FiLM sederhana: tambahkan global context (avg-pool seluruh 3 hari)
         ke tiap posisi waktu → model tetap tahu trend/fault 3 hari terakhir.
      4. Head point-wise (Conv1d kernel=1) memetakan (n_ch -> n_features) di SETIAP
         timestep secara independen → hasilnya adalah delta (koreksi).
      5. Prediksi akhir = sinyal Day 3 (persistence baseline) + delta yang dipelajari.
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
        self.head = nn.Sequential(
            nn.Conv1d(n_ch, n_ch // 2, 1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_ch // 2, n_features, 1)
        )

    def forward(self, x):
        feat = self.tcn(x.transpose(1, 2))          # (B, n_ch, 3*FUTURE)
        ctx  = self.pool(feat).squeeze(-1)           # (B, n_ch) global context
        last_day_feat = feat[:, :, -FUTURE:]         # (B, n_ch, FUTURE) fitur Day 3, selaras waktu
        fused = last_day_feat + ctx.unsqueeze(-1)    # FiLM-style: injeksi context global
        delta = self.head(fused).transpose(1, 2)     # (B, FUTURE, n_features)
        persistence = x[:, -FUTURE:, :]              # (B, FUTURE, n_features) sinyal Day 3 (scaled)
        pred = persistence + delta
        return pred, ctx

model     = TCNForecaster(n_features).to(device)
n_params  = sum(p.numel() for p in model.parameters())
print(f"[Stage 1] Total parameter model: {n_params:,}")
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8, verbose=True)
criterion = nn.MSELoss()

# =============================
# CHECKPOINT & RESUME (disederhanakan — early stopping state tidak di-resume)
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

log(f"\n{'='*60}\nSTAGE 1 TCN FORECASTER (IMPROVED v2) | {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"Train days: {n_train_days} | Test days: {n_test_days}")
log(f"Window fit: {len(X_fit)} | Window val(internal): {len(X_val)} | Window test: {len(X_seq_test_list)}")
log(f"Epoch: {N_EPOCHS} | Batch: {BATCH_SIZE} | Total parameter: {n_params:,}")
log(f"{'='*60}")

# =============================
# TRAINING + VALIDATION + EARLY STOPPING
# =============================
best_val   = float('inf')
best_state = None
bad_epochs = 0

if start_epoch <= N_EPOCHS:
    for epoch in range(start_epoch, N_EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        avg_train = total / len(dataloader)

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        scheduler.step(val_loss)

        log(f"Epoch {epoch:4d}/{N_EPOCHS} | Train MSE: {avg_train:.7f} | Val MSE: {val_loss:.7f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            p = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, p)
            log(f"   -> Checkpoint: {p}")

        if bad_epochs >= EARLY_STOP_PATIENCE:
            log(f"[EarlyStopping] Val MSE tidak membaik {EARLY_STOP_PATIENCE} epoch berturut-turut. Stop di epoch {epoch}. Best val MSE: {best_val:.7f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"[EarlyStopping] Restore ke bobot terbaik (Val MSE: {best_val:.7f})")
    log("=== STAGE 1 TRAINING SELESAI ===\n")

# =============================
# EVALUASI TRAINING SET (seluruh window fit+val, untuk perbandingan apple-to-apple dgn versi lama)
# =============================
log("=== EVALUASI TRAINING SET (Stage 1) - skala normalized [-0.1, 1.1] ===")
model.eval()
with torch.no_grad():
    X_all_train = torch.FloatTensor(X_scaled).to(device)
    pred_train, _ = model(X_all_train)
    pred_train    = pred_train.cpu().numpy()
y_scaled_np   = y_scaled
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
    log("=== EVALUASI TEST SET (Stage 1) - skala normalized [-0.1, 1.1] ===")
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
    log(f"Test MSE     : {mse:.6f}")
    log(f"Test RMSE    : {rmse:.6f}")
    log(f"Test MAE     : {mae:.6f}")
    log(f"Test MAPE    : {mape:.2f}%")
    log("=" * 40)
else:
    log("[Stage 1] Tidak ada test windows")

torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model_stage1_forecaster.pth"))
log("model_stage1_forecaster.pth disimpan")

# =============================
# PLOT KURVA LOSS (Train vs Val)
# =============================
_ep, _tr_l, _va_l = [], [], []
try:
    with open(LOG_FILE, 'r', encoding='utf-8') as _lf:
        for _line in _lf:
            _m = re.search(r'Epoch\s+(\d+)/\d+\s*\|\s*Train MSE:\s*([\d.]+)\s*\|\s*Val MSE:\s*([\d.]+)', _line)
            if _m:
                _ep.append(int(_m.group(1))); _tr_l.append(float(_m.group(2))); _va_l.append(float(_m.group(3)))
except Exception as _e:
    print(f"[Plot] Gagal parse log stage1: {_e}")

if _ep:
    _d = {}
    for _e, _t, _v in zip(_ep, _tr_l, _va_l): _d[_e] = (_t, _v)
    _es = sorted(_d.keys())
    _ts = [_d[e][0] for e in _es]
    _vs = [_d[e][1] for e in _es]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(_es, _ts, 'b-', linewidth=1.5, label='MSE Training (fit)')
    ax.plot(_es, _vs, 'r-', linewidth=1.5, label='MSE Validation (internal)')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(
        f"Kurva Epoch vs MSE — TCN Forecaster IMPROVED v2 ({max(_es)} epoch, early-stop-aware)\n"
        f"Train awal: {_ts[0]:.4f} -> akhir: {_ts[-1]:.4f} | Val awal: {_vs[0]:.4f} -> terbaik: {min(_vs):.4f}",
        fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _out = os.path.join(EVIDENCE_DIR, "kurva_loss_stage1.png")
    plt.savefig(_out, dpi=300, bbox_inches='tight')
    plt.close()
    log("kurva_loss_stage1.png disimpan")

# =============================
# INFO TXT — Stage 1
# =============================
_dates = [os.path.basename(f)[:8] for f in csv_files]
_lines = [
    "=" * 68,
    "INFORMASI — TCN FORECASTER STAGE 1 (IMPROVED v2)",
    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "=" * 68,
    "",
    "[PERBEDAAN UTAMA vs VERSI ASLI]",
    "  1. Decoder point-wise per-timestep (bukan global-avg-pool + Linear raksasa)",
    "  2. Residual/persistence connection (prediksi = Day3 + delta)",
    "  3. FiLM-style fusion context global ke fitur per-timestep",
    "  4. Validation split internal (4 window terakhir) + early stopping",
    f"  5. Total parameter model: {n_params:,}",
    "",
    "[DATASET & SPLIT]",
    f"  Total hari CSV               : {len(compressed_dfs)}",
    f"  Train days (TRAIN_RATIO=0.8) : {n_train_days}",
    f"  Test days                    : {n_test_days}",
    f"  Window training (total)      : {len(X_seq)}",
    f"  Window fit (gradient update)  : {len(X_fit)}",
    f"  Window validation (internal) : {len(X_val)}",
    f"  Window testing (held-out)    : {len(X_seq_test_list)}",
    "",
    "[HASIL AKHIR]",
    f"  Train MSE / RMSE / MAE / MAPE : {tr_mse:.6f} / {tr_rmse:.6f} / {tr_mae:.6f} / {tr_mape:.2f}%",
]
if X_scaled_test is not None:
    _lines.append(f"  Test  MSE / RMSE / MAE / MAPE : {mse:.6f} / {rmse:.6f} / {mae:.6f} / {mape:.2f}%")
_lines.append(f"  Best internal Val MSE          : {best_val:.6f}")

_out_txt = os.path.join(EVIDENCE_DIR, "info_stage1.txt")
with open(_out_txt, 'w', encoding='utf-8') as _f:
    _f.write('\n'.join(_lines))
log("info_stage1.txt disimpan")

print("\nSTAGE 1 (IMPROVED v2) SELESAI! Jalankan stage2_classifier.py berikutnya.")
