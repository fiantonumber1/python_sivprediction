# =============================
# STAGE 1 — TCN FORECASTER (IMPROVED v3 — ISOLATED DECODER TEST)
#
# TUJUAN v3: v2 gagal (kalah dari model lama di semua metrik). v2 menggabungkan
# 3 perubahan sekaligus (decoder baru + validation split + skip early-stop di
# akhir), sehingga tidak jelas bagian mana yang jadi penyebab kegagalan.
#
# v3 mengisolasi SATU variabel saja: DECODER-nya (point-wise + persistence +
# FiLM context, sama persis dengan v2). SEMUA hal lain dikembalikan PERSIS
# seperti model lama (original "Setup 100 epoch"):
#   - TIDAK ada validation split internal -- seluruh 26 window dipakai training
#     (identik protokol lama)
#   - Scheduler ReduceLROnPlateau memantau TRAIN loss (sama seperti lama),
#     patience=30 (sama seperti lama)
#   - TIDAK ada early stopping -- selalu jalan penuh 100 epoch
#   - Bobot yang dipakai = bobot epoch TERAKHIR (100), bukan best-checkpoint
#     (sama seperti lama)
#
# Kalau v3 MASIH kalah dari model lama -> decoder barunya sendiri yang bermasalah
# (bukan validation-split/early-stop tadi). Kalau v3 MENANG -> validation split
# di v2 kemarin yang jadi biang kegagalan (kehilangan 4 dari 26 window training
# itu ternyata terlalu mahal untuk dataset sekecil ini).
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
DATA_DIR             = os.path.join(SCRIPT_DIR, "..", "Setup 100 epoch", "data")  # referensi data asli (read-only)
CHECKPOINT_DIR      = os.path.join(SCRIPT_DIR, "checkpoints_stage1")
LOG_FILE            = os.path.join(SCRIPT_DIR, "log_stage1.txt")
EVIDENCE_DIR        = os.path.join(SCRIPT_DIR, "evidence")

N_EPOCHS            = 100
BATCH_SIZE          = 3
CHECKPOINT_INTERVAL = 50
TRAIN_RATIO         = 0.8
# ==================================================================

N_TAKE     = 200_000
FUTURE     = N_TAKE
START_TIME = time(3, 0, 0)
END_TIME                   = time(18, 16, 35)
N_DROP_FIRST               = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(os.cpu_count() or 1)
print(f"[Stage 1 v3] Device: {device} | CPU threads: {torch.get_num_threads()}")
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

print(f"[Stage 1 v3] Data dir : {DATA_DIR}")
print(f"[Stage 1 v3] File CSV : {len(csv_files)} ditemukan")

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

print(f"[Stage 1 v3] Total hari: {len(compressed_dfs)}")
if len(compressed_dfs) < 4:
    raise ValueError("Minimal 4 hari CSV!")

# =============================
# TRAIN / TEST SPLIT (kronologis) -- identik versi asli
# =============================
n_train_days = max(4, int(len(compressed_dfs) * TRAIN_RATIO))
n_test_days  = len(compressed_dfs) - n_train_days
train_dfs    = compressed_dfs[:n_train_days]
print(f"[Stage 1 v3] Train days: {n_train_days} | Test days: {n_test_days}")

# =============================
# SLIDING WINDOW: 3 hari -> 1 hari (hanya dari data training)
# =============================
X_seq, y_signal = [], []
for i in range(len(train_dfs) - 3):
    seq = np.concatenate([df[target_columns].values for df in train_dfs[i:i+3]], axis=0)
    X_seq.append(seq)
    y_signal.append(train_dfs[i+3][target_columns].values)

X_seq    = np.array(X_seq,    dtype=np.float32)
y_signal = np.array(y_signal, dtype=np.float32)
print(f"[Stage 1 v3] Window training: {len(X_seq)}")

X_seq_test_list, y_signal_test_list = [], []
for i in range(n_train_days - 3, len(compressed_dfs) - 3):
    seq = np.concatenate([df[target_columns].values for df in compressed_dfs[i:i+3]], axis=0)
    X_seq_test_list.append(seq)
    y_signal_test_list.append(compressed_dfs[i+3][target_columns].values)
print(f"[Stage 1 v3] Window testing : {len(X_seq_test_list)}")

scaler   = MinMaxScaler(feature_range=(-0.1, 1.1))
X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
y_scaled = scaler.transform(y_signal.reshape(-1, n_features)).reshape(y_signal.shape)
joblib.dump(scaler, os.path.join(SCRIPT_DIR, "scaler_stage1.pkl"))
print("[Stage 1 v3] scaler_stage1.pkl disimpan")

X_scaled_test      = None
y_signal_test_orig = None
if X_seq_test_list:
    X_seq_test         = np.array(X_seq_test_list,    dtype=np.float32)
    y_signal_test_orig = np.array(y_signal_test_list, dtype=np.float32)
    X_scaled_test      = scaler.transform(X_seq_test.reshape(-1, n_features)).reshape(X_seq_test.shape)

# TIDAK ADA validation split -- seluruh window training dipakai (identik lama)
X_tensor     = torch.FloatTensor(X_scaled).to(device)
y_sig_tensor = torch.FloatTensor(y_scaled).to(device)

class ForecastDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dataloader = DataLoader(ForecastDataset(X_tensor, y_sig_tensor),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# =============================
# MODEL TCN -- decoder SAMA PERSIS dengan v2 (point-wise + persistence + FiLM)
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
        feat = self.tcn(x.transpose(1, 2))
        ctx  = self.pool(feat).squeeze(-1)
        last_day_feat = feat[:, :, -FUTURE:]
        fused = last_day_feat + ctx.unsqueeze(-1)
        delta = self.head(fused).transpose(1, 2)
        persistence = x[:, -FUTURE:, :]
        pred = persistence + delta
        return pred, ctx

model     = TCNForecaster(n_features).to(device)
n_params  = sum(p.numel() for p in model.parameters())
print(f"[Stage 1 v3] Total parameter model: {n_params:,}")
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
# Scheduler monitor TRAIN loss, patience=30 -- PERSIS protokol model lama
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
        print(f"[Stage 1 v3] Resume epoch {start_epoch}")
    else:
        model.load_state_dict(torch.load(cp_path, map_location=device)['model'])
        print(f"[Stage 1 v3] Sudah selesai epoch {latest}")

def log(t):
    print(t)
    with open(LOG_FILE, 'a', encoding='utf-8') as f: f.write(t + '\n')

log(f"\n{'='*60}\nSTAGE 1 TCN FORECASTER (IMPROVED v3 - isolated decoder) | {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"Train days: {n_train_days} | Test days: {n_test_days}")
log(f"Window train: {len(X_tensor)} | Window test: {len(X_seq_test_list)}")
log(f"Epoch: {N_EPOCHS} | Batch: {BATCH_SIZE} | Total parameter: {n_params:,}")
log(f"{'='*60}")

# =============================
# TRAINING -- PERSIS protokol lama (tanpa val, tanpa early stop, full 100 epoch)
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
            log(f"   -> Checkpoint: {p}")
    log("=== STAGE 1 TRAINING SELESAI ===\n")

# =============================
# EVALUASI TRAINING SET
# =============================
log("=== EVALUASI TRAINING SET (Stage 1) - skala normalized [-0.1, 1.1] ===")
model.eval()
with torch.no_grad():
    pred_train, _ = model(X_tensor)
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
    log("[Stage 1 v3] Tidak ada test windows")

torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model_stage1_forecaster.pth"))
log("model_stage1_forecaster.pth disimpan")

# =============================
# PLOT KURVA LOSS
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
        f"Kurva Epoch vs MSE - TCN Forecaster IMPROVED v3 (isolated decoder, {max(_ep_sorted)} epoch)\n"
        f"MSE Awal: {_mse_sorted[0]:.4f} -> MSE Akhir: {_mse_sorted[-1]:.4f}",
        fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _out = os.path.join(EVIDENCE_DIR, "kurva_loss_stage1.png")
    plt.savefig(_out, dpi=300, bbox_inches='tight')
    plt.close()
    log("kurva_loss_stage1.png disimpan")

# =============================
# INFO TXT
# =============================
_lines = [
    "=" * 68,
    "INFORMASI - TCN FORECASTER STAGE 1 (IMPROVED v3, isolated decoder)",
    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "=" * 68,
    "",
    "[PERBEDAAN vs v2]",
    "  - TIDAK ada validation split internal (v2 menyisihkan 4/26 window)",
    "  - TIDAK ada early stopping (v2 patience 15 tapi dinonaktifkan manual)",
    "  - Scheduler monitor TRAIN loss, patience=30 (identik model lama)",
    "  - Bobot dipakai = epoch 100 (terakhir), bukan best-checkpoint",
    "  - Decoder arsitektur SAMA PERSIS dengan v2 (point-wise + persistence + FiLM)",
    f"  - Total parameter: {n_params:,}",
    "",
    "[HASIL AKHIR]",
    f"  Train MSE / RMSE / MAE / MAPE : {tr_mse:.6f} / {tr_rmse:.6f} / {tr_mae:.6f} / {tr_mape:.2f}%",
]
if X_scaled_test is not None:
    _lines.append(f"  Test  MSE / RMSE / MAE / MAPE : {mse:.6f} / {rmse:.6f} / {mae:.6f} / {mape:.2f}%")

_out_txt = os.path.join(EVIDENCE_DIR, "info_stage1.txt")
with open(_out_txt, 'w', encoding='utf-8') as _f:
    _f.write('\n'.join(_lines))
log("info_stage1.txt disimpan")

print("\nSTAGE 1 (IMPROVED v3) SELESAI!")
