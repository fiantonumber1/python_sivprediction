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
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==================================================================
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
DATA_DIR            = os.path.join(SCRIPT_DIR, "data")          # folder CSV terisolasi
CHECKPOINT_DIR      = os.path.join(SCRIPT_DIR, "checkpoints_stage1")
LOG_FILE            = os.path.join(SCRIPT_DIR, "log_stage1.txt")

N_EPOCHS            = 100
BATCH_SIZE          = 3
CHECKPOINT_INTERVAL = 50
COMPRESSION_FACTOR  = 1
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
# SLIDING WINDOW: 3 hari → 1 hari (semua data dipakai training)
# =============================
X_seq, y_signal = [], []
for i in range(len(compressed_dfs) - 3):
    seq = np.concatenate([df[target_columns].values for df in compressed_dfs[i:i+3]], axis=0)
    X_seq.append(seq)
    y_signal.append(compressed_dfs[i+3][target_columns].values)

X_seq    = np.array(X_seq,    dtype=np.float32)
y_signal = np.array(y_signal, dtype=np.float32)
print(f"[Stage 1] Window training: {len(X_seq)}")

scaler   = MinMaxScaler(feature_range=(-0.1, 1.1))
X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
y_scaled = scaler.transform(y_signal.reshape(-1, n_features)).reshape(y_signal.shape)
joblib.dump(scaler, os.path.join(SCRIPT_DIR, "scaler_stage1.pkl"))
print("[Stage 1] scaler_stage1.pkl disimpan")

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
log(f"Window: {len(X_tensor)} | Epoch: {N_EPOCHS} | Batch: {BATCH_SIZE}\n{'='*60}")

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
    log("=== STAGE 1 SELESAI ===\n")

torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model_stage1_forecaster.pth"))
log("model_stage1_forecaster.pth disimpan")
print("\nSTAGE 1 SELESAI! Jalankan stage2_classifier.py berikutnya.")
