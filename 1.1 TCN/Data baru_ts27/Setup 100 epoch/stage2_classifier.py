# =============================
# STAGE 2 — MLP CLASSIFIER
# Per hari: Day1→Status1, Day2→Status2, dst (independen dari Stage 1)
# Input   : ringkasan 21 param sensor 1 hari (mean + std + max + min)
# Output  : model_stage2_classifier.pth
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
CHECKPOINT_DIR      = os.path.join(SCRIPT_DIR, "checkpoints_stage2")
LOG_FILE            = os.path.join(SCRIPT_DIR, "log_stage2.txt")

N_EPOCHS            = 100
BATCH_SIZE          = 3
CHECKPOINT_INTERVAL = 50
COMPRESSION_FACTOR  = 1
# ==================================================================

N_TAKE                    = 200_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
START_TIME                 = time(3, 0, 0)
END_TIME                   = time(18, 16, 35)
N_DROP_FIRST               = 3600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Stage 2] Device: {device}")
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

# Fitur per hari = mean + std + max + min untuk setiap parameter → 21×4 = 84 dim
N_STATS   = 4
INPUT_DIM = n_features * N_STATS  # 84

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

print(f"[Stage 2] Data dir : {DATA_DIR}")
print(f"[Stage 2] File CSV : {len(csv_files)} ditemukan")

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

def compress_day(df_raw):
    chunks, ts_mid = [], []
    for i in range(COMPRESSED_POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunks.append(df_raw[target_columns + fault_columns].iloc[s:e].mean())
        ts_mid.append(df_raw['ts_date'].iloc[s + COMPRESSION_FACTOR // 2])
    df_c = pd.DataFrame(chunks, columns=target_columns + fault_columns)
    df_c.insert(0, 'ts_date', ts_mid)
    return df_c

def day_to_feature_vector(df_day):
    vals = df_day[target_columns].values
    feat = np.concatenate([
        vals.mean(axis=0),
        vals.std(axis=0),
        vals.max(axis=0),
        vals.min(axis=0),
    ])
    return feat.astype(np.float32)

# =============================
# LABELING: per hari
# 0 fault aktif = Sehat, 1 fault aktif = Warning, >=2 fault aktif = Not Ready
# =============================
def label_day(df_day):
    n_active = sum(1 for col in fault_columns
                   if col in df_day.columns and (df_day[col] > 0).any())
    if n_active == 0:   return 0  # Sehat
    elif n_active == 1: return 1  # Warning
    else:               return 2  # Not Ready

compressed_dfs = []
for f in csv_files:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"  Skip {os.path.basename(f)}")
        continue
    compressed_dfs.append(compress_day(df_raw))

total_days = len(compressed_dfs)
print(f"[Stage 2] Total hari: {total_days}")
if total_days < 2:
    raise ValueError("Minimal 2 hari CSV!")

# Setiap hari → 1 sampel fitur + 1 label (semua data dipakai training)
X_cls, y_cls = [], []
status_map = {0: "Sehat", 1: "Warning", 2: "Not Ready"}
for i, df_day in enumerate(compressed_dfs):
    feat  = day_to_feature_vector(df_day)
    label = label_day(df_day)
    X_cls.append(feat)
    y_cls.append(label)
    print(f"  Day {i+1:2d} -> {status_map[label]}")

X_cls = np.array(X_cls, dtype=np.float32)
y_cls = np.array(y_cls, dtype=np.int64)
print(f"[Stage 2] Sampel training: {len(X_cls)} hari")

# Normalisasi fitur statistik
scaler_cls = MinMaxScaler(feature_range=(-0.1, 1.1))
X_scaled   = scaler_cls.fit_transform(X_cls)
joblib.dump(scaler_cls, os.path.join(SCRIPT_DIR, "scaler_stage2.pkl"))
print("[Stage 2] scaler_stage2.pkl disimpan")

X_tensor = torch.FloatTensor(X_scaled).to(device)
y_tensor = torch.LongTensor(y_cls).to(device)

class ClassDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dataloader = DataLoader(ClassDataset(X_tensor, y_tensor),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# =============================
# MODEL MLP CLASSIFIER
# =============================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=84, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

model     = MLPClassifier(INPUT_DIM).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30, verbose=True)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.5, 4.0]).to(device))

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
        print(f"[Stage 2] Resume epoch {start_epoch}")
    else:
        model.load_state_dict(torch.load(cp_path, map_location=device)['model'])
        print(f"[Stage 2] Sudah selesai epoch {latest}")

def log(t):
    print(t)
    with open(LOG_FILE, 'a', encoding='utf-8') as f: f.write(t + '\n')

log(f"\n{'='*60}\nSTAGE 2 MLP CLASSIFIER | {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"Sampel: {len(X_tensor)} hari | Epoch: {N_EPOCHS} | Batch: {BATCH_SIZE}")
log(f"Distribusi: Sehat={sum(y_cls==0)} Warning={sum(y_cls==1)} Not Ready={sum(y_cls==2)}")
log(f"{'='*60}")

# =============================
# TRAINING
# =============================
if start_epoch <= N_EPOCHS:
    model.train()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        total, correct, samples = 0.0, 0, 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total   += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            samples += y.size(0)
        avg = total / len(dataloader)
        acc = 100.0 * correct / samples
        scheduler.step(avg)
        log(f"Epoch {epoch:4d}/{N_EPOCHS} | CE: {avg:.6f} | Acc: {acc:6.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            p = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, p)
            log(f"   → Checkpoint: {p}")
    log("=== STAGE 2 SELESAI ===\n")

torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model_stage2_classifier.pth"))
log("model_stage2_classifier.pth disimpan")
print("\nSTAGE 2 SELESAI! Jalankan stage3_inference.py untuk prediksi.")
