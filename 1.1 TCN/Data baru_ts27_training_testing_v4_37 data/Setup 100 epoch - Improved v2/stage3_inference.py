# =============================
# STAGE 3 — INFERENCE (IMPROVED v2, disederhanakan)
# Menjalankan 2 skenario seperti versi asli:
#   A) Training Inference : Day (n_train-2..n_train) -> prediksi Day (n_train+1)
#      -> ada ground truth -> MSE/RMSE/MAE bisa dihitung
#   B) Testing Inference  : 3 hari TERAKHIR dataset -> prediksi hari berikutnya
#      -> tidak ada ground truth -> hanya confidence
#
# CATATAN: versi ini SENGAJA jauh lebih ringkas daripada stage3_inference.py
# original (57KB, banyak plot untuk kebutuhan jurnal). Fokus v2 adalah
# memverifikasi bahwa model IMPROVED (arsitektur baru) bekerja end-to-end dan
# menghasilkan metrik yang bisa dibandingkan apple-to-apple dengan versi asli.
# Class TCNForecaster & MLPClassifier di bawah HARUS identik dengan yang dipakai
# di stage1_forecaster.py / stage2_classifier.py (supaya state_dict cocok).
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "..", "Setup 100 epoch", "data")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

TRAIN_RATIO = 0.8
N_TAKE      = 200_000
FUTURE      = N_TAKE
START_TIME  = time(3, 0, 0)
END_TIME    = time(18, 16, 35)

N_FEATURES  = 21
N_CH        = 96
N_STATS     = 4
INPUT_DIM_2 = N_FEATURES * N_STATS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Inference v2] Device: {device}")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
status_map = {0: "Healthy", 1: "Warning"}

MODEL_STAGE1  = os.path.join(BASE_DIR, "model_stage1_forecaster.pth")
SCALER_STAGE1 = os.path.join(BASE_DIR, "scaler_stage1.pkl")
MODEL_STAGE2  = os.path.join(BASE_DIR, "model_stage2_classifier.pth")
SCALER_STAGE2 = os.path.join(BASE_DIR, "scaler_stage2.pkl")

for fpath, label in [
    (MODEL_STAGE1,  "model_stage1_forecaster.pth"),
    (SCALER_STAGE1, "scaler_stage1.pkl"),
    (MODEL_STAGE2,  "model_stage2_classifier.pth"),
    (SCALER_STAGE2, "scaler_stage2.pkl"),
]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"'{label}' tidak ditemukan di {fpath}. Jalankan stage1 & stage2 dulu.")

# =============================
# MODEL DEFS — HARUS identik dengan stage1_forecaster.py & stage2_classifier.py
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

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=84, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),        nn.LayerNorm(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

# =============================
# LOAD MODELS
# =============================
tcn_model = TCNForecaster(N_FEATURES, N_CH).to(device)
tcn_model.load_state_dict(torch.load(MODEL_STAGE1, map_location=device))
tcn_model.eval()

mlp_state_dict_all = torch.load(MODEL_STAGE2, map_location=device)
mlp_folds = []
for key, sd in mlp_state_dict_all.items():
    m = MLPClassifier(INPUT_DIM_2).to(device)
    m.load_state_dict(sd)
    m.eval()
    mlp_folds.append(m)
print(f"[Inference v2] MLP ensemble: {len(mlp_folds)} fold model dimuat")

def mlp_ensemble_predict(X_np):
    X_t = torch.FloatTensor(X_np).to(device)
    probs_sum = None
    with torch.no_grad():
        for m in mlp_folds:
            probs = torch.softmax(m(X_t), dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / len(mlp_folds)).cpu().numpy()

scaler1 = joblib.load(SCALER_STAGE1)
scaler2 = joblib.load(SCALER_STAGE2)

# =============================
# BACA DATA (sama seperti stage1/stage2)
# =============================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if "hasil" not in os.path.basename(f).lower()
    and "prediksi" not in os.path.basename(f).lower()
    and "inference" not in os.path.basename(f).lower()
], key=extract_date)

_needed_cols = set(['ts_date'] + target_columns + fault_columns)

def read_and_crop(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';', decimal=',',
                      usecols=lambda c: c.strip() in _needed_cols,
                      low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'),
                                    format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['ts_date'])
    for col in target_columns + fault_columns:
        if col not in df.columns: df[col] = np.nan
        else: df[col] = pd.to_numeric(df[col], errors='coerce')
    df[target_columns + fault_columns] = df[target_columns + fault_columns].ffill().bfill()
    date0 = df['ts_date'].dt.date.iloc[0]
    df = df[(df['ts_date'] >= datetime.combine(date0, START_TIME)) & (df['ts_date'] <= datetime.combine(date0, END_TIME))]
    if len(df) < N_TAKE: return pd.DataFrame()
    return df.iloc[:N_TAKE].reset_index(drop=True)[['ts_date'] + target_columns + fault_columns]

compressed_dfs, dates = [], []
for f in csv_files:
    d = read_and_crop(f)
    if d.empty: continue
    compressed_dfs.append(d)
    dates.append(os.path.basename(f)[:8])

total_days = len(compressed_dfs)
n_train_days = max(4, int(total_days * TRAIN_RATIO))

def label_day(df_day):
    n_active = sum(1 for col in fault_columns if col in df_day.columns and (df_day[col] > 0).any())
    return 0 if n_active == 0 else 1

def day_to_feature_vector(df_day):
    vals = df_day[target_columns].values
    return np.concatenate([vals.mean(0), vals.std(0), vals.max(0), vals.min(0)]).astype(np.float32)

def run_inference(day_a, day_b, day_c, label="Scenario"):
    seq = np.concatenate([df[target_columns].values for df in [day_a, day_b, day_c]], axis=0).astype(np.float32)
    seq_sc = scaler1.transform(seq)
    with torch.no_grad():
        pred_sc, _ = tcn_model(torch.FloatTensor(seq_sc).unsqueeze(0).to(device))
    pred_sc_np = pred_sc.squeeze(0).cpu().numpy()   # (FUTURE, 21), skala normalized
    pred_orig  = scaler1.inverse_transform(pred_sc_np)

    fake_df = pd.DataFrame(pred_orig, columns=target_columns)
    feat = day_to_feature_vector(fake_df).reshape(1, -1)
    feat_sc = scaler2.transform(feat)
    probs = mlp_ensemble_predict(feat_sc)[0]
    pred_label = int(np.argmax(probs))
    return pred_sc_np, pred_orig, pred_label, probs

results = []

# --- A) Training Inference: Day(n_train-2..n_train) -> prediksi Day(n_train+1) ---
if n_train_days + 1 <= total_days:
    a, b, c = compressed_dfs[n_train_days-3], compressed_dfs[n_train_days-2], compressed_dfs[n_train_days-1]
    pred_sc_np, pred_orig, pred_label, probs = run_inference(a, b, c, "Training Inference")
    target_day = compressed_dfs[n_train_days]
    y_true_sc = scaler1.transform(target_day[target_columns].values.astype(np.float32))
    mse  = float(np.mean((pred_sc_np - y_true_sc) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(pred_sc_np - y_true_sc)))
    true_label = label_day(target_day)
    results.append((
        "TRAINING INFERENCE (ada ground truth)",
        dates[n_train_days-3], dates[n_train_days-2], dates[n_train_days-1], dates[n_train_days],
        pred_label, true_label, probs, mse, rmse, mae
    ))

# --- B) Testing Inference: 3 hari terakhir dataset -> prediksi hari depan (tanpa ground truth) ---
a, b, c = compressed_dfs[-3], compressed_dfs[-2], compressed_dfs[-1]
pred_sc_np, pred_orig, pred_label, probs = run_inference(a, b, c, "Testing Inference")
results.append((
    "TESTING INFERENCE (tanpa ground truth, prediksi masa depan)",
    dates[-3], dates[-2], dates[-1], "hari ke-" + str(total_days+1),
    pred_label, None, probs, None, None, None
))

# =============================
# TULIS HASIL
# =============================
lines = ["=" * 68, "HASIL INFERENCE — IMPROVED v2", f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 68, ""]
for label, da, db, dc, dtarget, pred_label, true_label, probs, mse, rmse, mae in results:
    lines.append(f"[{label}]")
    lines.append(f"  Input : {da}, {db}, {dc}")
    lines.append(f"  Target: {dtarget}")
    lines.append(f"  Prediksi Status : {status_map[pred_label]} (confidence {probs[pred_label]*100:.2f}%)")
    lines.append(f"  Prob Healthy/Warning: {probs[0]*100:.2f}% / {probs[1]*100:.2f}%")
    if true_label is not None:
        match = "BENAR" if pred_label == true_label else "SALAH"
        lines.append(f"  Status Aktual   : {status_map[true_label]}  -> {match}")
        lines.append(f"  MSE / RMSE / MAE (skala normalized): {mse:.6f} / {rmse:.6f} / {mae:.6f}")
    else:
        lines.append("  Ground Truth    : tidak tersedia (prediksi masa depan)")
    lines.append("")

out_path = os.path.join(EVIDENCE_DIR, "hasil_inference.txt")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print(f"\n[Inference v2] Hasil disimpan: {out_path}")

# =============================
# PLOT sederhana: 1 parameter representatif (training inference), downsampled
# =============================
if results and results[0][8] is not None:  # ada training inference
    a, b, c = compressed_dfs[n_train_days-3], compressed_dfs[n_train_days-2], compressed_dfs[n_train_days-1]
    seq = np.concatenate([df[target_columns].values for df in [a, b, c]], axis=0).astype(np.float32)
    seq_sc = scaler1.transform(seq)
    with torch.no_grad():
        pred_sc, _ = tcn_model(torch.FloatTensor(seq_sc).unsqueeze(0).to(device))
    pred_orig = scaler1.inverse_transform(pred_sc.squeeze(0).cpu().numpy())
    target_day = compressed_dfs[n_train_days]
    true_vals = target_day[target_columns].values

    param_idx = target_columns.index('SIV_T_Container')
    ds = 200
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(true_vals[::ds, param_idx], label='Aktual', color='#2196F3')
    ax.plot(pred_orig[::ds, param_idx], label='Prediksi', color='#FF9800', linestyle='--')
    ax.set_title(f"Training Inference — {target_columns[param_idx]} (Aktual vs Prediksi, IMPROVED v2)")
    ax.set_xlabel(f"Timestep (downsample {ds}x)"); ax.set_ylabel(target_columns[param_idx])
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCE_DIR, "plot_training_inference_contoh.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("[Inference v2] plot_training_inference_contoh.png disimpan")

print("\nSTAGE 3 (IMPROVED v2) SELESAI!")
