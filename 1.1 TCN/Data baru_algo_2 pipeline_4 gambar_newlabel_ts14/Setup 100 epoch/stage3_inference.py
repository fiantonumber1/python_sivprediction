# =============================
# STAGE 3 — INFERENCE PIPELINE
# CSV VERSION + 4 GAMBAR (identik logika Data Lama)
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# =========================================================
# BASE DIR & PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # folder CSV terisolasi

# =========================================================
# CONFIG
# =========================================================
COMPRESSION_FACTOR         = 1
PLOT_DOWNSAMPLE            = 10      # ambil setiap N-th point untuk plot (hemat memori)
N_TAKE                     = 150_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
FUTURE                     = COMPRESSED_POINTS_PER_DAY

START_TIME   = time(6, 0, 0)
END_TIME     = time(18, 16, 35)
N_DROP_FIRST = 3600

N_FEATURES  = 21
N_CH        = 96
N_STATS     = 4
INPUT_DIM_2 = N_FEATURES * N_STATS   # 84

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Inference] Device: {device}")

# =========================================================
# TARGET & FAULT COLUMNS
# =========================================================
target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']

status_map   = {0: "Sehat", 1: "Warning", 2: "Not Ready"}
status_color = {0: "green", 1: "orange", 2: "red"}

# =========================================================
# CEK FILE MODEL
# =========================================================
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
        raise FileNotFoundError(
            f"\nFile '{label}' tidak ditemukan di:\n{fpath}\n"
            f"Pastikan training stage1 (forecast/) dan stage2 (classifier/) sudah selesai."
        )

# =========================================================
# AUTO AMBIL CSV — sort berdasarkan tanggal di nama file
# =========================================================
def extract_date(f):
    try:
        return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")
    except Exception:
        return datetime.min

all_csv_files = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if "inference" not in os.path.basename(f).lower()
    and "prediksi"  not in os.path.basename(f).lower()
    and "hasil"     not in os.path.basename(f).lower()
], key=extract_date)

print("\n[DEBUG] CSV ditemukan:")
for f in all_csv_files:
    print("  ", os.path.basename(f))

if len(all_csv_files) < 4:
    raise ValueError(
        f"Minimal harus ada 4 file CSV (untuk gambar1 identik Data Lama)!\n"
        f"Ditemukan: {len(all_csv_files)}"
    )

# =========================================================
# MODEL DEFINITION
# =========================================================
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
        self.c1   = CausalConv1d(in_ch, out_ch, ks, dilation)
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
    def __init__(self, n_features=21, n_ch=96, ks=3, n_blocks=7, dropout=0.3):
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
        ctx  = self.pool(self.tcn(x.transpose(1, 2))).squeeze(-1)
        n_features = x.shape[-1]
        pred = self.dec(ctx).view(-1, FUTURE, n_features)
        return pred, ctx

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

# =========================================================
# LOAD MODEL & SCALER
# =========================================================
scaler = joblib.load(SCALER_STAGE1)    # identik dgn scaler Data Lama
scaler2 = joblib.load(SCALER_STAGE2)
print("[Inference] Scaler loaded")

tcn_model = TCNForecaster(N_FEATURES, N_CH).to(device)
tcn_model.load_state_dict(torch.load(MODEL_STAGE1, map_location=device))
tcn_model.eval()

mlp_model = MLPClassifier(INPUT_DIM_2).to(device)
mlp_model.load_state_dict(torch.load(MODEL_STAGE2, map_location=device))
mlp_model.eval()
print("[Inference] Model TCN dan MLP loaded")

# =========================================================
# READ CSV FUNCTION
# =========================================================
def read_csv_day(filepath):
    print(f"\n  Membaca: {os.path.basename(filepath)}")

    with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
        first_line = f.readline()
    sep = ';' if first_line.count(';') > first_line.count(',') else ','
    print(f"    Separator: '{sep}'")

    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', sep=sep, engine='python', on_bad_lines='skip')
    except Exception as e:
        raise ValueError(f"\nGagal membaca CSV:\n{filepath}\n\n{str(e)}")

    df.columns = [str(c).strip() for c in df.columns]

    # Cari kolom timestamp
    ts_col = None
    for c in df.columns:
        c_lower = c.lower()
        if 'date' in c_lower or 'time' in c_lower or 'timestamp' in c_lower or 'ts' in c_lower:
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]

    df['ts_date'] = pd.to_datetime(
        df[ts_col].astype(str).str.replace(',', '.'), errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])

    for col in target_columns + fault_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan
    df[target_columns + fault_columns] = df[target_columns + fault_columns].ffill().bfill()

    date0 = df['ts_date'].dt.date.iloc[0]
    df = df[
        (df['ts_date'] >= datetime.combine(date0, START_TIME)) &
        (df['ts_date'] <= datetime.combine(date0, END_TIME))
    ]

    min_required = int(N_DROP_FIRST + N_TAKE * 0.8)
    if len(df) < min_required:
        raise ValueError(f"Data terlalu sedikit: {len(df)} rows (min {min_required})")

    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)

    chunks, ts_mid = [], []
    n_pts = min(COMPRESSED_POINTS_PER_DAY, len(df) // max(COMPRESSION_FACTOR, 1))
    for i in range(n_pts):
        s = i * COMPRESSION_FACTOR
        e = (i + 1) * COMPRESSION_FACTOR
        chunks.append(df[target_columns + fault_columns].iloc[s:e].mean())
        mid_idx = min(s + COMPRESSION_FACTOR // 2, len(df) - 1)
        ts_mid.append(df['ts_date'].iloc[mid_idx])

    df_c = pd.DataFrame(chunks, columns=target_columns + fault_columns)
    df_c.insert(0, 'ts_date', ts_mid)
    print(f"    → {len(df_c)} titik terkompresi")
    return df_c

# =========================================================
# HEALTH STATUS LABELING (sama persis Data Lama)
# =========================================================
def label_health_status(df_day):
    n_active = sum(1 for col in fault_columns
                   if col in df_day.columns and (df_day[col] > 0).any())
    if n_active == 0:   return 0  # Sehat
    elif n_active == 1: return 1  # Warning
    else:               return 2  # Not Ready

# =========================================================
# LOAD SEMUA CSV
# =========================================================
print("\n[Inference] Membaca semua CSV...")
compressed_dfs = []
for f in all_csv_files:
    try:
        df_c = read_csv_day(f)
        compressed_dfs.append(df_c)
    except Exception as e:
        print(f"  Skip {os.path.basename(f)}: {e}")

total_days = len(compressed_dfs)
print(f"\n[Inference] Total hari valid: {total_days}")

if total_days < 4:
    raise ValueError("Minimal 4 hari data valid!")

# Health status tiap hari (sama persis Data Lama)
health_status = []
for i, df in enumerate(compressed_dfs):
    stat = label_health_status(df)
    health_status.append(stat)
    print(f"  Day {i+1:2d} → {status_map[stat]}")

# 3 hari terakhir sebagai input inference
CSV_DAY_A = os.path.basename(all_csv_files[-3])
CSV_DAY_B = os.path.basename(all_csv_files[-2])
CSV_DAY_C = os.path.basename(all_csv_files[-1])
df_A = compressed_dfs[-3]
df_B = compressed_dfs[-2]
df_C = compressed_dfs[-1]

print(f"\n[Inference] Input: {CSV_DAY_A} + {CSV_DAY_B} + {CSV_DAY_C}")

# =========================================================
# STAGE 1 — TCN FORECASTER
# =========================================================
print("\n[Stage 1] TCN Forecaster")
seq_raw = np.concatenate([
    df_A[target_columns].values,
    df_B[target_columns].values,
    df_C[target_columns].values,
], axis=0).astype(np.float32)

seq_scaled = scaler.transform(seq_raw.reshape(-1, N_FEATURES)).reshape(seq_raw.shape)
x_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)

with torch.no_grad():
    pred_scaled, ctx = tcn_model(x_tensor)
    pred_scaled = pred_scaled.cpu().numpy()[0]

pred_signal = scaler.inverse_transform(
    pred_scaled.reshape(-1, N_FEATURES)
).reshape(FUTURE, N_FEATURES)

print(f"  → Shape prediksi: {pred_signal.shape}")

# =========================================================
# STAGE 2 — MLP CLASSIFIER
# =========================================================
print("\n[Stage 2] MLP Classifier")
feat = np.concatenate([
    pred_signal.mean(axis=0),
    pred_signal.std(axis=0),
    pred_signal.max(axis=0),
    pred_signal.min(axis=0),
]).astype(np.float32)

feat_scaled = scaler2.transform(feat.reshape(1, -1))
feat_tensor = torch.FloatTensor(feat_scaled).to(device)

with torch.no_grad():
    logits = mlp_model(feat_tensor)
    prob   = torch.softmax(logits, dim=1).cpu().numpy()[0]

pred_status     = int(np.argmax(prob))
pred_confidence = float(prob[pred_status] * 100)

print("\n" + "="*60)
print("HASIL PREDIKSI HARI D")
print("="*60)
print(f"Input  : {CSV_DAY_A} + {CSV_DAY_B} + {CSV_DAY_C}")
print(f"Status : {status_map[pred_status]}  ({pred_confidence:.2f}% confidence)")
print(f"Prob Sehat     : {prob[0]*100:.2f}%")
print(f"Prob Warning   : {prob[1]*100:.2f}%")
print(f"Prob Not Ready : {prob[2]*100:.2f}%")
print("="*60)

# =========================================================
# PLOT — IDENTIK LOGIKA DATA LAMA
#
# Kunci: setiap parameter dinormalisasi PER KOLOM
# sehingga rentang berbeda (mis. Voltage vs Current)
# tetap muncul semua dalam skala [0-1] atau [-0.1, 1.1]
#
# plot_all → per-kolom min-max dari data REAL saja (bukan termasuk pred)
#            → identik Data Lama
# gambar1,2,3 → pakai scaler training (fit pada data real)
#               → identik Data Lama
# =========================================================

# --- Helper normalisasi per kolom (untuk plot_all) ---
# SAMA PERSIS dengan Data Lama:
#   mn, mx = col.min(), col.max()
#   normalized = (col - mn) / (mx - mn)   jika mx-mn > 1e-8
#   else 0
def normalize_per_col_data_lama(df_concat):
    norm = df_concat[target_columns].copy()
    for col in target_columns:
        mn, mx = norm[col].min(), norm[col].max()
        if mx - mn > 1e-8:
            norm[col] = (norm[col] - mn) / (mx - mn)
        else:
            norm[col] = 0
    return norm

# --- Helper scaler transform (untuk gambar 1,2,3) ---
def scale_arr(arr):
    return scaler.transform(arr.reshape(-1, N_FEATURES)).reshape(arr.shape)

# =========================================================
# PLOT_ALL — semua hari real, per-kolom min-max
# IDENTIK Data Lama (hanya data real, tidak ada pred di sini)
# =========================================================
print("\n[Plot] plot_all_parameters_TCN.png ...")

df_all  = pd.concat(compressed_dfs, ignore_index=True)
norm_all = normalize_per_col_data_lama(df_all)   # per-kolom [0,1]

PPD_ds = COMPRESSED_POINTS_PER_DAY // PLOT_DOWNSAMPLE   # titik per hari setelah downsample
norm_ds = norm_all.iloc[::PLOT_DOWNSAMPLE].reset_index(drop=True)
x = np.arange(len(norm_ds))
fig, ax = plt.subplots(figsize=(24, 10))
for col in target_columns:
    ax.plot(x, norm_ds[col], linewidth=0.9, alpha=0.7)

day_bounds = np.arange(0, (total_days + 1) * PPD_ds, PPD_ds)
for b in day_bounds[1:-1]:
    ax.axvline(b, color='red', linestyle='--', alpha=0.8)
mid_points = [(day_bounds[i] + day_bounds[i+1]) // 2 for i in range(total_days)]
for i, mid in enumerate(mid_points):
    ax.text(mid, 1.05, f'Day {i+1}', ha='center', color='red', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(mid, 1.15, status_map[health_status[i]], ha='center',
            color=['green', 'orange', 'red'][health_status[i]], fontweight='bold',
            transform=ax.get_xaxis_transform())

ax.set_title(f"21 Parameter + Health Status (TCN Pipeline) - {total_days} Hari", fontsize=16)
ax.set_ylabel("Normalized [0-1]")
ax.grid(alpha=0.3)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plot_all_parameters_TCN.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → plot_all_parameters_TCN.png")

# =========================================================
# Persiapan gambar 1, 2, 3
# Gunakan 4 HARI REAL TERAKHIR — identik Data Lama
# =========================================================
last4_dfs = compressed_dfs[-4:]
df4       = pd.concat(last4_dfs, ignore_index=True)
real4     = df4[target_columns].values

# Gambar 2 & 3: training scaler (skala sama untuk real vs prediksi)
norm4_sc   = scale_arr(real4)          # scaler training → [-0.1, 1.1]
pred_norm  = scale_arr(pred_signal)    # scaler training → [-0.1, 1.1]

# Gambar 1: pakai min/max dari semua 12 hari (sama dengan plot_all) → apple to apple
norm4_g1   = norm_all.iloc[-4 * COMPRESSED_POINTS_PER_DAY:].values  # (4*PPD, 21)

PPD_ds = COMPRESSED_POINTS_PER_DAY // PLOT_DOWNSAMPLE   # titik per hari setelah downsample

# Downsample semua array untuk plot
norm4_sc_ds  = norm4_sc[::PLOT_DOWNSAMPLE]              # (4*PPD_ds, 21)
norm4_g1_ds  = norm4_g1[::PLOT_DOWNSAMPLE]              # (4*PPD_ds, 21)
pred_norm_ds = pred_norm[::PLOT_DOWNSAMPLE]              # (PPD_ds, 21)
x4_ds = np.arange(4 * PPD_ds)
n3_ds = 3 * PPD_ds

# setup_plot — pakai PPD_ds (downsampled)
# override_last: jika diisi int (0/1/2), hari ke-4 (slot prediksi) pakai status ini
def setup_plot(ax, title, ylim_low=-0.1, override_last=None):
    bounds = np.arange(0, 5 * PPD_ds, PPD_ds)
    for b in bounds[1:]:
        if b < len(x4_ds):
            ax.axvline(b, color='red', linestyle='--', alpha=0.8)
    mids = [(bounds[i] + bounds[i+1]) // 2 for i in range(4)]
    for i, m in enumerate(mids):
        day_idx = total_days - 4 + i
        # hari ke-4 (i==3): pakai pred_status jika override_last diberikan
        stat = override_last if (i == 3 and override_last is not None) else health_status[day_idx]
        label_txt = status_map[stat]
        if i == 3 and override_last is not None:
            label_txt = f"Pred: {status_map[stat]}"
        ax.text(m, 1.05, f'Day {day_idx+1}', ha='center', color='red', fontweight='bold',
                transform=ax.get_xaxis_transform())
        ax.text(m, 1.15, label_txt, ha='center',
                color=['green', 'orange', 'red'][stat], fontweight='bold',
                transform=ax.get_xaxis_transform())
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.3)
    ax.set_ylim(ylim_low, 1.3)

# =========================================================
# GAMBAR 1 — 4 Hari Real Data + Health Status
# Normalisasi per-kolom min-max (konsisten dengan plot_all)
# =========================================================
print("[Plot] gambar1_4hari_real_TCN.png ...")

fig, ax = plt.subplots(figsize=(24, 10))
for i, col in enumerate(target_columns):
    ax.plot(x4_ds, norm4_g1_ds[:, i], linewidth=1, alpha=0.8)
setup_plot(ax, "GAMBAR 1: 4 Hari Real Data + Health Status (TCN Pipeline)", ylim_low=0)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "gambar1_4hari_real_TCN.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → gambar1_4hari_real_TCN.png")

# =========================================================
# GAMBAR 2 — 3 Hari Input + 1 Hari Prediksi
# Training scaler → skala real dan prediksi konsisten
# =========================================================
print("[Plot] gambar2_input_plus_prediksi_TCN.png ...")

fig, ax = plt.subplots(figsize=(24, 10))
for i, col in enumerate(target_columns):
    ax.plot(x4_ds[:n3_ds], norm4_sc_ds[:n3_ds, i], linewidth=1.2, label=col)
for i, col in enumerate(target_columns):
    color = ax.get_lines()[i].get_color()
    ax.plot(x4_ds[n3_ds:], pred_norm_ds[:, i], '--', linewidth=2.8, color=color, alpha=0.95)
setup_plot(ax, "GAMBAR 2: 3 Hari Input + 1 Hari Prediksi (TCN Pipeline)", override_last=pred_status)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "gambar2_input_plus_prediksi_TCN.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → gambar2_input_plus_prediksi_TCN.png")

# =========================================================
# GAMBAR 3 — Real vs Prediksi Hari Terakhir
# Training scaler → skala real dan prediksi konsisten
#   • 3 hari input → solid tipis
#   • real day 4 → solid tebal
#   • prediksi day 4 → dashed tebal (overlay)
# =========================================================
print("[Plot] gambar3_real_vs_prediksi_TCN.png ...")

fig, ax = plt.subplots(figsize=(24, 10))
# 3 hari input (solid)
for i, col in enumerate(target_columns):
    ax.plot(x4_ds[:n3_ds], norm4_sc_ds[:n3_ds, i], linewidth=1.2)
# Real day 4 (solid tebal)
for i, col in enumerate(target_columns):
    ax.plot(x4_ds[n3_ds:], norm4_sc_ds[n3_ds:, i], linewidth=1.8, alpha=0.9)
# Prediksi (dashed tebal, warna sama)
for i, col in enumerate(target_columns):
    color = ax.get_lines()[i].get_color()
    ax.plot(x4_ds[n3_ds:], pred_norm_ds[:, i], '--', linewidth=3, color=color, alpha=0.95,
            label=f'Pred {col}' if i == 0 else None)
setup_plot(ax, "GAMBAR 3: Real vs Prediksi Hari Terakhir (TCN Pipeline)")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "gambar3_real_vs_prediksi_TCN.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → gambar3_real_vs_prediksi_TCN.png")

# =========================================================
# SAVE CSV
# =========================================================
result_sensor = pd.DataFrame(pred_signal, columns=target_columns)
result_sensor.insert(0, 'timestep', range(len(result_sensor)))
result_sensor.to_csv(os.path.join(BASE_DIR, "inference_prediksi_sensor.csv"), index=False)

result_status = pd.DataFrame([{
    'input_day_a'         : CSV_DAY_A,
    'input_day_b'         : CSV_DAY_B,
    'input_day_c'         : CSV_DAY_C,
    'health_status'       : status_map[pred_status],
    'status_code'         : pred_status,
    'confidence_pct'      : round(pred_confidence, 2),
    'prob_sehat_pct'      : round(float(prob[0]*100), 2),
    'prob_pre_anomali_pct': round(float(prob[1]*100), 2),
    'prob_warning_pct'    : round(float(prob[2]*100), 2),
    'generated_at'        : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}])
result_status.to_csv(os.path.join(BASE_DIR, "inference_health_status.csv"), index=False)

print("\n" + "="*70)
print("SELESAI! Output disimpan di folder ini:")
print("  plot_all_parameters_TCN.png")
print("  gambar1_4hari_real_TCN.png")
print("  gambar2_input_plus_prediksi_TCN.png")
print("  gambar3_real_vs_prediksi_TCN.png")
print("  inference_prediksi_sensor.csv")
print("  inference_health_status.csv")
print("="*70)
