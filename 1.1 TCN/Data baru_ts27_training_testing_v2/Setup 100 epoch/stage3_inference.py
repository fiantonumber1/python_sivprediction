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
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")   # folder CSV terisolasi
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

# =========================================================
# CONFIG
# =========================================================
COMPRESSION_FACTOR         = 1
PLOT_DOWNSAMPLE            = 10      # ambil setiap N-th point untuk plot (hemat memori)
TRAIN_RATIO                = 0.8    # konsisten dengan stage1 & stage2
N_TAKE        = 200_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
FUTURE                     = COMPRESSED_POINTS_PER_DAY

START_TIME                 = time(3, 0, 0)
END_TIME     = time(18, 16, 35)
N_DROP_FIRST = 3600

N_FEATURES  = 21
N_CH        = 96
N_STATS     = 4
INPUT_DIM_2 = N_FEATURES * N_STATS   # 84

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Inference] Device: {device}")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

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

status_map   = {0: "Healthy", 1: "Warning"}
status_color = {0: "green", 1: "orange"}

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
            nn.Linear(64, 2)
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

    min_required = N_DROP_FIRST + N_TAKE
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
    if n_active == 0: return 0  # Healthy
    else:             return 1  # Warning

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

# =========================================================
# TRAIN / TEST SPLIT (kronologis, konsisten dengan stage1 & stage2)
# =========================================================
n_train_days = max(4, int(total_days * TRAIN_RATIO))
n_test_days  = total_days - n_train_days
train_dfs    = compressed_dfs[:n_train_days]
test_dfs     = compressed_dfs[n_train_days:]

print(f"\n{'='*65}")
print(f"PEMBAGIAN DATA TRAIN / TEST (TRAIN_RATIO={TRAIN_RATIO})")
print(f"{'='*65}")
print(f"  Total hari valid  : {total_days}")
print(f"  Hari TRAINING     : {n_train_days} hari (80%)")
print(f"  Hari TESTING      : {n_test_days} hari (20%)")
print(f"\n  {'Day':>4}  {'Tanggal':>10}  {'Set':>7}  {'Status Kesehatan':>14}")
print(f"  {'-'*42}")
for _i, (_hs, _f) in enumerate(zip(health_status, all_csv_files)):
    _tag = "TRAIN" if _i < n_train_days else "TEST"
    _tgl = os.path.basename(_f)[:8]
    print(f"  {_i+1:>4}  {_tgl:>10}  {_tag:>7}  {status_map[_hs]:>14}")
print(f"  {'-'*42}")
print(f"\n  DATA TRAINING  → Day 1 s/d Day {n_train_days} "
      f"({os.path.basename(all_csv_files[0])[:8]} – {os.path.basename(all_csv_files[n_train_days-1])[:8]})")
print(f"  DATA TESTING   → Day {n_train_days+1} s/d Day {total_days} "
      f"({os.path.basename(all_csv_files[n_train_days])[:8]} – {os.path.basename(all_csv_files[-1])[:8]})")
print(f"{'='*65}")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_per_col(df_concat):
    norm = df_concat[target_columns].copy()
    for col in target_columns:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - mn) / (mx - mn) if mx - mn > 1e-8 else 0
    return norm

def scale_arr(arr):
    return scaler.transform(arr.reshape(-1, N_FEATURES)).reshape(arr.shape)

def run_tcn(df_a, df_b, df_c):
    seq_raw = np.concatenate([
        df_a[target_columns].values,
        df_b[target_columns].values,
        df_c[target_columns].values,
    ], axis=0).astype(np.float32)
    seq_sc = scaler.transform(seq_raw.reshape(-1, N_FEATURES)).reshape(seq_raw.shape)
    with torch.no_grad():
        pred_sc, _ = tcn_model(torch.FloatTensor(seq_sc).unsqueeze(0).to(device))
        pred_sc = pred_sc.cpu().numpy()[0]
    return scaler.inverse_transform(pred_sc.reshape(-1, N_FEATURES)).reshape(FUTURE, N_FEATURES)

def run_mlp(pred_signal):
    feat = np.concatenate([
        pred_signal.mean(axis=0), pred_signal.std(axis=0),
        pred_signal.max(axis=0),  pred_signal.min(axis=0),
    ]).astype(np.float32)
    with torch.no_grad():
        logits = mlp_model(torch.FloatTensor(scaler2.transform(feat.reshape(1, -1))).to(device))
        prob   = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_st = int(np.argmax(prob))
    return pred_st, float(prob[pred_st] * 100), prob

PPD_ds  = COMPRESSED_POINTS_PER_DAY // PLOT_DOWNSAMPLE
df_all  = pd.concat(compressed_dfs, ignore_index=True)
norm_all = normalize_per_col(df_all)

# =========================================================
# PLOT_ALL — semua hari + label [Tr]/[Te]
# =========================================================
print("\n[Plot] plot_all_parameters_TCN.png ...")
norm_ds  = norm_all.iloc[::PLOT_DOWNSAMPLE].reset_index(drop=True)
x        = np.arange(len(norm_ds))
fig, ax  = plt.subplots(figsize=(24, 10))
colors21 = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors) + list(plt.cm.Dark2.colors[:3])
for j, col in enumerate(target_columns):
    ax.plot(x, norm_ds[col], linewidth=0.9, alpha=0.7, color=colors21[j])
day_bounds = np.arange(0, (total_days + 1) * PPD_ds, PPD_ds)
for b in day_bounds[1:-1]:
    ax.axvline(b, color='red', linestyle='--', alpha=0.8)
for i, mid in enumerate([(day_bounds[i] + day_bounds[i+1]) // 2 for i in range(total_days)]):
    tag = "[Tr]" if i < n_train_days else "[Te]"
    ax.text(mid, 1.05, f'Day {i+1} {tag}', ha='center', color='red', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(mid, 1.15, status_map[health_status[i]], ha='center',
            color=['green','orange'][health_status[i]], fontweight='bold',
            transform=ax.get_xaxis_transform())
ax.set_title(f"21 Parameter + Health Status — {total_days} Hari (Train={n_train_days} | Test={n_test_days})", fontsize=15)
ax.set_ylabel("Normalized [0-1]")
ax.grid(alpha=0.3)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "plot_all_parameters_TCN.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → plot_all_parameters_TCN.png  (supplementary)")

# =========================================================
# GAMBAR 6 — Skema Pembentukan Dataset Forecasting 3 Hari → 1 Hari
# Visualisasi sliding window: tiap window = 3 hari input → 1 hari target
# Train windows dan test windows ditampilkan dengan warna berbeda
# =========================================================
print("\n[Plot] jurnal_sliding_window.png (Jurnal Gambar 6) ...")

_sw_n_days      = total_days
_sw_n_train     = n_train_days
_sw_n_test      = n_test_days
_sw_win_size    = 3   # jumlah hari input
_sw_pred_size   = 1   # jumlah hari target

# Kumpulkan semua window: (day_a, day_b, day_c, day_target, set_tag)
_sw_windows = []
# Train windows: input sepenuhnya dari data training
for _wi in range(_sw_n_train - _sw_win_size):
    _sw_windows.append((_wi, _wi+1, _wi+2, _wi+3, 'Train'))
# Test windows: input mencakup batas train-test (overlap diizinkan untuk input)
for _wi in range(_sw_n_test):
    _a = _sw_n_train - _sw_win_size + _wi
    _sw_windows.append((_a, _a+1, _a+2, _a+3, 'Test'))

_n_win    = len(_sw_windows)
_bw       = 0.7    # lebar box hari
_bh       = 0.4    # tinggi box
_row_h    = 1.1    # jarak antar baris window
_fig_h    = max(6, _n_win * _row_h + 2.5)
fig, ax   = plt.subplots(figsize=(min(18, _sw_n_days * 1.05 + 3), _fig_h))

# Warna per hari (train=biru, test=oranye)
_day_colors = ['#1565C0' if d < _sw_n_train else '#E65100' for d in range(_sw_n_days + 1)]
_day_dates  = [os.path.basename(f)[:8] for f in all_csv_files]

# Gambar kotak hari di baris atas (header)
for _d in range(_sw_n_days):
    _xc = _d + 0.5
    rect = plt.Rectangle((_d, _fig_h - 1.35), _bw, _bh,
                         facecolor=_day_colors[_d], edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(_xc, _fig_h - 1.12, f"D{_d+1}", ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white', zorder=4)
    ax.text(_xc, _fig_h - 1.52, _day_dates[_d], ha='center', va='center',
            fontsize=5.5, color=_day_colors[_d], zorder=4)

# Garis pemisah train-test di header
ax.axvline(_sw_n_train, color='#B71C1C', linewidth=2.5, linestyle='--', alpha=0.9, zorder=2)
ax.text(_sw_n_train, _fig_h - 0.6, '◄ TRAIN  |  TEST ►',
        ha='center', va='center', fontsize=9, color='#B71C1C', fontweight='bold')

# Gambar tiap baris window
for _ri, (_a, _b, _c, _tgt, _stag) in enumerate(_sw_windows):
    _y_top = _fig_h - 2.1 - _ri * _row_h
    _fc_in  = '#90CAF9' if _stag == 'Train' else '#FFCC80'
    _fc_tgt = '#1565C0' if _stag == 'Train' else '#E65100'
    _ec     = '#1565C0' if _stag == 'Train' else '#E65100'
    # Kotak 3 hari input
    for _di, _dx in enumerate([_a, _b, _c]):
        rect_in = plt.Rectangle((_dx, _y_top - _bh), _bw, _bh,
                                facecolor=_fc_in, edgecolor=_ec, linewidth=1.2, zorder=3)
        ax.add_patch(rect_in)
        ax.text(_dx + 0.35, _y_top - _bh/2, f"D{_dx+1}", ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=_ec, zorder=4)
    # Bracket { di kiri kotak input
    _bx_left = _a - 0.05
    ax.annotate('', xy=(_a, _y_top - _bh/2),
                xytext=(_bx_left - 0.15, _y_top - _bh/2),
                arrowprops=dict(arrowstyle='->', color=_ec, lw=1.5), zorder=3)
    ax.text(_bx_left - 0.18, _y_top - _bh/2, f"W{_ri+1}\n({_stag})",
            ha='right', va='center', fontsize=6.5, color=_ec, fontweight='bold')
    # Panah → ke kotak target
    ax.annotate('', xy=(_tgt + 0.05, _y_top - _bh/2),
                xytext=(_c + _bw + 0.05, _y_top - _bh/2),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2.0), zorder=3)
    # Kotak target
    if _tgt < _sw_n_days:
        rect_tgt = plt.Rectangle((_tgt, _y_top - _bh), _bw, _bh,
                                  facecolor=_fc_tgt, edgecolor='#333333', linewidth=1.5, zorder=3)
        ax.add_patch(rect_tgt)
        ax.text(_tgt + 0.35, _y_top - _bh/2, f"D{_tgt+1}", ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', zorder=4)
    else:
        # Target di luar range (future)
        rect_tgt = plt.Rectangle((_tgt, _y_top - _bh), _bw, _bh,
                                  facecolor='#CFD8DC', edgecolor='#455A64', linewidth=1.5,
                                  linestyle='--', zorder=3)
        ax.add_patch(rect_tgt)
        ax.text(_tgt + 0.35, _y_top - _bh/2, f"D{_tgt+1}\n(future)", ha='center', va='center',
                fontsize=6.5, color='#455A64', zorder=4)

# Legend
from matplotlib.patches import Patch
_leg_items = [
    Patch(facecolor='#90CAF9', edgecolor='#1565C0', label=f'Input — Train window ({_sw_n_train-3} window)'),
    Patch(facecolor='#FFCC80', edgecolor='#E65100', label=f'Input — Test window ({_sw_n_test} window)'),
    Patch(facecolor='#1565C0', label='Target — Train'),
    Patch(facecolor='#E65100', label='Target — Test'),
]
ax.legend(handles=_leg_items, loc='lower right', fontsize=8, framealpha=0.9)

ax.set_xlim(-2.0, _sw_n_days + 1.2)
ax.set_ylim(-0.3, _fig_h)
ax.axis('off')
ax.set_title(
    f"Gambar 6. Pembentukan Dataset Forecasting — Skema Sliding Window 3 Hari → 1 Hari\n"
    f"Total {_n_win} window ({_sw_n_train-3} train window + {_sw_n_test} test window) "
    f"dari {_sw_n_days} hari data",
    fontsize=11, fontweight='bold', pad=10
)
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "jurnal_sliding_window.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → jurnal_sliding_window.png  ← Jurnal Gambar 6")

# =========================================================
# GAMBAR 5 — Perbandingan Data Sebelum dan Sesudah Normalisasi
# Ambil 1 hari training (Day 1) sebagai sampel representatif
# Tampilkan 6 parameter paling representatif (2 suhu, 2 arus, 2 tegangan)
# =========================================================
print("\n[Plot] jurnal_normalisasi_comparison.png (Jurnal Gambar 5) ...")
_sample_df   = compressed_dfs[0]  # Day 1 (training)
_sample_raw  = _sample_df[target_columns].values.astype(np.float32)
_sample_norm = scaler.transform(_sample_raw.reshape(-1, N_FEATURES)).reshape(_sample_raw.shape)
_sample_ds   = slice(None, None, PLOT_DOWNSAMPLE)

# Pilih 6 parameter representatif: 2 suhu, 2 arus, 2 tegangan
_rep_params = [
    'SIV_T_HS_InConv_1',   # suhu
    'SIV_T_Container',     # suhu
    'SIV_I_L1',            # arus
    'SIV_I_Battery',       # arus
    'SIV_U_DC_In',         # tegangan
    'SIV_U_L1',            # tegangan
]
_rep_idx = [target_columns.index(p) for p in _rep_params]
_rep_colors = ['#e41a1c','#ff7f00','#377eb8','#4daf4a','#984ea3','#a65628']

_xn = np.arange(len(_sample_raw[_sample_ds]))
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Subplot atas — SEBELUM normalisasi (nilai asli)
ax_raw = axes[0]
for k, (ci, col) in enumerate(zip(_rep_idx, _rep_params)):
    ax_raw.plot(_xn, _sample_raw[_sample_ds, ci], linewidth=1.5,
                color=_rep_colors[k], label=col, alpha=0.85)
ax_raw.set_title("Sebelum Normalisasi — Nilai Asli (satuan berbeda per parameter)", fontsize=12)
ax_raw.set_ylabel("Nilai Asli")
ax_raw.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax_raw.grid(alpha=0.3)

# Subplot bawah — SESUDAH normalisasi (MinMax → [−0.1, 1.1])
ax_nrm = axes[1]
for k, (ci, col) in enumerate(zip(_rep_idx, _rep_params)):
    ax_nrm.plot(_xn, _sample_norm[_sample_ds, ci], linewidth=1.5,
                color=_rep_colors[k], label=col, alpha=0.85)
ax_nrm.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6, label='Min (0)')
ax_nrm.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Max (1)')
ax_nrm.set_title("Sesudah Normalisasi — MinMax Scaler (semua parameter dalam rentang yang sama)", fontsize=12)
ax_nrm.set_ylabel("Nilai Normalisasi")
ax_nrm.set_xlabel("Timestep")
ax_nrm.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax_nrm.grid(alpha=0.3)

_tgl_day1 = os.path.basename(all_csv_files[0])[:8]
fig.suptitle(
    f"Gambar 5. Perbandingan Data Sebelum dan Setelah Normalisasi\n"
    f"(Day 1 — {_tgl_day1}, 6 parameter representatif dari 21 parameter total)",
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "jurnal_normalisasi_comparison.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("  → jurnal_normalisasi_comparison.png  ← Jurnal Gambar 5")

# =========================================================
# BAGIAN 1 — TRAINING INFERENCE
# Input : 3 hari terakhir TRAINING
# Target: hari pertama TEST (ada ground truth → bisa dibandingkan)
# =========================================================
print("\n" + "="*65)
print("BAGIAN 1: TRAINING INFERENCE")
print("Konsep: gunakan 3 hari TERAKHIR data TRAINING → prediksi hari ke-(n_train+1)")
print("        Hari ke-(n_train+1) adalah hari PERTAMA data TESTING → ada ground truth")
print("="*65)

if n_train_days >= 3:
    TR_CSV_A = os.path.basename(all_csv_files[n_train_days - 3])
    TR_CSV_B = os.path.basename(all_csv_files[n_train_days - 2])
    TR_CSV_C = os.path.basename(all_csv_files[n_train_days - 1])
    df_trA, df_trB, df_trC = train_dfs[-3], train_dfs[-2], train_dfs[-1]

    print(f"\n  [INPUT — dari data TRAINING]")
    print(f"    Hari A (Day {n_train_days-2}): {TR_CSV_A}  Status: {status_map[health_status[n_train_days-3]]}")
    print(f"    Hari B (Day {n_train_days-1}): {TR_CSV_B}  Status: {status_map[health_status[n_train_days-2]]}")
    print(f"    Hari C (Day {n_train_days}  ): {TR_CSV_C}  Status: {status_map[health_status[n_train_days-1]]}")

    pred_sig_tr = run_tcn(df_trA, df_trB, df_trC)
    pred_st_tr, pred_cf_tr, prob_tr = run_mlp(pred_sig_tr)

    has_gt = n_test_days >= 1
    if has_gt:
        GT_CSV      = os.path.basename(all_csv_files[n_train_days])
        df_gt       = test_dfs[0]
        real_sig_gt = df_gt[target_columns].values.astype(np.float32)
        real_st_gt  = label_health_status(df_gt)
        # Metrik di normalized space (konsisten dengan training MSE stage1)
        pred_sig_tr_sc = scale_arr(pred_sig_tr)
        real_sig_gt_sc = scale_arr(real_sig_gt)
        mse_tr   = float(np.mean((pred_sig_tr_sc - real_sig_gt_sc) ** 2))
        rmse_tr  = float(np.sqrt(mse_tr))
        mae_tr   = float(np.mean(np.abs(pred_sig_tr_sc - real_sig_gt_sc)))
        _mask_gt = real_sig_gt_sc != 0
        mape_tr  = float(np.mean(np.abs(
            (pred_sig_tr_sc[_mask_gt] - real_sig_gt_sc[_mask_gt]) / real_sig_gt_sc[_mask_gt]
        )) * 100) if _mask_gt.any() else float('nan')
        _clf_match  = pred_st_tr == real_st_gt
        print(f"\n  [TARGET — hari pertama data TESTING (ground truth tersedia)]")
        print(f"    Hari D (Day {n_train_days+1}): {GT_CSV}  Status Aktual: {status_map[real_st_gt]}")
        print(f"\n  [HASIL TRAINING INFERENCE]")
        print(f"    Prediksi Status : {status_map[pred_st_tr]}  ({pred_cf_tr:.2f}% confidence)")
        print(f"    Status Aktual   : {status_map[real_st_gt]}")
        print(f"    Klasifikasi     : {'BENAR ✓' if _clf_match else 'SALAH ✗'}")
        print(f"    Prob Healthy    : {prob_tr[0]*100:.2f}%")
        print(f"    Prob Warning    : {prob_tr[1]*100:.2f}%")
        print(f"\n  [METRIK REGRESI SINYAL — skala normalized [-0.1, 1.1]]")
        print(f"    MSE   = {mse_tr:.8f}")
        print(f"    RMSE  = {rmse_tr:.8f}")
        print(f"    MAE   = {mae_tr:.8f}")
        print(f"    MAPE  = {mape_tr:.2f}%")
    else:
        print(f"\n  [HASIL TRAINING INFERENCE — tanpa ground truth]")
        print(f"    Prediksi hari ke-{n_train_days+1}: {status_map[pred_st_tr]} ({pred_cf_tr:.2f}%)")

    # Siapkan array plot
    input3_tr_sc = scale_arr(np.concatenate([
        df_trA[target_columns].values,
        df_trB[target_columns].values,
        df_trC[target_columns].values,
    ], axis=0).astype(np.float32))[::PLOT_DOWNSAMPLE]
    pred_tr_sc   = scale_arr(pred_sig_tr)[::PLOT_DOWNSAMPLE]
    if has_gt:
        real_gt_sc = scale_arr(real_sig_gt)[::PLOT_DOWNSAMPLE]
    x3 = np.arange(3 * PPD_ds)
    x1 = np.arange(3 * PPD_ds, 4 * PPD_ds)

    # --- Gambar Train-1: 3 hari input + prediksi (+ actual overlay) ---
    print("[Plot] gambar_train1_input_prediksi.png ...")
    fig, ax = plt.subplots(figsize=(24, 10))
    for j, col in enumerate(target_columns):
        line, = ax.plot(x3, input3_tr_sc[:, j], linewidth=1.2)
        ax.plot(x1, pred_tr_sc[:, j], '--', linewidth=2.5, color=line.get_color(), alpha=0.9)
        if has_gt:
            ax.plot(x1, real_gt_sc[:, j], '-', linewidth=1.5, color=line.get_color(), alpha=0.4)
    for b in [PPD_ds, 2*PPD_ds, 3*PPD_ds]:
        ax.axvline(b, color='red', linestyle='--', alpha=0.8)
    tr_labels = [
        (f"Day {n_train_days-2}", health_status[n_train_days-3]),
        (f"Day {n_train_days-1}", health_status[n_train_days-2]),
        (f"Day {n_train_days}",   health_status[n_train_days-1]),
        (f"Day {n_train_days+1} (Pred)", pred_st_tr),
    ]
    for i, (lbl, stat) in enumerate(tr_labels):
        mid = int((i + 0.5) * PPD_ds)
        ax.text(mid, 1.05, lbl, ha='center', color='red', fontweight='bold',
                transform=ax.get_xaxis_transform())
        ax.text(mid, 1.15, status_map[stat], ha='center',
                color=['green','orange'][stat], fontweight='bold',
                transform=ax.get_xaxis_transform())
    note = " | Solid tipis=Aktual, Dashed=Pred" if has_gt else " | Dashed=Pred"
    ax.set_title(f"Gambar 9. TRAINING INFERENCE — 3 Hari Input (Train) + Prediksi Day {n_train_days+1}{note}", fontsize=14)
    ax.set_ylabel("Scaled"); ax.grid(alpha=0.3); ax.set_ylim(-0.2, 1.3)
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCE_DIR, "gambar_train1_input_prediksi.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  → gambar_train1_input_prediksi.png")

    # --- Gambar Train-2: Real vs Pred hari pertama test (hanya jika ada ground truth) ---
    if has_gt:
        print("[Plot] gambar_train2_real_vs_pred.png ...")
        fig, ax = plt.subplots(figsize=(18, 8))
        x1_only = np.arange(PPD_ds)
        for j, col in enumerate(target_columns):
            line, = ax.plot(x1_only, real_gt_sc[:, j], '-', linewidth=1.8, alpha=0.85, label=col)
            ax.plot(x1_only, pred_tr_sc[:, j], '--', linewidth=2.5, color=line.get_color(), alpha=0.9)
        ax.set_title(
            f"Gambar 11. Kurva Aktual (Y_Testing) vs Prediksi — Day {n_train_days+1}\n"
            f"Aktual: {status_map[real_st_gt]} | Pred: {status_map[pred_st_tr]} | "
            f"MSE={mse_tr:.4f}  RMSE={rmse_tr:.4f}  MAE={mae_tr:.4f}  MAPE={mape_tr:.2f}%", fontsize=13)
        ax.set_ylabel("Scaled"); ax.grid(alpha=0.3); ax.set_ylim(-0.2, 1.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(EVIDENCE_DIR, "gambar_train2_real_vs_pred.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("  → gambar_train2_real_vs_pred.png")

    # Save CSV training inference
    pd.DataFrame(pred_sig_tr, columns=target_columns).assign(
        timestep=range(FUTURE)
    )[['timestep'] + target_columns].to_csv(
        os.path.join(EVIDENCE_DIR, "inference_train_prediksi_sensor.csv"), index=False)
    row_tr = {
        'mode': 'training_inference',
        'input_day_a': TR_CSV_A, 'input_day_b': TR_CSV_B, 'input_day_c': TR_CSV_C,
        'pred_status': status_map[pred_st_tr], 'pred_status_code': pred_st_tr,
        'confidence_pct': round(pred_cf_tr, 2),
        'prob_healthy_pct': round(float(prob_tr[0]*100), 2),
        'prob_warning_pct': round(float(prob_tr[1]*100), 2),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if has_gt:
        row_tr.update({'ground_truth_csv': GT_CSV, 'actual_status': status_map[real_st_gt],
                       'mse': round(mse_tr, 6), 'rmse': round(rmse_tr, 6),
                       'mae': round(mae_tr, 6), 'mape_pct': round(mape_tr, 4)})
    pd.DataFrame([row_tr]).to_csv(os.path.join(EVIDENCE_DIR, "inference_train_health_status.csv"), index=False)
else:
    print("  [Skip] Training days < 3")


# =========================================================
# BAGIAN 2 — TESTING INFERENCE
# Input : 3 hari terakhir SEMUA data (hari test terbaru)
# Output: prediksi hari masa depan (tidak ada ground truth)
# =========================================================
print("\n" + "="*65)
print("BAGIAN 2: TESTING INFERENCE")
print("Konsep: gunakan 3 hari TERAKHIR dari SEMUA data → prediksi hari MASA DEPAN")
print("        Tidak ada ground truth → hanya prediksi + confidence")
print("="*65)

CSV_DAY_A = os.path.basename(all_csv_files[-3])
CSV_DAY_B = os.path.basename(all_csv_files[-2])
CSV_DAY_C = os.path.basename(all_csv_files[-1])
df_A, df_B, df_C = compressed_dfs[-3], compressed_dfs[-2], compressed_dfs[-1]
_idx_a = total_days - 3
_idx_b = total_days - 2
_idx_c = total_days - 1

print(f"\n  [INPUT — dari data TESTING (hari terakhir dataset)]")
print(f"    Hari A (Day {_idx_a+1}): {CSV_DAY_A}  Status: {status_map[health_status[_idx_a]]}")
print(f"    Hari B (Day {_idx_b+1}): {CSV_DAY_B}  Status: {status_map[health_status[_idx_b]]}")
print(f"    Hari C (Day {_idx_c+1}): {CSV_DAY_C}  Status: {status_map[health_status[_idx_c]]}")

pred_signal = run_tcn(df_A, df_B, df_C)
pred_status, pred_confidence, prob = run_mlp(pred_signal)

print(f"\n  [HASIL TESTING INFERENCE — prediksi hari ke-{total_days+1} (masa depan)]")
print(f"    Prediksi Status : {status_map[pred_status]}  ({pred_confidence:.2f}% confidence)")
print(f"    Prob Healthy      : {prob[0]*100:.2f}%")
print(f"    Prob Warning    : {prob[1]*100:.2f}%")
print(f"    Ground Truth    : TIDAK ADA (hari masa depan)")
print(f"    Metrik MSE/RMSE : TIDAK DAPAT DIHITUNG (tidak ada data aktual)")
print("="*65)

# Persiapan array plot testing (4 hari terakhir real + prediksi)
last4_dfs    = compressed_dfs[-4:]
real4        = pd.concat(last4_dfs, ignore_index=True)[target_columns].values
norm4_sc     = scale_arr(real4)
pred_norm    = scale_arr(pred_signal)
norm4_g1     = norm_all.iloc[-4 * COMPRESSED_POINTS_PER_DAY:].values
norm4_sc_ds  = norm4_sc[::PLOT_DOWNSAMPLE]
norm4_g1_ds  = norm4_g1[::PLOT_DOWNSAMPLE]
pred_norm_ds = pred_norm[::PLOT_DOWNSAMPLE]
x4_ds = np.arange(4 * PPD_ds)
n3_ds = 3 * PPD_ds

def setup_plot(ax, title, ylim_low=-0.1, override_last=None):
    bounds = np.arange(0, 5 * PPD_ds, PPD_ds)
    for b in bounds[1:]:
        if b < len(x4_ds):
            ax.axvline(b, color='red', linestyle='--', alpha=0.8)
    for i, m in enumerate([(bounds[i] + bounds[i+1]) // 2 for i in range(4)]):
        day_idx = total_days - 4 + i
        stat = override_last if (i == 3 and override_last is not None) else health_status[day_idx]
        label_txt = f"Pred: {status_map[stat]}" if (i == 3 and override_last is not None) else status_map[stat]
        ax.text(m, 1.05, f'Day {day_idx+1}', ha='center', color='red', fontweight='bold',
                transform=ax.get_xaxis_transform())
        ax.text(m, 1.15, label_txt, ha='center',
                color=['green','orange'][stat], fontweight='bold',
                transform=ax.get_xaxis_transform())
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.3)
    ax.set_ylim(ylim_low, 1.3)

# Supplementary — 4 hari real (konteks testing, bukan gambar jurnal bernomor)
print("[Plot] gambar1_4hari_real_TCN.png (supplementary) ...")
fig, ax = plt.subplots(figsize=(24, 10))
for i, col in enumerate(target_columns):
    ax.plot(x4_ds, norm4_g1_ds[:, i], linewidth=1, alpha=0.8)
setup_plot(ax, "[Supplementary] Testing Inference — 4 Hari Real Data + Health Status", ylim_low=0)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "gambar1_4hari_real_TCN.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → gambar1_4hari_real_TCN.png")

# Gambar 11 — Testing Inference: 3 hari input + prediksi D+1
print("[Plot] gambar2_input_plus_prediksi_TCN.png (Jurnal Gambar 15) ...")
fig, ax = plt.subplots(figsize=(24, 10))
for i, col in enumerate(target_columns):
    ax.plot(x4_ds[:n3_ds], norm4_sc_ds[:n3_ds, i], linewidth=1.2, label=col)
for i, col in enumerate(target_columns):
    ax.plot(x4_ds[n3_ds:], pred_norm_ds[:, i], '--', linewidth=2.8,
            color=ax.get_lines()[i].get_color(), alpha=0.95)
setup_plot(ax, f"Gambar 15. Hasil Forecasting — Input {_idx_a+1} s/d {_idx_c+1} ({CSV_DAY_A}–{CSV_DAY_C}), 3 Hari → Prediksi Label Kelas: {status_map[pred_status]}",
           override_last=pred_status)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "gambar2_input_plus_prediksi_TCN.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → gambar2_input_plus_prediksi_TCN.png  ← Jurnal Gambar 15")

# Supplementary — Testing Inference: real hari terakhir vs prediksi D+1
print("[Plot] gambar3_real_vs_prediksi_TCN.png (supplementary) ...")
fig, ax = plt.subplots(figsize=(24, 10))
for i, col in enumerate(target_columns):
    ax.plot(x4_ds[:n3_ds], norm4_sc_ds[:n3_ds, i], linewidth=1.2)
for i, col in enumerate(target_columns):
    ax.plot(x4_ds[n3_ds:], norm4_sc_ds[n3_ds:, i], linewidth=1.8, alpha=0.9)
for i, col in enumerate(target_columns):
    ax.plot(x4_ds[n3_ds:], pred_norm_ds[:, i], '--', linewidth=3,
            color=ax.get_lines()[i].get_color(), alpha=0.95,
            label='Pred' if i == 0 else None)
setup_plot(ax, "[Supplementary] Testing Inference — Real Hari Terakhir vs Prediksi Hari D+1")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "gambar3_real_vs_prediksi_TCN.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  → gambar3_real_vs_prediksi_TCN.png")

# Save CSV testing inference
pd.DataFrame(pred_signal, columns=target_columns).assign(
    timestep=range(FUTURE)
)[['timestep'] + target_columns].to_csv(
    os.path.join(EVIDENCE_DIR, "inference_prediksi_sensor.csv"), index=False)
pd.DataFrame([{
    'mode': 'testing_inference',
    'input_day_a': CSV_DAY_A, 'input_day_b': CSV_DAY_B, 'input_day_c': CSV_DAY_C,
    'health_status': status_map[pred_status], 'status_code': pred_status,
    'confidence_pct': round(pred_confidence, 2),
    'prob_healthy_pct': round(float(prob[0]*100), 2),
    'prob_warning_pct': round(float(prob[1]*100), 2),
    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}]).to_csv(os.path.join(EVIDENCE_DIR, "inference_health_status.csv"), index=False)

print("\n" + "="*70)
print("RINGKASAN PERBANDINGAN: TRAINING INFERENCE vs TESTING INFERENCE")
print("="*70)
print(f"  {'Aspek':<28} {'Training Inference':>20}  {'Testing Inference':>18}")
print(f"  {'-'*70}")
if n_train_days >= 3:
    _ti_input = f"Day {n_train_days-2}–{n_train_days} (Train)"
    _ti_target = f"Day {n_train_days+1} (1st Test)"
    _ti_status = f"{status_map[pred_st_tr]} ({pred_cf_tr:.1f}%)"
    if has_gt:
        _ti_gt = status_map[real_st_gt]
        _ti_match = "BENAR ✓" if pred_st_tr == real_st_gt else "SALAH ✗"
        _ti_mse = f"{mse_tr:.6f}"
        _ti_rmse = f"{rmse_tr:.6f}"
        _ti_mae = f"{mae_tr:.6f}"
    else:
        _ti_gt = "N/A"; _ti_match = "N/A"; _ti_mse = "N/A"; _ti_rmse = "N/A"; _ti_mae = "N/A"
else:
    _ti_input="N/A"; _ti_target="N/A"; _ti_status="N/A"; _ti_gt="N/A"; _ti_match="N/A"
    _ti_mse="N/A"; _ti_rmse="N/A"; _ti_mae="N/A"
_te_input  = f"Day {total_days-2}–{total_days} (Test)"
_te_target = f"Day {total_days+1} (Future)"
_te_status = f"{status_map[pred_status]} ({pred_confidence:.1f}%)"
print(f"  {'Input Days':<28} {_ti_input:>20}  {_te_input:>18}")
print(f"  {'Target':<28} {_ti_target:>20}  {_te_target:>18}")
print(f"  {'Prediksi Status':<28} {_ti_status:>20}  {_te_status:>18}")
print(f"  {'Ground Truth':<28} {_ti_gt:>20}  {'TIDAK ADA':>18}")
print(f"  {'Klasifikasi Benar?':<28} {_ti_match:>20}  {'Tidak dpt diukur':>18}")
print(f"  {'MSE':<28} {_ti_mse:>20}  {'Tidak dpt diukur':>18}")
print(f"  {'RMSE':<28} {_ti_rmse:>20}  {'Tidak dpt diukur':>18}")
print(f"  {'MAE':<28} {_ti_mae:>20}  {'Tidak dpt diukur':>18}")
print(f"  {'-'*70}")
print(f"\n  NOTE: Training inference BISA diukur akurasi/MSE karena ground truth")
print(f"        hari D+1 (data test) tersedia. Testing inference TIDAK BISA diukur")
print(f"        karena prediksi ke hari masa depan yang belum ada datanya.")
print("="*70)

print("\nSELESAI! Output disimpan ke folder evidence/:")
print("\n[TRAINING INFERENCE — Gambar 11 Jurnal]")
if n_train_days >= 3:
    print("  gambar_train1_input_prediksi.png   (supplementary — 3 hari train input + prediksi)")
    if has_gt:
        print("  gambar_train2_real_vs_pred.png     ← Jurnal Gambar 11 (Aktual Y_Testing vs Prediksi)")
    print("  inference_train_prediksi_sensor.csv")
    print("  inference_train_health_status.csv")
print("\n[TESTING INFERENCE — Gambar 15 Jurnal]")
print("  jurnal_normalisasi_comparison.png      ← Jurnal Gambar 5")
print("  jurnal_sliding_window.png              ← Jurnal Gambar 6")
print("  plot_all_parameters_TCN.png            (supplementary)")
print("  gambar2_input_plus_prediksi_TCN.png    ← Jurnal Gambar 11 (3 hari test + prediksi D+1)")
print("  inference_prediksi_sensor.csv")
print("  inference_health_status.csv")
print("\n[SUPPLEMENTARY — tidak bernomor gambar jurnal]")
print("  gambar1_4hari_real_TCN.png             (konteks 4 hari real testing)")
print("  gambar3_real_vs_prediksi_TCN.png       (overlay real vs pred testing)")
print("="*70)

# =========================================================
# FLOWCHART — Jurnal Gambar 1 & 2
# =========================================================
def draw_flowchart(steps, title, filename):
    n = len(steps)
    fig_h = n * 1.45 + 1.2
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_xlim(0, 10); ax.set_ylim(0, fig_h); ax.axis('off')
    bw, bh, x0 = 7.6, 0.82, 1.2
    y_top = fig_h - 0.85
    for i, (num, text) in enumerate(steps):
        y = y_top - i * 1.45
        is_edge = (i == 0 or i == n - 1)
        fc = '#1565C0' if is_edge else '#E3F2FD'
        tc = 'white'   if is_edge else '#0D47A1'
        rect = plt.Rectangle((x0, y - bh/2), bw, bh,
                              facecolor=fc, edgecolor='#1565C0', linewidth=1.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + bw/2, y, f"{num}. {text}",
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=tc, zorder=3, wrap=True)
        if i < n - 1:
            ya = y - bh/2
            yb = y - 1.45 + bh/2
            ax.annotate('', xy=(x0 + bw/2, yb + 0.03),
                        xytext=(x0 + bw/2, ya - 0.03),
                        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.0), zorder=1)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1565C0', pad=6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → {os.path.basename(filename)}")

print("\n[Plot] Membuat flowchart jurnal ...")

draw_flowchart([
    ("1", "Baca CSV 13 hari (200.000 baris/hari, 21 parameter + 3 fault kolom)"),
    ("2", "Preprocessing: drop 3.600 warmup, crop 04:00–18:16, ffill/bfill"),
    ("3", "Train/Test Split kronologis — 80%/20% (10 train | 3 test hari)"),
    ("4", "Sliding Window 3→1 hari: 7 window train + 3 window test"),
    ("5", "MinMax Scaler [−0,1 ; 1,1] — fit hanya dari data training"),
    ("6", "TCNForecaster: 7 Residual Block, dilation [1,2,4,8,16,32,64], n_ch=96"),
    ("7", "Training Loop: 100 epoch, batch=3, AdamW, gradient clipping 1,0"),
    ("8", "Evaluasi Test Set (MSE / RMSE / MAE) → Simpan model + scaler"),
], "Gambar 1. Flowchart Pipeline TCN Forecasting (Stage 1)",
   os.path.join(EVIDENCE_DIR, "jurnal_flowchart_stage1.png"))

draw_flowchart([
    ("1", "Baca CSV 13 hari (sama seperti Stage 1)"),
    ("2", "Preprocessing: drop warmup, crop waktu, ffill/bfill"),
    ("3", "Train/Test Split kronologis — 80%/20% (10 train | 3 test hari)"),
    ("4", "Feature Extraction: mean+std+max+min per hari → 84 dimensi"),
    ("5", "Labeling: 0 fault→Healthy (0) | ≥1 fault→Warning (1)"),
    ("6", "MinMax Scaler [−0,1 ; 1,1] — fit hanya dari data training"),
    ("7", "MLPClassifier: 84→256→128→64→2, CrossEntropy, 100 epoch"),
    ("8", "Evaluasi Test Set (Accuracy per hari) → Simpan model + scaler"),
], "Gambar 2. Flowchart Pipeline MLP Classifier (Stage 2)",
   os.path.join(EVIDENCE_DIR, "jurnal_flowchart_stage2.png"))

print("\n[FLOWCHART JURNAL]")
print("  jurnal_flowchart_stage1.png   ← Jurnal Gambar 1")
print("  jurnal_flowchart_stage2.png   ← Jurnal Gambar 2")

# =========================================================
# JURNAL INFORMASI LENGKAP TXT
# =========================================================
_j3_csv_dates = [os.path.basename(f)[:8] for f in all_csv_files]

# Baca stage1 & stage2 info jika ada
def _baca_info(fname):
    fp = os.path.join(BASE_DIR, fname)
    if os.path.exists(fp):
        with open(fp, 'r', encoding='utf-8') as _f: return _f.read()
    return f"(File {fname} belum ada — jalankan stage terkait terlebih dahulu)"

_j3_lines = [
    "=" * 68,
    "INFORMASI LENGKAP UNTUK PENULISAN JURNAL — TCN SIV TS27",
    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "=" * 68,
    "",
    "FILE INI BERISI SEMUA DATA NUMERIK YANG DIBUTUHKAN JURNAL.",
    "Cukup copy-paste nilai-nilai di bawah ke dokumen jurnal.",
    "",
    "=" * 68,
    "BAGIAN A — DATA UMUM",
    "=" * 68,
    "",
    "[A1. DATASET]",
    f"  Total hari CSV valid         : {total_days}",
    f"  Train days (80%)             : {n_train_days}",
    f"  Test days  (20%)             : {n_test_days}",
    f"  Baris per hari               : 200,000",
    f"  Total baris digunakan        : {200000 * total_days:,}",
    f"  Warmup di-drop per hari      : 3,600 baris",
    f"  Window TCN (train)           : {n_train_days - 3} window",
    f"  Window TCN (test)            : {n_test_days} window (overlap 3 hari terakhir train)",
    "",
    "[A2. DETAIL HARI TRAINING vs TESTING]",
    f"  {'Day':>4}  {'Tanggal':>10}  {'Set':>7}  {'Status Kesehatan':>14}",
    "  " + "-" * 42,
]
for _idx, (_hs, _f) in enumerate(zip(health_status, all_csv_files)):
    _set_tag = "TRAIN" if _idx < n_train_days else "TEST"
    _tgl     = os.path.basename(_f)[:8]
    _j3_lines.append(f"  {_idx+1:>4}  {_tgl:>10}  {_set_tag:>7}  {status_map[_hs]:>14}")
_j3_lines += [
    "  " + "-" * 42,
    f"  Data TRAINING → Day 1 s/d Day {n_train_days}  "
    f"({os.path.basename(all_csv_files[0])[:8]} – {os.path.basename(all_csv_files[n_train_days-1])[:8]})",
    f"  Data TESTING  → Day {n_train_days+1} s/d Day {total_days}  "
    f"({os.path.basename(all_csv_files[n_train_days])[:8]} – {os.path.basename(all_csv_files[-1])[:8]})",
    "",
    "[A3. NORMALISASI — untuk Tabel Normalisasi Jurnal]",
    f"  Metode                       : Min-Max Scaler",
    f"  Rentang output               : [-0.1, 1.1]",
    f"  Jumlah parameter             : {N_FEATURES}",
    f"  Fit dari                     : {n_train_days} hari training saja",
]
# Tampilkan rentang per parameter dari scaler
_j3_lines += ["", f"  {'Parameter':<28} {'Min Data':>12} {'Max Data':>12}", "  " + "-" * 54]
for _ci, _col in enumerate(target_columns):
    try:
        _mn, _mx = scaler.data_min_[_ci], scaler.data_max_[_ci]
        _j3_lines.append(f"  {_col:<28} {_mn:>12.4f} {_mx:>12.4f}")
    except: pass

_j3_stat_count = {0: 0, 1: 0}
_j3_stat_dates = {0: [], 1: []}
for _idx, (_hs, _f) in enumerate(zip(health_status, all_csv_files)):
    _j3_stat_count[_hs] += 1
    _j3_stat_dates[_hs].append(os.path.basename(_f)[:8])
_j3_lines += [
    "",
    "[A4. LABEL KONDISI — untuk Kalimat Jurnal Para 175 & Tabel Distribusi]",
    f"  Total hari dilabel           : {total_days}",
    f"  Healthy (0)   : {_j3_stat_count[0]} hari → {', '.join(_j3_stat_dates[0])}",
    f"  Warning (1)   : {_j3_stat_count[1]} hari → {', '.join(_j3_stat_dates[1]) or '-'}",
]

_j3_lines += [
    "",
    "=" * 68,
    "BAGIAN B — TRAINING INFERENCE",
    "Konsep: 3 hari terakhir DATA TRAINING → prediksi hari pertama DATA TESTING",
    "        Ground truth tersedia → metrik MSE/RMSE/MAE DAPAT dihitung",
    "=" * 68,
    "",
]
if n_train_days >= 3:
    _b_a = os.path.basename(all_csv_files[n_train_days-3])[:8]
    _b_b = os.path.basename(all_csv_files[n_train_days-2])[:8]
    _b_c = os.path.basename(all_csv_files[n_train_days-1])[:8]
    _j3_lines += [
        f"  [INPUT — dari data TRAINING]",
        f"    Hari A (Day {n_train_days-2}): {_b_a}  Status: {status_map[health_status[n_train_days-3]]}",
        f"    Hari B (Day {n_train_days-1}): {_b_b}  Status: {status_map[health_status[n_train_days-2]]}",
        f"    Hari C (Day {n_train_days}  ): {_b_c}  Status: {status_map[health_status[n_train_days-1]]}",
        "",
        f"  Prediksi hari D (Day {n_train_days+1})     : {status_map[pred_st_tr]} ({pred_cf_tr:.2f}% confidence)",
        f"  Prob Healthy                   : {prob_tr[0]*100:.2f}%",
        f"  Prob Warning                 : {prob_tr[1]*100:.2f}%",
    ]
    if has_gt:
        _j3_lines += [
            "",
            f"  [GROUND TRUTH — hari pertama data TESTING]",
            f"    Hari D (Day {n_train_days+1}): {os.path.basename(all_csv_files[n_train_days])[:8]}  "
            f"Status Aktual: {status_map[real_st_gt]}",
            "",
            f"  Status Prediksi              : {status_map[pred_st_tr]}",
            f"  Status Aktual                : {status_map[real_st_gt]}",
            f"  Klasifikasi                  : {'BENAR ✓' if pred_st_tr == real_st_gt else 'SALAH ✗'}",
            "",
            f"  [METRIK REGRESI SINYAL]",
            f"  MSE Prediksi vs Aktual       : {mse_tr:.8f}  ({mse_tr*100:.4f}%)",
            f"  RMSE                         : {rmse_tr:.8f}",
            f"  MAE                          : {mae_tr:.8f}",
            "",
            "  → Untuk Para 170 Jurnal:",
            f"    'Training inference menghasilkan MSE={mse_tr:.6f}, RMSE={rmse_tr:.6f}, MAE={mae_tr:.6f}'",
            f"    'Prediksi status kesehatan: {status_map[pred_st_tr]} — {'sesuai' if pred_st_tr == real_st_gt else 'tidak sesuai'} ground truth ({status_map[real_st_gt]})'",
        ]
    else:
        _j3_lines.append("  (Tidak ada data test tersedia untuk ground truth)")
else:
    _j3_lines.append("  (Tidak cukup training days)")

_j3_lines += [
    "",
    "=" * 68,
    "BAGIAN C — TESTING INFERENCE",
    "Konsep: 3 hari terakhir DATA TESTING → prediksi hari MASA DEPAN",
    "        Tidak ada ground truth → metrik MSE/RMSE/MAE TIDAK dapat dihitung",
    "=" * 68,
    "",
    f"  [INPUT — dari data TESTING (hari terakhir dataset)]",
    f"    Hari A (Day {total_days-2}): {CSV_DAY_A}  Status: {status_map[health_status[total_days-3]]}",
    f"    Hari B (Day {total_days-1}): {CSV_DAY_B}  Status: {status_map[health_status[total_days-2]]}",
    f"    Hari C (Day {total_days}  ): {CSV_DAY_C}  Status: {status_map[health_status[total_days-1]]}",
    "",
    f"  Prediksi hari ke-{total_days+1} (masa depan) : {status_map[pred_status]}",
    f"  Confidence                   : {pred_confidence:.2f}%",
    f"  Prob Healthy                   : {prob[0]*100:.2f}%",
    f"  Prob Warning                 : {prob[1]*100:.2f}%",
    f"  Ground Truth                 : TIDAK ADA",
    f"  Metrik MSE/RMSE/MAE          : TIDAK DAPAT DIHITUNG",
    "",
    "  → Untuk Para 224 Jurnal (caption gambar forecasting):",
    f"    'Hasil forecasting dari input tanggal {CSV_DAY_A}, {CSV_DAY_B}, {CSV_DAY_C}",
    f"     didapatkan prediksi hari ke-{total_days+1} dengan Label Kelas: {status_map[pred_status]}'",
    "",
    "=" * 68,
    "BAGIAN D — PERBANDINGAN TRAINING INFERENCE vs TESTING INFERENCE",
    "=" * 68,
    "",
    f"  {'Aspek':<32} {'Training Inference':>18}  {'Testing Inference':>18}",
    "  " + "-" * 72,
]
_ti_in  = f"Day {n_train_days-2}–{n_train_days} (Train)"   if n_train_days >= 3 else "N/A"
_te_in  = f"Day {total_days-2}–{total_days} (Test)"
_ti_tgt = f"Day {n_train_days+1} (1st Test)"               if n_train_days >= 3 else "N/A"
_te_tgt = f"Day {total_days+1} (Future)"
_ti_st  = f"{status_map[pred_st_tr]} ({pred_cf_tr:.1f}%)" if n_train_days >= 3 else "N/A"
_te_st  = f"{status_map[pred_status]} ({pred_confidence:.1f}%)"
_ti_gt2 = status_map[real_st_gt] if (n_train_days >= 3 and has_gt) else "N/A"
_ti_ok  = ("BENAR ✓" if pred_st_tr == real_st_gt else "SALAH ✗") if (n_train_days >= 3 and has_gt) else "N/A"
_ti_mse2= f"{mse_tr:.6f}"  if (n_train_days >= 3 and has_gt) else "N/A"
_ti_rm2 = f"{rmse_tr:.6f}" if (n_train_days >= 3 and has_gt) else "N/A"
_ti_ma2 = f"{mae_tr:.6f}"  if (n_train_days >= 3 and has_gt) else "N/A"
_j3_lines += [
    f"  {'Input Days':<32} {_ti_in:>18}  {_te_in:>18}",
    f"  {'Target':<32} {_ti_tgt:>18}  {_te_tgt:>18}",
    f"  {'Prediksi Status':<32} {_ti_st:>18}  {_te_st:>18}",
    f"  {'Ground Truth':<32} {_ti_gt2:>18}  {'TIDAK ADA':>18}",
    f"  {'Klasifikasi Benar?':<32} {_ti_ok:>18}  {'Tdk dpt diukur':>18}",
    f"  {'MSE':<32} {_ti_mse2:>18}  {'Tdk dpt diukur':>18}",
    f"  {'RMSE':<32} {_ti_rm2:>18}  {'Tdk dpt diukur':>18}",
    f"  {'MAE':<32} {_ti_ma2:>18}  {'Tdk dpt diukur':>18}",
    "  " + "-" * 72,
    "",
    "  KESIMPULAN:",
    "  - Training Inference memiliki metrik kuantitatif (MSE/RMSE/MAE) karena",
    "    hari yang diprediksi (D+1) sudah ada datanya sebagai ground truth.",
    "  - Testing Inference hanya memiliki confidence score karena memprediksi",
    "    hari masa depan yang belum ada datanya. Akurasi tidak dapat diukur.",
    "  - Kedua inference menggunakan model yang SAMA (hasil training stage1 & stage2).",
    "    Perbedaan hanya pada DATA INPUT dan ketersediaan GROUND TRUTH.",
    "",
    "=" * 68,
    "BAGIAN E — DAFTAR SEMUA FILE OUTPUT",
    "=" * 68,
    "",
    "[GAMBAR — siap dimasukkan ke jurnal]",
    f"  Gambar 1  : jurnal_flowchart_stage1.png     (Pipeline TCN)",
    f"  Gambar 2  : jurnal_flowchart_stage2.png     (Pipeline MLP)",
    f"  Gambar 3  : MANUAL — screenshot LeadMind",
    f"  Gambar 4  : MANUAL — laporan gangguan sarana",
    f"  Gambar 5  : MANUAL — detail insiden SIV TS27",
    f"  Gambar 5  : jurnal_normalisasi_comparison.png (sebelum vs sesudah normalisasi)",
    f"  Gambar 6  : jurnal_sliding_window.png        (skema sliding window 3 hari → 1 hari)",
    f"  Suppl.    : plot_all_parameters_TCN.png      (21 param semua hari — supplementary)",
    f"  Gambar 10 : jurnal_kurva_loss_stage1.png     (Kurva Epoch vs MSE Training TCN)",
    f"  Gambar 10 : jurnal_kurva_loss_stage1.png      (Kurva Epoch vs MSE Training)",
    f"  Gambar 11 : gambar_train2_real_vs_pred.png    [TRAINING INFERENCE] Aktual Y_Testing vs Prediksi",
    f"  Gambar 15 : gambar2_input_plus_prediksi_TCN.png [TESTING INFERENCE] Hasil Forecasting + Label Kelas",
    f"  Suppl.    : gambar_train1_input_prediksi.png  (3 hari train input + prediksi — Training Inference)",
    f"  Gambar 12 : jurnal_kurva_loss_acc_stage2.png  CE Loss + Accuracy MLP",
    f"  Suppl.    : gambar1_4hari_real_TCN.png        (konteks 4 hari real testing)",
    f"  Suppl.    : gambar3_real_vs_prediksi_TCN.png  (overlay real vs pred testing)",
    f"  Suppl.    : jurnal_confusion_matrix_train.png  (4 metrik: Acc/Prec/Rec/F1 — train)",
    f"  Suppl.    : jurnal_confusion_matrix_test.png   (4 metrik: Acc/Prec/Rec/F1 — test)",
    "",
    "[DATA NUMERIK]",
    f"  jurnal_info_stage1.txt       (Hyperpar, MSE train/test, sequence table)",
    f"  jurnal_info_stage2.txt       (Hyperpar, CE/Acc, confusion matrix, label dates)",
    f"  jurnal_fitur_statistik.csv   (mean/std/max/min per parameter per hari)",
    f"  inference_train_health_status.csv",
    f"  inference_health_status.csv",
]

_j3_out = os.path.join(EVIDENCE_DIR, "jurnal_informasi_lengkap.txt")
with open(_j3_out, 'w', encoding='utf-8') as _fj3:
    _fj3.write('\n'.join(_j3_lines))
print(f"\n[INFO] jurnal_informasi_lengkap.txt disimpan ← BACA INI untuk semua data jurnal")
print("\nCatatan: Gambar 3 (LeadMind), Gambar 4 (Laporan Gangguan),")
print("         Gambar 5 (Detail insiden) → perlu screenshot manual.")
print("         Gambar 7 & 8 → jalankan stage1 (jurnal_kurva_loss_stage1.png).")
print("         Gambar 12   → jalankan stage2 (jurnal_kurva_loss_acc_stage2.png).")
