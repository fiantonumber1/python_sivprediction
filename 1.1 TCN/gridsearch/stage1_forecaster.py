# ============================================================
# GRID SEARCH — TCN FORECASTER
# Mencari window size terbaik dari kandidat [1, 2, 3, 4, 5, 6]
# menggunakan TimeSeriesSplit (walk-forward, tanpa shuffle).
#
# Output:
#   evidence/gs_table.png          — tabel hasil grid search
#   evidence/gs_bar_rmse.png       — bar chart RMSE per window size
#   evidence/gs_bar_mae.png        — bar chart MAE per window size
#   evidence/gs_bar_mape.png       — bar chart MAPE per window size
#   evidence/gs_comparison.png     — subplot RMSE + MAE + MAPE
#   evidence/gs_fold_detail.png    — RMSE per fold tiap window size
#   gridsearch_results.csv         — hasil lengkap grid search
# ============================================================

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
DATA_DIR         = os.path.join(SCRIPT_DIR, "data")
EVIDENCE_DIR     = os.path.join(SCRIPT_DIR, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Grid Search
WINDOW_CANDIDATES = [1, 2, 3, 4, 5, 6]   # kandidat window size (jumlah hari input)
GS_EPOCHS         = 30                    # epoch per fold (cukup untuk perbandingan relatif)
GS_FOLDS          = 3                     # jumlah fold TimeSeriesSplit
GS_METRIC         = "rmse"               # metrik utama: "rmse" | "mae" | "mape"

# Data & model
N_TAKE       = 200_000
FUTURE       = N_TAKE
TRAIN_RATIO  = 0.8
BATCH_SIZE   = 3
START_TIME   = time(3, 0, 0)
END_TIME     = time(18, 16, 35)

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
n_features    = len(target_columns)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(os.cpu_count() or 1)
print(f"Device : {device}  |  Threads : {torch.get_num_threads()}")

# ─────────────────────────────────────────────────────────────
# 1. BACA & PREPROCESSING DATA
# ─────────────────────────────────────────────────────────────
def extract_date(f):
    from datetime import datetime as _dt
    return _dt.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not any(x in os.path.basename(f).lower() for x in ["hasil", "prediksi", "inference"])
], key=extract_date)

print(f"\nData dir  : {DATA_DIR}")
print(f"File CSV  : {len(csv_files)} ditemukan")

_needed = set(['ts_date'] + target_columns + fault_columns)

def read_and_crop(filepath):
    df = pd.read_csv(
        filepath, encoding='utf-8-sig', sep=';', decimal=',',
        usecols=lambda c: c.strip() in _needed, low_memory=False, on_bad_lines='skip'
    )
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])
    for col in target_columns + fault_columns:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors='coerce')
    df[target_columns + fault_columns] = df[target_columns + fault_columns].ffill().bfill()
    d0  = df['ts_date'].dt.date.iloc[0]
    df  = df[(df['ts_date'] >= datetime.combine(d0, START_TIME)) &
              (df['ts_date'] <= datetime.combine(d0, END_TIME))]
    if len(df) < N_TAKE:
        return pd.DataFrame()
    return df.iloc[:N_TAKE].reset_index(drop=True)[['ts_date'] + target_columns + fault_columns]

all_dfs = []
for f in csv_files:
    d = read_and_crop(f)
    if d.empty:
        print(f"  Skip : {os.path.basename(f)}")
    else:
        all_dfs.append(d)

print(f"Total hari valid : {len(all_dfs)}")
if len(all_dfs) < max(WINDOW_CANDIDATES) + 2:
    raise ValueError(f"Perlu minimal {max(WINDOW_CANDIDATES)+2} hari CSV!")

n_train = max(max(WINDOW_CANDIDATES) + 2, int(len(all_dfs) * TRAIN_RATIO))
n_test  = len(all_dfs) - n_train
train_dfs = all_dfs[:n_train]
print(f"Train days : {n_train}  |  Test days : {n_test}\n")

# ─────────────────────────────────────────────────────────────
# 2. ARSITEKTUR TCN (identik untuk semua kandidat)
# ─────────────────────────────────────────────────────────────
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
        dilations  = [1, 2, 4, 8, 16, 32, 64]
        layers, ic = [], n_features
        for i in range(n_blocks):
            layers.append(ResidualBlock(ic, n_ch, ks, dilations[i], dropout))
            ic = n_ch
        self.tcn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dec  = nn.Sequential(
            nn.Linear(n_ch, n_ch), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_ch, n_features * FUTURE)
        )
    def forward(self, x):
        ctx  = self.pool(self.tcn(x.transpose(1, 2))).squeeze(-1)
        pred = self.dec(ctx).view(-1, FUTURE, n_features)
        return pred, ctx

class ForecastDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

criterion = nn.MSELoss()

# ─────────────────────────────────────────────────────────────
# 3. FUNGSI BANTU
# ─────────────────────────────────────────────────────────────
def build_sequences(dfs, ws):
    """Sliding window: ws hari input → 1 hari ke depan."""
    X, y = [], []
    for i in range(len(dfs) - ws):
        seq = np.concatenate([df[target_columns].values for df in dfs[i:i+ws]], axis=0)
        X.append(seq)
        y.append(dfs[i + ws][target_columns].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def scale(X, y, X_val, y_val):
    """MinMax fit dari training fold, transform keduanya."""
    sc     = MinMaxScaler(feature_range=(-0.1, 1.1))
    X_s    = sc.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
    y_s    = sc.transform(y.reshape(-1, n_features)).reshape(y.shape)
    X_vs   = sc.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    y_vs   = sc.transform(y_val.reshape(-1, n_features)).reshape(y_val.shape)
    return X_s, y_s, X_vs, y_vs

def train_one_fold(X_s, y_s, epochs):
    """Latih model TCN baru untuk satu fold."""
    model = TCNForecaster(n_features).to(device)
    opt   = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)
    dl    = DataLoader(
        ForecastDataset(torch.FloatTensor(X_s).to(device),
                        torch.FloatTensor(y_s).to(device)),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    model.train()
    for _ in range(epochs):
        total = sum(
            (opt.zero_grad() or True) and
            (lambda xb, yb: (
                p := model(xb)[0],
                l := criterion(p, yb),
                l.backward() or True,
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) or True,
                opt.step() or True,
                l.item()
            )[-1])(xb, yb)
            for xb, yb in dl
        )
        sched.step(total / len(dl))
    return model

def evaluate(model, X_s, y_s):
    """Hitung RMSE, MAE, MAPE (ruang normalized)."""
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(X_s).to(device))[0].cpu().numpy()
    rmse = float(np.sqrt(np.mean((p - y_s) ** 2)))
    mae  = float(np.mean(np.abs(p - y_s)))
    mask = y_s != 0
    mape = float(np.mean(np.abs((p[mask] - y_s[mask]) / y_s[mask])) * 100) if mask.any() else float('nan')
    return rmse, mae, mape

# ─────────────────────────────────────────────────────────────
# 4. GRID SEARCH
# ─────────────────────────────────────────────────────────────
sep = "─" * 62

print(sep)
print(f"  GRID SEARCH — WINDOW SIZE  [{datetime.now():%H:%M:%S}]")
print(f"  Kandidat  : {WINDOW_CANDIDATES}")
print(f"  Folds     : {GS_FOLDS} (TimeSeriesSplit / walk-forward)")
print(f"  Epoch/fold: {GS_EPOCHS}")
print(f"  Metrik    : {GS_METRIC.upper()}")
print(sep + "\n")

gs_rows   = []   # ringkasan per window size
fold_rows = []   # detail per fold

for ws in WINDOW_CANDIDATES:
    X_all, y_all = build_sequences(train_dfs, ws)
    n_samp       = len(X_all)

    print(f"  WS = {ws} hari  (sampel: {n_samp})")

    if n_samp < GS_FOLDS + 1:
        print(f"       SKIP — sampel tidak cukup untuk {GS_FOLDS} fold\n")
        gs_rows.append({'ws': ws, 'n_samples': n_samp,
                        'rmse': np.nan, 'mae': np.nan, 'mape': np.nan})
        continue

    tscv = TimeSeriesSplit(n_splits=GS_FOLDS)
    f_rmse, f_mae, f_mape = [], [], []

    for fi, (tr_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_s, y_s, X_vs, y_vs = scale(
            X_all[tr_idx], y_all[tr_idx],
            X_all[val_idx], y_all[val_idx]
        )
        model = train_one_fold(X_s, y_s, GS_EPOCHS)
        rmse, mae, mape = evaluate(model, X_vs, y_vs)

        f_rmse.append(rmse); f_mae.append(mae); f_mape.append(mape)
        fold_rows.append({'ws': ws, 'fold': fi+1,
                          'n_train': len(tr_idx), 'n_val': len(val_idx),
                          'rmse': rmse, 'mae': mae, 'mape': mape})

        print(f"       Fold {fi+1}/{GS_FOLDS} — "
              f"train={len(tr_idx):2d}  val={len(val_idx):2d}  |  "
              f"RMSE={rmse:.5f}  MAE={mae:.5f}  MAPE={mape:.2f}%")

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    avg_rmse = float(np.mean(f_rmse))
    avg_mae  = float(np.mean(f_mae))
    avg_mape = float(np.nanmean(f_mape))
    gs_rows.append({'ws': ws, 'n_samples': n_samp,
                    'rmse': avg_rmse, 'mae': avg_mae, 'mape': avg_mape})
    print(f"       Avg   — RMSE={avg_rmse:.5f}  MAE={avg_mae:.5f}  MAPE={avg_mape:.2f}%\n")

# ─────────────────────────────────────────────────────────────
# 5. TABEL HASIL (terminal)
# ─────────────────────────────────────────────────────────────
df_gs   = pd.DataFrame(gs_rows)
df_fold = pd.DataFrame(fold_rows)

valid   = df_gs.dropna(subset=[GS_METRIC]).copy()
valid['rank'] = valid[GS_METRIC].rank(method='min').astype(int)
df_gs   = df_gs.merge(valid[['ws', 'rank']], on='ws', how='left')
df_gs['rank'] = df_gs['rank'].fillna('-')

best     = valid.loc[valid[GS_METRIC].idxmin()]
BEST_WS  = int(best['ws'])

print(sep)
print("  TABEL HASIL GRID SEARCH")
print(sep)
header = f"  {'Rank':>4}  {'WS (hari)':>9}  {'N Sampel':>8}  {'RMSE':>10}  {'MAE':>10}  {'MAPE (%)':>10}"
print(header)
print("  " + "─" * 56)
for _, r in df_gs.iterrows():
    rk   = f"{int(r['rank'])}" if r['rank'] != '-' else "-"
    rmse = f"{r['rmse']:.6f}" if not np.isnan(r['rmse']) else "     N/A"
    mae  = f"{r['mae']:.6f}"  if not np.isnan(r['mae'])  else "     N/A"
    mape = f"{r['mape']:.2f}" if not np.isnan(r['mape']) else "    N/A"
    mark = " ◄ TERBAIK" if int(r['ws']) == BEST_WS else ""
    print(f"  {rk:>4}  {int(r['ws']):>9}  {int(r['n_samples']):>8}  {rmse:>10}  {mae:>10}  {mape:>10}{mark}")
print(sep)
print(f"\n  Window size terpilih : {BEST_WS} hari")
print(f"  Interpretasi         : input {BEST_WS} hari → prediksi 1 hari ke depan")
print(f"  {GS_METRIC.upper()} terbaik (GS)  : {best[GS_METRIC]:.6f}")
print(sep + "\n")

# Simpan CSV
df_gs.to_csv(os.path.join(SCRIPT_DIR, "gridsearch_results.csv"), index=False)
df_fold.to_csv(os.path.join(SCRIPT_DIR, "gridsearch_fold_detail.csv"), index=False)
print("  gridsearch_results.csv      disimpan")
print("  gridsearch_fold_detail.csv  disimpan\n")

# ─────────────────────────────────────────────────────────────
# 6. VISUALISASI
# ─────────────────────────────────────────────────────────────
ws_labels  = [str(ws) for ws in WINDOW_CANDIDATES]
bar_colors = ['#d62728' if ws == BEST_WS else '#4878d0' for ws in WINDOW_CANDIDATES]

def _bar_val(df, col):
    return [float(df.loc[df['ws'] == ws, col].values[0])
            if not np.isnan(float(df.loc[df['ws'] == ws, col].values[0])) else 0
            for ws in WINDOW_CANDIDATES]

rmse_vals = _bar_val(df_gs, 'rmse')
mae_vals  = _bar_val(df_gs, 'mae')
mape_vals = _bar_val(df_gs, 'mape')

# ── 6a. Subplot RMSE + MAE + MAPE ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Hasil Grid Search Window Size — TCN Forecaster\n"
    f"Terbaik: WS = {BEST_WS} hari  "
    f"(RMSE={best['rmse']:.5f}, MAE={best['mae']:.5f}, MAPE={best['mape']:.2f}%)",
    fontsize=13, fontweight='bold'
)

metrics_info = [
    (axes[0], rmse_vals, 'RMSE',     'RMSE (normalized)', '#4878d0'),
    (axes[1], mae_vals,  'MAE',      'MAE (normalized)',  '#6acc65'),
    (axes[2], mape_vals, 'MAPE (%)', 'MAPE (%)',          '#ee854a'),
]

for ax, vals, metric_name, ylabel, base_color in metrics_info:
    colors_m = ['#d62728' if ws == BEST_WS else base_color for ws in WINDOW_CANDIDATES]
    bars     = ax.bar(ws_labels, vals, color=colors_m, edgecolor='white', linewidth=0.8, width=0.6)
    ax.set_xlabel("Window Size (hari)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    max_v = max(v for v in vals if v > 0) if any(v > 0 for v in vals) else 1
    for bar, val, ws in zip(bars, vals, WINDOW_CANDIDATES):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_v * 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold' if ws == BEST_WS else 'normal')

    best_idx = WINDOW_CANDIDATES.index(BEST_WS)
    ax.annotate(
        f'TERBAIK\nWS={BEST_WS}',
        xy=(best_idx, vals[best_idx]),
        xytext=(best_idx + (1 if best_idx < len(WINDOW_CANDIDATES)-2 else -1), max_v * 0.75),
        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
        fontsize=8.5, color='#d62728', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0', edgecolor='#d62728', alpha=0.8)
    )

plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "gs_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  evidence/gs_comparison.png          disimpan")

# ── 6b. Detail RMSE per fold ─────────────────────────────────
if not df_fold.empty:
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fold_colors = ['#4878d0', '#6acc65', '#ee854a']
    x    = np.arange(len(WINDOW_CANDIDATES))
    w    = 0.22
    offset = [-(GS_FOLDS-1)/2 * w + i*w for i in range(GS_FOLDS)]

    for fi in range(GS_FOLDS):
        fold_rmse_per_ws = []
        for ws in WINDOW_CANDIDATES:
            row = df_fold[(df_fold['ws'] == ws) & (df_fold['fold'] == fi+1)]
            fold_rmse_per_ws.append(float(row['rmse'].values[0]) if len(row) > 0 else 0)
        ax2.bar(x + offset[fi], fold_rmse_per_ws, width=w,
                label=f'Fold {fi+1}', color=fold_colors[fi % len(fold_colors)],
                edgecolor='white', linewidth=0.6, alpha=0.85)

    # Garis rata-rata
    ax2.plot(x, rmse_vals, 'k--o', linewidth=1.5, markersize=6, label='Rata-rata', zorder=5)
    ax2.axvline(x=WINDOW_CANDIDATES.index(BEST_WS), color='#d62728',
                linestyle=':', linewidth=2, alpha=0.7, label=f'Best WS={BEST_WS}')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'WS={ws}' for ws in WINDOW_CANDIDATES], fontsize=11)
    ax2.set_xlabel("Window Size (hari)", fontsize=12)
    ax2.set_ylabel("RMSE (normalized)", fontsize=12)
    ax2.set_title(
        f"RMSE per Fold — TimeSeriesSplit Walk-Forward Validation\n"
        f"Window Size Terbaik: WS = {BEST_WS} hari",
        fontsize=13, fontweight='bold'
    )
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCE_DIR, "gs_fold_detail.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  evidence/gs_fold_detail.png         disimpan")

# ── 6c. Tabel sebagai gambar ─────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 3.2))
ax3.axis('off')

col_labels  = ['Rank', 'Window Size\n(hari)', 'Jumlah\nSampel', 'RMSE', 'MAE', 'MAPE (%)', 'Keterangan']
table_data  = []
row_colors  = []

for _, r in df_gs.iterrows():
    rk   = str(int(r['rank'])) if r['rank'] != '-' else '-'
    rmse = f"{r['rmse']:.6f}" if not np.isnan(r['rmse']) else "N/A"
    mae  = f"{r['mae']:.6f}"  if not np.isnan(r['mae'])  else "N/A"
    mape = f"{r['mape']:.2f}" if not np.isnan(r['mape']) else "N/A"
    ket  = "TERBAIK ★" if int(r['ws']) == BEST_WS else ""
    table_data.append([rk, str(int(r['ws'])), str(int(r['n_samples'])), rmse, mae, mape, ket])
    row_colors.append(['#fff7cc' if int(r['ws']) == BEST_WS else 'white'] * len(col_labels))

tbl = ax3.table(
    cellText=table_data,
    colLabels=col_labels,
    cellColours=row_colors,
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.15, 2.0)

# Header style
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#2c5f8a')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

ax3.set_title(
    f"Tabel Grid Search Window Size — TCN Forecaster\n"
    f"Folds: {GS_FOLDS} (TimeSeriesSplit)  |  Epoch/fold: {GS_EPOCHS}  |  "
    f"Metrik utama: {GS_METRIC.upper()}",
    fontsize=11, fontweight='bold', pad=18
)
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "gs_table.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  evidence/gs_table.png               disimpan")

# ── 6d. Line plot rata-rata metrik ───────────────────────────
valid_ws    = [r['ws'] for _, r in df_gs.iterrows() if not np.isnan(r['rmse'])]
valid_rmse  = [r['rmse'] for _, r in df_gs.iterrows() if not np.isnan(r['rmse'])]
valid_mae   = [r['mae']  for _, r in df_gs.iterrows() if not np.isnan(r['mae'])]
valid_mape  = [r['mape'] for _, r in df_gs.iterrows() if not np.isnan(r['mape'])]

if len(valid_ws) > 1:
    fig4, ax4a = plt.subplots(figsize=(10, 4))
    ax4b = ax4a.twinx()

    l1, = ax4a.plot(valid_ws, valid_rmse, 'o-', color='#4878d0', linewidth=2,
                    markersize=7, label='RMSE (kiri)')
    l2, = ax4a.plot(valid_ws, valid_mae,  's--', color='#6acc65', linewidth=2,
                    markersize=7, label='MAE (kiri)')
    l3, = ax4b.plot(valid_ws, valid_mape, '^:', color='#ee854a', linewidth=2,
                    markersize=7, label='MAPE % (kanan)')

    ax4a.axvline(x=BEST_WS, color='#d62728', linestyle='--', linewidth=1.5,
                 alpha=0.7, label=f'Best WS={BEST_WS}')
    ax4a.scatter([BEST_WS], [best['rmse']], color='#d62728', s=120, zorder=5)

    ax4a.set_xlabel("Window Size (hari)", fontsize=12)
    ax4a.set_ylabel("RMSE / MAE (normalized)", fontsize=11, color='#333')
    ax4b.set_ylabel("MAPE (%)", fontsize=11, color='#ee854a')
    ax4a.set_xticks(valid_ws)
    ax4a.set_title(
        f"Tren Metrik Evaluasi per Window Size\nWindow Size Terbaik: {BEST_WS} hari",
        fontsize=13, fontweight='bold'
    )
    ax4a.legend(handles=[l1, l2, l3,
                          plt.Line2D([0],[0], color='#d62728', linestyle='--',
                                     label=f'Best WS={BEST_WS}')],
                fontsize=10, loc='upper right')
    ax4a.grid(alpha=0.3, linestyle='--')
    ax4a.spines[['top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCE_DIR, "gs_trend.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  evidence/gs_trend.png               disimpan")

# ─────────────────────────────────────────────────────────────
# 7. RINGKASAN AKHIR
# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("  RINGKASAN GRID SEARCH")
print(sep)
print(f"  {'WS':>4}  {'N Samp':>6}  {'RMSE':>10}  {'MAE':>10}  {'MAPE(%)':>9}  {'Rank':>4}")
print("  " + "─" * 48)
for _, r in df_gs.sort_values('ws').iterrows():
    rk   = str(int(r['rank'])) if r['rank'] != '-' else '-'
    rmse = f"{r['rmse']:.6f}" if not np.isnan(r['rmse']) else "       N/A"
    mae  = f"{r['mae']:.6f}"  if not np.isnan(r['mae'])  else "       N/A"
    mape = f"{r['mape']:.2f}" if not np.isnan(r['mape']) else "    N/A"
    mark = " ◄" if int(r['ws']) == BEST_WS else ""
    print(f"  {int(r['ws']):>4}  {int(r['n_samples']):>6}  {rmse:>10}  {mae:>10}  {mape:>9}  {rk:>4}{mark}")
print(sep)
print(f"\n  ★  WINDOW SIZE TERBAIK  : {BEST_WS} hari")
print(f"     Konfigurasi          : input {BEST_WS} hari → prediksi 1 hari ke depan")
print(f"     {GS_METRIC.upper()} terbaik (avg CV)  : {best[GS_METRIC]:.6f}")
print(f"\n  Gunakan BEST_WS = {BEST_WS} pada model akhir (stage1_train_final.py)\n")
print(sep)
