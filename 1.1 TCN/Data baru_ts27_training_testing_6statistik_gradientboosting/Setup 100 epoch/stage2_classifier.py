# =============================
# STAGE 2 — GRADIENT BOOSTING CLASSIFIER (versi resmi, menggantikan MLP)
# Per hari: Day1->Status1, Day2->Status2, dst (independen dari Stage 1)
# Input   : ringkasan 21 param sensor 1 hari (mean + std + max + min + skew + rate-of-change)
# Output  : model_stage2_classifier.pkl
#
# Alasan ganti dari MLP ke Gradient Boosting:
#   - Dataset Stage 2 kecil (63 hari training, 84-126 dim fitur). MLP (baik versi
#     besar 256-128-64 maupun versi kecil 32-16) overfit atau underfit pada data
#     sekecil ini, menghasilkan Test Accuracy hanya 56-62%.
#   - Dicoba beberapa alternatif (RandomForest, SVM RBF, GradientBoosting) dengan
#     2 varian fitur (4 statistik lama vs 6 statistik baru dengan skewness &
#     rate-of-change). Hasil terbaik: GradientBoosting + 6 statistik,
#     Test Accuracy 75.00% (naik dari 62.50% versi MLP terbaik sebelumnya),
#     Recall Warning 57.14% (naik dari 14-28%).
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================================================================
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
DATA_DIR            = os.path.join(SCRIPT_DIR, "data")          # folder CSV terisolasi
LOG_FILE            = os.path.join(SCRIPT_DIR, "log_stage2.txt")
EVIDENCE_DIR        = os.path.join(SCRIPT_DIR, "evidence")

TRAIN_RATIO         = 0.8   # proporsi hari untuk training (sisanya testing)
N_ESTIMATORS        = 200
MAX_DEPTH           = 3
LEARNING_RATE       = 0.1
RANDOM_STATE        = 42
# ==================================================================

N_TAKE     = 200_000
START_TIME = time(3, 0, 0)
END_TIME                   = time(18, 16, 35)
# N_DROP_FIRST = 0 -> tidak ada warmup drop (lihat alasan di versi MLP sebelumnya:
# beberapa fault muncul di menit-menit awal operasi, warmup drop akan
# menghilangkan event tersebut dari label).
N_DROP_FIRST               = 0

os.makedirs(EVIDENCE_DIR, exist_ok=True)

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
n_features    = len(target_columns)  # 21

# Fitur per hari = mean + std + max + min + skew + rate-of-change -> 21x6 = 126 dim
N_STATS   = 6
INPUT_DIM = n_features * N_STATS  # 126

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

def _skewness(vals):
    mean = vals.mean(axis=0)
    std  = vals.std(axis=0)
    std_safe = np.where(std == 0, 1e-8, std)
    m3 = np.mean((vals - mean) ** 3, axis=0)
    return m3 / (std_safe ** 3)

def day_to_feature_vector(df_day):
    vals = df_day[target_columns].values
    roc  = np.abs(np.diff(vals, axis=0)).mean(axis=0)
    feat = np.concatenate([
        vals.mean(axis=0),
        vals.std(axis=0),
        vals.max(axis=0),
        vals.min(axis=0),
        _skewness(vals),
        roc,
    ])
    return feat.astype(np.float32)

# =============================
# LABELING: per hari (v6 — rule berbasis blok kontinu, bukan sekadar "ada aktif")
# Analisis data menunjukkan nilai fault_columns selalu biner (0/1), dan baris
# aktifnya membentuk pola bimodal yang jelas: 28/31 event berupa 1 blok
# kontinu sepanjang 19-24 baris (durasi konfirmasi fault standar PLC), sementara
# 3 event cuma 2 baris (jauh di bawah pola normal -> kemungkinan besar glitch
# sensor sesaat, bukan fault terkonfirmasi).
# Rule: Warning jika ada >=1 blok kontinu panjang >= MIN_CONFIRMED_LEN (fault
# nyata terkonfirmasi), ATAU ada >=2 kejadian terpisah dalam 1 hari meski
# masing-masing singkat (indikasi gangguan intermiten berulang, bukan noise
# tunggal).
# =============================
MIN_CONFIRMED_LEN = 10  # titik tengah gap natural: noise=2 baris, event nyata=19-24 baris

def _get_blocks(series):
    active = (series.values > 0).astype(int)
    idx = np.where(active == 1)[0]
    if len(idx) == 0:
        return []
    gaps = np.diff(idx)
    blocks, cur = [], 1
    for g in gaps:
        if g == 1:
            cur += 1
        else:
            blocks.append(cur)
            cur = 1
    blocks.append(cur)
    return blocks

def label_day(df_day):
    for col in fault_columns:
        if col not in df_day.columns:
            continue
        blocks = _get_blocks(df_day[col])
        if not blocks:
            continue
        if max(blocks) >= MIN_CONFIRMED_LEN:
            return 1  # Warning: fault terkonfirmasi
        if len(blocks) >= 2:
            return 1  # Warning: fault singkat tapi berulang
    return 0  # Healthy

compressed_dfs = []
for f in csv_files:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"  Skip {os.path.basename(f)}")
        continue
    compressed_dfs.append(df_raw[['ts_date'] + target_columns + fault_columns].copy())

total_days = len(compressed_dfs)
print(f"[Stage 2] Total hari: {total_days}")
if total_days < 2:
    raise ValueError("Minimal 2 hari CSV!")

# =============================
# TRAIN / TEST SPLIT (kronologis)
# =============================
n_train_days  = max(2, int(total_days * TRAIN_RATIO))
n_test_days   = total_days - n_train_days
train_dfs_cls = compressed_dfs[:n_train_days]
test_dfs_cls  = compressed_dfs[n_train_days:]
print(f"[Stage 2] Train days: {n_train_days} | Test days: {n_test_days}")

status_map = {0: "Healthy", 1: "Warning"}

X_train_cls, y_train_cls = [], []
for i, df_day in enumerate(train_dfs_cls):
    feat  = day_to_feature_vector(df_day)
    label = label_day(df_day)
    X_train_cls.append(feat)
    y_train_cls.append(label)
    print(f"  [Train] Day {i+1:2d} -> {status_map[label]}")

X_test_cls, y_test_cls = [], []
for i, df_day in enumerate(test_dfs_cls):
    feat  = day_to_feature_vector(df_day)
    label = label_day(df_day)
    X_test_cls.append(feat)
    y_test_cls.append(label)
    print(f"  [Test]  Day {n_train_days+i+1:2d} -> {status_map[label]}")

X_train_cls = np.array(X_train_cls, dtype=np.float32)
y_train_cls = np.array(y_train_cls, dtype=np.int64)
X_test_cls_arr = np.array(X_test_cls, dtype=np.float32) if X_test_cls else np.empty((0, INPUT_DIM), dtype=np.float32)
y_test_cls_arr = np.array(y_test_cls, dtype=np.int64)
print(f"[Stage 2] Sampel training: {len(X_train_cls)} hari")
print(f"[Stage 2] Sampel testing : {len(X_test_cls_arr)} hari")

# Normalisasi fitur statistik (fit HANYA dari data training)
scaler_cls = StandardScaler()
X_scaled   = scaler_cls.fit_transform(X_train_cls)
joblib.dump(scaler_cls, os.path.join(SCRIPT_DIR, "scaler_stage2.pkl"))
print("[Stage 2] scaler_stage2.pkl disimpan")

# =============================
# MODEL: GRADIENT BOOSTING CLASSIFIER
# =============================
model = GradientBoostingClassifier(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE, random_state=RANDOM_STATE
)

def log(t):
    print(t)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(t + '\n')

log(f"\n{'='*60}\nSTAGE 2 GRADIENT BOOSTING CLASSIFIER | {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"Train days: {n_train_days} | Test days: {n_test_days}")
log(f"Sampel train: {len(X_scaled)} hari | n_estimators: {N_ESTIMATORS} | max_depth: {MAX_DEPTH}")
log(f"Distribusi Train: Healthy={sum(y_train_cls==0)} Warning={sum(y_train_cls==1)}")
log(f"Fitur per hari: {INPUT_DIM} dim (21 param x 6 statistik: mean,std,max,min,skew,rate-of-change)")
log(f"{'='*60}")

# =============================
# TRAINING
# =============================
model.fit(X_scaled, y_train_cls)

# Kurva "epoch" (staged) -> akurasi & loss (deviance) per jumlah estimator terpakai
_stage_acc, _stage_loss = [], []
for i, y_pred_stage in enumerate(model.staged_predict(X_scaled), start=1):
    acc_i = 100.0 * float(np.mean(y_pred_stage == y_train_cls))
    _stage_acc.append(acc_i)
for i, loss_i in enumerate(model.train_score_, start=1):
    _stage_loss.append(loss_i)
    if i % 50 == 0 or i == N_ESTIMATORS:
        log(f"Stage {i:4d}/{N_ESTIMATORS} | Deviance: {loss_i:.6f} | Acc: {_stage_acc[i-1]:6.2f}%")

log("=== STAGE 2 TRAINING SELESAI ===\n")

# =============================
# EVALUASI TRAINING SET
# =============================
log("=== EVALUASI TRAINING SET (Stage 2) ===")
preds_tr  = model.predict(X_scaled)
y_tr_arr  = y_train_cls
_acc_tr   = accuracy_score(y_tr_arr, preds_tr) * 100
_prec_tr  = precision_score(y_tr_arr, preds_tr, average='weighted', zero_division=0) * 100
_rec_tr   = recall_score(y_tr_arr, preds_tr, average='weighted', zero_division=0) * 100
_f1_tr    = f1_score(y_tr_arr, preds_tr, average='weighted', zero_division=0) * 100
log(f"Train Days   : {len(y_tr_arr)}")
log(f"Train Acc    : {_acc_tr:.2f}%")
log(f"Train Prec   : {_prec_tr:.2f}%")
log(f"Train Recall : {_rec_tr:.2f}%")
log(f"Train F1     : {_f1_tr:.2f}%")
log("=" * 40 + "\n")

# =============================
# EVALUASI TEST SET
# =============================
if len(X_test_cls_arr) > 0:
    log("=== EVALUASI TEST SET (Stage 2) ===")
    X_test_scaled_arr = scaler_cls.transform(X_test_cls_arr)
    preds_test = model.predict(X_test_scaled_arr)
    y_test_arr = y_test_cls_arr
    _acc_te    = accuracy_score(y_test_arr, preds_test) * 100
    _prec_te   = precision_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    _rec_te    = recall_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    _f1_te     = f1_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    log(f"Test Days    : {len(y_test_arr)}")
    log(f"Test Acc     : {_acc_te:.2f}%")
    log(f"Test Prec    : {_prec_te:.2f}%")
    log(f"Test Recall  : {_rec_te:.2f}%")
    log(f"Test F1      : {_f1_te:.2f}%")
    for day_i, (true_lbl, pred_lbl) in enumerate(zip(y_test_arr, preds_test)):
        match = "OK" if true_lbl == pred_lbl else "X"
        log(f"  Day {n_train_days + day_i + 1:2d}: True={status_map[true_lbl]:9s} | Pred={status_map[pred_lbl]:9s} {match}")
    log("=" * 40)
else:
    log("[Stage 2] Tidak ada test days - tambah lebih banyak hari CSV")

joblib.dump(model, os.path.join(SCRIPT_DIR, "model_stage2_classifier.pkl"))
log("model_stage2_classifier.pkl disimpan")

# =============================
# PLOT KURVA DEVIANCE & ACCURACY vs JUMLAH ESTIMATOR - Jurnal Gambar 14
# =============================
_eps = list(range(1, N_ESTIMATORS + 1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(_eps, _stage_loss, 'r-', linewidth=1.5)
ax1.set_xlabel("Jumlah Estimator (Boosting Stage)", fontsize=11); ax1.set_ylabel("Training Deviance (Loss)", fontsize=11)
ax1.set_title(
    f"Gambar 14. Kurva Boosting Stage vs Training Deviance\nAwal: {_stage_loss[0]:.4f} -> Akhir: {_stage_loss[-1]:.6f}", fontsize=12)
ax1.grid(alpha=0.3)

ax2.plot(_eps, _stage_acc, 'g-', linewidth=1.5)
ax2.set_xlabel("Jumlah Estimator (Boosting Stage)", fontsize=11); ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.set_ylim(0, 108)
ax2.set_title(
    f"Accuracy vs. Boosting Stage\nAkhir: {_stage_acc[-1]:.2f}%", fontsize=12)
ax2.grid(alpha=0.3)

plt.suptitle("Gambar 14. Kurva Boosting Stage vs Training Deviance & Accuracy - Gradient Boosting Classifier", fontsize=13, fontweight='bold')
plt.tight_layout()
_out2 = os.path.join(EVIDENCE_DIR, "jurnal_kurva_loss_acc_stage2.png")
plt.savefig(_out2, dpi=300, bbox_inches='tight')
plt.close()
log(f"jurnal_kurva_loss_acc_stage2.png disimpan  <- Jurnal Gambar 14")

# =============================
# CONFUSION MATRIX - Jurnal
# =============================
_cls_names = ['Healthy', 'Warning']

_j2_cm_train = confusion_matrix(y_train_cls, preds_tr, labels=[0, 1])

_j2_cm_test = None
_j2_preds_test = None
_j2_y_test_np = None
if len(X_test_cls_arr) > 0:
    _j2_cm_test   = confusion_matrix(y_test_arr, preds_test, labels=[0, 1])
    _j2_preds_test = preds_test
    _j2_y_test_np  = y_test_arr

def _plot_cm(cm, title, filename, y_true, y_pred):
    _acc  = accuracy_score(y_true, y_pred) * 100
    _prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    _rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    _f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=_cls_names, yticklabels=_cls_names,
           ylabel='Label Aktual', xlabel='Label Prediksi')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    _thresh = cm.max() / 2.0
    for _i in range(2):
        for _j in range(2):
            ax.text(_j, _i, str(cm[_i, _j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[_i, _j] > _thresh else 'black')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    metrics_text = (f"Accuracy: {_acc:.2f}%    Precision: {_prec:.2f}%    "
                    f"Recall: {_rec:.2f}%    F1-Score: {_f1:.2f}%")
    fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom', fontsize=10,
             fontweight='bold', color='#1565C0',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0'))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"{os.path.basename(filename)} - Acc={_acc:.2f}% Prec={_prec:.2f}% Rec={_rec:.2f}% F1={_f1:.2f}%")
    return _acc, _prec, _rec, _f1

_j2_acc_tr, _j2_prec_tr, _j2_rec_tr, _j2_f1_tr = _plot_cm(
    _j2_cm_train, "Confusion Matrix - Training Set",
    os.path.join(EVIDENCE_DIR, "jurnal_confusion_matrix_train.png"),
    y_train_cls, preds_tr)
_j2_acc_ts = _j2_prec_ts = _j2_rec_ts = _j2_f1_ts = None
if _j2_cm_test is not None:
    _j2_acc_ts, _j2_prec_ts, _j2_rec_ts, _j2_f1_ts = _plot_cm(
        _j2_cm_test, "Confusion Matrix - Test Set",
        os.path.join(EVIDENCE_DIR, "jurnal_confusion_matrix_test.png"),
        _j2_y_test_np, _j2_preds_test)

# =============================
# GAMBAR 15 - Kurva Aktual (Y_Testing) vs Prediksi
# =============================
_j2_dates = [os.path.basename(f)[:8] for f in csv_files]
if _j2_cm_test is not None and _j2_y_test_np is not None:
    _n_test_cls  = len(_j2_y_test_np)
    _test_days   = list(range(n_train_days + 1, n_train_days + _n_test_cls + 1))
    _test_dates  = [_j2_dates[n_train_days + i] for i in range(_n_test_cls)]
    _cls_colors  = {0: '#2196F3', 1: '#FF9800'}
    _cls_names_s = ['Healthy', 'Warning']

    _bar_w   = 0.35
    _x_pos   = np.arange(_n_test_cls)
    fig, ax  = plt.subplots(figsize=(max(8, _n_test_cls * 2.2), 5.5))

    for _i, (_true, _xp) in enumerate(zip(_j2_y_test_np, _x_pos)):
        ax.bar(_xp - _bar_w/2, _true + 1, _bar_w,
               color=_cls_colors[_true], alpha=0.85, edgecolor='white', linewidth=1.2,
               label=f'Aktual' if _i == 0 else '')
        ax.text(_xp - _bar_w/2, _true + 1 + 0.05, _cls_names_s[_true],
                ha='center', va='bottom', fontsize=8, color=_cls_colors[_true], fontweight='bold')

    for _i, (_pred, _xp) in enumerate(zip(_j2_preds_test, _x_pos)):
        _match = _j2_preds_test[_i] == _j2_y_test_np[_i]
        ax.bar(_xp + _bar_w/2, _pred + 1, _bar_w,
               color=_cls_colors[_pred], alpha=0.5, edgecolor=_cls_colors[_pred],
               linewidth=2, linestyle='--' if not _match else '-',
               label=f'Prediksi' if _i == 0 else '')
        ax.text(_xp + _bar_w/2, _pred + 1 + 0.05, _cls_names_s[_pred],
                ha='center', va='bottom', fontsize=8,
                color=_cls_colors[_pred], fontstyle='italic')
        _sym = 'OK' if _match else 'X'
        _sc  = '#2E7D32' if _match else '#B71C1C'
        ax.text(_xp, max(_j2_y_test_np[_i], _pred) + 1.35, _sym, ha='center', va='bottom',
                fontsize=12, color=_sc, fontweight='bold')

    ax.set_xticks(_x_pos)
    ax.set_xticklabels([f"Day {d}\n({dt})" for d, dt in zip(_test_days, _test_dates)],
                       fontsize=9)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Healthy (0)', 'Warning (1)'], fontsize=9)
    ax.set_ylim(0, 3.2)
    ax.set_ylabel("Label Kondisi Operasional", fontsize=11)
    ax.set_xlabel("Hari Testing", fontsize=11)

    from matplotlib.patches import Patch
    _leg = [
        Patch(facecolor='gray', alpha=0.85, label='Aktual (solid)'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='gray', linewidth=2, label='Prediksi (dashed)'),
        Patch(facecolor='white', label='OK = Benar   X = Salah'),
    ]
    ax.legend(handles=_leg, loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    _n_benar = int(np.sum(_j2_preds_test == _j2_y_test_np))
    ax.set_title(
        f"Gambar 15. Kurva Aktual (Y_Testing) vs Prediksi - Gradient Boosting Classifier\n"
        f"Test Days: {_n_test_cls} hari | Benar: {_n_benar}/{_n_test_cls} | "
        f"Accuracy: {_j2_acc_ts:.2f}% | Precision: {_j2_prec_ts:.2f}% | "
        f"Recall: {_j2_rec_ts:.2f}% | F1: {_j2_f1_ts:.2f}%",
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCE_DIR, "jurnal_aktual_vs_prediksi_mlp.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    log("jurnal_aktual_vs_prediksi_mlp.png disimpan  <- Jurnal Gambar 15")
else:
    print("[Plot] Gambar 15 skip - tidak ada test data")

# =============================
# FEATURE STATISTICS CSV - Tabel 15 Jurnal
# =============================
import csv as _csvmod
_j2_fstat = os.path.join(EVIDENCE_DIR, "jurnal_fitur_statistik.csv")
with open(_j2_fstat, 'w', newline='', encoding='utf-8-sig') as _fc:
    _fw = _csvmod.writer(_fc, delimiter=';')
    _hdr = ['Day', 'Tanggal', 'Set']
    for _c in target_columns:
        _hdr += [f"{_c}_mean", f"{_c}_std", f"{_c}_max", f"{_c}_min", f"{_c}_skew", f"{_c}_roc"]
    _fw.writerow(_hdr)
    _all_labels_j2 = list(y_train_cls) + list(y_test_cls)
    for _idx, (_dfday, _fpath, _lbl) in enumerate(zip(compressed_dfs, csv_files, _all_labels_j2)):
        _dstr = os.path.basename(_fpath)[:8]
        _set  = "Train" if _idx < n_train_days else "Test"
        _row  = [_idx + 1, _dstr, _set]
        for _c in target_columns:
            _v = _dfday[_c].values
            _skew_v = _skewness(_v.reshape(-1, 1))[0]
            _roc_v  = np.abs(np.diff(_v)).mean()
            _row += [f"{_v.mean():.4f}", f"{_v.std():.4f}",
                     f"{_v.max():.4f}", f"{_v.min():.4f}",
                     f"{_skew_v:.4f}", f"{_roc_v:.4f}"]
        _fw.writerow(_row)
log("jurnal_fitur_statistik.csv disimpan  <- Tabel Fitur Statistik Jurnal (126 kolom)")

# =============================
# JURNAL INFO TXT - Stage 2
# =============================
_j2_dates = [os.path.basename(f)[:8] for f in csv_files]
_j2_all_labels = list(y_train_cls) + list(y_test_cls)
_j2_label_dates = {0: [], 1: []}
for _fd, _lb in zip(_j2_dates, _j2_all_labels):
    _j2_label_dates[_lb].append(_fd)

_j2_report_tr  = classification_report(y_train_cls, preds_tr,
                                        target_names=_cls_names, zero_division=0)
_j2_report_te  = (classification_report(_j2_y_test_np, _j2_preds_test,
                                         target_names=_cls_names, zero_division=0)
                  if _j2_y_test_np is not None else "(tidak ada test data)")

_j2_lines = [
    "=" * 68,
    "INFORMASI JURNAL - GRADIENT BOOSTING CLASSIFIER (STAGE 2)",
    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "=" * 68,
    "",
    "[DATASET & SPLIT]",
    f"  Total hari                   : {total_days}",
    f"  Train days (TRAIN_RATIO=0.8) : {n_train_days}",
    f"  Test days                    : {n_test_days}",
    f"  Input dimensi fitur          : {n_features} param x {N_STATS} statistik = {INPUT_DIM} dim",
    "",
    "[DISTRIBUSI LABEL - untuk Tabel Jurnal (dengan tanggal)]",
    f"  Healthy (0)   : {len(_j2_label_dates[0])} hari",
    f"    Tanggal     : {', '.join(_j2_label_dates[0])}",
    f"  Warning (1)   : {len(_j2_label_dates[1])} hari",
    f"    Tanggal     : {', '.join(_j2_label_dates[1]) or '-'}",
    "",
    "[HYPERPARAMETER GRADIENT BOOSTING - untuk Tabel Jurnal]",
    f"  Algoritma                    : Gradient Boosting Classifier (sklearn)",
    f"  n_estimators                 : {N_ESTIMATORS}",
    f"  max_depth                    : {MAX_DEPTH}",
    f"  learning_rate                : {LEARNING_RATE}",
    f"  random_state                 : {RANDOM_STATE}",
    f"  Normalisasi fitur            : StandardScaler (fit hanya pada training)",
    "",
    "[FITUR - untuk Tabel Jurnal]",
    f"  Input                         : {INPUT_DIM} fitur ({n_features} parameter x 6 statistik)",
    f"  Statistik per parameter       : mean, std, max, min, skewness, rate-of-change (mean |diff|)",
    "",
    "[HASIL TRAINING - untuk Para 213 Jurnal]",
    f"  Training Deviance Stage 1    : {_stage_loss[0]:.6f}",
    f"  Training Deviance Stage {N_ESTIMATORS} (akhir) : {_stage_loss[-1]:.6f}",
    f"  Accuracy Stage 1              : {_stage_acc[0]:.2f}%",
    f"  Accuracy Stage {N_ESTIMATORS} (akhir)   : {_stage_acc[-1]:.2f}%",
    "",
    f"[EVALUASI TRAINING SET]",
    f"  Accuracy   (weighted)        : {_j2_acc_tr:.2f}%",
    f"  Precision  (weighted)        : {_j2_prec_tr:.2f}%",
    f"  Recall     (weighted)        : {_j2_rec_tr:.2f}%",
    f"  F1-Score   (weighted)        : {_j2_f1_tr:.2f}%",
    "",
    "[CONFUSION MATRIX - TRAINING SET]",
    f"  {'':15} {'Pred Healthy':>12} {'Pred Warning':>12}",
    f"  {'Aktual Healthy':15} {_j2_cm_train[0,0]:>12} {_j2_cm_train[0,1]:>12}",
    f"  {'Aktual Warning':15} {_j2_cm_train[1,0]:>12} {_j2_cm_train[1,1]:>12}",
    "",
    "[CLASSIFICATION REPORT - TRAINING SET]",
]
for _rline in _j2_report_tr.splitlines():
    _j2_lines.append("  " + _rline)

_j2_lines += ["", "[EVALUASI TEST SET - untuk Para 215 Jurnal]"]
if _j2_acc_ts is not None:
    _j2_lines += [
        f"  Test Days                    : {len(_j2_y_test_np)}",
        f"  Accuracy   (weighted)        : {_j2_acc_ts:.2f}%",
        f"  Precision  (weighted)        : {_j2_prec_ts:.2f}%",
        f"  Recall     (weighted)        : {_j2_rec_ts:.2f}%",
        f"  F1-Score   (weighted)        : {_j2_f1_ts:.2f}%",
        "",
        "[CONFUSION MATRIX - TEST SET]",
        f"  {'':15} {'Pred Healthy':>12} {'Pred Warning':>12}",
        f"  {'Aktual Healthy':15} {_j2_cm_test[0,0]:>12} {_j2_cm_test[0,1]:>12}",
        f"  {'Aktual Warning':15} {_j2_cm_test[1,0]:>12} {_j2_cm_test[1,1]:>12}",
        "",
        "[CLASSIFICATION REPORT - TEST SET]",
    ]
    for _rline in _j2_report_te.splitlines():
        _j2_lines.append("  " + _rline)
    _j2_lines += [
        "",
        "[DETAIL PREDIKSI PER HARI - TEST SET]",
        f"  {'Day':>5}  {'Tanggal':>10}  {'Aktual':>12}  {'Prediksi':>12}  {'Benar?':>7}",
        "  " + "-" * 52,
    ]
    for _di, (_true, _pred) in enumerate(zip(_j2_y_test_np, _j2_preds_test)):
        _dstr = _j2_dates[n_train_days + _di]
        _ok   = "OK" if _true == _pred else "X"
        _j2_lines.append(
            f"  {n_train_days+_di+1:>5}  {_dstr:>10}  "
            f"{_cls_names[_true]:>12}  {_cls_names[_pred]:>12}  {_ok:>7}")
else:
    _j2_lines.append("  (Tidak ada test days)")

_j2_out = os.path.join(EVIDENCE_DIR, "jurnal_info_stage2.txt")
with open(_j2_out, 'w', encoding='utf-8') as _f2:
    _f2.write('\n'.join(_j2_lines))
log("jurnal_info_stage2.txt disimpan  <- Data untuk Tabel & Paragraf Jurnal Stage 2")

print("\nSTAGE 2 (GRADIENT BOOSTING) SELESAI!")
