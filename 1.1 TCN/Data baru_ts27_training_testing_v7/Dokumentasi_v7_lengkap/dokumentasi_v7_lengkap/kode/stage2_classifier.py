# =============================
# STAGE 2 — MLP CLASSIFIER BERBASIS WINDOW (v7)
# Independen dari Stage 1.
# Granularitas : 1 sampel = 1 window 30 detik (117 baris @ 256 ms)
# Fitur        : 21 parameter x 6 statistik (mean, std, max, min, skew, rate-of-change) = 126 dim
# Label window : Warning jika ada fault_columns > 0 di dalam window, selain itu Healthy
# Split        : kronologis per HARI (80/20), window mengikuti hari induknya (tanpa leakage)
# Keputusan hari: Warning jika minimal 1 window diprediksi Warning
# Output       : model_stage2_classifier.pkl, scaler_stage2.pkl, evidence/
# =============================

import os
import glob
import csv
import json
import warnings
from datetime import datetime, time

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

# ==================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
EVIDENCE_DIR = os.path.join(SCRIPT_DIR, "evidence")
LOG_FILE     = os.path.join(SCRIPT_DIR, "log_stage2.txt")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

TRAIN_RATIO     = 0.8
N_TAKE          = 200_000
START_TIME      = time(3, 0, 0)
END_TIME        = time(18, 16, 35)
SAMPLE_INTERVAL = 0.256            # detik per baris (hasil ukur dari timestamp)
WINDOW_SECONDS  = 30
ROWS_PER_WINDOW = int(WINDOW_SECONDS / SAMPLE_INTERVAL)   # 117
WINDOWS_PER_DAY = N_TAKE // ROWS_PER_WINDOW               # 1709

HIDDEN          = (64, 32)
ALPHA           = 1e-3
MAX_ITER        = 500
VAL_FRACTION    = 0.15
MAIN_SEED       = 42
EXTRA_SEEDS     = [1, 7, 21, 99]    # untuk melaporkan variasi antar seed
CV_FOLDS        = 5
# ==================================================================

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
STAT_NAMES = ['mean', 'std', 'max', 'min', 'skew', 'roc']
n_params = len(target_columns)
INPUT_DIM = n_params * len(STAT_NAMES)
_needed = set(['ts_date'] + target_columns + fault_columns)


def log(t=""):
    print(t)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(t + '\n')


def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")


def read_and_crop(fp):
    df = pd.read_csv(fp, encoding='utf-8-sig', sep=';', decimal=',',
                     usecols=lambda c: c.strip() in _needed, low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'),
                                   format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['ts_date'])
    for c in target_columns + fault_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce') if c in df.columns else np.nan
    df[target_columns + fault_columns] = df[target_columns + fault_columns].ffill().bfill()
    d0 = df['ts_date'].dt.date.iloc[0]
    df = df[(df['ts_date'] >= datetime.combine(d0, START_TIME)) &
            (df['ts_date'] <= datetime.combine(d0, END_TIME))]
    if len(df) < N_TAKE:
        return None
    return df.iloc[:N_TAKE].reset_index(drop=True)


def day_to_windows(df):
    """Satu hari -> (X [W,126], y [W]) secara vektorisasi."""
    n_use = WINDOWS_PER_DAY * ROWS_PER_WINDOW
    v = df[target_columns].values[:n_use].astype(np.float64).reshape(WINDOWS_PER_DAY, ROWS_PER_WINDOW, n_params)
    f = df[fault_columns].values[:n_use].reshape(WINDOWS_PER_DAY, ROWS_PER_WINDOW, len(fault_columns))
    mean = v.mean(axis=1)
    std = v.std(axis=1)
    mx = v.max(axis=1)
    mn = v.min(axis=1)
    std_safe = np.where(std == 0, 1e-8, std)
    skew = ((v - mean[:, None, :]) ** 3).mean(axis=1) / (std_safe ** 3)
    roc = np.abs(np.diff(v, axis=1)).mean(axis=1)
    X = np.concatenate([mean, std, mx, mn, skew, roc], axis=1).astype(np.float32)
    y = (f > 0).any(axis=(1, 2)).astype(np.int64)
    return X, y


def day_metrics(y_true, y_pred):
    return dict(
        acc=accuracy_score(y_true, y_pred) * 100,
        f1=f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        rec_warn=recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
        prec_warn=precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
        cm=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    )


def aggregate_days(pred_windows, n_days):
    p = pred_windows.reshape(n_days, WINDOWS_PER_DAY)
    return (p == 1).any(axis=1).astype(int), (p == 1).sum(axis=1)


def new_mlp(seed):
    return MLPClassifier(hidden_layer_sizes=HIDDEN, solver='adam', alpha=ALPHA, max_iter=MAX_ITER,
                         early_stopping=True, validation_fraction=VAL_FRACTION, random_state=seed)


# =============================
# BACA DATA
# =============================
log(f"\n{'=' * 70}\nSTAGE 2 v7 MLP WINDOW {WINDOW_SECONDS} DETIK | {datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 70}")
csv_files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
                    if "hasil" not in os.path.basename(f).lower()
                    and "prediksi" not in os.path.basename(f).lower()
                    and "inference" not in os.path.basename(f).lower()], key=extract_date)
log(f"File CSV ditemukan : {len(csv_files)}")
log(f"Window             : {WINDOW_SECONDS} detik = {ROWS_PER_WINDOW} baris | {WINDOWS_PER_DAY} window/hari")

X_days, y_days, dates, skipped = [], [], [], []
for fp in csv_files:
    d = read_and_crop(fp)
    ds = os.path.basename(fp)[:8]
    if d is None:
        skipped.append(ds)
        continue
    Xd, yd = day_to_windows(d)
    X_days.append(Xd)
    y_days.append(yd)
    dates.append(ds)

total_days = len(dates)
log(f"Hari terpakai      : {total_days} | dilewati (baris < {N_TAKE}): {len(skipped)} {skipped}")
day_label = np.array([int(y.any()) for y in y_days])

n_train_days = max(2, int(total_days * TRAIN_RATIO))
n_test_days = total_days - n_train_days
log(f"Split kronologis   : train {n_train_days} hari ({dates[0]}..{dates[n_train_days - 1]}) | "
    f"test {n_test_days} hari ({dates[n_train_days]}..{dates[-1]})")

X_train = np.vstack(X_days[:n_train_days])
y_train = np.concatenate(y_days[:n_train_days])
X_test = np.vstack(X_days[n_train_days:])
y_test = np.concatenate(y_days[n_train_days:])
train_day_label, test_day_label = day_label[:n_train_days], day_label[n_train_days:]

log(f"Hari Healthy/Warning : train {int((train_day_label == 0).sum())}/{int((train_day_label == 1).sum())} | "
    f"test {int((test_day_label == 0).sum())}/{int((test_day_label == 1).sum())}")
log(f"Window train       : {len(X_train)} (Healthy={int((y_train == 0).sum())}, Warning={int((y_train == 1).sum())})")
log(f"Window test        : {len(X_test)} (Healthy={int((y_test == 0).sum())}, Warning={int((y_test == 1).sum())})")

scaler = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
joblib.dump(scaler, os.path.join(SCRIPT_DIR, "scaler_stage2.pkl"))

# =============================
# MODEL UTAMA
# =============================
log("\n--- Training model utama ---")
mlp = new_mlp(MAIN_SEED)
mlp.fit(Xtr_s, y_train)
log(f"Iterasi sampai berhenti : {mlp.n_iter_} (early stopping, validation_fraction={VAL_FRACTION})")
log(f"Loss akhir              : {mlp.loss_curve_[-1]:.6f} (awal {mlp.loss_curve_[0]:.6f})")
joblib.dump(mlp, os.path.join(SCRIPT_DIR, "model_stage2_classifier.pkl"))
log("model_stage2_classifier.pkl & scaler_stage2.pkl disimpan")

pred_tr = mlp.predict(Xtr_s)
pred_te = mlp.predict(Xte_s)

# --- level window
cm_w_tr = confusion_matrix(y_train, pred_tr, labels=[0, 1])
cm_w_te = confusion_matrix(y_test, pred_te, labels=[0, 1])
log("\n=== EVALUASI LEVEL WINDOW ===")
log(f"Train : Acc={accuracy_score(y_train, pred_tr) * 100:.2f}% "
    f"RecallWarn={recall_score(y_train, pred_tr, pos_label=1, zero_division=0) * 100:.2f}% CM={cm_w_tr.tolist()}")
log(f"Test  : Acc={accuracy_score(y_test, pred_te) * 100:.2f}% "
    f"RecallWarn={recall_score(y_test, pred_te, pos_label=1, zero_division=0) * 100:.2f}% "
    f"PrecWarn={precision_score(y_test, pred_te, pos_label=1, zero_division=0) * 100:.2f}% CM={cm_w_te.tolist()}")
rep_w = classification_report(y_test, pred_te, target_names=['Healthy', 'Warning'], zero_division=0)
log(rep_w)

# --- level hari
day_pred_tr, nwarn_tr = aggregate_days(pred_tr, n_train_days)
day_pred_te, nwarn_te = aggregate_days(pred_te, n_test_days)
m_tr = day_metrics(train_day_label, day_pred_tr)
m_te = day_metrics(test_day_label, day_pred_te)
log("\n=== EVALUASI LEVEL HARI (keputusan: >=1 window Warning) ===")
log(f"Train : Acc={m_tr['acc']:.2f}% F1={m_tr['f1']:.2f}% RecallWarn={m_tr['rec_warn']:.2f}% CM={m_tr['cm']}")
log(f"Test  : Acc={m_te['acc']:.2f}% F1={m_te['f1']:.2f}% RecallWarn={m_te['rec_warn']:.2f}% "
    f"PrecWarn={m_te['prec_warn']:.2f}% CM={m_te['cm']}")
rep_d = classification_report(test_day_label, day_pred_te, target_names=['Healthy', 'Warning'], zero_division=0)
log(rep_d)

names = {0: 'Healthy', 1: 'Warning'}
log("Detail per hari (TEST):")
log(f"  {'Tanggal':>10} {'Aktual':>8} {'Prediksi':>9} {'Benar':>6} {'WinWarnAktual':>14} {'WinWarnPrediksi':>16}")
test_dates = dates[n_train_days:]
per_day_rows = []
for i, ds in enumerate(test_dates):
    n_true_w = int(y_days[n_train_days + i].sum())
    ok = 'OK' if test_day_label[i] == day_pred_te[i] else 'X'
    log(f"  {ds:>10} {names[test_day_label[i]]:>8} {names[day_pred_te[i]]:>9} {ok:>6} {n_true_w:>14} {int(nwarn_te[i]):>16}")
    per_day_rows.append([ds, names[test_day_label[i]], names[day_pred_te[i]], ok, n_true_w, int(nwarn_te[i])])

with open(os.path.join(EVIDENCE_DIR, "hasil_per_hari_test.csv"), 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh, delimiter=';')
    w.writerow(['Tanggal', 'Aktual', 'Prediksi', 'Benar', 'Window_Warning_Aktual', 'Window_Warning_Prediksi'])
    w.writerows(per_day_rows)

# =============================
# VARIASI ANTAR SEED (model dilatih ulang, split sama)
# =============================
log("\n=== VARIASI ANTAR SEED (split sama) ===")
seed_rows = [(MAIN_SEED, m_te['acc'], m_te['f1'], m_te['rec_warn'], m_te['cm'])]
for sd in EXTRA_SEEDS:
    mm = new_mlp(sd).fit(Xtr_s, y_train)
    dp, _ = aggregate_days(mm.predict(Xte_s), n_test_days)
    mt = day_metrics(test_day_label, dp)
    seed_rows.append((sd, mt['acc'], mt['f1'], mt['rec_warn'], mt['cm']))
for r in seed_rows:
    log(f"  seed={r[0]:>3} DayAcc={r[1]:6.2f}% DayF1={r[2]:6.2f}% DayRecWarn={r[3]:6.2f}% CM={r[4]}")
accs = np.array([r[1] for r in seed_rows])
log(f"  Rata-rata DayAcc = {accs.mean():.2f}% | std = {accs.std():.2f}% | min = {accs.min():.2f}% | max = {accs.max():.2f}%")

# =============================
# CROSS-VALIDATION PER HARI (stratified, seluruh hari; window ikut hari induknya)
# =============================
log(f"\n=== {CV_FOLDS}-FOLD CROSS-VALIDATION PER HARI (seluruh {total_days} hari) ===")
skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=MAIN_SEED)
cv_rows = []
all_true, all_pred = np.zeros(total_days, dtype=int), np.zeros(total_days, dtype=int)
for k, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(total_days), day_label), start=1):
    Xk_tr = np.vstack([X_days[i] for i in tr_idx])
    yk_tr = np.concatenate([y_days[i] for i in tr_idx])
    Xk_te = np.vstack([X_days[i] for i in te_idx])
    sck = StandardScaler().fit(Xk_tr)
    mk = new_mlp(MAIN_SEED).fit(sck.transform(Xk_tr), yk_tr)
    dp, _ = aggregate_days(mk.predict(sck.transform(Xk_te)), len(te_idx))
    mt = day_metrics(day_label[te_idx], dp)
    all_true[te_idx], all_pred[te_idx] = day_label[te_idx], dp
    cv_rows.append(mt)
    log(f"  fold {k}: n_test={len(te_idx):>2} DayAcc={mt['acc']:6.2f}% DayRecWarn={mt['rec_warn']:6.2f}% CM={mt['cm']}")
cv_accs = np.array([r['acc'] for r in cv_rows])
cv_all = day_metrics(all_true, all_pred)
log(f"  Rata-rata fold DayAcc = {cv_accs.mean():.2f}% (std {cv_accs.std():.2f}%)")
log(f"  Gabungan seluruh fold : Acc={cv_all['acc']:.2f}% F1={cv_all['f1']:.2f}% "
    f"RecallWarn={cv_all['rec_warn']:.2f}% PrecWarn={cv_all['prec_warn']:.2f}% CM={cv_all['cm']}")

# =============================
# PLOT
# =============================
def plot_cm(cm, title, fn, subtitle):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Healthy', 'Warning'],
           yticklabels=['Healthy', 'Warning'], xlabel='Label Prediksi', ylabel='Label Aktual')
    ax.set_title(title, fontweight='bold')
    cm = np.array(cm)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center', fontsize=15, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.text(0.5, 0.01, subtitle, ha='center', fontsize=9, fontweight='bold', color='#1565C0')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(os.path.join(EVIDENCE_DIR, fn), dpi=300, bbox_inches='tight')
    plt.close()


plot_cm(m_tr['cm'], "Confusion Matrix Level Hari - Training", "cm_hari_train.png",
        f"Acc {m_tr['acc']:.2f}%  F1 {m_tr['f1']:.2f}%  Recall Warning {m_tr['rec_warn']:.2f}%")
plot_cm(m_te['cm'], "Confusion Matrix Level Hari - Test", "cm_hari_test.png",
        f"Acc {m_te['acc']:.2f}%  F1 {m_te['f1']:.2f}%  Recall Warning {m_te['rec_warn']:.2f}%")
plot_cm(cm_w_te, "Confusion Matrix Level Window - Test", "cm_window_test.png",
        f"{len(y_test):,} window uji, hanya {int(y_test.sum())} window Warning")
plot_cm(cv_all['cm'], f"Confusion Matrix Level Hari - {CV_FOLDS}-Fold CV (gabungan)", "cm_hari_cv.png",
        f"Acc {cv_all['acc']:.2f}%  F1 {cv_all['f1']:.2f}%  Recall Warning {cv_all['rec_warn']:.2f}%")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, 'r-', label='Training loss')
if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_ is not None:
    ax2 = ax.twinx()
    ax2.plot(range(1, len(mlp.validation_scores_) + 1), mlp.validation_scores_, 'g-', label='Akurasi validasi')
    ax2.set_ylabel('Akurasi validasi')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy loss'); ax.grid(alpha=0.3)
ax.set_title('Kurva Training MLP Stage 2 (window 30 detik)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "kurva_loss_stage2.png"), dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(max(9, n_test_days * 0.8), 4.8))
xp = np.arange(n_test_days)
cc = {0: '#2196F3', 1: '#FF9800'}
ax.bar(xp - 0.2, test_day_label + 1, 0.4, color=[cc[v] for v in test_day_label], alpha=0.9, label='Aktual')
ax.bar(xp + 0.2, day_pred_te + 1, 0.4, color=[cc[v] for v in day_pred_te], alpha=0.5,
       edgecolor='k', label='Prediksi')
for i in range(n_test_days):
    ax.text(xp[i], max(test_day_label[i], day_pred_te[i]) + 1.15,
            'OK' if test_day_label[i] == day_pred_te[i] else 'X', ha='center', fontweight='bold',
            color='#2E7D32' if test_day_label[i] == day_pred_te[i] else '#B71C1C')
ax.set_xticks(xp); ax.set_xticklabels(test_dates, rotation=60, ha='right', fontsize=8)
ax.set_yticks([1, 2]); ax.set_yticklabels(['Healthy', 'Warning']); ax.set_ylim(0, 3)
ax.set_title(f"Aktual vs Prediksi per Hari (Test) - Acc {m_te['acc']:.2f}%", fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "aktual_vs_prediksi_hari.png"), dpi=300)
plt.close()

# =============================
# RINGKASAN JSON + TXT UNTUK LAPORAN
# =============================
summary = dict(
    generated=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    total_days=total_days, skipped_days=skipped, train_days=n_train_days, test_days=n_test_days,
    first_date=dates[0], last_date=dates[-1],
    window_seconds=WINDOW_SECONDS, rows_per_window=ROWS_PER_WINDOW, windows_per_day=WINDOWS_PER_DAY,
    n_features=INPUT_DIM, stats=STAT_NAMES,
    window_counts=dict(train=len(y_train), train_warning=int(y_train.sum()),
                       test=len(y_test), test_warning=int(y_test.sum())),
    day_counts=dict(train_healthy=int((train_day_label == 0).sum()), train_warning=int((train_day_label == 1).sum()),
                    test_healthy=int((test_day_label == 0).sum()), test_warning=int((test_day_label == 1).sum())),
    hyperparameters=dict(hidden=list(HIDDEN), alpha=ALPHA, solver='adam', max_iter=MAX_ITER,
                         early_stopping=True, validation_fraction=VAL_FRACTION, seed=MAIN_SEED,
                         iterations_run=int(mlp.n_iter_)),
    window_test=dict(acc=accuracy_score(y_test, pred_te) * 100,
                     rec_warn=recall_score(y_test, pred_te, pos_label=1, zero_division=0) * 100,
                     prec_warn=precision_score(y_test, pred_te, pos_label=1, zero_division=0) * 100,
                     cm=cm_w_te.tolist()),
    day_train=m_tr, day_test=m_te,
    seeds=[dict(seed=r[0], acc=r[1], f1=r[2], rec_warn=r[3], cm=r[4]) for r in seed_rows],
    seeds_mean_acc=float(accs.mean()), seeds_std_acc=float(accs.std()),
    cv=dict(folds=CV_FOLDS, fold_acc=cv_accs.tolist(), mean_acc=float(cv_accs.mean()),
            std_acc=float(cv_accs.std()), pooled=cv_all),
)
with open(os.path.join(EVIDENCE_DIR, "ringkasan_stage2.json"), 'w', encoding='utf-8') as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)

log("\nevidence/ : cm_hari_train.png, cm_hari_test.png, cm_window_test.png, cm_hari_cv.png, "
    "kurva_loss_stage2.png, aktual_vs_prediksi_hari.png, hasil_per_hari_test.csv, ringkasan_stage2.json")
log("STAGE 2 v7 SELESAI.")
