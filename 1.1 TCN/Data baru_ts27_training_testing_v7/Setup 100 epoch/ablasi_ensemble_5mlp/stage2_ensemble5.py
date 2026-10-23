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
DATA_DIR     = os.path.join(SCRIPT_DIR, "..", "data")
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
MAX_ITER        = 100
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
                         early_stopping=False, tol=0.0, n_iter_no_change=1000, random_state=seed)


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


ENS_SEEDS = [42, 1, 7, 21, 99]     # ditetapkan sebelum melihat hasil; dijalankan SEKALI

def ens_fit(Xs, y):
    return [new_mlp(s).fit(Xs, y) for s in ENS_SEEDS]

def ens_pred(models, Xs):
    p = np.mean([m.predict_proba(Xs)[:, 1] for m in models], axis=0)
    return (p >= 0.5).astype(int)

def plot_cm2(cm, title, fn, subtitle):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Healthy', 'Warning']); ax.set_yticklabels(['Healthy', 'Warning'])
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual'); ax.set_title(title + "
" + subtitle, fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(EVIDENCE_DIR, fn), dpi=300); plt.close()

log("
--- ENSEMBLE 5 MLP (seed %s), 100 epoch tanpa early stopping, rata-rata probabilitas ---" % ENS_SEEDS)
models = ens_fit(Xtr_s, y_train)
pw_te = ens_pred(models, Xte_s); pw_tr = ens_pred(models, Xtr_s)
dp_te, nw_te = aggregate_days(pw_te, n_test_days); dp_tr, _ = aggregate_days(pw_tr, n_train_days)
m_te = day_metrics(test_day_label, dp_te); m_tr = day_metrics(train_day_label, dp_tr)
log(f"Window test : Acc={accuracy_score(y_test, pw_te)*100:.2f}% RecallWarn={recall_score(y_test, pw_te, pos_label=1)*100:.2f}% CM={confusion_matrix(y_test, pw_te, labels=[0,1]).tolist()}")
log(f"Hari train  : Acc={m_tr['acc']:.2f}% F1={m_tr['f1']:.2f}% RecallWarn={m_tr['rec_warn']:.2f}% CM={m_tr['cm']}")
log(f"Hari TEST   : Acc={m_te['acc']:.2f}% F1={m_te['f1']:.2f}% RecallWarn={m_te['rec_warn']:.2f}% PrecWarn={m_te['prec_warn']:.2f}% CM={m_te['cm']}")
for i in range(n_test_days):
    log(f"  {dates[n_train_days+i]}  aktual={'Warning' if test_day_label[i] else 'Healthy'}  prediksi={'Warning' if dp_te[i] else 'Healthy'}  win_warn_pred={int(nw_te[i])}")
plot_cm2(m_te['cm'], "Confusion Matrix Hari - Test (Ensemble 5 MLP)", "cm_hari_test_ensemble.png",
         f"Acc {m_te['acc']:.2f}%  F1 {m_te['f1']:.2f}%  Recall Warning {m_te['rec_warn']:.2f}%")
joblib.dump(models, os.path.join(SCRIPT_DIR, "model_stage2_ensemble5.pkl"))

log(f"
=== {CV_FOLDS}-FOLD CV PER HARI, ENSEMBLE 5 MLP (seluruh {total_days} hari) ===")
skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=MAIN_SEED)
all_true, all_pred = np.zeros(total_days, dtype=int), np.zeros(total_days, dtype=int); accs = []
for k, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(total_days), day_label), start=1):
    Xk_tr = np.vstack([X_days[i] for i in tr_idx]); yk_tr = np.concatenate([y_days[i] for i in tr_idx])
    Xk_te = np.vstack([X_days[i] for i in te_idx]); sck = StandardScaler().fit(Xk_tr)
    mk = ens_fit(sck.transform(Xk_tr), yk_tr)
    dp, _ = aggregate_days(ens_pred(mk, sck.transform(Xk_te)), len(te_idx))
    mt = day_metrics(day_label[te_idx], dp); accs.append(mt['acc'])
    all_true[te_idx], all_pred[te_idx] = day_label[te_idx], dp
    log(f"  fold {k}: n_test={len(te_idx):>2} DayAcc={mt['acc']:6.2f}% DayRecWarn={mt['rec_warn']:6.2f}% CM={mt['cm']}")
cv_all = day_metrics(all_true, all_pred)
log(f"  Rata-rata fold DayAcc = {np.mean(accs):.2f}% (std {np.std(accs):.2f}%)")
log(f"  Gabungan: Acc={cv_all['acc']:.2f}% F1={cv_all['f1']:.2f}% RecallWarn={cv_all['rec_warn']:.2f}% PrecWarn={cv_all['prec_warn']:.2f}% CM={cv_all['cm']}")
plot_cm2(cv_all['cm'], "Confusion Matrix Hari - 5-Fold CV (Ensemble 5 MLP)", "cm_hari_cv_ensemble.png",
         f"Acc {cv_all['acc']:.2f}%  F1 {cv_all['f1']:.2f}%  Recall Warning {cv_all['rec_warn']:.2f}%")
log("ENSEMBLE SELESAI.")
