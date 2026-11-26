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
import copy
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
SCRIPT_DIR   = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
EVIDENCE_DIR = os.path.join(SCRIPT_DIR, "evidence")
LOG_FILE     = os.path.join(SCRIPT_DIR, "laporan", "akurasi_per_epoch", "log_tmp.txt")
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

scaler = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=HIDDEN, solver='adam', alpha=ALPHA, max_iter=1, warm_start=True,
                    early_stopping=False, tol=0.0, n_iter_no_change=1000, random_state=MAIN_SEED)
rows = []
for ep in range(1, MAX_ITER + 1):
    mlp.fit(Xtr_s, y_train)
    ptr, pte = mlp.predict(Xtr_s), mlp.predict(Xte_s)
    dtr, _ = aggregate_days(ptr, n_train_days); dte, _ = aggregate_days(pte, n_test_days)
    rows.append(dict(epoch=ep, loss=float(mlp.loss_curve_[-1]),
                     train_window_acc=accuracy_score(y_train, ptr)*100, test_window_acc=accuracy_score(y_test, pte)*100,
                     train_hari_acc=accuracy_score(train_day_label, dtr)*100, test_hari_acc=accuracy_score(test_day_label, dte)*100))
df = pd.DataFrame(rows); df.to_csv(os.path.join(SCRIPT_DIR, "laporan", "akurasi_per_epoch", "akurasi_per_epoch.csv"), index=False)
print(df.loc[[0,1,2,67,68,69,99]].round(4).to_string())
