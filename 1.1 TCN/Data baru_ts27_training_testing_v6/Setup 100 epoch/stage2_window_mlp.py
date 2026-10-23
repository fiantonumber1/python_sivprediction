import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
N_TAKE, START_TIME, END_TIME = 200_000, time(3, 0, 0), time(18, 16, 35)
TRAIN_RATIO = 0.8

WINDOW_MINUTES  = 1
SAMPLE_INTERVAL = 0.256  # detik per baris (dari analisis timestamp)
ROWS_PER_WINDOW = int(WINDOW_MINUTES * 60 / SAMPLE_INTERVAL)  # ~1171
WINDOWS_PER_DAY = N_TAKE // ROWS_PER_WINDOW  # ~170

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy', 'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
_needed = set(['ts_date'] + target_columns + fault_columns)


def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")


csv_files = sorted(
    [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
     if "hasil" not in f.lower() and "prediksi" not in f.lower() and "inference" not in f.lower()],
    key=extract_date
)
print(f"[Stage 2 - Window] File CSV: {len(csv_files)} ditemukan")
print(f"[Stage 2 - Window] Rows/window: {ROWS_PER_WINDOW} | Windows/day: {WINDOWS_PER_DAY} | Window = {WINDOW_MINUTES} menit")


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
    df = df[(df['ts_date'] >= datetime.combine(d0, START_TIME)) & (df['ts_date'] <= datetime.combine(d0, END_TIME))]
    if len(df) < N_TAKE:
        return pd.DataFrame()
    return df.iloc[:N_TAKE].reset_index(drop=True)[['ts_date'] + target_columns + fault_columns]


def skewness(v):
    m, s = v.mean(0), v.std(0)
    s = np.where(s == 0, 1e-8, s)
    return (np.mean((v - m) ** 3, 0)) / (s ** 3)


def window_feat_and_label(win_df):
    v = win_df[target_columns].values
    roc = np.abs(np.diff(v, axis=0)).mean(0) if len(v) > 1 else np.zeros(v.shape[1])
    feat = np.concatenate([v.mean(0), v.std(0), v.max(0), v.min(0), skewness(v), roc]).astype(np.float32)
    label = 1 if any((win_df[c] > 0).any() for c in fault_columns if c in win_df.columns) else 0
    return feat, label


days, dates = [], []
for f in csv_files:
    d = read_and_crop(f)
    if not d.empty:
        days.append(d)
        dates.append(os.path.basename(f)[:8])

total_days = len(days)
n_train_days = max(2, int(total_days * TRAIN_RATIO))
n_test_days = total_days - n_train_days
print(f"[Stage 2 - Window] Total hari: {total_days} | Train days: {n_train_days} | Test days: {n_test_days}")


def days_to_windows(day_list):
    X, y = [], []
    for d in day_list:
        for w in range(WINDOWS_PER_DAY):
            seg = d.iloc[w * ROWS_PER_WINDOW:(w + 1) * ROWS_PER_WINDOW]
            feat, label = window_feat_and_label(seg)
            X.append(feat)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


X_train, y_train = days_to_windows(days[:n_train_days])
X_test, y_test = days_to_windows(days[n_train_days:])
print(f"[Stage 2 - Window] Sampel train: {len(X_train)} (Healthy={sum(y_train == 0)}, Warning={sum(y_train == 1)})")
print(f"[Stage 2 - Window] Sampel test : {len(X_test)} (Healthy={sum(y_test == 0)}, Warning={sum(y_test == 1)})")

scaler = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
joblib.dump(scaler, os.path.join(SCRIPT_DIR, "scaler_stage2_window.pkl"))

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), solver='adam', alpha=0.001, max_iter=500,
                     early_stopping=True, validation_fraction=0.15, random_state=42)
mlp.fit(Xtr_s, y_train)

pred_tr = mlp.predict(Xtr_s)
pred_te = mlp.predict(Xte_s)

print("\n=== EVALUASI TRAIN (per-window) ===")
print(f"Acc={accuracy_score(y_train, pred_tr) * 100:.2f}% F1={f1_score(y_train, pred_tr, average='weighted') * 100:.2f}%")
print(confusion_matrix(y_train, pred_tr, labels=[0, 1]))

print("\n=== EVALUASI TEST (per-window) ===")
acc_w = accuracy_score(y_test, pred_te) * 100
f1_w = f1_score(y_test, pred_te, average='weighted', zero_division=0) * 100
rec_w = recall_score(y_test, pred_te, pos_label=1, zero_division=0) * 100
prec_w = precision_score(y_test, pred_te, pos_label=1, zero_division=0) * 100
print(f"Acc={acc_w:.2f}% F1={f1_w:.2f}% RecallWarning={rec_w:.2f}% PrecisionWarning={prec_w:.2f}%")
print(confusion_matrix(y_test, pred_te, labels=[0, 1]))
print(classification_report(y_test, pred_te, target_names=['Healthy', 'Warning'], zero_division=0))

print("\n=== AGREGASI KE LEVEL HARI (dari prediksi window) ===")
day_true, day_pred = [], []
idx = 0
test_dates = dates[n_train_days:]
for di, d in enumerate(days[n_train_days:]):
    day_preds = pred_te[idx:idx + WINDOWS_PER_DAY]
    day_true_lbl = 1 if any((d[c] > 0).any() for c in fault_columns if c in d.columns) else 0
    day_pred_lbl = 1 if (day_preds == 1).any() else 0
    day_true.append(day_true_lbl)
    day_pred.append(day_pred_lbl)
    match = "OK" if day_true_lbl == day_pred_lbl else "X"
    true_str = 'Warning' if day_true_lbl else 'Healthy'
    pred_str = 'Warning' if day_pred_lbl else 'Healthy'
    n_warn_pred = int((day_preds == 1).sum())
    print(f"  {test_dates[di]}: True={true_str:8s} Pred={pred_str:8s} {match} (n_window_warning_pred={n_warn_pred}/{WINDOWS_PER_DAY})")
    idx += WINDOWS_PER_DAY

day_true = np.array(day_true)
day_pred = np.array(day_pred)
acc_day = accuracy_score(day_true, day_pred) * 100
f1_day = f1_score(day_true, day_pred, average='weighted', zero_division=0) * 100
rec_day = recall_score(day_true, day_pred, pos_label=1, zero_division=0) * 100
print(f"\nAkurasi level HARI (agregasi dari window): Acc={acc_day:.2f}% F1={f1_day:.2f}% RecallWarning={rec_day:.2f}%")
print(confusion_matrix(day_true, day_pred, labels=[0, 1]))
