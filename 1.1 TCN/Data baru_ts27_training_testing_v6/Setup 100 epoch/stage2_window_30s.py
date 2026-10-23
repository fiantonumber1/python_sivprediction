import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
N_TAKE, START_TIME, END_TIME = 200_000, time(3, 0, 0), time(18, 16, 35)
TRAIN_RATIO = 0.8
SAMPLE_INTERVAL = 0.256

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


print("Membaca semua hari (sekali saja, dipakai ulang untuk semua ukuran window)...")
days, dates = [], []
for f in csv_files:
    d = read_and_crop(f)
    if not d.empty:
        days.append(d)
        dates.append(os.path.basename(f)[:8])

total_days = len(days)
n_train_days = max(2, int(total_days * TRAIN_RATIO))
n_test_days = total_days - n_train_days
print(f"Total hari: {total_days} | Train days: {n_train_days} | Test days: {n_test_days}")

train_days_data = days[:n_train_days]
test_days_data = days[n_train_days:]
test_dates = dates[n_train_days:]

# Precompute label ground-truth per hari (dari fault_columns asli, bukan prediksi)
day_true_all = [1 if any((d[c] > 0).any() for c in fault_columns if c in d.columns) else 0
                for d in days]
train_day_true = day_true_all[:n_train_days]
test_day_true = day_true_all[n_train_days:]


def window_feat_and_label(win_df):
    v = win_df[target_columns].values
    roc = np.abs(np.diff(v, axis=0)).mean(0) if len(v) > 1 else np.zeros(v.shape[1])
    feat = np.concatenate([v.mean(0), v.std(0), v.max(0), v.min(0), skewness(v), roc]).astype(np.float32)
    label = 1 if any((win_df[c] > 0).any() for c in fault_columns if c in win_df.columns) else 0
    return feat, label


def days_to_windows(day_list, rows_per_window, windows_per_day):
    X, y = [], []
    for d in day_list:
        for w in range(windows_per_day):
            seg = d.iloc[w * rows_per_window:(w + 1) * rows_per_window]
            feat, label = window_feat_and_label(seg)
            X.append(feat)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


results = []
for window_minutes in [0.5]:
    rows_per_window = int(window_minutes * 60 / SAMPLE_INTERVAL)
    windows_per_day = N_TAKE // rows_per_window

    X_train, y_train = days_to_windows(train_days_data, rows_per_window, windows_per_day)
    X_test, y_test = days_to_windows(test_days_data, rows_per_window, windows_per_day)

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), solver='adam', alpha=0.001, max_iter=500,
                         early_stopping=True, validation_fraction=0.15, random_state=42)
    mlp.fit(Xtr_s, y_train)
    pred_te = mlp.predict(Xte_s)

    acc_w = accuracy_score(y_test, pred_te) * 100
    rec_w = recall_score(y_test, pred_te, pos_label=1, zero_division=0) * 100

    # Agregasi ke level hari
    day_pred = []
    idx = 0
    for d in test_days_data:
        day_preds = pred_te[idx:idx + windows_per_day]
        day_pred.append(1 if (day_preds == 1).any() else 0)
        idx += windows_per_day

    acc_day = accuracy_score(test_day_true, day_pred) * 100
    f1_day = f1_score(test_day_true, day_pred, average='weighted', zero_division=0) * 100
    rec_day = recall_score(test_day_true, day_pred, pos_label=1, zero_division=0) * 100
    prec_day = None
    cm_day = confusion_matrix(test_day_true, day_pred, labels=[0, 1]).tolist()

    results.append((window_minutes, rows_per_window, windows_per_day, len(X_train), sum(y_train),
                     acc_w, rec_w, acc_day, f1_day, rec_day, cm_day))
    print(f"[{window_minutes:5.2f} menit] rows/win={rows_per_window:5d} win/day={windows_per_day:4d} "
          f"n_train={len(X_train):6d} n_warn_train={sum(y_train):3d} | "
          f"WindowAcc={acc_w:6.2f}% WindowRecWarn={rec_w:6.2f}% || "
          f"DayAcc={acc_day:6.2f}% DayF1={f1_day:6.2f}% DayRecWarn={rec_day:6.2f}% CM={cm_day}")

print("\n=== RINGKASAN (urut DayAcc tertinggi) ===")
results.sort(key=lambda r: -r[7])
for r in results:
    print(f"Window {r[0]:5.2f} menit | DayAcc={r[7]:6.2f}% DayF1={r[8]:6.2f}% DayRecWarn={r[9]:6.2f}% CM={r[10]}")
