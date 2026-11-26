import pandas as pd
import numpy as np
import os, glob
from datetime import datetime, time
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
N_TAKE, START_TIME, END_TIME = 200_000, time(3,0,0), time(18,16,35)

target_columns = [
    'SIV_T_HS_InConv_1','SIV_T_HS_InConv_2','SIV_T_HS_Inv_1','SIV_T_HS_Inv_2','SIV_T_Container',
    'SIV_I_L1','SIV_I_L2','SIV_I_L3','SIV_I_Battery','SIV_I_DC_In',
    'SIV_U_Battery','SIV_U_DC_In','SIV_U_DC_Out','SIV_U_L1','SIV_U_L2','SIV_U_L3',
    'SIV_InConv_InEnergy','SIV_Output_Energy','PLC_OpenACOutputCont','PLC_OpenInputCont','SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres','SIV_MajorInputConvFltPres','SIV_MajorInverterFltPres']
_needed = set(['ts_date'] + target_columns + fault_columns)

def extract_date(f): return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")
csv_files = sorted([f for f in glob.glob(os.path.join(DATA_DIR,"*.csv"))
                     if "hasil" not in f.lower() and "prediksi" not in f.lower() and "inference" not in f.lower()],
                    key=extract_date)

def read_and_crop(fp):
    df = pd.read_csv(fp, encoding='utf-8-sig', sep=';', decimal=',',
                      usecols=lambda c: c.strip() in _needed, low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',','.'),
                                    format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['ts_date'])
    for c in target_columns+fault_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce') if c in df.columns else np.nan
    df[target_columns+fault_columns] = df[target_columns+fault_columns].ffill().bfill()
    d0 = df['ts_date'].dt.date.iloc[0]
    df = df[(df['ts_date']>=datetime.combine(d0,START_TIME)) & (df['ts_date']<=datetime.combine(d0,END_TIME))]
    if len(df) < N_TAKE: return pd.DataFrame()
    return df.iloc[:N_TAKE].reset_index(drop=True)[['ts_date']+target_columns+fault_columns]

def label_day(df):
    return 1 if any((df[c]>0).any() for c in fault_columns if c in df.columns) else 0

def skewness(v):
    m, s = v.mean(0), v.std(0); s = np.where(s==0,1e-8,s)
    return (np.mean((v-m)**3,0))/(s**3)

def slope(v):
    # linear trend coefficient per kolom (least squares vs index waktu)
    n = v.shape[0]
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    denom = np.sum(x**2)
    out = np.empty(v.shape[1])
    for j in range(v.shape[1]):
        y = v[:, j] - v[:, j].mean()
        out[j] = np.sum(x*y)/denom if denom != 0 else 0.0
    return out

def feat_vec_full(df):
    v = df[target_columns].values.astype(np.float64)
    mean = v.mean(0); std = v.std(0); mx = v.max(0); mn = v.min(0)
    median = np.median(v, axis=0)
    q25 = np.percentile(v, 25, axis=0); q75 = np.percentile(v, 75, axis=0)
    iqr = q75 - q25
    skew = skewness(v)
    kurt = np.nan_to_num(kurtosis(v, axis=0, fisher=True, bias=False, nan_policy="omit"), nan=0.0, posinf=0.0, neginf=0.0)
    roc  = np.abs(np.diff(v, axis=0)).mean(0)
    sl   = slope(v)
    last = v[-1, :]
    rng  = mx - mn
    return np.concatenate([mean, std, mx, mn, median, iqr, skew, kurt, roc, sl, last, rng]).astype(np.float32)

FEATURE_NAMES_GROUPS = ["mean","std","max","min","median","iqr","skew","kurt","roc","slope","last","range"]

days = []
for f in csv_files:
    d = read_and_crop(f)
    if not d.empty: days.append(d)
print(f"Total hari: {len(days)}")

n_train = int(len(days)*0.8)
train_days, test_days = days[:n_train], days[n_train:]

X_tr_full = np.array([feat_vec_full(d) for d in train_days])
y_tr = np.array([label_day(d) for d in train_days])
X_te_full = np.array([feat_vec_full(d) for d in test_days])
y_te = np.array([label_day(d) for d in test_days])
print(f"Full feature dim: {X_tr_full.shape[1]} ({len(FEATURE_NAMES_GROUPS)} grup x 21 param)")

n_param = len(target_columns)
def subset(groups):
    idxs = []
    for gi, g in enumerate(FEATURE_NAMES_GROUPS):
        if g in groups:
            idxs += list(range(gi*n_param, (gi+1)*n_param))
    return np.array(idxs)

combos = {
    "6stat(mean,std,max,min,skew,roc)": ["mean","std","max","min","skew","roc"],
    "8stat(+median,iqr)": ["mean","std","max","min","skew","roc","median","iqr"],
    "10stat(+kurt,slope)": ["mean","std","max","min","skew","roc","median","iqr","kurt","slope"],
    "ALL12stat": FEATURE_NAMES_GROUPS,
}

results = []
for combo_name, groups in combos.items():
    idx = subset(groups)
    X_tr = X_tr_full[:, idx]; X_te = X_te_full[:, idx]
    sc = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = sc.transform(X_tr), sc.transform(X_te)

    models = {
        "GB_d3_lr0.1_n200": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
        "GB_d2_lr0.05_n300": GradientBoostingClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=42),
        "GB_d4_lr0.1_n150": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42),
        "RF_d5_n300": RandomForestClassifier(n_estimators=300, max_depth=5, class_weight='balanced', random_state=42),
        "RF_d8_n500": RandomForestClassifier(n_estimators=500, max_depth=8, class_weight='balanced', random_state=42),
        "ExtraTrees_d6": ExtraTreesClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42),
        "LogReg_balanced": LogisticRegression(class_weight='balanced', max_iter=2000, C=0.5),
        "SVM_RBF_bal": SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=42),
    }
    for name, clf in models.items():
        clf.fit(Xtr_s, y_tr)
        pred_tr = clf.predict(Xtr_s)
        pred_te = clf.predict(Xte_s)
        acc_tr = accuracy_score(y_tr, pred_tr)*100
        acc_te = accuracy_score(y_te, pred_te)*100
        f1_te  = f1_score(y_te, pred_te, average='weighted', zero_division=0)*100
        rec_warn = recall_score(y_te, pred_te, pos_label=1, zero_division=0)*100
        prec_warn = precision_score(y_te, pred_te, pos_label=1, zero_division=0)*100
        cm = confusion_matrix(y_te, pred_te, labels=[0,1])
        results.append((combo_name, name, acc_tr, acc_te, f1_te, rec_warn, prec_warn, cm.tolist()))

results.sort(key=lambda r: -r[3])
print(f"\n{'Combo':35s} {'Model':20s} {'TrainAcc':>9s} {'TestAcc':>8s} {'F1':>7s} {'RecW':>6s} {'PrecW':>6s}  CM")
for r in results:
    print(f"{r[0]:35s} {r[1]:20s} {r[2]:8.2f}% {r[3]:7.2f}% {r[4]:6.2f}% {r[5]:5.2f}% {r[6]:5.2f}%  {r[7]}")
