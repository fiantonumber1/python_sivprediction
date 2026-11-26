import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
from scipy.stats import kurtosis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

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

def label_day(df): return 1 if any((df[c]>0).any() for c in fault_columns if c in df.columns) else 0
def skewness(v):
    m, s = v.mean(0), v.std(0); s = np.where(s==0,1e-8,s)
    return (np.mean((v-m)**3,0))/(s**3)
def slope(v):
    n = v.shape[0]; x = np.arange(n, dtype=np.float64); x = x - x.mean(); denom = np.sum(x**2)
    out = np.empty(v.shape[1])
    for j in range(v.shape[1]):
        y = v[:, j] - v[:, j].mean()
        out[j] = np.sum(x*y)/denom if denom != 0 else 0.0
    return out
def feat_vec_full(df):
    v = df[target_columns].values.astype(np.float64)
    mean=v.mean(0); std=v.std(0); mx=v.max(0); mn=v.min(0)
    median=np.median(v,axis=0); q25=np.percentile(v,25,axis=0); q75=np.percentile(v,75,axis=0); iqr=q75-q25
    skew=skewness(v)
    kurt=np.nan_to_num(kurtosis(v,axis=0,fisher=True,bias=False,nan_policy="omit"),nan=0.0,posinf=0.0,neginf=0.0)
    roc=np.abs(np.diff(v,axis=0)).mean(0)
    sl=slope(v); last=v[-1,:]; rng=mx-mn
    return np.concatenate([mean,std,mx,mn,median,iqr,skew,kurt,roc,sl,last,rng]).astype(np.float32)
GROUPS = ["mean","std","max","min","median","iqr","skew","kurt","roc","slope","last","range"]
n_param = len(target_columns)
def subset(groups):
    idx=[]
    for gi,g in enumerate(GROUPS):
        if g in groups: idx += list(range(gi*n_param,(gi+1)*n_param))
    return np.array(idx)

days = []
for f in csv_files:
    d = read_and_crop(f)
    if not d.empty: days.append(d)
n_train = int(len(days)*0.8)
train_days, test_days = days[:n_train], days[n_train:]
X_tr_full = np.array([feat_vec_full(d) for d in train_days]); y_tr = np.array([label_day(d) for d in train_days])
X_te_full = np.array([feat_vec_full(d) for d in test_days]);  y_te = np.array([label_day(d) for d in test_days])
print(f"Train: {len(X_tr_full)} | Test: {len(X_te_full)} | full dim: {X_tr_full.shape[1]}")

feature_sets = {
    "4stat":  ["mean","std","max","min"],
    "6stat":  ["mean","std","max","min","skew","roc"],
    "8stat":  ["mean","std","max","min","skew","roc","median","iqr"],
    "10stat": ["mean","std","max","min","skew","roc","median","iqr","kurt","slope"],
    "12stat": GROUPS,
}

param_grid = {
    'hidden_layer_sizes': [(8,), (16,), (32,), (16,8)],
    'activation': ['relu', 'tanh'],
    'solver': ['lbfgs'],
    'alpha': [0.1, 1.0, 3.0],
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)

all_results = []
for fname, groups in feature_sets.items():
    idx = subset(groups)
    Xtr, Xte = X_tr_full[:, idx], X_te_full[:, idx]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    gs = GridSearchCV(MLPClassifier(max_iter=3000, random_state=42), param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    gs.fit(Xtr_s, y_tr)
    best = gs.best_estimator_
    pred_te = best.predict(Xte_s)
    acc_te = accuracy_score(y_te, pred_te)*100
    f1_te  = f1_score(y_te, pred_te, average='weighted', zero_division=0)*100
    rec_w  = recall_score(y_te, pred_te, pos_label=1, zero_division=0)*100
    cm = confusion_matrix(y_te, pred_te, labels=[0,1]).tolist()
    all_results.append((fname, gs.best_params_, gs.best_score_*100, acc_te, f1_te, rec_w, cm))
    print(f"\n[{fname}] dim={len(idx)} | Best CV(train): {gs.best_params_} CV_Acc={gs.best_score_*100:.2f}%")
    print(f"         TEST -> Acc={acc_te:.2f}% F1={f1_te:.2f}% RecWarn={rec_w:.2f}% CM={cm}")

all_results.sort(key=lambda r: -r[3])
print(f"\n=== RINGKASAN (urut TestAcc tertinggi) ===")
for r in all_results:
    print(f"{r[0]:8s} {str(r[1]):55s} CV={r[2]:6.2f}% Test={r[3]:6.2f}% F1={r[4]:6.2f}% RecWarn={r[5]:6.2f}%")
