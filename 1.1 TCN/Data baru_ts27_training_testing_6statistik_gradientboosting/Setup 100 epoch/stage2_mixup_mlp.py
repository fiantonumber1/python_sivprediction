import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
def feat_vec(df):
    v = df[target_columns].values
    roc = np.abs(np.diff(v, axis=0)).mean(0)
    return np.concatenate([v.mean(0), v.std(0), v.max(0), v.min(0), skewness(v), roc]).astype(np.float32)

days = []
for f in csv_files:
    d = read_and_crop(f)
    if not d.empty: days.append(d)
n_train = int(len(days)*0.8)
train_days, test_days = days[:n_train], days[n_train:]
X_tr = np.array([feat_vec(d) for d in train_days]); y_tr = np.array([label_day(d) for d in train_days])
X_te = np.array([feat_vec(d) for d in test_days]);  y_te = np.array([label_day(d) for d in test_days])
sc = StandardScaler().fit(X_tr)
Xtr_s, Xte_s = sc.transform(X_tr), sc.transform(X_te)
print(f"Train: {len(X_tr)} (Healthy={sum(y_tr==0)}, Warning={sum(y_tr==1)}) | Test: {len(X_te)}")

def ev(name, pred_te, cv_acc=None):
    acc_te = accuracy_score(y_te, pred_te)*100
    f1_te  = f1_score(y_te, pred_te, average='weighted', zero_division=0)*100
    rec_w  = recall_score(y_te, pred_te, pos_label=1, zero_division=0)*100
    cm = confusion_matrix(y_te, pred_te, labels=[0,1]).tolist()
    cv_str = f"CV={cv_acc:6.2f}% " if cv_acc is not None else ""
    print(f"{name:45s} {cv_str}Test={acc_te:6.2f}% F1={f1_te:6.2f}% RecWarn={rec_w:6.2f}% CM={cm}")

def mixup_augment(X, y, n_extra_per_sample, alpha=0.4, seed=42):
    rng = np.random.RandomState(seed)
    X_aug, y_aug = [X], [y]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < 2: continue
        n_new = n_extra_per_sample * len(idx)
        i1 = rng.choice(idx, n_new); i2 = rng.choice(idx, n_new)
        lam = rng.beta(alpha, alpha, size=(n_new, 1))
        Xnew = lam * X[i1] + (1 - lam) * X[i2]
        ynew = np.full(n_new, cls)
        X_aug.append(Xnew); y_aug.append(ynew)
    return np.vstack(X_aug), np.concatenate(y_aug)

mlp_cfg = dict(hidden_layer_sizes=(32,), solver='adam', alpha=0.01, max_iter=2000,
               early_stopping=True, validation_fraction=0.2, random_state=42)

mlp0 = MLPClassifier(**mlp_cfg)
cv0 = cross_val_score(mlp0, Xtr_s, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
mlp0.fit(Xtr_s, y_tr)
ev("MLP tanpa augmentasi (baseline)", mlp0.predict(Xte_s), cv0)

for n_extra in [1, 2, 3, 5]:
    for alpha_mix in [0.2, 0.4]:
        Xtr_m, ytr_m = mixup_augment(Xtr_s, y_tr, n_extra, alpha=alpha_mix)
        mlp = MLPClassifier(**mlp_cfg)
        try:
            cv_acc = cross_val_score(mlp, Xtr_m, ytr_m, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
        except Exception:
            cv_acc = None
        mlp.fit(Xtr_m, ytr_m)
        ev(f"Mixup-MLP (n_extra={n_extra}x, alpha={alpha_mix}, total={len(Xtr_m)})", mlp.predict(Xte_s), cv_acc)

gb0 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
cv_gb = cross_val_score(gb0, Xtr_s, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
gb0.fit(Xtr_s, y_tr)
ev("GB biasa (baseline resmi saat ini)", gb0.predict(Xte_s), cv_gb)
