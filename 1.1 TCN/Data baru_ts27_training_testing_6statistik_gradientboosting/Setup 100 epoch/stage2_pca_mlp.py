import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
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
def feat_6stat(df):
    v = df[target_columns].values
    roc = np.abs(np.diff(v, axis=0)).mean(0)
    return np.concatenate([v.mean(0), v.std(0), v.max(0), v.min(0), skewness(v), roc]).astype(np.float32)

N_SEG = 200  # downsample tiap hari jadi 200 titik per parameter (rata-rata per blok)
def feat_downsampled(df):
    v = df[target_columns].values.astype(np.float64)
    n = v.shape[0]
    seg_len = n // N_SEG
    segs = v[:seg_len*N_SEG].reshape(N_SEG, seg_len, v.shape[1]).mean(axis=1)  # (N_SEG, 21)
    return segs.flatten().astype(np.float32)  # 200*21 = 4200 dim

days = []
for f in csv_files:
    d = read_and_crop(f)
    if not d.empty: days.append(d)
n_train = int(len(days)*0.8)
train_days, test_days = days[:n_train], days[n_train:]
y_tr = np.array([label_day(d) for d in train_days])
y_te = np.array([label_day(d) for d in test_days])

def ev(name, pred_te, cv_acc=None):
    acc_te = accuracy_score(y_te, pred_te)*100
    f1_te  = f1_score(y_te, pred_te, average='weighted', zero_division=0)*100
    rec_w  = recall_score(y_te, pred_te, pos_label=1, zero_division=0)*100
    cm = confusion_matrix(y_te, pred_te, labels=[0,1]).tolist()
    cv_str = f"CV={cv_acc:6.2f}% " if cv_acc is not None else ""
    print(f"{name:45s} {cv_str}Test={acc_te:6.2f}% F1={f1_te:6.2f}% RecWarn={rec_w:6.2f}% CM={cm}")

mlp_cfg = dict(hidden_layer_sizes=(32,), solver='adam', alpha=0.01, max_iter=2000,
               early_stopping=True, validation_fraction=0.2, random_state=42)

print("=== VERSI A: PCA dari 6-statistik (126 dim) ===")
X_tr6 = np.array([feat_6stat(d) for d in train_days])
X_te6 = np.array([feat_6stat(d) for d in test_days])
sc6 = StandardScaler().fit(X_tr6)
Xtr6_s, Xte6_s = sc6.transform(X_tr6), sc6.transform(X_te6)

mlp0 = MLPClassifier(**mlp_cfg)
cv0 = cross_val_score(mlp0, Xtr6_s, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
mlp0.fit(Xtr6_s, y_tr)
ev("MLP tanpa PCA (baseline 6stat)", mlp0.predict(Xte6_s), cv0)

for n_comp in [5, 10, 15, 20, 30, 40]:
    pca = PCA(n_components=n_comp, random_state=42).fit(Xtr6_s)
    Xtr_p, Xte_p = pca.transform(Xtr6_s), pca.transform(Xte6_s)
    var_explained = pca.explained_variance_ratio_.sum()*100
    mlp = MLPClassifier(**mlp_cfg)
    cv_acc = cross_val_score(mlp, Xtr_p, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
    mlp.fit(Xtr_p, y_tr)
    ev(f"PCA({n_comp}comp,var={var_explained:.1f}%)-MLP [6stat]", mlp.predict(Xte_p), cv_acc)

print("\n=== VERSI B: PCA dari sinyal mentah downsampled (4200 dim, tanpa pilih statistik) ===")
X_trD = np.array([feat_downsampled(d) for d in train_days])
X_teD = np.array([feat_downsampled(d) for d in test_days])
scD = StandardScaler().fit(X_trD)
XtrD_s, XteD_s = scD.transform(X_trD), scD.transform(X_teD)

for n_comp in [5, 10, 15, 20, 30, 40, 50]:
    pca = PCA(n_components=n_comp, random_state=42).fit(XtrD_s)
    Xtr_p, Xte_p = pca.transform(XtrD_s), pca.transform(XteD_s)
    var_explained = pca.explained_variance_ratio_.sum()*100
    mlp = MLPClassifier(**mlp_cfg)
    cv_acc = cross_val_score(mlp, Xtr_p, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
    mlp.fit(Xtr_p, y_tr)
    ev(f"PCA({n_comp}comp,var={var_explained:.1f}%)-MLP [raw downsampled]", mlp.predict(Xte_p), cv_acc)
