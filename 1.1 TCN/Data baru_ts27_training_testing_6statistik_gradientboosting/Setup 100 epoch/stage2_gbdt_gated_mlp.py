import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
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
print(f"Train: {len(X_tr)} | Test: {len(X_te)} | dim: {X_tr.shape[1]}")

def ev(name, pred_te_fn_result, cv_fn=None):
    pred_te = pred_te_fn_result
    acc_te = accuracy_score(y_te, pred_te)*100
    f1_te  = f1_score(y_te, pred_te, average='weighted', zero_division=0)*100
    rec_w  = recall_score(y_te, pred_te, pos_label=1, zero_division=0)*100
    cm = confusion_matrix(y_te, pred_te, labels=[0,1]).tolist()
    cv_str = f"CV={cv_fn:6.2f}% " if cv_fn is not None else ""
    print(f"{name:45s} {cv_str}Test={acc_te:6.2f}% F1={f1_te:6.2f}% RecWarn={rec_w:6.2f}% CM={cm}")
    return acc_te

# ====== STEP 1: GBDT untuk menghitung feature importance (gate) ======
gbdt_gate = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
gbdt_gate.fit(Xtr_s, y_tr)
importances = gbdt_gate.feature_importances_
print(f"\nGBDT feature importance: min={importances.min():.5f} max={importances.max():.5f} "
      f"nonzero={np.sum(importances>0)}/{len(importances)}")

# ====== STEP 2: coba beberapa skema gating untuk MLP ======
for top_k in [10, 20, 30, 50]:
    idx_sorted = np.argsort(importances)[::-1][:top_k]
    Xtr_gated = Xtr_s[:, idx_sorted]
    Xte_gated = Xte_s[:, idx_sorted]
    mlp = MLPClassifier(hidden_layer_sizes=(32,), solver='adam', alpha=0.01, max_iter=2000,
                         early_stopping=True, validation_fraction=0.2, random_state=42)
    cv_acc = cross_val_score(mlp, Xtr_gated, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
    mlp.fit(Xtr_gated, y_tr)
    pred = mlp.predict(Xte_gated)
    ev(f"GBDT-Gated-MLP (top_k={top_k} fitur)", pred, cv_acc)

# ====== STEP 3: soft-gating -> kalikan fitur dengan importance (bukan hard select) ======
gate_weights = importances / (importances.max() + 1e-12)
Xtr_soft = Xtr_s * gate_weights
Xte_soft = Xte_s * gate_weights
mlp_soft = MLPClassifier(hidden_layer_sizes=(32,), solver='adam', alpha=0.01, max_iter=2000,
                          early_stopping=True, validation_fraction=0.2, random_state=42)
cv_acc = cross_val_score(mlp_soft, Xtr_soft, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
mlp_soft.fit(Xtr_soft, y_tr)
pred_soft = mlp_soft.predict(Xte_soft)
ev("GBDT-SoftGated-MLP (weighted by importance)", pred_soft, cv_acc)

# baseline pembanding
mlp_plain = MLPClassifier(hidden_layer_sizes=(32,), solver='adam', alpha=0.01, max_iter=2000,
                           early_stopping=True, validation_fraction=0.2, random_state=42)
cv_acc = cross_val_score(mlp_plain, Xtr_s, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
mlp_plain.fit(Xtr_s, y_tr)
pred_plain = mlp_plain.predict(Xte_s)
ev("MLP biasa (tanpa gating, baseline)", pred_plain, cv_acc)

gb_plain = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
cv_acc = cross_val_score(gb_plain, Xtr_s, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
gb_plain.fit(Xtr_s, y_tr)
pred_gb = gb_plain.predict(Xte_s)
ev("GB biasa (baseline resmi saat ini)", pred_gb, cv_acc)
