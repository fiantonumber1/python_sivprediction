import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
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

param_grid = {
    'hidden_layer_sizes': [(8,), (16,), (32,), (8,4), (16,8), (32,16), (64,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['lbfgs'],
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
gs = GridSearchCV(MLPClassifier(max_iter=3000, random_state=42), param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
gs.fit(Xtr_s, y_tr)
print(f"\nBest CV (train-only) config: {gs.best_params_} | CV Acc: {gs.best_score_*100:.2f}%")

best_model = gs.best_estimator_
pred_te = best_model.predict(Xte_s)
acc_te = accuracy_score(y_te, pred_te)*100
f1_te  = f1_score(y_te, pred_te, average='weighted', zero_division=0)*100
rec_w  = recall_score(y_te, pred_te, pos_label=1, zero_division=0)*100
cm = confusion_matrix(y_te, pred_te, labels=[0,1]).tolist()
print(f"Best model TEST eval -> Acc={acc_te:.2f}% F1={f1_te:.2f}% RecallWarn={rec_w:.2f}% CM={cm}")

# Tampilkan top-10 config berdasar CV score (bukan test) untuk transparansi
results_df = pd.DataFrame(gs.cv_results_)
top10 = results_df.sort_values('mean_test_score', ascending=False).head(10)
print("\nTop-10 config by CV(train) score:")
for _, row in top10.iterrows():
    print(f"  {row['params']}  CV_Acc={row['mean_test_score']*100:.2f}%")

# evaluasi top-5 CV config langsung ke test (untuk observasi, bukan untuk pilih model baru)
print("\nEvaluasi top-5 CV config ke TEST (observasi):")
for _, row in top10.head(5).iterrows():
    p = row['params']
    m = MLPClassifier(max_iter=3000, random_state=42, **p)
    m.fit(Xtr_s, y_tr)
    pr = m.predict(Xte_s)
    a = accuracy_score(y_te, pr)*100
    f = f1_score(y_te, pr, average='weighted', zero_division=0)*100
    r = recall_score(y_te, pr, pos_label=1, zero_division=0)*100
    c = confusion_matrix(y_te, pr, labels=[0,1]).tolist()
    print(f"  {p} -> TestAcc={a:.2f}% F1={f:.2f}% RecWarn={r:.2f}% CM={c}")
