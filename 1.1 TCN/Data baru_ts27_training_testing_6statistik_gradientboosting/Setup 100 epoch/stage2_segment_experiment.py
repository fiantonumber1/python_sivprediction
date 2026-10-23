import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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

def skewness(v):
    m, s = v.mean(0), v.std(0); s = np.where(s==0,1e-8,s)
    return (np.mean((v-m)**3,0))/(s**3)

def feat_vec(v):
    roc = np.abs(np.diff(v, axis=0)).mean(0) if v.shape[0] > 1 else np.zeros(v.shape[1])
    return np.concatenate([v.mean(0), v.std(0), v.max(0), v.min(0), skewness(v), roc]).astype(np.float32)

def label_seg(seg):
    return 1 if any((seg[c] > 0).any() for c in fault_columns if c in seg.columns) else 0

days = []
for f in csv_files:
    d = read_and_crop(f)
    if not d.empty: days.append(d)
n_train_days = int(len(days)*0.8)
print(f"Total hari: {len(days)} | Train days: {n_train_days} | Test days: {len(days)-n_train_days}")

for n_seg in [2,3,4,6,8]:
    seg_len = N_TAKE // n_seg
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for i, d in enumerate(days):
        v_full = d[target_columns].values.astype(np.float64)
        for s in range(n_seg):
            seg_v = v_full[s*seg_len:(s+1)*seg_len]
            seg_df = d.iloc[s*seg_len:(s+1)*seg_len]
            feat = feat_vec(seg_v)
            lbl  = label_seg(seg_df)
            if i < n_train_days:
                X_tr.append(feat); y_tr.append(lbl)
            else:
                X_te.append(feat); y_te.append(lbl)
    X_tr=np.array(X_tr); y_tr=np.array(y_tr); X_te=np.array(X_te); y_te=np.array(y_te)
    sc = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = sc.transform(X_tr), sc.transform(X_te)

    models = {
        "GB_d3_lr0.1_n200": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
        "RF_d6_n300_bal": RandomForestClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42),
        "LogReg_bal": LogisticRegression(class_weight='balanced', max_iter=3000),
    }
    print(f"\n--- n_seg={n_seg} (train_samples={len(X_tr)}, test_samples={len(X_te)}, "
          f"train_warn={sum(y_tr)}/{len(y_tr)}, test_warn={sum(y_te)}/{len(y_te)}) ---")
    for name, clf in models.items():
        # pilih model hanya pakai CV di TRAIN (tidak lihat test), lalu evaluasi sekali di test
        cv_acc = cross_val_score(clf, Xtr_s, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy').mean()*100
        clf.fit(Xtr_s, y_tr)
        pred_te = clf.predict(Xte_s)
        acc_te = accuracy_score(y_te, pred_te)*100
        f1_te  = f1_score(y_te, pred_te, average='weighted', zero_division=0)*100
        rec_w  = recall_score(y_te, pred_te, pos_label=1, zero_division=0)*100
        cm = confusion_matrix(y_te, pred_te, labels=[0,1]).tolist()
        print(f"  {name:18s} CV(train)Acc={cv_acc:6.2f}%  TestAcc={acc_te:6.2f}% F1={f1_te:6.2f}% RecallWarn={rec_w:6.2f}%  CM={cm}")
