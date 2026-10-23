import pandas as pd, numpy as np, os, glob
from datetime import datetime, time
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

results = []
for f in csv_files:
    d = read_and_crop(f)
    if d.empty: continue
    dstr = os.path.basename(f)[:8]
    for c in fault_columns:
        active_idx = np.where(d[c].values > 0)[0]
        if len(active_idx) == 0: continue
        vals = d[c].values[active_idx]
        # cek konsekutif: hitung jumlah "blok" kontinu
        gaps = np.diff(active_idx)
        n_blocks = 1 + np.sum(gaps > 1)
        max_block_len = 0; cur = 1
        for g in gaps:
            if g == 1: cur += 1
            else: max_block_len = max(max_block_len, cur); cur = 1
        max_block_len = max(max_block_len, cur)
        results.append({
            'date': dstr, 'fault_col': c, 'n_active_rows': len(active_idx),
            'val_min': vals.min(), 'val_max': vals.max(), 'val_unique': len(np.unique(vals)),
            'n_blocks': n_blocks, 'max_block_len': max_block_len,
            'first_idx': active_idx.min(), 'last_idx': active_idx.max(),
        })

dfr = pd.DataFrame(results)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 100)
print(dfr.to_string(index=False))
print("\n=== Ringkasan nilai fault (0/1 biner atau magnitude?) ===")
print(f"val_unique (jumlah nilai unik saat aktif): {dfr['val_unique'].value_counts().to_dict()}")
print(f"val_min range: {dfr['val_min'].min()} - {dfr['val_min'].max()}")
print(f"val_max range: {dfr['val_max'].min()} - {dfr['val_max'].max()}")
print("\n=== Pola blok (kontinu vs tersebar) ===")
print(f"n_blocks: min={dfr['n_blocks'].min()} max={dfr['n_blocks'].max()} median={dfr['n_blocks'].median()}")
print(f"max_block_len: min={dfr['max_block_len'].min()} max={dfr['max_block_len'].max()} median={dfr['max_block_len'].median()}")
