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

rows = []
for f in csv_files:
    d = read_and_crop(f)
    if d.empty:
        continue
    dstr = os.path.basename(f)[:8]
    row = {'date': dstr, 'total_rows': len(d)}
    total_active = 0
    for c in fault_columns:
        n_active = int((d[c] > 0).sum())
        row[c] = n_active
        total_active += n_active
    row['total_active_rows'] = total_active
    row['pct_active'] = 100*total_active/len(d)
    row['label'] = 1 if total_active > 0 else 0
    rows.append(row)

df_summary = pd.DataFrame(rows)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 200)
print(df_summary.to_string(index=False))

print("\n=== STATISTIK HARI WARNING (label=1) ===")
warn = df_summary[df_summary['label']==1]
print(f"Jumlah hari Warning: {len(warn)}")
print(f"Jumlah baris fault aktif per hari Warning -> min={warn['total_active_rows'].min()}, "
      f"max={warn['total_active_rows'].max()}, median={warn['total_active_rows'].median()}")
print(f"Persentase waktu aktif fault per hari -> min={warn['pct_active'].min():.6f}%, "
      f"max={warn['pct_active'].max():.6f}%, median={warn['pct_active'].median():.6f}%")
print("\nDistribusi jumlah baris aktif (urut menaik):")
print(warn[['date','total_active_rows','pct_active']].sort_values('total_active_rows').to_string(index=False))
