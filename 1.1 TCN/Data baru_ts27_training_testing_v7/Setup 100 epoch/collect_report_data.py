import os, glob, json
from datetime import datetime, time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "laporan")
os.makedirs(OUT_DIR, exist_ok=True)

N_TAKE, START_TIME, END_TIME = 200_000, time(3, 0, 0), time(18, 16, 35)
TRAIN_RATIO = 0.8
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
need = {'ts_date', *fault_columns}


def blocks(series):
    idx = np.where(series > 0)[0]
    if len(idx) == 0:
        return []
    out, cur = [], 1
    for g in np.diff(idx):
        if g == 1:
            cur += 1
        else:
            out.append(cur)
            cur = 1
    out.append(cur)
    return out


files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")),
               key=lambda f: datetime.strptime(os.path.basename(f)[:8], "%d%m%Y"))
rows = []
for f in files:
    ds = os.path.basename(f)[:8]
    df = pd.read_csv(f, encoding='utf-8-sig', sep=';', decimal=',',
                     usecols=lambda c: c.strip() in need, low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    n_raw = len(df)
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'),
                                   format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['ts_date'])
    for c in fault_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce') if c in df.columns else np.nan
    df[fault_columns] = df[fault_columns].ffill().bfill()
    d0 = df['ts_date'].dt.date.iloc[0]
    dfc = df[(df['ts_date'] >= datetime.combine(d0, START_TIME)) & (df['ts_date'] <= datetime.combine(d0, END_TIME))]
    n_crop = len(dfc)
    used = n_crop >= N_TAKE
    row = dict(tanggal=ds, baris_mentah=n_raw, baris_setelah_crop=n_crop,
               dipakai=('ya' if used else 'tidak (baris < 200000)'))
    if used:
        d = dfc.iloc[:N_TAKE]
        tot = 0
        allb = []
        for c in fault_columns:
            v = d[c].values
            row[c] = int((v > 0).sum())
            tot += row[c]
            allb += blocks(v)
        row['total_baris_fault'] = tot
        row['persen_baris_fault'] = round(100 * tot / N_TAKE, 4)
        row['jumlah_blok_fault'] = len(allb)
        row['panjang_blok_maks'] = max(allb) if allb else 0
        row['label_hari'] = 'Warning' if tot > 0 else 'Healthy'
    rows.append(row)
    print(ds, row.get('label_hari', '-'), flush=True)

inv = pd.DataFrame(rows)
used_idx = inv.index[inv['dipakai'] == 'ya'].tolist()
n_train = max(2, int(len(used_idx) * TRAIN_RATIO))
split = {i: ('train' if k < n_train else 'test') for k, i in enumerate(used_idx)}
inv['split'] = [split.get(i, '-') for i in inv.index]
inv.to_csv(os.path.join(OUT_DIR, "inventaris_data_per_hari.csv"), index=False, sep=';', encoding='utf-8-sig')

u = inv[inv['dipakai'] == 'ya']
w = u[u['label_hari'] == 'Warning']
summary = dict(
    file_csv=len(inv), hari_dipakai=len(u), hari_dilewati=inv[inv['dipakai'] != 'ya']['tanggal'].tolist(),
    tanggal_pertama=u['tanggal'].iloc[0], tanggal_terakhir=u['tanggal'].iloc[-1],
    hari_healthy=int((u['label_hari'] == 'Healthy').sum()), hari_warning=int(len(w)),
    baris_fault_per_hari_warning=dict(min=int(w['total_baris_fault'].min()), median=float(w['total_baris_fault'].median()),
                                      max=int(w['total_baris_fault'].max())),
    persen_fault_per_hari_warning=dict(min=float(w['persen_baris_fault'].min()),
                                       median=float(w['persen_baris_fault'].median()),
                                       max=float(w['persen_baris_fault'].max())),
    hari_warning_per_kolom={c: int((u[c] > 0).sum()) for c in fault_columns},
    hari_dengan_lebih_dari_satu_blok=int((w['jumlah_blok_fault'] > 1).sum()),
    panjang_blok_maks=dict(min=int(w['panjang_blok_maks'].min()), median=float(w['panjang_blok_maks'].median()),
                           max=int(w['panjang_blok_maks'].max())),
)
with open(os.path.join(OUT_DIR, "ringkasan_data.json"), 'w', encoding='utf-8') as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
