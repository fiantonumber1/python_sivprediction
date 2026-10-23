"""Evaluasi rantai Stage 1 -> Stage 2 (v7 final) pada SEMUA hari uji.
Sinyal prediksi TCN dipecah 30 detik lalu diklasifikasi MLP Stage 2."""
import os, sys, joblib
from datetime import datetime, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(6)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = r"C:\Users\User\Documents\GitHub\siv-prediction\1.1 TCN"
V5 = os.path.join(B, "Data baru_ts27_training_testing_v5_next generation", "Setup 100 epoch")
V7 = os.path.join(B, "Data baru_ts27_training_testing_v7", "Setup 100 epoch")
N_TAKE, START_TIME, END_TIME = 200_000, time(3, 0, 0), time(18, 16, 35)
ROWS, WPD = 117, 200_000 // 117
FUT = N_TAKE
cols = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy', 'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive']
faults = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
need = set(['ts_date'] + cols + faults)


class CausalConv1d(nn.Module):
    def __init__(s, i, o, ks, d=1):
        super().__init__(); s.pad = (ks - 1) * d; s.conv = nn.Conv1d(i, o, ks, padding=s.pad, dilation=d)
    def forward(s, x):
        o = s.conv(x); return o[:, :, :-s.pad] if s.pad > 0 else o

class Res(nn.Module):
    def __init__(s, i, o, ks=3, d=1, dr=0.3):
        super().__init__()
        s.c1 = CausalConv1d(i, o, ks, d); s.n1 = nn.BatchNorm1d(o); s.r1 = nn.ReLU(); s.d1 = nn.Dropout(dr)
        s.c2 = CausalConv1d(o, o, ks, d); s.n2 = nn.BatchNorm1d(o); s.r2 = nn.ReLU(); s.d2 = nn.Dropout(dr)
        s.skip = nn.Conv1d(i, o, 1) if i != o else nn.Identity()
    def forward(s, x):
        r = s.skip(x); o = s.d1(s.r1(s.n1(s.c1(x)))); o = s.d2(s.r2(s.n2(s.c2(o)))); return o + r

class TCN(nn.Module):
    def __init__(s, nf=21, nc=96, ks=3, nb=7, dr=0.3):
        super().__init__()
        dil = [1, 2, 4, 8, 16, 32, 64]; L = []; i = nf
        for k in range(nb):
            L.append(Res(i, nc, ks, dil[k], dr)); i = nc
        s.tcn = nn.Sequential(*L); s.pool = nn.AdaptiveAvgPool1d(1)
        s.dec = nn.Sequential(nn.Linear(nc, nc), nn.ReLU(), nn.Dropout(dr), nn.Linear(nc, nf * FUT))
    def forward(s, x):
        ctx = s.pool(s.tcn(x.transpose(1, 2))).squeeze(-1)
        return s.dec(ctx).view(-1, FUT, x.shape[-1]), ctx


def read_day(ds):
    fp = os.path.join(V7, "data", ds + ".csv")
    df = pd.read_csv(fp, encoding='utf-8-sig', sep=';', decimal=',',
                     usecols=lambda c: c.strip() in need, low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'),
                                   format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['ts_date'])
    for c in cols + faults:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[cols + faults] = df[cols + faults].ffill().bfill()
    d0 = df['ts_date'].dt.date.iloc[0]
    df = df[(df['ts_date'] >= datetime.combine(d0, START_TIME)) & (df['ts_date'] <= datetime.combine(d0, END_TIME))]
    return df.iloc[:N_TAKE].reset_index(drop=True)


def feats(sig):
    v = np.asarray(sig[:WPD * ROWS], dtype=np.float64).reshape(WPD, ROWS, 21)
    mean, std, mx, mn = v.mean(1), v.std(1), v.max(1), v.min(1)
    sk = ((v - mean[:, None, :]) ** 3).mean(1) / (np.where(std == 0, 1e-8, std) ** 3)
    roc = np.abs(np.diff(v, axis=1)).mean(1)
    return np.concatenate([mean, std, mx, mn, sk, roc], axis=1).astype(np.float32)


inv = pd.read_csv(os.path.join(V7, "laporan", "inventaris_data_per_hari.csv"), sep=';', encoding='utf-8-sig', dtype=str)
used = inv[inv['dipakai'] == 'ya'].reset_index(drop=True)
dates = used['tanggal'].tolist()
n_train = used.index[used['split'] == 'test'][0]
print("hari terpakai:", len(dates), "| train:", n_train, "| test pertama:", dates[n_train], flush=True)

sc1 = joblib.load(os.path.join(V7, "scaler_stage1.pkl"))
sc2 = joblib.load(os.path.join(V7, "scaler_stage2.pkl"))
clf = joblib.load(os.path.join(V7, "model_stage2_classifier.pkl"))
tcn = TCN()
tcn.load_state_dict(torch.load(os.path.join(V7, "model_stage1_forecaster.pth"), map_location="cpu"))
tcn.to(dev)
tcn.eval()

cache = {}
def day(ds):
    if ds not in cache:
        cache[ds] = read_day(ds)
    return cache[ds]


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TARGET = os.environ.get("TARGET_DAY", "19062026")
t = dates.index(TARGET)
a,b,c = dates[t-3],dates[t-2],dates[t-1]
seq = np.concatenate([day(x)[cols].values for x in (a,b,c)],axis=0).astype(np.float32)
with torch.no_grad():
    p,_ = tcn(torch.FloatTensor(sc1.transform(seq)).unsqueeze(0).to(dev))
pn = p.cpu().numpy()[0]; rn = sc1.transform(day(TARGET)[cols].values.astype(np.float32))
mse_all = float(((pn-rn)**2).mean())
fig, axs = plt.subplots(7, 3, figsize=(22, 24)); S = 20
for j, ax in enumerate(axs.ravel()):
    ax.plot(rn[::S, j], color="#1f77b4", lw=0.7, label="Real")
    ax.plot(pn[::S, j], color="#d62728", lw=1.0, label="Prediksi TCN")
    m = float(((pn[:, j]-rn[:, j])**2).mean())
    ax.set_title(f"{cols[j]}  (MSE {m:.4f})", fontsize=10); ax.grid(alpha=0.3)
    if j == 0: ax.legend(fontsize=9)
fig.suptitle(f"Real vs Prediksi TCN, hari {TARGET} (input: {a}, {b}, {c}) | MSE ternormalisasi total {mse_all:.5f} | skala normalisasi Stage 1", fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0,0,1,0.98])
out = os.path.join(V7, "evidence", f"gambar_real_vs_prediksi_21parameter_{TARGET}.png")
plt.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out, mse_all)
