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
SEL_NAMES = os.environ.get("SEL_PARAMS")
SEL = [cols.index(n) for n in SEL_NAMES.split(",")] if SEL_NAMES else list(range(21))
TARGET = os.environ.get("TARGET_DAY")
t = dates.index(TARGET) if TARGET else len(dates) - 1   # default: hari uji terakhir
ds = [dates[t-3], dates[t-2], dates[t-1], dates[t]]
real4 = [day(x)[cols].values.astype(np.float32) for x in ds]
lab4 = ["Warning" if (day(x)[faults].values > 0).any() else "Healthy" for x in ds]
seq = np.concatenate(real4[:3], axis=0)
with torch.no_grad():
    p, _ = tcn(torch.FloatTensor(sc1.transform(seq)).unsqueeze(0).to(dev))
pred = sc1.inverse_transform(p.cpu().numpy()[0])
pp = clf.predict_proba(sc2.transform(feats(pred)))[:, 1]
pred_lbl = "Warning" if (pp >= .5).any() else "Healthy"
N = 200_000; STEP = 10
def prep(a): return sc1.transform(a)[::STEP]
row1 = [prep(r) for r in real4]
row2 = [prep(r) for r in real4[:3]] + [prep(pred)]
cm = plt.get_cmap("tab20"); L = 200_000 // STEP
fig, axs = plt.subplots(2, 1, figsize=(22, 14), sharey=True)
titles = ["Baris 1: Real hari n, n+1, n+2, n+3", "Baris 2: Real hari n, n+1, n+2 + Prediksi TCN hari n+3"]
for ax, row, ttl in zip(axs, (row1, row2), titles):
    for k, a in enumerate(row):
        x = np.arange(k * L, (k + 1) * L)
        for j in SEL:
            ax.plot(x, a[:, j], color=cm(j % 20), lw=0.6, ls="--" if (ax is axs[1] and k == 3) else "-")
        if k: ax.axvline(k * L, color="red", ls="--", lw=1)
    for k in range(4):
        if ax is axs[1] and k == 3: txt = f"Prediksi: {pred_lbl}"
        else: txt = lab4[k]
        ax.text(k * L + L / 2, 1.22, ds[k] + "\n" + txt, ha="center", fontsize=11, fontweight="bold",
                color="#E65100" if "Warning" in txt else "#2E7D32")
    ax.set_title(ttl + (f"  [{len(SEL)} dari 21 parameter yang forecast-nya cocok]" if SEL_NAMES else ""), fontsize=14, fontweight="bold", pad=38); ax.set_ylim(-0.1, 1.35); ax.grid(alpha=0.3)
axs[0].legend([cols[j] for j in SEL], ncol=1, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
out = os.path.join(V7, "evidence", "gambar_forecast_real_vs_prediksi_bertingkat" + ("_" + TARGET if TARGET else "") + ("_param_terpilih" if SEL_NAMES else "") + ".png")
plt.savefig(out, dpi=150, bbox_inches="tight"); print("saved", ds, lab4, "pred:", pred_lbl, "max_p", float(pp.max()))

