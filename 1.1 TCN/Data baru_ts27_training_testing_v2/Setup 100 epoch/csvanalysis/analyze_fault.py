# =============================
# CSV FAULT ANALYSIS — Data Baru TS14
# Tampilan ALL DATA: split training/validasi + health status per hari
# Mirip tampilan plot_all dari stage_3 (dalam bentuk tabel teks)
# =============================

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, date, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================================================
# KONFIGURASI
# =========================================================
FAILURE_DATE  = None          # Belum ada failure date untuk Data Baru TS14
                              # Isi jika sudah diketahui, misal: date(2026, 3, 15)
CSV_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")  # folder data/ terisolasi

# Parameter crop — identik dengan stage1 & stage2
START_TIME    = time(3, 0, 0)
END_TIME      = time(18, 16, 35)
N_DROP_FIRST  = 3600
N_TAKE        = 200_000
MIN_ROWS      = N_DROP_FIRST + N_TAKE   # 203_600

FAULT_COLUMNS = [
    'SIV_MajorBCFltPres',
    'SIV_MajorInputConvFltPres',
    'SIV_MajorInverterFltPres',
]

TARGET_COLUMNS = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]

PLOT_DOWNSAMPLE = 10   # ambil setiap N-th point untuk plot (hemat memori, visual tetap sama)

# =========================================================
# BACA FILE — format DDMMYYYY.csv (setelah extract_csv.py)
# =========================================================
files = sorted(
    [f for f in glob.glob(os.path.join(CSV_DIR, "*.csv"))
     if "inference" not in os.path.basename(f).lower()
     and "prediksi"  not in os.path.basename(f).lower()
     and "hasil"     not in os.path.basename(f).lower()],
    key=lambda f: datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")
)

if not files:
    print(f"\nTidak ada file CSV di: {os.path.abspath(CSV_DIR)}")
    print("Pastikan sudah menjalankan extract_csv.py terlebih dahulu.")
    exit(1)

n_total = len(files)

# =========================================================
# ANALISIS PER FILE
# =========================================================
rows      = []
plot_data = []   # data per hari untuk plot (hanya file Layak)
for idx, f in enumerate(files):
    basename = os.path.basename(f).replace(".csv", "")
    try:
        file_date = datetime.strptime(basename[:8], "%d%m%Y").date()
    except ValueError:
        continue

    # TTF hanya jika failure date diketahui
    if FAILURE_DATE is not None:
        ttf_days = (FAILURE_DATE - file_date).days
        ttf_str  = f"{ttf_days} hr"
    else:
        ttf_str = "-"

    # Baca CSV (coba separator ; dulu, fallback ke ,)
    try:
        df = pd.read_csv(f, encoding='utf-8-sig', sep=';', low_memory=False, on_bad_lines='skip')
        if df.shape[1] < 3:
            df = pd.read_csv(f, encoding='utf-8-sig', sep=',', low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"  ERROR membaca {basename}: {e}")
        continue

    df.columns = [c.strip() for c in df.columns]

    # Deteksi kolom timestamp
    ts_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ['ts_date', 'timestamp', 'date', 'time']):
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]

    df['_ts'] = pd.to_datetime(
        df[ts_col].astype(str).str.replace(',', '.'), errors='coerce'
    )
    df = df.dropna(subset=['_ts'])
    if df.empty:
        continue

    date0   = df['_ts'].dt.date.iloc[0]
    df_crop = df[
        (df['_ts'] >= datetime.combine(date0, START_TIME)) &
        (df['_ts'] <= datetime.combine(date0, END_TIME))
    ]
    rows_after_crop = len(df_crop)
    rows_after_drop = max(0, rows_after_crop - N_DROP_FIRST)
    eligible        = rows_after_crop >= MIN_ROWS
    el_str          = "Layak" if eligible else "SKIP"

    # Hitung fault count dari df_crop (selaras dengan window training)
    counts = {}
    for col in FAULT_COLUMNS:
        if col in df_crop.columns:
            s = pd.to_numeric(df_crop[col].astype(str).str.replace(',', '.'), errors='coerce')
            counts[col] = int((s > 0).sum())
        else:
            counts[col] = "N/A"

    # Health status per hari (sama logika stage_3)
    n_active = sum(1 for c in FAULT_COLUMNS
                   if isinstance(counts[c], int) and counts[c] > 0)
    if n_active == 0:   health = "Sehat"
    elif n_active == 1: health = "Warning"
    else:               health = "Not Ready"

    rows.append({
        "idx"       : idx + 1,
        "file"      : basename,
        "date"      : file_date.strftime("%d %b %Y"),
        "ttf"       : ttf_str,
        "after_crop": rows_after_crop,
        "after_drop": rows_after_drop,
        "eligible"  : el_str,
        "health"    : health,
        "BC"        : counts['SIV_MajorBCFltPres'],
        "InputConv" : counts['SIV_MajorInputConvFltPres'],
        "Inverter"  : counts['SIV_MajorInverterFltPres'],
    })

    # Simpan data time-series untuk plot (hanya file Layak)
    if eligible:
        df_p = df_crop.copy()
        for col in TARGET_COLUMNS:
            if col in df_p.columns:
                df_p[col] = pd.to_numeric(df_p[col].astype(str).str.replace(',', '.'), errors='coerce')
            else:
                df_p[col] = np.nan
        df_p[TARGET_COLUMNS] = df_p[TARGET_COLUMNS].ffill().bfill()
        df_u = df_p.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)
        arr = df_u[TARGET_COLUMNS].values[::PLOT_DOWNSAMPLE].astype(np.float32)
        plot_data.append({'health': health, 'file': basename, 'data': arr})

if not rows:
    print("Tidak ada data valid untuk dianalisis.")
    exit(1)

# =========================================================
# CETAK TABEL — ALL DATA (seperti tampilan stage_3 plot_all)
# =========================================================
col_idx    = max(len("#"),             max(len(str(r["idx"]))           for r in rows))
col_file   = max(len("File"),          max(len(r["file"])               for r in rows))
col_date   = max(len("Tanggal"),       max(len(r["date"])               for r in rows))
col_crop   = max(len("Rows(crop)"),    max(len(str(r["after_crop"]))    for r in rows))
col_drop   = max(len("Rows(usable)"),  max(len(str(r["after_drop"]))    for r in rows))
col_el     = max(len("Eligible"),      max(len(r["eligible"])           for r in rows))
col_health = max(len("Health"),        max(len(r["health"])             for r in rows))
col_bc     = max(len("BC"),            max(len(str(r["BC"]))            for r in rows))
col_ic     = max(len("InputConv"),     max(len(str(r["InputConv"]))     for r in rows))
col_inv    = max(len("Inverter"),      max(len(str(r["Inverter"]))      for r in rows))

def make_sep():
    return (f"+-{'-'*col_idx}-+-{'-'*col_file}-+-{'-'*col_date}-+"
            f"-{'-'*col_crop}-+-{'-'*col_drop}-+-{'-'*col_el}-+-{'-'*col_health}-+"
            f"-{'-'*col_bc}-+-{'-'*col_ic}-+-{'-'*col_inv}-+")

def make_row(idx, file, date_, crop, drop, eligible, health, bc, ic, inv):
    return (f"| {str(idx):>{col_idx}} | {str(file):<{col_file}} | {str(date_):<{col_date}}"
            f" | {str(crop):>{col_crop}} | {str(drop):>{col_drop}}"
            f" | {str(eligible):<{col_el}} | {str(health):<{col_health}}"
            f" | {str(bc):<{col_bc}} | {str(ic):<{col_ic}} | {str(inv):<{col_inv}} |")

sep_line = make_sep()

print()
print("=" * len(sep_line))
print(f"  ALL DATA — TS14 Pipeline  (Total: {n_total} hari)")
print("=" * len(sep_line))
print(f"  CSV folder  : {os.path.abspath(CSV_DIR)}")
print(f"  Min rows    : {MIN_ROWS:,}  (drop {N_DROP_FIRST:,} + take {N_TAKE:,})")
if FAILURE_DATE is not None:
    print(f"  Failure date: {FAILURE_DATE}")
print()

print(sep_line)
print(make_row("#", "File", "Tanggal", "Rows(crop)", "Rows(usable)",
               "Eligible", "Health", "BC", "InputConv", "Inverter"))
print(sep_line)
for r in rows:
    print(make_row(r["idx"], r["file"], r["date"],
                   r["after_crop"], r["after_drop"], r["eligible"], r["health"],
                   r["BC"], r["InputConv"], r["Inverter"]))
print(sep_line)
print()

# =========================================================
# RINGKASAN
# =========================================================
n_layak   = sum(1 for r in rows if r["eligible"] == "Layak")
n_skip    = sum(1 for r in rows if r["eligible"] == "SKIP")
n_sehat    = sum(1 for r in rows if r["health"] == "Sehat")
n_warning  = sum(1 for r in rows if r["health"] == "Warning")
n_notready = sum(1 for r in rows if r["health"] == "Not Ready")

total_bc  = sum(r["BC"]        if isinstance(r["BC"],        int) else 0 for r in rows)
total_ic  = sum(r["InputConv"] if isinstance(r["InputConv"], int) else 0 for r in rows)
total_inv = sum(r["Inverter"]  if isinstance(r["Inverter"],  int) else 0 for r in rows)
days_bc   = sum(1 for r in rows if isinstance(r["BC"],        int) and r["BC"]        > 0)
days_ic   = sum(1 for r in rows if isinstance(r["InputConv"], int) and r["InputConv"] > 0)
days_inv  = sum(1 for r in rows if isinstance(r["Inverter"],  int) and r["Inverter"]  > 0)

print(f"  Kelayakan data : {n_layak} Layak  |  {n_skip} Skip")
print(f"  Health status  : {n_sehat} Sehat  |  {n_warning} Warning  |  {n_notready} Not Ready")
print(f"  Fault BC       : {total_bc:>5} count  |  {days_bc} hari aktif")
print(f"  Fault InputConv: {total_ic:>5} count  |  {days_ic} hari aktif")
print(f"  Fault Inverter : {total_inv:>5} count  |  {days_inv} hari aktif")
print()
print(f"  ---- Semua hari ({n_total} hari / training) ----")
for r in rows:
    mark = "  [SKIP]" if r["eligible"] == "SKIP" else ""
    print(f"    [{r['idx']:2d}] {r['file']}  {r['date']}  ->  {r['health']:<12}{mark}")
print()

# =========================================================
# PLOT — ALL DATA (identik logika plot_all dari stage_3)
# =========================================================
if not plot_data:
    print("[Plot] Tidak ada hari Layak untuk diplot.")
else:
    print(f"[Plot] Membuat plot_all_data.png  ({len(plot_data)} hari valid) ...")

    n_plot = len(plot_data)
    PPD    = N_TAKE // PLOT_DOWNSAMPLE   # points per day setelah downsample

    # Gabungkan semua hari
    all_arr = np.concatenate([p['data'] for p in plot_data], axis=0)   # (n_plot*PPD, 21)

    # Normalisasi per kolom — identik normalize_per_col_data_lama di stage_3
    norm_arr = all_arr.copy()
    for j in range(norm_arr.shape[1]):
        mn, mx = norm_arr[:, j].min(), norm_arr[:, j].max()
        if mx - mn > 1e-8:
            norm_arr[:, j] = (norm_arr[:, j] - mn) / (mx - mn)
        else:
            norm_arr[:, j] = 0.0

    x = np.arange(len(norm_arr))

    fig, ax = plt.subplots(figsize=(26, 10))

    # Plot 21 parameter — pakai colormap agar semua warna berbeda
    colors = (list(plt.cm.tab10.colors) +        # 10 warna
              list(plt.cm.Set2.colors) +          # 8 warna
              list(plt.cm.Dark2.colors[:3]))      # 3 warna → total 21
    for j, col in enumerate(TARGET_COLUMNS):
        ax.plot(x, norm_arr[:, j], linewidth=0.8, alpha=0.7, label=col, color=colors[j])

    # Garis batas antar hari (merah putus-putus)
    day_bounds = np.arange(0, (n_plot + 1) * PPD, PPD)
    for b in day_bounds[1:-1]:
        ax.axvline(b, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    # Label hari + health status di atas plot (identik stage_3)
    STATUS_COLOR = {'Sehat': 'green', 'Warning': 'orange', 'Not Ready': 'red'}
    mid_pts = [(day_bounds[i] + day_bounds[i + 1]) // 2 for i in range(n_plot)]
    for i, (mid, p) in enumerate(zip(mid_pts, plot_data)):
        ax.text(mid, 1.05, f"Day {i+1}", ha='center', va='bottom',
                color='red', fontsize=7, fontweight='bold',
                transform=ax.get_xaxis_transform())
        ax.text(mid, 1.12, p['file'], ha='center', va='bottom',
                color='dimgray', fontsize=5.5,
                transform=ax.get_xaxis_transform())
        ax.text(mid, 1.19, p['health'], ha='center', va='bottom',
                color=STATUS_COLOR.get(p['health'], 'black'), fontsize=7.5, fontweight='bold',
                transform=ax.get_xaxis_transform())

    ax.set_title(
        f"ALL DATA — 21 Parameter + Health Status  (TS14 Pipeline)\n"
        f"{n_plot} hari valid  |  Downsample 1/{PLOT_DOWNSAMPLE}",
        fontsize=13
    )
    ax.set_ylabel("Normalized [0-1]")
    ax.set_ylim(-0.05, 1.35)
    ax.grid(alpha=0.3)
    ax.legend(TARGET_COLUMNS, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize='small', ncol=2)
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_all_data.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Tersimpan: {out_png}")
print()
