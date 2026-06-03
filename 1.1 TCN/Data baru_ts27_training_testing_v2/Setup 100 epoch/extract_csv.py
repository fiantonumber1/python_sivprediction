# =============================
# EXTRACT CSV — Dekompresi file .gz dan rename ke format DDMMYYYY.csv
# Jalankan SEKALI sebelum stage1, stage2, stage3
# =============================

import os
import gzip
import shutil
import glob
import re

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

_script  = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
folder   = os.path.join(_script, "data")   # file gz & output CSV ada di data/
gz_files = sorted(glob.glob(os.path.join(folder, "*.gz")))

if not gz_files:
    print("Tidak ada file .gz ditemukan di folder ini.")
else:
    print(f"Ditemukan {len(gz_files)} file .gz\n")
    for gz_path in gz_files:
        basename = os.path.basename(gz_path)

        # Ambil stem (hapus .csv.gz atau .gz)
        stem = basename
        for ext in ['.csv.gz', '.gz']:
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break

        # Parse pola: TS27_D+ Mon YYYY  (misal: TS27_4 May 2026, TS27_14 May 2026)
        m = re.match(r'TS\d+_(\d{1,2})\s+(\w{3})\s+(\d{4})', stem)
        if not m:
            print(f"  Skip (pola tidak cocok): {basename}")
            continue

        day  = m.group(1).zfill(2)
        mon  = MONTH_MAP.get(m.group(2))
        year = m.group(3)

        if mon is None:
            print(f"  Skip (bulan tidak dikenal: {m.group(2)}): {basename}")
            continue

        out_name = f"{day}{mon}{year}.csv"
        out_path = os.path.join(folder, out_name)

        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"  Sudah ada (skip): {out_name}  ({size_mb:.1f} MB)")
            continue

        print(f"  Ekstrak: {basename:40s}  ->  {out_name}")
        try:
            with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"    Selesai ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"    ERROR: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)

print("\nEkstraksi selesai!")
print("Langkah berikutnya:")
print("  1. Jalankan stage1_forecaster.py")
print("  2. Jalankan stage2_classifier.py")
print("  3. Jalankan stage3_inference.py")
