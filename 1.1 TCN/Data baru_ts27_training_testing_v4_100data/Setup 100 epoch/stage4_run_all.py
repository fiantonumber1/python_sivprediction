# =============================
# STAGE 4 — RUN ALL PIPELINE
# Jalankan stage1 → stage2 → stage3 secara otomatis
# Cukup run file ini saja.
# =============================

import subprocess
import sys
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

stages = [
    ("STAGE 1 — TCN Forecaster Training",  "stage1_forecaster.py"),
    ("STAGE 2 — MLP Classifier Training",  "stage2_classifier.py"),
    ("STAGE 3 — Inference & Plot",          "stage3_inference.py"),
]

print("=" * 65)
print("  PIPELINE TCN SIV PREDICTION — AUTO RUN ALL STAGES")
print(f"  Mulai: {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 65)

for idx, (label, script) in enumerate(stages, 1):
    script_path = os.path.join(SCRIPT_DIR, script)
    print(f"\n{'='*65}")
    print(f"  [{idx}/3] {label}")
    print(f"  Script : {script}")
    print(f"  Waktu  : {datetime.now():%H:%M:%S}")
    print(f"{'='*65}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=SCRIPT_DIR,
    )

    if result.returncode != 0:
        print(f"\n[ERROR] {script} gagal (exit code {result.returncode})")
        print(f"Pipeline dihentikan. Periksa error di atas.")
        sys.exit(result.returncode)

    print(f"\n[OK] {label} selesai — {datetime.now():%H:%M:%S}")

print(f"\n{'='*65}")
print(f"  SEMUA STAGE SELESAI — {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  Output ada di folder: evidence/")
print(f"{'='*65}\n")
