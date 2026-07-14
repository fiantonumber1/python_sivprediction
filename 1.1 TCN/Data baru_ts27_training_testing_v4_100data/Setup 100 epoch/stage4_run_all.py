# =============================
# STAGE 4 — RUN ALL PIPELINE
# Jalankan stage1 → stage2 → stage3 secara otomatis
#
# SETTING: Pilih stage mana yang dijalankan
# True  = jalankan
# False = skip
# =============================

RUN_STAGE1 = True    # TCN Forecaster Training
RUN_STAGE2 = True    # MLP Classifier Training
RUN_STAGE3 = True    # Inference & Plot

# =============================

import subprocess
import sys
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_STAGES = [
    (1, "STAGE 1 — TCN Forecaster Training", "stage1_forecaster.py", RUN_STAGE1),
    (2, "STAGE 2 — MLP Classifier Training", "stage2_classifier.py", RUN_STAGE2),
    (3, "STAGE 3 — Inference & Plot",         "stage3_inference.py", RUN_STAGE3),
]

pipeline_start = datetime.now()

print("=" * 65)
print("  PIPELINE TCN SIV PREDICTION")
print(f"  Mulai        : {pipeline_start:%Y-%m-%d %H:%M:%S}")
active = [f"Stage {n}" for n, _, _, run in ALL_STAGES if run]
skipped = [f"Stage {n}" for n, _, _, run in ALL_STAGES if not run]
print(f"  Dijalankan   : {', '.join(active) if active else '-'}")
print(f"  Di-skip      : {', '.join(skipped) if skipped else '-'}")
print("=" * 65)

stage_times = []

for num, label, script, should_run in ALL_STAGES:
    if not should_run:
        print(f"\n  [SKIP] Stage {num} — {label}")
        continue

    script_path = os.path.join(SCRIPT_DIR, script)
    t_start = datetime.now()

    print(f"\n{'='*65}")
    print(f"  [Stage {num}/3] {label}")
    print(f"  Script       : {script}")
    print(f"  Mulai        : {t_start:%Y-%m-%d %H:%M:%S}")
    print(f"{'='*65}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=SCRIPT_DIR,
    )

    t_end = datetime.now()
    duration = t_end - t_start
    total_sec = int(duration.total_seconds())
    dur_str = f"{total_sec // 3600:02d}:{(total_sec % 3600) // 60:02d}:{total_sec % 60:02d}"

    if result.returncode != 0:
        print(f"\n{'='*65}")
        print(f"  [ERROR] Stage {num} GAGAL")
        print(f"  Mulai    : {t_start:%Y-%m-%d %H:%M:%S}")
        print(f"  Berhenti : {t_end:%Y-%m-%d %H:%M:%S}")
        print(f"  Durasi   : {dur_str}")
        print(f"  Pipeline dihentikan. Periksa error di atas.")
        print(f"{'='*65}")
        sys.exit(result.returncode)

    stage_times.append((num, label, t_start, t_end, dur_str))

    print(f"\n{'='*65}")
    print(f"  [OK] Stage {num} selesai")
    print(f"  Mulai    : {t_start:%Y-%m-%d %H:%M:%S}")
    print(f"  Selesai  : {t_end:%Y-%m-%d %H:%M:%S}")
    print(f"  Durasi   : {dur_str}")
    print(f"{'='*65}")

pipeline_end = datetime.now()
pipeline_dur = pipeline_end - pipeline_start
total_sec = int(pipeline_dur.total_seconds())
pipeline_dur_str = f"{total_sec // 3600:02d}:{(total_sec % 3600) // 60:02d}:{total_sec % 60:02d}"

print(f"\n{'='*65}")
print(f"  PIPELINE SELESAI")
print(f"{'='*65}")
for num, label, t_start, t_end, dur_str in stage_times:
    print(f"  Stage {num}  Mulai: {t_start:%H:%M:%S}  Selesai: {t_end:%H:%M:%S}  Durasi: {dur_str}")
print(f"  {'-'*61}")
print(f"  TOTAL    Mulai: {pipeline_start:%H:%M:%S}  Selesai: {pipeline_end:%H:%M:%S}  Durasi: {pipeline_dur_str}")
print(f"  Output ada di folder: evidence/")
print(f"{'='*65}\n")
