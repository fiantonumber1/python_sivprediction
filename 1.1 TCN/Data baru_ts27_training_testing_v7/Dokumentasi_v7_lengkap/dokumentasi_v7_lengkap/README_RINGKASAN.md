# Dokumentasi Lengkap SIV Predictive Maintenance v7

Run akhir: 2026-10-21 s.d. 2026-10-23. Pipeline `stage4_run_all.py` (Stage 1 → 2 → 3).

## 1. Data
- 95 file CSV, 88 hari terpakai (7 hari dilewati karena baris < 200.000): 11-07-2025 s.d. 19-06-2026.
- 54 hari Healthy, 34 hari Warning. Label hari = Warning jika ada kolom fault > 0.
- Split kronologis per hari: train 70 hari (44 Healthy / 26 Warning), test 18 hari (10 Healthy / 8 Warning).
- Kejadian fault hanya ~5 detik (2–66 baris) per hari Warning, sehingga tenggelam pada statistik harian.
- Detail: `data_laporan/inventaris_data_per_hari.csv`, `data_laporan/ringkasan_data.json`.

## 2. Stage 1 — TCN Forecaster (algoritma tidak diubah dari v5)
- Input 3 hari → output 1 hari (21 parameter, 200.000 baris/hari). 7 blok residual, dilasi 1–64, 96 kanal, MSE, 100 epoch, batch 3.
- Window: train 67, test 18.

| | MSE | RMSE | MAE |
|---|---|---|---|
| Train | 0,006679 | 0,081727 | 0,032266 |
| Test (18 window) | 0,005940 | 0,077069 | 0,029585 |

- MSE akhir training epoch 100: 0,0085145 (nilai rata-rata per epoch, ada dropout).
- MAPE (581.544% / 88.193%) tidak bermakna karena pembagian nilai ≈ 0; jangan dipakai.
- Catatan: bug resume (checkpoint epoch 100 melatih ulang dari epoch 1) diperbaiki dengan satu baris `start_epoch = N_EPOCHS + 1`; algoritma tidak berubah. Run yang salah dibatalkan; log tersimpan di `log/log_stage1_run_ulang_batal.txt`.
- Training sempat berhenti ±8 jam (laptop hibernate) dan evaluasi pertama crash cuDNN (`CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH`); model dimuat ulang dari `checkpoint_epoch_100.pth`.

## 3. Stage 2 — MLP Classifier (window 30 detik)
- `MLPClassifier(hidden_layer_sizes=(64,32), solver='adam', alpha=1e-3, early_stopping=True, validation_fraction=0.15, random_state=42)`, StandardScaler.
- Fitur: 21 parameter × 6 statistik (mean, std, max, min, skew, rate-of-change) = 126 per window 30 detik (117 baris, 1.709 window/hari).
- Label window = 1 jika ada fault di window. Keputusan hari = Warning jika ≥ 1 window diprediksi Warning.
- Window train 119.630 (Warning 41), window test 30.762 (Warning 13).

| Evaluasi | Hasil |
|---|---|
| Test 18 hari, seed 42 | Acc 83,33%, F1 82,50%, Precision Warning 100%, Recall Warning 62,5% (5/8), CM [[10,0],[3,5]] |
| Train 70 hari | Acc 94,29%, F1 94,23% |
| Level window (test) | Acc 99,98%, Recall Warning 53,85% (7/13), Precision 100% |
| 5 seed (42,1,7,21,99) | 83,33 / 61,11 / 66,67 / 83,33 / 72,22 → rata-rata 73,33% ± 8,89 |
| 5-fold CV per hari (88 hari) | rata-rata 86,34% ± 2,88; gabungan Acc 86,36%, F1 85,92%, Recall Warning 70,59%, Precision 92,31%, CM [[52,2],[10,24]] |

Perbandingan versi (hari uji; MLP per hari): v4 (37 data) Acc 62,5% (recall 62,5%, F1 48,1%); v5 (84 data) Recall 62,5%, F1 53,1%; v6 (95 data) 50,0%; v7 (window 30 detik) 83,33%.
Sweep ukuran window (akurasi hari): 1/2/3/5 menit 66,67%; 10 menit 55,56%; 15 menit 61,11%; 20 menit 61,11%; 30 menit 50%; 45 menit 50%; 60 menit 55,56%; 30 detik 83,33%; 15 detik 77,78%.

## 4. Stage 3 — Inferensi berantai (TCN → MLP)
- Contoh bawaan Stage 3 (1 sampel uji): prediksi Healthy, confidence 88,52% (`hasil_csv/inference_health_status.csv`).
- Evaluasi seluruh 18 hari uji (`kode/evaluasi_rantai_semua_hari_uji.py`, `data_laporan/evaluasi_rantai_semua_hari_uji.csv`):

| Masukan ke MLP | Acc | TP | FN | FP | TN |
|---|---|---|---|---|---|
| Sinyal real | 83,33% | 5 | 3 | 0 | 10 |
| Sinyal prediksi TCN (rantai) | **55,56%** | **0** | 8 | 0 | 10 |

- Rantai memprediksi semua hari uji sebagai Healthy; probabilitas Warning maksimum pada sinyal prediksi 0,0–0,44 (< 0,5).
- Penyebab (dari cek awal): window sinyal prediksi TCN berbeda statistik dari sinyal real (median std ≈ 1 vs ≈ 0 pada sinyal real, skala ternormalisasi), padahal MLP dilatih hanya pada sinyal real. Selain itu kejadian fault ~5 detik tidak dapat direproduksi oleh forecaster MSE.

## 5. Keterbatasan (untuk ditulis di laporan)
1. Test set kecil: 18 hari (8 Warning); 1 hari = 5,6 poin akurasi.
2. Hanya 41 window Warning pada train (13 pada test).
3. Hasil sensitif seed (61–83%); angka paling stabil adalah CV 86,36% ± 2,88.
4. Recall Warning masih 62,5% (test) / 70,59% (CV).
5. Rantai TCN → MLP belum mampu mendeteksi Warning (0/8); akurasi 55,56% hanya berasal dari kelas Healthy.
6. Sweep window dilakukan dengan melihat hasil pada test set yang sama; angka 83,33% berpotensi optimistis. CV lebih jujur.

## 6. Isi folder
- `gambar/` semua gambar (termasuk nama setara v4: `jurnal_confusion_matrix_*`, `jurnal_kurva_loss_acc_stage2`, `jurnal_aktual_vs_prediksi_mlp`; dan `evaluasi_rantai_semua_hari_uji.png`).
- `kode/` stage1–4, pengumpul data, evaluasi rantai.
- `log/` semua log. `hasil_csv/` hasil per hari & ringkasan. `data_laporan/` inventaris data.
- `model_kecil/` model Stage 2 dan scaler. Tidak disertakan (ukuran besar): `model_stage1_forecaster.pth` (1,5 GB), checkpoint (9,3 GB), 95 CSV mentah (13 GB), CSV prediksi sensor (86 MB).
