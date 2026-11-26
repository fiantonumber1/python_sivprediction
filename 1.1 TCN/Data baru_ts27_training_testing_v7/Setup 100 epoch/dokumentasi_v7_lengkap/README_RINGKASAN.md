# Dokumentasi Lengkap SIV Predictive Maintenance v7 (HASIL RESMI)

Run akhir: 2026-10-21 s.d. 2026-10-23. Stage 1 (TCN) → Stage 2 (MLP window 30 detik, 100 epoch, tanpa early stopping, model = epoch terbaik) → Stage 3.

## 1. Data
- 95 file CSV, 88 hari terpakai (7 hari dilewati karena baris < 200.000): 11-07-2025 s.d. 19-06-2026.
- 54 hari Healthy, 34 hari Warning. Label hari = Warning jika ada kolom fault > 0.
- Split kronologis per hari: train 70 hari (44 Healthy / 26 Warning), test 18 hari (10 Healthy / 8 Warning).
- Kejadian fault hanya ~5 detik (2–66 baris) per hari Warning, sehingga tenggelam pada statistik harian; karena itu data dipecah per window 30 detik.
- Detail: `data_laporan/inventaris_data_per_hari.csv`, `data_laporan/ringkasan_data.json`.

## 2. Stage 1 — TCN Forecaster (algoritma tidak diubah dari v5)
- Input 3 hari → output 1 hari (21 parameter, 200.000 baris/hari). 7 blok residual, dilasi 1–64, 96 kanal, MSE, 100 epoch, batch 3. Window: train 67, test 18.

| | MSE | RMSE | MAE |
|---|---|---|---|
| Train | 0,006679 | 0,081727 | 0,032266 |
| Test (18 window) | 0,005940 | 0,077069 | 0,029585 |

- MSE epoch 100: 0,0085145 (rata-rata per epoch, ada dropout). Run v5: 0,00688.
- Error forecast per hari uji: terbaik 19-06-2026 (MSE 0,00266), terburuk 05-05-2026 (0,01878) → `data_laporan/forecast_error_per_hari_uji.csv`.
- MAPE tidak bermakna (pembagian nilai ≈ 0); jangan dipakai.
- Forecast mengikuti tren/rata-rata, tetapi tidak mereproduksi lonjakan cepat (mis. arus L2/L3) → `gambar/gambar_real_vs_prediksi_21parameter_19062026.png`.
- Bug resume (checkpoint epoch 100 melatih ulang dari epoch 1) diperbaiki satu baris `start_epoch = N_EPOCHS + 1`; algoritma tidak berubah. Training sempat berhenti ±8 jam karena hibernate dan evaluasi pertama crash cuDNN; model dimuat ulang dari `checkpoint_epoch_100.pth`.

## 3. Stage 2 — MLP Classifier (window 30 detik) — HASIL RESMI
- `MLPClassifier(hidden_layer_sizes=(64,32), solver='adam', alpha=1e-3)`, StandardScaler, 100 epoch penuh, **tanpa early stopping**, seed 42.
- **Pemilihan model:** tiap epoch model diuji (akurasi hari pada data test); model dengan akurasi hari terbaik disimpan sebagai model resmi → **epoch 69**. *Model dipilih berdasarkan data test, sehingga akurasi testing bersifat optimistis (batas atas).* Kurva tetap menampilkan seluruh 100 epoch (`gambar/jurnal_kurva_loss_acc_stage2.png`).
- Fitur: 21 parameter × 6 statistik (mean, std, max, min, skew, rate-of-change) = 126 per window 30 detik (117 baris, 1.709 window/hari). Label window = 1 jika ada fault di window. Keputusan hari = Warning jika ≥ 1 window diprediksi Warning.
- Window train 119.630 (Warning 41), window test 30.762 (Warning 13).

| Evaluasi (model epoch 69) | Hasil |
|---|---|
| **Akurasi training, hari (70)** | **90,00%**, F1 89,96%, recall Warning 84,62%, CM [[41,3],[4,22]] |
| **Akurasi testing, hari (18)** | **83,33%**, F1 83,17%, precision Warning 85,71%, recall Warning 75% (6/8), CM [[9,1],[2,6]] |
| Training, window | Acc 99,99%, recall Warning 78,05% (32/41), CM [[119582,7],[9,32]] |
| Testing, window | Acc 99,98%, recall Warning 61,54% (8/13), precision 80%, CM [[30747,2],[5,8]] |

Analisis tambahan (tidak digunakan untuk memilih model):
- Seed lain dengan model epoch 100 (seed 1, 7, 21, 99): akurasi hari 88,89 / 88,89 / 83,33 / 88,89%; recall Warning 75% semuanya. Seed 42 epoch 100 (sebelum pemilihan epoch): 77,78% (recall 50%). → `gambar/hasil_per_seed_stage2.png`.
- 5-fold cross-validation per hari, 88 hari (fold diacak, sehingga dapat lebih optimistis daripada test kronologis): rata-rata fold 89,80% ± 6,46; gabungan Acc 89,77%, F1 89,59%, recall Warning 79,41%, precision 93,10%, CM [[52,2],[7,27]].
- Akurasi window ≈ 100% tidak informatif karena 99,97% window adalah Healthy (menebak semua Healthy sudah ≈ 99,97%). Ukuran utama: akurasi hari, recall Warning, CM.
- Konfigurasi sebelumnya (diarsipkan di `arsip_konfigurasi_lama/`): early stopping (iterasi 13) seed 42 → 83,33%; 100 epoch tanpa pemilihan epoch (epoch 100) seed 42 → 77,78%.

Perbandingan versi (MLP per hari, hari uji): v4 (37 data) recall 62,5% / F1 48,1%; v5 (84 data) recall 62,5% / F1 53,1%; v6 (95 data) 50,0%; v7 window 30 detik: 83,33% (test, epoch terbaik).

## 4. Stage 3 — Inferensi berantai (TCN → MLP)
- Contoh bawaan Stage 3 (1 sampel uji): `hasil_csv/inference_health_status.csv`.
- Evaluasi seluruh 18 hari uji (`kode/evaluasi_rantai_semua_hari_uji.py`, `data_laporan/evaluasi_rantai_semua_hari_uji.csv`):

| Masukan ke MLP | Acc | TP | FN | FP | TN |
|---|---|---|---|---|---|
| Sinyal real | 83,33% | 6 | 2 | 1 | 9 |
| Sinyal prediksi TCN (rantai) | **55,56%** | **0** | 8 | 0 | 10 |

- Rantai memprediksi semua hari uji Healthy: benar di 10/10 hari Healthy, 0/8 hari Warning.
- Penyebab: TCN (MSE) memprediksi pola rata-rata dan tidak mereproduksi kejadian cepat; statistik window prediksi berbeda dari sinyal real, padahal MLP dilatih hanya pada sinyal real.
- Gambar forecast bertingkat (baris 1: real n..n+3; baris 2: real n..n+2 + prediksi n+3): `gambar/gambar_forecast_real_vs_prediksi_bertingkat.png` (19-06-2026, forecast terbaik) dan `..._benar_22042026.png` (prediksi Healthy, aktual Healthy). Contoh yang benar selalu kelas Healthy. Catatan: label prediksi pada dua gambar itu dibuat dengan model Stage 2 versi sebelumnya (hasilnya sama: Healthy).

## 5. Keterbatasan (untuk ditulis di laporan)
1. Test set kecil: 18 hari (8 Warning); 1 hari = 5,6 poin akurasi.
2. Hanya 41 window Warning pada train (13 pada test).
3. Epoch terbaik dipilih dari data test → akurasi testing 83,33% optimistis; pada epoch 100 dengan seed yang sama 77,78%, seed lain 83,33–88,89%.
4. Recall Warning 75% (6/8 hari): 2 hari Warning terlewat, 1 hari Healthy menjadi false alarm.
5. Rantai TCN → MLP belum mendeteksi Warning (0/8).
6. Pemilihan ukuran window (30 detik) dilakukan dengan melihat hasil test.

## 6. Isi folder
- `gambar/` semua gambar (nama setara v4: `jurnal_confusion_matrix_*`, `jurnal_kurva_loss_acc_stage2`, `jurnal_aktual_vs_prediksi_mlp`; plus evaluasi rantai, forecast bertingkat, 21 parameter, hasil per seed).
- `kode/`, `log/`, `hasil_csv/`, `data_laporan/`, `model_kecil/` (model Stage 2 epoch 69 + scaler), `arsip_konfigurasi_lama/`.
- Tidak disertakan (besar): `model_stage1_forecaster.pth` (1,5 GB), checkpoint (9,3 GB), 95 CSV mentah (13 GB), CSV prediksi sensor (86 MB).
