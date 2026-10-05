# =============================
# STAGE 2 — MLP CLASSIFIER (IMPROVED v2)
# Per hari: Day1->Status1, Day2->Status2, dst (independen dari Stage 1)
# Input   : ringkasan 21 param sensor 1 hari (mean + std + max + min)
# Output  : model_stage2_classifier.pth (ensemble K-fold)
#
# MASALAH versi asli: overfitting parah (Train Acc 100% vs Test Acc 62.5%,
# Test F1 48.08%, 3/3 kasus Warning di test SALAH semua diprediksi Healthy).
# Penyebab: dataset sangat kecil (29 sampel, cuma 9 Warning) dilatih dengan
# MLP besar (84->256->128->64->2, ~62.600 parameter) TANPA validation/early
# stopping sama sekali selama 100 epoch penuh -> model menghafal training set.
#
# PERBAIKAN v2:
#   1) MODEL DIPERKECIL drastis: 84->32->16->2 (~3.200 parameter, turun ~19x)
#      supaya proporsional dengan jumlah sampel yang sangat sedikit.
#   2) REGULARISASI lebih kuat: dropout 0.3->0.5, weight_decay 1e-5->5e-4.
#   3) FEATURE JITTER AUGMENTATION: tiap epoch, sampel training (feature 84-dim
#      hasil normalisasi) diberi noise Gaussian kecil (std relatif ke std fitur)
#      -> membuat sampel "virtual" berbeda tiap epoch, regularisasi efektif untuk
#      kelas minoritas (Warning) yang cuma 9 sampel.
#   4) STRATIFIED K-FOLD CROSS-VALIDATION (K=5) di atas 29 hari training:
#      tiap fold dilatih dengan early stopping (monitor val weighted-F1),
#      menghasilkan 5 model. Prediksi akhir = ENSEMBLE (rata-rata softmax
#      5 model) -> jauh lebih stabil daripada 1 model yang gampang overfit
#      ke 29 sampel. Test set (8 hari) TIDAK PERNAH dipakai selama CV/training,
#      cuma dipakai sekali di evaluasi akhir.
#   5) Class weight [Healthy=0.5, Warning=2.5] dipertahankan.
# =============================

import copy
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib
import re
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==================================================================
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
DATA_DIR             = os.path.join(SCRIPT_DIR, "..", "Setup 100 epoch", "data")  # referensi data asli (read-only)
CHECKPOINT_DIR      = os.path.join(SCRIPT_DIR, "checkpoints_stage2")
LOG_FILE            = os.path.join(SCRIPT_DIR, "log_stage2.txt")
EVIDENCE_DIR        = os.path.join(SCRIPT_DIR, "evidence")

N_EPOCHS            = 100
BATCH_SIZE          = 3
TRAIN_RATIO         = 0.8
K_FOLDS             = 5
EARLY_STOP_PATIENCE = 20
JITTER_STD          = 0.05   # std noise augmentation, relatif thd skala fitur ternormalisasi
# ==================================================================

N_TAKE     = 200_000
START_TIME = time(3, 0, 0)
END_TIME                   = time(18, 16, 35)
N_DROP_FIRST               = 0

# Dipaksa CPU: model kecil (~3.200 parameter/fold, 29 sampel) sehingga CPU sudah
# cukup cepat, dan supaya tidak berebut VRAM dengan Stage 1 TCN yang sedang
# memakai ~97% GPU memory secara bersamaan.
device = torch.device('cpu')
torch.set_num_threads(os.cpu_count() or 1)
print(f"[Stage 2] Device: {device} | CPU threads: {torch.get_num_threads()}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR,   exist_ok=True)

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres']
n_features    = len(target_columns)  # 21

N_STATS   = 4
INPUT_DIM = n_features * N_STATS  # 84

# =============================
# BACA & PREPROCESSING (identik versi asli)
# =============================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if "hasil"      not in os.path.basename(f).lower()
    and "prediksi"  not in os.path.basename(f).lower()
    and "inference" not in os.path.basename(f).lower()
], key=extract_date)

print(f"[Stage 2] Data dir : {DATA_DIR}")
print(f"[Stage 2] File CSV : {len(csv_files)} ditemukan")

_needed_cols = set(['ts_date'] + target_columns + fault_columns)

def read_and_crop(filepath):
    df = pd.read_csv(
        filepath,
        encoding='utf-8-sig', sep=';', decimal=',',
        usecols=lambda c: c.strip() in _needed_cols,
        low_memory=False, on_bad_lines='skip'
    )
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])
    for col in target_columns + fault_columns:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[target_columns + fault_columns] = df[target_columns + fault_columns].ffill().bfill()
    date0 = df['ts_date'].dt.date.iloc[0]
    df    = df[(df['ts_date'] >= datetime.combine(date0, START_TIME)) &
               (df['ts_date'] <= datetime.combine(date0, END_TIME))]
    if len(df) < N_TAKE:
        return pd.DataFrame()
    return df.iloc[:N_TAKE].reset_index(drop=True)[['ts_date'] + target_columns + fault_columns]

def day_to_feature_vector(df_day):
    vals = df_day[target_columns].values
    feat = np.concatenate([
        vals.mean(axis=0),
        vals.std(axis=0),
        vals.max(axis=0),
        vals.min(axis=0),
    ])
    return feat.astype(np.float32)

def label_day(df_day):
    n_active = sum(1 for col in fault_columns
                   if col in df_day.columns and (df_day[col] > 0).any())
    if n_active == 0: return 0  # Healthy
    else:             return 1  # Warning

compressed_dfs = []
for f in csv_files:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"  Skip {os.path.basename(f)}")
        continue
    compressed_dfs.append(df_raw[['ts_date'] + target_columns + fault_columns].copy())

total_days = len(compressed_dfs)
print(f"[Stage 2] Total hari: {total_days}")
if total_days < 2:
    raise ValueError("Minimal 2 hari CSV!")

n_train_days  = max(2, int(total_days * TRAIN_RATIO))
n_test_days   = total_days - n_train_days
train_dfs_cls = compressed_dfs[:n_train_days]
test_dfs_cls  = compressed_dfs[n_train_days:]
print(f"[Stage 2] Train days: {n_train_days} | Test days: {n_test_days}")

status_map = {0: "Healthy", 1: "Warning"}

X_train_cls, y_train_cls = [], []
for i, df_day in enumerate(train_dfs_cls):
    feat  = day_to_feature_vector(df_day)
    label = label_day(df_day)
    X_train_cls.append(feat)
    y_train_cls.append(label)
    print(f"  [Train] Day {i+1:2d} -> {status_map[label]}")

X_test_cls, y_test_cls = [], []
for i, df_day in enumerate(test_dfs_cls):
    feat  = day_to_feature_vector(df_day)
    label = label_day(df_day)
    X_test_cls.append(feat)
    y_test_cls.append(label)
    print(f"  [Test]  Day {n_train_days+i+1:2d} -> {status_map[label]}")

X_train_cls = np.array(X_train_cls, dtype=np.float32)
y_train_cls = np.array(y_train_cls, dtype=np.int64)
print(f"[Stage 2] Sampel training: {len(X_train_cls)} hari")
print(f"[Stage 2] Sampel testing : {len(X_test_cls)} hari")

# Scaler di-fit dari SELURUH data training (dipakai konsisten across folds & test)
scaler_cls = MinMaxScaler(feature_range=(-0.1, 1.1))
X_scaled_all = scaler_cls.fit_transform(X_train_cls)
joblib.dump(scaler_cls, os.path.join(SCRIPT_DIR, "scaler_stage2.pkl"))
print("[Stage 2] scaler_stage2.pkl disimpan")

X_test_scaled_arr = scaler_cls.transform(np.array(X_test_cls, dtype=np.float32)) if X_test_cls else None
y_test_arr = np.array(y_test_cls, dtype=np.int64) if y_test_cls else None

# =============================
# MODEL MLP CLASSIFIER — DIPERKECIL
# =============================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=84, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),        nn.LayerNorm(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

class ClassDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def log(t):
    print(t)
    with open(LOG_FILE, 'a', encoding='utf-8') as f: f.write(t + '\n')

_tmp_model = MLPClassifier(INPUT_DIM)
n_params = sum(p.numel() for p in _tmp_model.parameters())
del _tmp_model

log(f"\n{'='*60}\nSTAGE 2 MLP CLASSIFIER (IMPROVED v2) | {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"Train days: {n_train_days} | Test days: {n_test_days}")
log(f"K-Fold: {K_FOLDS} | Epoch(max): {N_EPOCHS} | Batch: {BATCH_SIZE} | Total parameter/model: {n_params:,}")
log(f"Distribusi Train: Healthy={int(np.sum(y_train_cls==0))} Warning={int(np.sum(y_train_cls==1))}")
log(f"{'='*60}")

# =============================
# K-FOLD CROSS VALIDATION TRAINING
# =============================
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_models   = []
fold_val_f1   = []
_all_fold_curves = []  # untuk plotting

feat_std = X_scaled_all.std(axis=0, keepdims=True) + 1e-6  # dipakai skala noise jitter

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_scaled_all, y_train_cls), start=1):
    X_tr, y_tr = X_scaled_all[tr_idx], y_train_cls[tr_idx]
    X_va, y_va = X_scaled_all[va_idx], y_train_cls[va_idx]

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    y_va_t = torch.LongTensor(y_va).to(device)
    feat_std_t = torch.FloatTensor(feat_std).to(device)

    loader = DataLoader(ClassDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model     = MLPClassifier(INPUT_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8, verbose=False)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.5]).to(device))

    best_f1, best_state, bad_epochs = -1.0, None, 0
    curve = {'epoch': [], 'ce': [], 'acc': [], 'val_f1': []}

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total, correct, samples = 0.0, 0, 0
        for x, y in loader:
            x_aug = x + torch.randn_like(x) * feat_std_t * JITTER_STD  # feature jitter augmentation
            optimizer.zero_grad()
            logits = model(x_aug)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total   += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            samples += y.size(0)
        avg = total / max(1, len(loader))
        acc = 100.0 * correct / max(1, samples)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_loss   = criterion(val_logits, y_va_t).item()
            val_preds  = val_logits.argmax(1).cpu().numpy()
        val_f1 = f1_score(y_va, val_preds, average='weighted', zero_division=0)
        scheduler.step(val_loss)

        curve['epoch'].append(epoch); curve['ce'].append(avg); curve['acc'].append(acc); curve['val_f1'].append(val_f1)

        if val_f1 > best_f1 + 1e-6:
            best_f1, best_state, bad_epochs = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if bad_epochs >= EARLY_STOP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    fold_models.append(model)
    fold_val_f1.append(best_f1)
    _all_fold_curves.append(curve)
    log(f"[Fold {fold_idx}/{K_FOLDS}] n_train={len(tr_idx)} n_val={len(va_idx)} | epoch berhenti={curve['epoch'][-1]} | Best Val F1={best_f1*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"fold{fold_idx}_best.pth"))

log(f"\nRata-rata Val F1 across {K_FOLDS} fold: {np.mean(fold_val_f1)*100:.2f}% (std {np.std(fold_val_f1)*100:.2f}%)")
log("=== STAGE 2 TRAINING (K-FOLD) SELESAI ===\n")

# =============================
# ENSEMBLE PREDICT — helper
# =============================
def ensemble_predict_proba(X_np):
    X_t = torch.FloatTensor(X_np).to(device)
    probs_sum = None
    with torch.no_grad():
        for m in fold_models:
            m.eval()
            logits = m(X_t)
            probs  = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    probs_avg = (probs_sum / len(fold_models)).cpu().numpy()
    return probs_avg

# =============================
# EVALUASI TRAINING SET (ensemble, seluruh 29 hari)
# =============================
log("=== EVALUASI TRAINING SET (Stage 2, ENSEMBLE) ===")
probs_tr  = ensemble_predict_proba(X_scaled_all)
preds_tr  = probs_tr.argmax(axis=1)
_acc_tr   = 100.0 * float(np.mean(preds_tr == y_train_cls))
_prec_tr  = precision_score(y_train_cls, preds_tr, average='weighted', zero_division=0) * 100
_rec_tr   = recall_score(y_train_cls, preds_tr, average='weighted', zero_division=0) * 100
_f1_tr    = f1_score(y_train_cls, preds_tr, average='weighted', zero_division=0) * 100
log(f"Train Days   : {len(y_train_cls)}")
log(f"Train Acc    : {_acc_tr:.2f}%")
log(f"Train Prec   : {_prec_tr:.2f}%")
log(f"Train Recall : {_rec_tr:.2f}%")
log(f"Train F1     : {_f1_tr:.2f}%")
log("=" * 40 + "\n")

# =============================
# EVALUASI TEST SET (ensemble, held-out murni, tak pernah dipakai training/CV)
# =============================
_acc_ts = _prec_ts = _rec_ts = _f1_ts = None
preds_test = None
if X_test_scaled_arr is not None:
    log("=== EVALUASI TEST SET (Stage 2, ENSEMBLE - held-out) ===")
    probs_test = ensemble_predict_proba(X_test_scaled_arr)
    preds_test = probs_test.argmax(axis=1)
    _acc_ts  = 100.0 * float(np.mean(preds_test == y_test_arr))
    _prec_ts = precision_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    _rec_ts  = recall_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    _f1_ts   = f1_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    log(f"Test Days    : {len(y_test_arr)}")
    log(f"Test Acc     : {_acc_ts:.2f}%")
    log(f"Test Prec    : {_prec_ts:.2f}%")
    log(f"Test Recall  : {_rec_ts:.2f}%")
    log(f"Test F1      : {_f1_ts:.2f}%")
    _dates_all = [os.path.basename(f)[:8] for f in csv_files]
    for day_i, (true_lbl, pred_lbl) in enumerate(zip(y_test_arr, preds_test)):
        match = "OK" if true_lbl == pred_lbl else "X"
        log(f"  Day {n_train_days + day_i + 1:2d} ({_dates_all[n_train_days+day_i]}): True={status_map[true_lbl]:9s} | Pred={status_map[pred_lbl]:9s} {match}")
    log("=" * 40)
else:
    log("[Stage 2] Tidak ada test days")

# Simpan ensemble sebagai satu file (state_dict per fold)
torch.save({f"fold{i+1}": m.state_dict() for i, m in enumerate(fold_models)},
           os.path.join(SCRIPT_DIR, "model_stage2_classifier.pth"))
log("model_stage2_classifier.pth disimpan (ensemble 5-fold)")

# =============================
# PLOT: Val F1 per fold + kurva CE fold pertama (representatif)
# =============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar([f"Fold {i+1}" for i in range(K_FOLDS)], [v*100 for v in fold_val_f1], color='#42A5F5')
axes[0].axhline(np.mean(fold_val_f1)*100, color='red', linestyle='--', label=f'Rata-rata: {np.mean(fold_val_f1)*100:.1f}%')
axes[0].set_ylabel("Val F1-Score (%)"); axes[0].set_title("Val F1 per Fold (early-stopping)")
axes[0].legend(); axes[0].grid(alpha=0.3)

_c0 = _all_fold_curves[0]
axes[1].plot(_c0['epoch'], _c0['ce'], 'r-', label='CE Loss (fold 1)')
ax2b = axes[1].twinx()
ax2b.plot(_c0['epoch'], [f*100 for f in _c0['val_f1']], 'g-', label='Val F1 % (fold 1)')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("CE Loss", color='r')
ax2b.set_ylabel("Val F1 (%)", color='g')
axes[1].set_title("Contoh Kurva Training (Fold 1, early-stopping-aware)")
axes[1].grid(alpha=0.3)

plt.suptitle("Stage 2 MLP Classifier IMPROVED v2 — K-Fold CV + Early Stopping", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EVIDENCE_DIR, "kurva_kfold_stage2.png"), dpi=300, bbox_inches='tight')
plt.close()
log("kurva_kfold_stage2.png disimpan")

# =============================
# CONFUSION MATRIX
# =============================
_cls_names = ['Healthy', 'Warning']

def _plot_cm(cm, title, filename, y_true, y_pred):
    _acc  = float(np.mean(np.array(y_pred) == np.array(y_true))) * 100
    _prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    _rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    _f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=_cls_names, yticklabels=_cls_names,
           ylabel='Label Aktual', xlabel='Label Prediksi')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    _thresh = cm.max() / 2.0
    for _i in range(2):
        for _j in range(2):
            ax.text(_j, _i, str(cm[_i, _j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[_i, _j] > _thresh else 'black')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    metrics_text = (f"Accuracy: {_acc:.2f}%    Precision: {_prec:.2f}%    "
                    f"Recall: {_rec:.2f}%    F1-Score: {_f1:.2f}%")
    fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom', fontsize=10,
             fontweight='bold', color='#1565C0',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0'))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"{os.path.basename(filename)} — Acc={_acc:.2f}% Prec={_prec:.2f}% Rec={_rec:.2f}% F1={_f1:.2f}%")

_cm_train = confusion_matrix(y_train_cls, preds_tr, labels=[0, 1])
_plot_cm(_cm_train, "Confusion Matrix — Training Set (Ensemble)",
         os.path.join(EVIDENCE_DIR, "confusion_matrix_train.png"), y_train_cls, preds_tr)

if preds_test is not None:
    _cm_test = confusion_matrix(y_test_arr, preds_test, labels=[0, 1])
    _plot_cm(_cm_test, "Confusion Matrix — Test Set (Ensemble, held-out)",
             os.path.join(EVIDENCE_DIR, "confusion_matrix_test.png"), y_test_arr, preds_test)

# =============================
# INFO TXT
# =============================
_lines = [
    "=" * 68,
    "INFORMASI — MLP CLASSIFIER STAGE 2 (IMPROVED v2)",
    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "=" * 68,
    "",
    "[PERBEDAAN UTAMA vs VERSI ASLI]",
    "  1. Model diperkecil: 84->32->16->2 (~3.200 parameter, vs ~62.600 sebelumnya)",
    "  2. Dropout 0.5, weight_decay 5e-4 (lebih kuat)",
    "  3. Feature jitter augmentation (noise Gaussian tiap epoch)",
    f"  4. Stratified {K_FOLDS}-Fold CV + early stopping (patience {EARLY_STOP_PATIENCE}) per fold",
    "  5. Prediksi akhir = ENSEMBLE rata-rata softmax dari semua fold",
    "",
    "[DATASET & SPLIT]",
    f"  Total hari       : {total_days}",
    f"  Train days       : {n_train_days} (Healthy={int(np.sum(y_train_cls==0))}, Warning={int(np.sum(y_train_cls==1))})",
    f"  Test days        : {n_test_days}",
    "",
    "[HASIL PER FOLD (Val F1, early-stopping)]",
]
for i, f1v in enumerate(fold_val_f1, start=1):
    _lines.append(f"  Fold {i}: {f1v*100:.2f}%")
_lines += [
    f"  Rata-rata: {np.mean(fold_val_f1)*100:.2f}% (std {np.std(fold_val_f1)*100:.2f}%)",
    "",
    "[EVALUASI TRAINING SET - ENSEMBLE]",
    f"  Accuracy  : {_acc_tr:.2f}%",
    f"  Precision : {_prec_tr:.2f}%",
    f"  Recall    : {_rec_tr:.2f}%",
    f"  F1-Score  : {_f1_tr:.2f}%",
]
if _acc_ts is not None:
    _lines += [
        "",
        "[EVALUASI TEST SET - ENSEMBLE, HELD-OUT MURNI]",
        f"  Accuracy  : {_acc_ts:.2f}%",
        f"  Precision : {_prec_ts:.2f}%",
        f"  Recall    : {_rec_ts:.2f}%",
        f"  F1-Score  : {_f1_ts:.2f}%",
    ]

_out_txt = os.path.join(EVIDENCE_DIR, "info_stage2.txt")
with open(_out_txt, 'w', encoding='utf-8') as _f:
    _f.write('\n'.join(_lines))
log("info_stage2.txt disimpan")

print("\nSTAGE 2 (IMPROVED v2) SELESAI! Jalankan stage3_inference.py untuk prediksi.")
