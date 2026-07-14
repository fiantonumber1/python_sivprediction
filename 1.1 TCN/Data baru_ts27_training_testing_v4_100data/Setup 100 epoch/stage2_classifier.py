# =============================
# STAGE 2 — MLP CLASSIFIER
# Per hari: Day1→Status1, Day2→Status2, dst (independen dari Stage 1)
# Input   : ringkasan 21 param sensor 1 hari (mean + std + max + min)
# Output  : model_stage2_classifier.pth
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
from sklearn.preprocessing import MinMaxScaler
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
DATA_DIR            = os.path.join(SCRIPT_DIR, "data")          # folder CSV terisolasi
CHECKPOINT_DIR      = os.path.join(SCRIPT_DIR, "checkpoints_stage2")
LOG_FILE            = os.path.join(SCRIPT_DIR, "log_stage2.txt")
EVIDENCE_DIR        = os.path.join(SCRIPT_DIR, "evidence")

N_EPOCHS            = 100
BATCH_SIZE          = 3
CHECKPOINT_INTERVAL = 50
TRAIN_RATIO         = 0.8   # proporsi hari untuk training (sisanya testing)
# ==================================================================

N_TAKE     = 200_000
START_TIME = time(3, 0, 0)
END_TIME                   = time(18, 16, 35)
# N_DROP_FIRST = 0 → tidak ada warmup drop.
# Alasan: beberapa hari memiliki fault (SIV_MajorInverterFltPres) yang terjadi
# di menit-menit pertama operasi (row 32–166 setelah filter waktu). Dengan
# N_DROP_FIRST=3600, fault tersebut seluruhnya terbuang sehingga label Warning
# tidak terdeteksi. Dengan N_DROP_FIRST=0, data diambil langsung dari awal
# operasi sehingga semua event fault masuk ke dalam window yang digunakan,
# dan label health status konsisten antara signal maupun ground truth.
N_DROP_FIRST               = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Fitur per hari = mean + std + max + min untuk setiap parameter → 21×4 = 84 dim
N_STATS   = 4
INPUT_DIM = n_features * N_STATS  # 84

# =============================
# BACA & PREPROCESSING
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

# =============================
# LABELING: per hari
# 0 fault aktif = Healthy, >=1 fault aktif = Warning
# =============================
def label_day(df_day):
    n_active = sum(1 for col in fault_columns
                   if col in df_day.columns and (df_day[col] > 0).any())
    if n_active == 0: return 0  # Healthy
    else:             return 1  # Warning (1 atau lebih fault)

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

# =============================
# TRAIN / TEST SPLIT (kronologis)
# =============================
n_train_days  = max(2, int(total_days * TRAIN_RATIO))
n_test_days   = total_days - n_train_days
train_dfs_cls = compressed_dfs[:n_train_days]
test_dfs_cls  = compressed_dfs[n_train_days:]
print(f"[Stage 2] Train days: {n_train_days} | Test days: {n_test_days}")

status_map = {0: "Healthy", 1: "Warning"}

# Setiap hari → 1 sampel fitur + 1 label
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

# Normalisasi fitur statistik (fit HANYA dari data training)
scaler_cls = MinMaxScaler(feature_range=(-0.1, 1.1))
X_scaled   = scaler_cls.fit_transform(X_train_cls)
joblib.dump(scaler_cls, os.path.join(SCRIPT_DIR, "scaler_stage2.pkl"))
print("[Stage 2] scaler_stage2.pkl disimpan")

X_tensor = torch.FloatTensor(X_scaled).to(device)
y_tensor = torch.LongTensor(y_train_cls).to(device)

class ClassDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dataloader = DataLoader(ClassDataset(X_tensor, y_tensor),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        pin_memory=(device.type == 'cuda'))

# =============================
# MODEL MLP CLASSIFIER
# =============================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=84, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model     = MLPClassifier(INPUT_DIM).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30, verbose=True)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.5]).to(device))

# =============================
# CHECKPOINT & RESUME
# =============================
start_epoch = 1
cp_files    = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
if cp_files:
    latest  = max(int(os.path.basename(f).split('_')[-1].replace('.pth','')) for f in cp_files)
    cp_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{latest}.pth")
    if latest < N_EPOCHS:
        cp = torch.load(cp_path, map_location=device)
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler']); start_epoch = latest + 1
        print(f"[Stage 2] Resume epoch {start_epoch}")
    else:
        model.load_state_dict(torch.load(cp_path, map_location=device)['model'])
        print(f"[Stage 2] Sudah selesai epoch {latest}")

def log(t):
    print(t)
    with open(LOG_FILE, 'a', encoding='utf-8') as f: f.write(t + '\n')

log(f"\n{'='*60}\nSTAGE 2 MLP CLASSIFIER | {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"Train days: {n_train_days} | Test days: {n_test_days}")
log(f"Sampel train: {len(X_tensor)} hari | Epoch: {N_EPOCHS} | Batch: {BATCH_SIZE}")
log(f"Distribusi Train: Healthy={sum(y_train_cls==0)} Warning={sum(y_train_cls==1)}")
log(f"{'='*60}")

# =============================
# TRAINING
# =============================
if start_epoch <= N_EPOCHS:
    model.train()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        total, correct, samples = 0.0, 0, 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total   += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            samples += y.size(0)
        avg = total / len(dataloader)
        acc = 100.0 * correct / samples
        scheduler.step(avg)
        log(f"Epoch {epoch:4d}/{N_EPOCHS} | CE: {avg:.6f} | Acc: {acc:6.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            p = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, p)
            log(f"   → Checkpoint: {p}")
    log("=== STAGE 2 TRAINING SELESAI ===\n")

# =============================
# EVALUASI TRAINING SET
# =============================
log("=== EVALUASI TRAINING SET (Stage 2) ===")
model.eval()
with torch.no_grad():
    logits_tr  = model(X_tensor)
    preds_tr   = logits_tr.argmax(1).cpu().numpy()
y_tr_arr   = np.array(y_train_cls, dtype=np.int64)
_ce_tr     = float(criterion(logits_tr, torch.LongTensor(y_tr_arr).to(device)).item())
_acc_tr    = 100.0 * float(np.mean(preds_tr == y_tr_arr))
_prec_tr   = precision_score(y_tr_arr, preds_tr, average='weighted', zero_division=0) * 100
_rec_tr    = recall_score(y_tr_arr, preds_tr, average='weighted', zero_division=0) * 100
_f1_tr     = f1_score(y_tr_arr, preds_tr, average='weighted', zero_division=0) * 100
log(f"Train Days   : {len(y_tr_arr)}")
log(f"Train CE     : {_ce_tr:.6f}")
log(f"Train Acc    : {_acc_tr:.2f}%")
log(f"Train Prec   : {_prec_tr:.2f}%")
log(f"Train Recall : {_rec_tr:.2f}%")
log(f"Train F1     : {_f1_tr:.2f}%")
log("=" * 40 + "\n")

# =============================
# EVALUASI TEST SET
# =============================
if X_test_cls:
    log("=== EVALUASI TEST SET (Stage 2) ===")
    X_test_scaled_arr = scaler_cls.transform(np.array(X_test_cls, dtype=np.float32))
    X_test_tensor     = torch.FloatTensor(X_test_scaled_arr).to(device)
    with torch.no_grad():
        logits_test = model(X_test_tensor)
        preds_test  = logits_test.argmax(1).cpu().numpy()
    y_test_arr = np.array(y_test_cls, dtype=np.int64)
    _ce_te     = float(criterion(logits_test, torch.LongTensor(y_test_arr).to(device)).item())
    _acc_te    = 100.0 * float(np.mean(preds_test == y_test_arr))
    _prec_te   = precision_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    _rec_te    = recall_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    _f1_te     = f1_score(y_test_arr, preds_test, average='weighted', zero_division=0) * 100
    log(f"Test Days    : {len(y_test_arr)}")
    log(f"Test CE      : {_ce_te:.6f}")
    log(f"Test Acc     : {_acc_te:.2f}%")
    log(f"Test Prec    : {_prec_te:.2f}%")
    log(f"Test Recall  : {_rec_te:.2f}%")
    log(f"Test F1      : {_f1_te:.2f}%")
    for day_i, (true_lbl, pred_lbl) in enumerate(zip(y_test_arr, preds_test)):
        match = "✓" if true_lbl == pred_lbl else "✗"
        log(f"  Day {n_train_days + day_i + 1:2d}: True={status_map[true_lbl]:9s} | Pred={status_map[pred_lbl]:9s} {match}")
    log("=" * 40)
else:
    log("[Stage 2] Tidak ada test days — tambah lebih banyak hari CSV")

torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model_stage2_classifier.pth"))
log("model_stage2_classifier.pth disimpan")

# =============================
# PLOT KURVA CE LOSS & ACCURACY — Jurnal Gambar 14
# Parse dari log_stage2.txt
# =============================
_ep2, _ce2, _acc2 = [], [], []
try:
    with open(LOG_FILE, 'r', encoding='utf-8') as _lf:
        for _line in _lf:
            _m = re.search(
                r'Epoch\s+(\d+)/\d+\s*\|\s*CE:\s*([\d.]+)\s*\|\s*Acc:\s*([\d.]+)%', _line)
            if _m:
                _ep2.append(int(_m.group(1)))
                _ce2.append(float(_m.group(2)))
                _acc2.append(float(_m.group(3)))
except Exception as _e:
    print(f"[Plot] Gagal parse log stage2: {_e}")

if _ep2:
    _d2 = {}
    for _ep, _ce, _ac in zip(_ep2, _ce2, _acc2):
        _d2[_ep] = (_ce, _ac)
    _eps   = sorted(_d2.keys())
    _ces   = [_d2[e][0] for e in _eps]
    _accs  = [_d2[e][1] for e in _eps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(_eps, _ces, 'r-', linewidth=1.5)
    ax1.set_xlabel("Epoch", fontsize=11); ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title(
        f"Gambar 14. Kurva Epoch vs Cross-Entropy Loss Training\nAwal: {_ces[0]:.4f} → Akhir: {_ces[-1]:.6f}", fontsize=12)
    ax1.grid(alpha=0.3)

    ax2.plot(_eps, _accs, 'g-', linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=11); ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 108)
    ax2.set_title(
        f"Accuracy vs. Epoch\nAkhir: {_accs[-1]:.2f}%", fontsize=12)
    ax2.grid(alpha=0.3)

    plt.suptitle("Gambar 14. Kurva Epoch vs Cross-Entropy Loss & Accuracy — MLP Classifier", fontsize=13, fontweight='bold')
    plt.tight_layout()
    _out2 = os.path.join(EVIDENCE_DIR, "jurnal_kurva_loss_acc_stage2.png")
    plt.savefig(_out2, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"jurnal_kurva_loss_acc_stage2.png disimpan  ← Jurnal Gambar 14")
else:
    print("[Plot] Log stage2 belum ada data epoch")

# =============================
# CONFUSION MATRIX — Jurnal
# =============================
_cls_names = ['Healthy', 'Warning']

# Prediksi training set
model.eval()
with torch.no_grad():
    _j2_logits_tr = model(X_tensor)
    _j2_preds_tr  = _j2_logits_tr.argmax(1).cpu().numpy()
_j2_cm_train = confusion_matrix(y_train_cls, _j2_preds_tr, labels=[0, 1])
_j2_acc_train = float(np.mean(_j2_preds_tr == y_train_cls)) * 100
_j2_ce_train  = float(nn.CrossEntropyLoss(
    weight=torch.tensor([0.5, 2.5]).to(device))(
    _j2_logits_tr,
    torch.LongTensor(y_train_cls).to(device)).item())

# Prediksi test set (jika ada)
_j2_cm_test = _j2_acc_test = _j2_ce_test = None
_j2_preds_test = _j2_y_test_np = None
if X_test_cls:
    _j2_X_te   = torch.FloatTensor(
        scaler_cls.transform(np.array(X_test_cls, dtype=np.float32))).to(device)
    _j2_y_te   = np.array(y_test_cls, dtype=np.int64)
    with torch.no_grad():
        _j2_logits_te  = model(_j2_X_te)
        _j2_preds_test = _j2_logits_te.argmax(1).cpu().numpy()
    _j2_cm_test   = confusion_matrix(_j2_y_te, _j2_preds_test, labels=[0, 1])
    _j2_acc_test  = float(np.mean(_j2_preds_test == _j2_y_te)) * 100
    _j2_ce_test   = float(nn.CrossEntropyLoss(
        weight=torch.tensor([0.5, 2.5]).to(device))(
        _j2_logits_te,
        torch.LongTensor(_j2_y_te).to(device)).item())
    _j2_y_test_np = _j2_y_te

def _plot_cm(cm, title, filename, y_true, y_pred):
    # Hitung 4 metrik
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
    # Tampilkan 4 metrik di bawah plot
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
    return _acc, _prec, _rec, _f1

_j2_acc_tr, _j2_prec_tr, _j2_rec_tr, _j2_f1_tr = _plot_cm(
    _j2_cm_train, "Confusion Matrix — Training Set",
    os.path.join(EVIDENCE_DIR, "jurnal_confusion_matrix_train.png"),
    y_train_cls, _j2_preds_tr)
_j2_acc_ts = _j2_prec_ts = _j2_rec_ts = _j2_f1_ts = None
if _j2_cm_test is not None:
    _j2_acc_ts, _j2_prec_ts, _j2_rec_ts, _j2_f1_ts = _plot_cm(
        _j2_cm_test, "Confusion Matrix — Test Set",
        os.path.join(EVIDENCE_DIR, "jurnal_confusion_matrix_test.png"),
        _j2_y_test_np, _j2_preds_test)

# =============================
# GAMBAR 15 — Kurva Aktual (Y_Testing) vs Prediksi — MLP Classifier
# Bar chart per hari test: status aktual vs status prediksi
# =============================
_j2_dates = [os.path.basename(f)[:8] for f in csv_files]
if _j2_cm_test is not None and _j2_y_test_np is not None:
    _n_test_cls  = len(_j2_y_test_np)
    _test_days   = list(range(n_train_days + 1, n_train_days + _n_test_cls + 1))
    _test_dates  = [_j2_dates[n_train_days + i] for i in range(_n_test_cls)]
    _cls_colors  = {0: '#2196F3', 1: '#FF9800'}  # biru, oranye
    _cls_names_s = ['Healthy', 'Warning']

    _bar_w   = 0.35
    _x_pos   = np.arange(_n_test_cls)
    fig, ax  = plt.subplots(figsize=(max(8, _n_test_cls * 2.2), 5.5))

    # Bar aktual
    for _i, (_true, _xp) in enumerate(zip(_j2_y_test_np, _x_pos)):
        ax.bar(_xp - _bar_w/2, _true + 1, _bar_w,
               color=_cls_colors[_true], alpha=0.85, edgecolor='white', linewidth=1.2,
               label=f'Aktual' if _i == 0 else '')
        ax.text(_xp - _bar_w/2, _true + 1 + 0.05, _cls_names_s[_true],
                ha='center', va='bottom', fontsize=8, color=_cls_colors[_true], fontweight='bold')

    # Bar prediksi
    for _i, (_pred, _xp) in enumerate(zip(_j2_preds_test, _x_pos)):
        _match = _j2_preds_test[_i] == _j2_y_test_np[_i]
        ax.bar(_xp + _bar_w/2, _pred + 1, _bar_w,
               color=_cls_colors[_pred], alpha=0.5, edgecolor=_cls_colors[_pred],
               linewidth=2, linestyle='--' if not _match else '-',
               label=f'Prediksi' if _i == 0 else '')
        ax.text(_xp + _bar_w/2, _pred + 1 + 0.05, _cls_names_s[_pred],
                ha='center', va='bottom', fontsize=8,
                color=_cls_colors[_pred], fontstyle='italic')
        # Tanda benar/salah di atas
        _sym = '✓' if _match else '✗'
        _sc  = '#2E7D32' if _match else '#B71C1C'
        ax.text(_xp, max(_j2_y_test_np[_i], _pred) + 1.35, _sym, ha='center', va='bottom',
                fontsize=14, color=_sc, fontweight='bold')

    ax.set_xticks(_x_pos)
    ax.set_xticklabels([f"Day {d}\n({dt})" for d, dt in zip(_test_days, _test_dates)],
                       fontsize=9)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Healthy (0)', 'Warning (1)'], fontsize=9)
    ax.set_ylim(0, 3.2)
    ax.set_ylabel("Label Kondisi Operasional", fontsize=11)
    ax.set_xlabel("Hari Testing", fontsize=11)

    # Legend manual
    from matplotlib.patches import Patch
    _leg = [
        Patch(facecolor='gray', alpha=0.85, label='Aktual (solid)'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='gray', linewidth=2, label='Prediksi (dashed)'),
        Patch(facecolor='white', label='✓ = Benar   ✗ = Salah'),
    ]
    ax.legend(handles=_leg, loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    _n_benar = int(np.sum(_j2_preds_test == _j2_y_test_np))
    ax.set_title(
        f"Gambar 15. Kurva Aktual (Y_Testing) vs Prediksi — MLP Classifier\n"
        f"Test Days: {_n_test_cls} hari | Benar: {_n_benar}/{_n_test_cls} | "
        f"Accuracy: {_j2_acc_ts:.2f}% | Precision: {_j2_prec_ts:.2f}% | "
        f"Recall: {_j2_rec_ts:.2f}% | F1: {_j2_f1_ts:.2f}%",
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EVIDENCE_DIR, "jurnal_aktual_vs_prediksi_mlp.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    log("jurnal_aktual_vs_prediksi_mlp.png disimpan  ← Jurnal Gambar 15")
else:
    print("[Plot] Gambar 15 skip — tidak ada test data")

# =============================
# FEATURE STATISTICS CSV — Tabel 15 Jurnal
# =============================
import csv as _csvmod
_j2_fstat = os.path.join(EVIDENCE_DIR, "jurnal_fitur_statistik.csv")
with open(_j2_fstat, 'w', newline='', encoding='utf-8-sig') as _fc:
    _fw = _csvmod.writer(_fc, delimiter=';')
    _hdr = ['Day', 'Tanggal', 'Set']
    for _c in target_columns:
        _hdr += [f"{_c}_mean", f"{_c}_std", f"{_c}_max", f"{_c}_min"]
    _fw.writerow(_hdr)
    _all_labels_j2 = list(y_train_cls) + list(y_test_cls)
    for _idx, (_dfday, _fpath, _lbl) in enumerate(zip(compressed_dfs, csv_files, _all_labels_j2)):
        _dstr = os.path.basename(_fpath)[:8]
        _set  = "Train" if _idx < n_train_days else "Test"
        _row  = [_idx + 1, _dstr, _set]
        for _c in target_columns:
            _v = _dfday[_c].values
            _row += [f"{_v.mean():.4f}", f"{_v.std():.4f}",
                     f"{_v.max():.4f}", f"{_v.min():.4f}"]
        _fw.writerow(_row)
log("jurnal_fitur_statistik.csv disimpan  ← Tabel Fitur Statistik Jurnal (84 kolom)")

# =============================
# JURNAL INFO TXT — Stage 2
# =============================
_j2_dates = [os.path.basename(f)[:8] for f in csv_files]
_j2_all_labels = list(y_train_cls) + list(y_test_cls)
_j2_label_dates = {0: [], 1: []}
for _fd, _lb in zip(_j2_dates, _j2_all_labels):
    _j2_label_dates[_lb].append(_fd)

_j2_ep1_ce = _j2_epN_ce = _j2_ep1_acc = _j2_epN_acc = None
if _ep2:
    _j2_d2 = {}
    for _e, _c, _a in zip(_ep2, _ce2, _acc2): _j2_d2[_e] = (_c, _a)
    _j2_ep_s = sorted(_j2_d2.keys())
    _j2_ep1_ce, _j2_ep1_acc = _j2_d2[_j2_ep_s[0]]
    _j2_epN_ce, _j2_epN_acc = _j2_d2[_j2_ep_s[-1]]

_j2_report_tr  = classification_report(y_train_cls, _j2_preds_tr,
                                        target_names=_cls_names, zero_division=0)
_j2_report_te  = (classification_report(_j2_y_test_np, _j2_preds_test,
                                         target_names=_cls_names, zero_division=0)
                  if _j2_y_test_np is not None else "(tidak ada test data)")

_j2_lines = [
    "=" * 68,
    "INFORMASI JURNAL — MLP CLASSIFIER (STAGE 2)",
    f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "=" * 68,
    "",
    "[DATASET & SPLIT]",
    f"  Total hari                   : {total_days}",
    f"  Train days (TRAIN_RATIO=0.8) : {n_train_days}",
    f"  Test days                    : {n_test_days}",
    f"  Input dimensi fitur          : {n_features} param x 4 statistik = {n_features*4} dim",
    "",
    "[DISTRIBUSI LABEL — untuk Tabel Jurnal (dengan tanggal)]",
    f"  Healthy (0)   : {len(_j2_label_dates[0])} hari",
    f"    Tanggal     : {', '.join(_j2_label_dates[0])}",
    f"  Warning (1)   : {len(_j2_label_dates[1])} hari",
    f"    Tanggal     : {', '.join(_j2_label_dates[1]) or '-'}",
    "",
    "[HYPERPARAMETER MLP — untuk Tabel Jurnal]",
    f"  Optimizer                    : AdamW",
    f"  Learning Rate (awal)         : 0.001",
    f"  Batch Size                   : {BATCH_SIZE}",
    f"  Epoch                        : {N_EPOCHS}",
    f"  Loss Function                : Cross-Entropy Loss",
    f"  Class Weight                 : [Healthy=0.5, Warning=2.5]",
    f"  Dropout                      : 0.3",
    f"  Weight Decay                 : 1e-5",
    f"  Scheduler                    : ReduceLROnPlateau (factor=0.5, patience=30)",
    "",
    "[ARSITEKTUR MLP — untuk Tabel Jurnal]",
    f"  Input Layer                  : {n_features * N_STATS} neuron (84 fitur statistik)",
    f"  Hidden Layer 1               : 256 neuron + LayerNorm + ReLU + Dropout",
    f"  Hidden Layer 2               : 128 neuron + LayerNorm + ReLU + Dropout",
    f"  Hidden Layer 3               : 64 neuron  + LayerNorm + ReLU + Dropout",
    f"  Output Layer                 : 2 neuron (Softmax) — Healthy/Warning",
    "",
    "[HASIL TRAINING — untuk Para 213 Jurnal]",
]
if _j2_ep1_ce is not None:
    _j2_pen_ce = (1 - _j2_epN_ce / _j2_ep1_ce) * 100
    _j2_lines += [
        f"  CE Loss Epoch 1              : {_j2_ep1_ce:.6f}",
        f"  CE Loss Epoch {_j2_ep_s[-1]} (akhir)   : {_j2_epN_ce:.6f}",
        f"  Penurunan CE Loss            : {_j2_pen_ce:.1f}%",
        f"  Accuracy Epoch 1             : {_j2_ep1_acc:.2f}%",
        f"  Accuracy Epoch {_j2_ep_s[-1]} (akhir)  : {_j2_epN_acc:.2f}%",
    ]
else:
    _j2_lines.append("  (Jalankan training untuk mengisi bagian ini)")

_j2_lines += [
    "",
    f"[EVALUASI TRAINING SET]",
    f"  CE Loss Training             : {_j2_ce_train:.6f}",
    f"  Accuracy   (weighted)        : {_j2_acc_tr:.2f}%",
    f"  Precision  (weighted)        : {_j2_prec_tr:.2f}%",
    f"  Recall     (weighted)        : {_j2_rec_tr:.2f}%",
    f"  F1-Score   (weighted)        : {_j2_f1_tr:.2f}%",
    "",
    "[CONFUSION MATRIX — TRAINING SET]",
    f"  {'':15} {'Pred Healthy':>12} {'Pred Warning':>12}",
    f"  {'Aktual Healthy':15} {_j2_cm_train[0,0]:>12} {_j2_cm_train[0,1]:>12}",
    f"  {'Aktual Warning':15} {_j2_cm_train[1,0]:>12} {_j2_cm_train[1,1]:>12}",
    "",
    "[CLASSIFICATION REPORT — TRAINING SET]",
]
for _rline in _j2_report_tr.splitlines():
    _j2_lines.append("  " + _rline)

_j2_lines += ["", "[EVALUASI TEST SET — untuk Para 215 Jurnal]"]
if _j2_acc_test is not None:
    _j2_selisih_ce = abs(_j2_epN_ce - _j2_ce_test) if _j2_epN_ce else None
    _j2_lines += [
        f"  Test Days                    : {len(_j2_y_test_np)}",
        f"  CE Loss Test                 : {_j2_ce_test:.6f}",
        f"  Accuracy   (weighted)        : {_j2_acc_ts:.2f}%",
        f"  Precision  (weighted)        : {_j2_prec_ts:.2f}%",
        f"  Recall     (weighted)        : {_j2_rec_ts:.2f}%",
        f"  F1-Score   (weighted)        : {_j2_f1_ts:.2f}%",
    ]
    if _j2_selisih_ce is not None:
        _j2_lines.append(f"  Selisih CE Loss (Train-Test)  : {_j2_selisih_ce:.6f}  → relatif kecil")
    _j2_lines += [
        "",
        "[CONFUSION MATRIX — TEST SET]",
        f"  {'':15} {'Pred Healthy':>12} {'Pred Warning':>12}",
        f"  {'Aktual Healthy':15} {_j2_cm_test[0,0]:>12} {_j2_cm_test[0,1]:>12}",
        f"  {'Aktual Warning':15} {_j2_cm_test[1,0]:>12} {_j2_cm_test[1,1]:>12}",
        "",
        "[CLASSIFICATION REPORT — TEST SET]",
    ]
    for _rline in _j2_report_te.splitlines():
        _j2_lines.append("  " + _rline)
    _j2_lines += [
        "",
        "[DETAIL PREDIKSI PER HARI — TEST SET]",
        f"  {'Day':>5}  {'Tanggal':>10}  {'Aktual':>12}  {'Prediksi':>12}  {'Benar?':>7}",
        "  " + "-" * 52,
    ]
    for _di, (_true, _pred) in enumerate(zip(_j2_y_test_np, _j2_preds_test)):
        _dstr = _j2_dates[n_train_days + _di]
        _ok   = "✓" if _true == _pred else "✗"
        _j2_lines.append(
            f"  {n_train_days+_di+1:>5}  {_dstr:>10}  "
            f"{_cls_names[_true]:>12}  {_cls_names[_pred]:>12}  {_ok:>7}")
else:
    _j2_lines.append("  (Tidak ada test days)")

_j2_out = os.path.join(EVIDENCE_DIR, "jurnal_info_stage2.txt")
with open(_j2_out, 'w', encoding='utf-8') as _f2:
    _f2.write('\n'.join(_j2_lines))
log("jurnal_info_stage2.txt disimpan  ← Data untuk Tabel & Paragraf Jurnal Stage 2")

print("\nSTAGE 2 SELESAI! Jalankan stage3_inference.py untuk prediksi.")
