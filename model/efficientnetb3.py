"""
EfficientNet-B3 — Leukemia Cell Classification
Dataset : CNMC-2019
Protocol: 5-Fold Stratified Cross-Validation
Outputs :
  OUTPUT_DIR/
    fold_{k}/
      best_model.pth          <- best val-AUC checkpoint per fold
      training_history.csv    <- epoch-level loss/acc/auc per fold
    all_folds_metrics.csv     <- per-fold + mean±std summary
    training_history_all.csv  <- all folds concatenated (for plotting)
    confusion_matrices.png    <- 2x3 grid of confusion matrices
    efficientnetb3_best_model.pth <- overall best model from all folds
"""

import os
import random
import time
import shutil  # <-- Added to handle copying the best model
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  —  edit only this section
# ─────────────────────────────────────────────────────────────
DATA_DIR   = '/home/eecommu06/Documents/BT/CNMC2019/image'
OUTPUT_DIR = '/home/eecommu06/Desktop/Bee/ALL/output/efficientnetB3'

NUM_FOLDS     = 5
NUM_EPOCHS    = 50
BATCH_SIZE    = 16
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10          # early-stopping patience (epochs without val-AUC improvement)
NUM_WORKERS   = 4
SEED          = 42
IMG_SIZE      = 224         # EfficientNet-B3 native resolution
NUM_CLASSES   = 2           # 0 = normal (hem), 1 = blast (all)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()

# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class LeukemiaDataset(Dataset):
    """
    Expects DATA_DIR with sub-folders per class, e.g.:
        DATA_DIR/
            all/   (or blast/)   -> label 1
            hem/   (or normal/)  -> label 0
    Class names are sorted alphabetically; adjust CLASS_MAP if needed.
    """
    CLASS_MAP = None

    def __init__(self, root, transform=None):
        self.root      = root
        self.transform = transform
        self.samples   = []
        self.classes   = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        LeukemiaDataset.CLASS_MAP = {c: i for i, c in enumerate(self.classes)}
        print(f"[Dataset] Classes found : {LeukemiaDataset.CLASS_MAP}")

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            label   = LeukemiaDataset.CLASS_MAP[cls]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        # per-class count
        for cls, idx in LeukemiaDataset.CLASS_MAP.items():
            n = sum(1 for _, l in self.samples if l == idx)
            print(f"  {cls}: {n} images")
        print(f"[Dataset] Total samples : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────────────────────────────────────────
# METRICS HELPER
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    ppv         = tp / (tp + fp + 1e-9)
    npv         = tn / (tn + fn + 1e-9)
    return {
        "accuracy"   : accuracy_score(y_true, y_pred),
        "f1"         : f1_score(y_true, y_pred, zero_division=0),
        "auc"        : roc_auc_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision"  : ppv,
        "npv"        : npv,
        "mcc"        : matthews_corrcoef(y_true, y_pred),
        "kappa"      : cohen_kappa_score(y_true, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH  (with tqdm batch bar)
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"  [Train] Epoch {epoch:3d}/{num_epochs}",
                leave=False, dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        n          += imgs.size(0)

        # live postfix on the batch bar
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.4f}")

    pbar.close()
    return total_loss / n, correct / n


# ─────────────────────────────────────────────────────────────
# VALIDATE / EVALUATE ONE EPOCH  (with tqdm batch bar)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="  [Val  ]"):
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        probs  = torch.softmax(outputs, dim=1)[:, 1]
        preds  = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        n += imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss/n:.4f}")

    pbar.close()
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics["loss"] = total_loss / n
    return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────────────────────────
# TRAIN ONE FOLD  (with tqdm epoch bar)
# ─────────────────────────────────────────────────────────────
def train_fold(fold, train_idx, val_idx, full_dataset, output_fold_dir):
    print(f"\n{'='*64}")
    print(f"  FOLD {fold+1}/{NUM_FOLDS}  |  "
          f"train={len(train_idx)}  val={len(val_idx)}")
    print(f"{'='*64}")
    os.makedirs(output_fold_dir, exist_ok=True)

    # datasets with correct transforms
    train_ds = Subset(_clone_dataset_with_transform(full_dataset, train_transform), train_idx)
    val_ds   = Subset(_clone_dataset_with_transform(full_dataset, val_transform),   val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # model / loss / optimizer / scheduler
    model     = build_model().to(DEVICE)
    labels_tr = [full_dataset.samples[i][1] for i in train_idx]
    n_neg     = labels_tr.count(0)
    n_pos     = labels_tr.count(1)
    pos_w     = n_neg / (n_pos + 1e-9)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_w], device=DEVICE)
    )
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history_rows   = []
    best_auc       = 0.0
    patience_count = 0

    # ── outer epoch progress bar ──────────────────────────────
    epoch_bar = tqdm(range(1, NUM_EPOCHS + 1),
                     desc=f"  Fold {fold+1} epochs",
                     dynamic_ncols=True,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for epoch in epoch_bar:
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, NUM_EPOCHS)

        val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, DEVICE,
            desc=f"  [Val  ] Epoch {epoch:3d}/{NUM_EPOCHS}")

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        # update outer bar postfix with key metrics
        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            val_auc=f"{val_metrics['auc']:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            best_auc=f"{best_auc:.4f}",
            patience=patience_count,
        )

        row = {
            "fold"        : fold + 1,
            "epoch"       : epoch,
            "lr"          : round(lr_now, 8),
            "train_loss"  : round(train_loss, 6),
            "train_acc"   : round(train_acc, 6),
            "val_loss"    : round(val_metrics["loss"], 6),
            "val_acc"     : round(val_metrics["accuracy"], 6),
            "val_f1"      : round(val_metrics["f1"], 6),
            "val_auc"     : round(val_metrics["auc"], 6),
            "val_sens"    : round(val_metrics["sensitivity"], 6),
            "val_spec"    : round(val_metrics["specificity"], 6),
            "val_prec"    : round(val_metrics["precision"], 6),
            "val_npv"     : round(val_metrics["npv"], 6),
            "val_mcc"     : round(val_metrics["mcc"], 6),
            "val_kappa"   : round(val_metrics["kappa"], 6),
            "epoch_time_s": round(elapsed, 2),
        }
        history_rows.append(row)

        # checkpoint: best val AUC
        if val_metrics["auc"] > best_auc:
            best_auc       = val_metrics["auc"]
            patience_count = 0
            torch.save({
                "fold"       : fold + 1,
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_auc"    : best_auc,
                "val_metrics": val_metrics,
                "class_map"  : LeukemiaDataset.CLASS_MAP,
            }, os.path.join(output_fold_dir, "best_model.pth"))
            tqdm.write(f"    ✓ [Fold {fold+1}] Epoch {epoch:3d} — "
                       f"New best model saved  val_AUC={best_auc:.4f}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                tqdm.write(f"    ✗ [Fold {fold+1}] Early stopping at epoch {epoch} "
                           f"(no improvement for {PATIENCE} epochs)")
                break

    epoch_bar.close()

    # save fold training history
    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(os.path.join(output_fold_dir, "training_history.csv"), index=False)

    # reload best checkpoint and compute final val metrics
    ckpt = torch.load(os.path.join(output_fold_dir, "best_model.pth"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    final_metrics, y_true, y_pred, y_prob = evaluate(
        model, val_loader, criterion, DEVICE, desc="  [Final eval]")

    tqdm.write(f"\n  ── Fold {fold+1} best-checkpoint results ──")
    tqdm.write(f"     AUC         : {final_metrics['auc']:.4f}")
    tqdm.write(f"     Accuracy    : {final_metrics['accuracy']:.4f}")
    tqdm.write(f"     F1          : {final_metrics['f1']:.4f}")
    tqdm.write(f"     Sensitivity : {final_metrics['sensitivity']:.4f}")
    tqdm.write(f"     Specificity : {final_metrics['specificity']:.4f}")
    tqdm.write(f"     MCC         : {final_metrics['mcc']:.4f}")

    return final_metrics, y_true, y_pred, hist_df


# ─────────────────────────────────────────────────────────────
# HELPER: clone dataset with new transform (no copy of images)
# ─────────────────────────────────────────────────────────────
class _TransformedDataset(Dataset):
    def __init__(self, original, transform):
        self.samples   = original.samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _clone_dataset_with_transform(original, transform):
    return _TransformedDataset(original, transform)


# ─────────────────────────────────────────────────────────────
# PLOT CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrices(cms, class_names, save_path):
    n    = len(cms)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for i, cm in enumerate(cms):
        ax = axes[i]
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"Fold {i+1}", fontsize=12)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Confusion matrices → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*64}")
    print(f"  EfficientNet-B3 — Leukemia Classification")
    print(f"{'='*64}")
    print(f"  Device     : {DEVICE}")
    print(f"  DATA_DIR   : {DATA_DIR}")
    print(f"  OUTPUT_DIR : {OUTPUT_DIR}")
    print(f"  Folds      : {NUM_FOLDS}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}")
    print(f"  Patience   : {PATIENCE}")
    print(f"{'='*64}\n")

    full_dataset = LeukemiaDataset(DATA_DIR, transform=None)
    all_labels   = [s[1] for s in full_dataset.samples]

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics_list = []
    confusion_mats    = []
    all_history_dfs   = []

    # <-- ADDED: Track the overall best model across all folds -->
    global_best_auc = -1.0
    global_best_model_src = None

    # ── outer fold progress bar ───────────────────────────────
    fold_bar = tqdm(enumerate(skf.split(np.zeros(len(full_dataset)), all_labels)),
                    total=NUM_FOLDS, desc="Overall folds",
                    bar_format="{l_bar}{bar}| fold {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for fold, (train_idx, val_idx) in fold_bar:
        fold_bar.set_description(f"Overall — Fold {fold+1}/{NUM_FOLDS}")
        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold+1}")
        metrics, y_true, y_pred, hist_df = train_fold(
            fold, train_idx, val_idx, full_dataset, fold_dir)

        cm = confusion_matrix(y_true, y_pred)
        confusion_mats.append(cm)
        all_history_dfs.append(hist_df)

        row = {"fold": fold + 1}
        row.update({k: round(v, 6) if isinstance(v, float) else v
                    for k, v in metrics.items()})
        fold_metrics_list.append(row)

        # <-- ADDED: Check if this fold has the highest AUC and store its path -->
        if metrics["auc"] > global_best_auc:
            global_best_auc = metrics["auc"]
            global_best_model_src = os.path.join(fold_dir, "best_model.pth")

        # update outer bar with running mean AUC
        run_auc = np.mean([r["auc"] for r in fold_metrics_list])
        fold_bar.set_postfix(mean_auc=f"{run_auc:.4f}")

    fold_bar.close()

    # ── aggregate metrics ─────────────────────────────────────
    metrics_df   = pd.DataFrame(fold_metrics_list)
    numeric_cols = [c for c in metrics_df.columns
                    if c != "fold" and pd.api.types.is_numeric_dtype(metrics_df[c])]
    mean_row = {"fold": "mean"}
    std_row  = {"fold": "std"}
    for c in numeric_cols:
        mean_row[c] = round(metrics_df[c].mean(), 6)
        std_row[c]  = round(metrics_df[c].std(),  6)

    summary_df = pd.concat([
        metrics_df,
        pd.DataFrame([mean_row]),
        pd.DataFrame([std_row]),
    ], ignore_index=True)

    metrics_path = os.path.join(OUTPUT_DIR, "all_folds_metrics.csv")
    summary_df.to_csv(metrics_path, index=False)
    print(f"\n[Saved] All-folds metrics       → {metrics_path}")

    # ── concatenated training history ─────────────────────────
    hist_all_df   = pd.concat(all_history_dfs, ignore_index=True)
    hist_all_path = os.path.join(OUTPUT_DIR, "training_history_all.csv")
    hist_all_df.to_csv(hist_all_path, index=False)
    print(f"[Saved] All training history     → {hist_all_path}")

    # ── confusion matrices ────────────────────────────────────
    class_names = list(LeukemiaDataset.CLASS_MAP.keys())
    cm_path     = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plot_confusion_matrices(confusion_mats, class_names, cm_path)
    
    # ── save overall best model ───────────────────────────────
    # <-- ADDED: copy the top performing checkpoint out into the root directory -->
    if global_best_model_src and os.path.exists(global_best_model_src):
        overall_best_path = os.path.join(OUTPUT_DIR, "efficientnetb3_best_model.pth")
        shutil.copy2(global_best_model_src, overall_best_path)
        print(f"[Saved] Overall best model (AUC: {global_best_auc:.4f}) → {overall_best_path}")

    # ── final summary ─────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  FINAL 5-FOLD SUMMARY — EfficientNet-B3")
    print(f"{'='*64}")
    key_metrics = ["accuracy", "f1", "auc", "sensitivity",
                   "specificity", "precision", "npv", "mcc", "kappa"]
    for m in key_metrics:
        if m in metrics_df.columns:
            vals = metrics_df[m].values
            print(f"  {m:<14}: {vals.mean():.4f} ± {vals.std():.4f}"
                  f"  [{vals.min():.4f} – {vals.max():.4f}]")
    print(f"{'='*64}")
    print(f"\n[Done] All outputs saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()