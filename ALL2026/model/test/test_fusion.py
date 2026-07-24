"""
=============================================================================
Cross-Dataset Blind Test Script — Fusion (ResNet50 + MobileNetV2)
Training set : CNMC-2019  (5-fold stratified CV)
Test set     : ALL-IDB2   (blind, never seen during training)
Model        : best_model.pth from fold_3
=============================================================================
Output files (all saved beside this script or in ./test_results/):
  • test_results/metrics_summary.csv       — per-class + macro/weighted metrics
  • test_results/confusion_matrix.png      — normalised heat-map
  • test_results/roc_curve.png             — ROC + AUC
  • test_results/precision_recall_curve.png
  • test_results/prediction_details.csv   — per-image ground-truth & prediction
=============================================================================
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on headless server)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

# ─────────────────────────────────────────────
#  CONFIGURATION  (edit here if paths change)
# ─────────────────────────────────────────────
DATASET_DIR  = "/home/eecommu06/Documents/BT/ALLIDB2"   # root: cancer/ non cancer/
MODEL_PATH   = "/home/eecommu06/Desktop/Bee/ALL/output/fusion_mobilenet_resnet/fusion_mobilenet_resnet_best_model.pth"
OUTPUT_DIR   = "./test/test_results/fusion_mobilenet_resnet"
BATCH_SIZE   = 32
NUM_WORKERS  = 4
IMG_SIZE     = 257        # Standard resolution for both ResNet50 and MobileNetV2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class order: torchvision ImageFolder sorts alphabetically
# "cancer" → index 0, "non cancer" → index 1
CLASS_NAMES  = ["cancer", "non cancer"]   # adjust if your folder names differ

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  TRANSFORMS  (same normalisation as training)
# ─────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  DATASET & DATALOADER
# ─────────────────────────────────────────────
print(f"\n[INFO] Loading dataset from: {DATASET_DIR}")
test_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS,
                          pin_memory=True)

# Remap torchvision class indices → our CLASS_NAMES order
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
print(f"[INFO] Class mapping (torchvision): {test_dataset.class_to_idx}")
print(f"[INFO] Total samples : {len(test_dataset)}")

# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class FusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FusionModel, self).__init__()
        
        # 1. Load base models
        self.resnet = models.resnet50(weights=None)
        self.mobilenet = models.mobilenet_v2(weights=None)
        
        # 2. Strip original classifiers to extract features
        # ResNet50 outputs 2048 features
        self.resnet.fc = nn.Identity()
        # MobileNetV2 outputs 1280 features
        self.mobilenet.classifier = nn.Identity()
        
        # 3. Define the custom Fusion Head
        combined_features = 2048 + 1280  # 3328 total features
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f_res = self.resnet(x)
        f_mob = self.mobilenet(x)
        
        # Concatenate features along the channel dimension
        f_concat = torch.cat((f_res, f_mob), dim=1)
        
        out = self.classifier(f_concat)
        return out

def build_model(num_classes: int = 2) -> nn.Module:
    return FusionModel(num_classes=num_classes)

print(f"\n[INFO] Loading model from: {MODEL_PATH}")
model = build_model(num_classes=len(CLASS_NAMES))
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# Auto-detect checkpoint format
if isinstance(checkpoint, dict):
    print(f"[INFO] Checkpoint keys: {list(checkpoint.keys())}")
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("model_state")
        or checkpoint.get("state_dict")
        or checkpoint.get("model")
    )
    if state_dict is None:
        raise KeyError(f"Cannot find model weights. Keys found: {list(checkpoint.keys())}")
    
    # Using strict=False just in case there are minor structural naming differences
    model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded | fold={checkpoint.get('fold','?')} "
          f"epoch={checkpoint.get('epoch','?')} val_auc={checkpoint.get('val_auc','?')}")
else:
    model.load_state_dict(checkpoint, strict=False)
    print("[INFO] Loaded raw state dict")

model = model.to(DEVICE)
model.eval()

# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
print(f"\n[INFO] Running inference on {DEVICE} …")
all_labels, all_preds, all_probs = [], [], []
image_paths = [s[0] for s in test_dataset.samples]

t0 = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

elapsed = time.time() - t0
all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)   # shape (N, 2)

print(f"[INFO] Inference done in {elapsed:.1f}s  ({len(all_labels)} samples)")

# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────
# Cancer = positive class (index 0 in sorted order)
pos_idx    = test_dataset.class_to_idx.get("cancer", 0)
cancer_prob = all_probs[:, pos_idx]   # probability of cancer class

acc        = accuracy_score(all_labels, all_preds)
prec_mac   = precision_score(all_labels, all_preds, average="macro",   zero_division=0)
rec_mac    = recall_score(all_labels,   all_preds, average="macro",   zero_division=0)
f1_mac     = f1_score(all_labels,       all_preds, average="macro",   zero_division=0)
prec_wt    = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
rec_wt     = recall_score(all_labels,   all_preds, average="weighted", zero_division=0)
f1_wt      = f1_score(all_labels,       all_preds, average="weighted", zero_division=0)
mcc        = matthews_corrcoef(all_labels, all_preds)
kappa      = cohen_kappa_score(all_labels, all_preds)

fpr, tpr, _ = roc_curve(all_labels, cancer_prob, pos_label=pos_idx)
roc_auc     = auc(fpr, tpr)
ap          = average_precision_score((all_labels == pos_idx).astype(int), cancer_prob)

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (None, None, None, None)

report_dict = classification_report(
    all_labels, all_preds,
    target_names=[idx_to_class[i] for i in sorted(idx_to_class)],
    output_dict=True, zero_division=0,
)

print("\n" + "="*60)
print("  TEST RESULTS — Fusion (ResNet50 + MobileNetV2) on ALL-IDB2")
print("="*60)
print(f"  Accuracy          : {acc:.4f}")
print(f"  Precision (macro) : {prec_mac:.4f}")
print(f"  Recall    (macro) : {rec_mac:.4f}")
print(f"  F1-score  (macro) : {f1_mac:.4f}")
print(f"  F1-score (weight) : {f1_wt:.4f}")
print(f"  ROC-AUC           : {roc_auc:.4f}")
print(f"  Avg Precision     : {ap:.4f}")
print(f"  MCC               : {mcc:.4f}")
print(f"  Cohen Kappa       : {kappa:.4f}")
if tp is not None:
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print("="*60 + "\n")

# ─────────────────────────────────────────────
#  SAVE metrics_summary.csv
# ─────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Model", "Fusion (ResNet50 + MobileNetV2)"])
    writer.writerow(["Test Dataset", "ALL-IDB2"])
    writer.writerow(["Model Path", MODEL_PATH])
    writer.writerow(["Total Samples", len(all_labels)])
    writer.writerow(["Inference Time (s)", f"{elapsed:.2f}"])
    writer.writerow([])
    writer.writerow(["--- Overall ---"])
    writer.writerow(["Accuracy",               f"{acc:.4f}"])
    writer.writerow(["Precision (macro)",      f"{prec_mac:.4f}"])
    writer.writerow(["Recall (macro)",         f"{rec_mac:.4f}"])
    writer.writerow(["F1-score (macro)",       f"{f1_mac:.4f}"])
    writer.writerow(["Precision (weighted)",  f"{prec_wt:.4f}"])
    writer.writerow(["Recall (weighted)",     f"{rec_wt:.4f}"])
    writer.writerow(["F1-score (weighted)",   f"{f1_wt:.4f}"])
    writer.writerow(["ROC-AUC",               f"{roc_auc:.4f}"])
    writer.writerow(["Average Precision (AP)", f"{ap:.4f}"])
    writer.writerow(["MCC",                   f"{mcc:.4f}"])
    writer.writerow(["Cohen Kappa",           f"{kappa:.4f}"])
    if tp is not None:
        writer.writerow(["TP", tp])
        writer.writerow(["TN", tn])
        writer.writerow(["FP", fp])
        writer.writerow(["FN", fn])
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        writer.writerow(["Sensitivity (Recall cancer)", f"{sens:.4f}"])
        writer.writerow(["Specificity (Recall non-cancer)", f"{spec:.4f}"])
    writer.writerow([])
    writer.writerow(["--- Per-Class ---"])
    writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])
    for cls_name, vals in report_dict.items():
        if isinstance(vals, dict):
            writer.writerow([
                cls_name,
                f"{vals['precision']:.4f}",
                f"{vals['recall']:.4f}",
                f"{vals['f1-score']:.4f}",
                int(vals['support']),
            ])

print(f"[SAVED] {csv_path}")

# ─────────────────────────────────────────────
#  SAVE prediction_details.csv
# ─────────────────────────────────────────────
detail_path = os.path.join(OUTPUT_DIR, "prediction_details.csv")
with open(detail_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image Path", "True Label", "Predicted Label",
                     "Prob_cancer", "Prob_non_cancer", "Correct"])
    for i, path in enumerate(image_paths):
        true_cls = idx_to_class[all_labels[i]]
        pred_cls = idx_to_class[all_preds[i]]
        writer.writerow([
            path, true_cls, pred_cls,
            f"{all_probs[i, 0]:.4f}",
            f"{all_probs[i, 1]:.4f}",
            "Yes" if all_labels[i] == all_preds[i] else "No",
        ])
print(f"[SAVED] {detail_path}")

# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────
PALETTE = {
    "primary":   "#2563EB",   # blue
    "secondary": "#DC2626",   # red
    "accent":    "#16A34A",   # green
    "bg":        "#F8FAFC",
    "grid":      "#E2E8F0",
    "text":      "#1E293B",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titleweight":  "bold",
})

display_names = [idx_to_class[i] for i in sorted(idx_to_class)]

# ─────────────────────────────────────────────
#  1. CONFUSION MATRIX
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Confusion Matrix — Fusion Model on ALL-IDB2",
             fontsize=14, fontweight="bold", y=1.01)

for ax, normalize, title in zip(
    axes,
    [None, "true"],
    ["Raw Counts", "Row-Normalised (Recall per class)"]
):
    cm_plot = confusion_matrix(all_labels, all_preds, normalize=normalize)
    fmt     = ".2f" if normalize else "d"
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, ax=ax,
        xticklabels=display_names, yticklabels=display_names,
        cmap="Blues", linewidths=0.5, linecolor=PALETTE["grid"],
        annot_kws={"size": 13, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"[SAVED] {cm_path}")

# ─────────────────────────────────────────────
#  2. ROC CURVE
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5,
        label=f"Fusion Model  (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color=PALETTE["grid"], lw=1.2, linestyle="--",
        label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)",        fontsize=12)
ax.set_title("ROC Curve — Cancer vs Non-Cancer\n(ALL-IDB2 blind test)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"[SAVED] {roc_path}")

# ─────────────────────────────────────────────
#  3. PRECISION–RECALL CURVE
# ─────────────────────────────────────────────
binary_labels = (all_labels == pos_idx).astype(int)
prec_curve, rec_curve, _ = precision_recall_curve(binary_labels, cancer_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_curve, prec_curve, color=PALETTE["secondary"], lw=2.5,
        label=f"Fusion Model  (AP = {ap:.4f})")
baseline = binary_labels.mean()
ax.axhline(y=baseline, color=PALETTE["grid"], lw=1.2, linestyle="--",
           label=f"Random baseline (prevalence = {baseline:.2f})")
ax.fill_between(rec_curve, prec_curve, alpha=0.08, color=PALETTE["secondary"])
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
ax.set_ylabel("Precision",             fontsize=12)
ax.set_title("Precision–Recall Curve — Cancer class\n(ALL-IDB2 blind test)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower left", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
pr_path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
plt.savefig(pr_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"[SAVED] {pr_path}")

# ─────────────────────────────────────────────
#  4. METRIC BAR CHART  (summary visual)
# ─────────────────────────────────────────────
metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)",
                 "F1\n(macro)", "F1\n(weighted)", "ROC-AUC", "AP", "MCC"]
metric_values = [acc, prec_mac, rec_mac, f1_mac, f1_wt, roc_auc, ap,
                 (mcc + 1) / 2]  # scale MCC [-1,1] → [0,1] for display
mcc_label_note = "(MCC scaled to [0,1])"

fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.bar(metric_labels, metric_values,
              color=[PALETTE["primary"]] * 7 + [PALETTE["accent"]],
              edgecolor="white", linewidth=0.8, width=0.6, zorder=3)
for bar, val, orig in zip(bars, metric_values,
                           [acc, prec_mac, rec_mac, f1_mac, f1_wt, roc_auc, ap, mcc]):
    label = f"{orig:.4f}" if orig != mcc else f"{mcc:.4f}\n{mcc_label_note}"
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012, label,
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color=PALETTE["text"])

ax.set_ylim(0, 1.18)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Performance Metrics Summary — Fusion Model on ALL-IDB2",
             fontsize=13, fontweight="bold", pad=12)
ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
ax.tick_params(axis="x", labelsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "metrics_bar_chart.png")
plt.savefig(bar_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"[SAVED] {bar_path}")

# ─────────────────────────────────────────────
#  5. PER-CLASS BAR CHART
# ─────────────────────────────────────────────
per_class_names = [k for k in report_dict if k not in
                   ("accuracy", "macro avg", "weighted avg")]
pc_prec = [report_dict[c]["precision"] for c in per_class_names]
pc_rec  = [report_dict[c]["recall"]    for c in per_class_names]
pc_f1   = [report_dict[c]["f1-score"]  for c in per_class_names]

x     = np.arange(len(per_class_names))
width = 0.25
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(x - width, pc_prec, width, label="Precision", color=PALETTE["primary"],   zorder=3)
ax.bar(x,         pc_rec,  width, label="Recall",    color=PALETTE["secondary"],  zorder=3)
ax.bar(x + width, pc_f1,   width, label="F1-score",  color=PALETTE["accent"],     zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(per_class_names, fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-Class Precision / Recall / F1\nFusion Model on ALL-IDB2",
             fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
pc_path = os.path.join(OUTPUT_DIR, "per_class_metrics.png")
plt.savefig(pc_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"[SAVED] {pc_path}")

# ─────────────────────────────────────────────
#  DONE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(f"  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
print("="*60)
print("  Files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath)
    print(f"    {fname:<40} {size:>8,} bytes")
print("="*60 + "\n")