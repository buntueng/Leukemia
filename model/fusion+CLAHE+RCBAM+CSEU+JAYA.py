"""
Fusion (MobileNetV2 + ResNet50) — Leukemia Cell Classification
Dataset : CNMC-2019
Protocol: 5-Fold Stratified Cross-Validation
Strategy: Pre-trained backbones + newly trained fusion head + CLAHE Preprocessing + RCBAM + CSEU + Enhanced-JAYA.
Outputs :
  OUTPUT_DIR/
    fold_{k}/
      best_model.pth          <- best val-AUC checkpoint per fold
      training_history.csv    <- epoch-level loss/acc/auc per fold
    all_folds_metrics.csv     <- per-fold + mean±std summary
    training_history_all.csv  <- all folds concatenated (for plotting)
    confusion_matrices.png    <- 2x3 grid of confusion matrices
    fusion_mobilenet_resnet_full_proposed_best_model.pth <- overall best model from all folds
"""

import os
import random
import time
import shutil  # <-- Added to handle copying the best model
import warnings
warnings.filterwarnings("ignore")

import cv2
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
    confusion_matrix, matthews_corrcoef, cohen_kappa_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torchvision.models import resnet50

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  —  edit only this section
# ─────────────────────────────────────────────────────────────
DATA_DIR   = '/home/eecommu06/Documents/BT/CNMC2019/image'
OUTPUT_DIR = '/home/eecommu06/Desktop/Bee/ALL/output/fusion_mobilenet_resnet_full_proposed'

# Paths to your previously trained individual models
RESNET_WEIGHTS_PATH    = '/home/eecommu06/Desktop/Bee/ALL/output/resnet50/resnet50_best_model.pth'
MOBILENET_WEIGHTS_PATH = '/home/eecommu06/Desktop/Bee/ALL/output/mobilenetv2/mobilenetv2_best_model.pth'

NUM_FOLDS    = 5
NUM_EPOCHS   = 50
BATCH_SIZE   = 32
LR           = 1e-4        # Standard LR for the head and attention modules
WEIGHT_DECAY = 1e-4
PATIENCE     = 10        
NUM_WORKERS  = 32         
SEED         = 42
IMG_SIZE     = 224       
NUM_CLASSES  = 2         
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# CLAHE PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        np_img = np.array(img)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l_channel)

        merged = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced_img)

# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
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
    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class LeukemiaDataset(Dataset):
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
# ATTENTION & OPTIMIZATION MODULES (RCBAM, CSEU, EJAYA)
# ─────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class RCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(RCBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return x + out  # Residual connection

class CSEU(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CSEU, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class EnhancedJAYA(nn.Module):
    """
    Enhanced JAYA-inspired Feature Refinement Layer.
    Translates the gradient-free JAYA optimization equation into a 
    differentiable layer to refine feature maps before classification.
    """
    def __init__(self, in_channels):
        super(EnhancedJAYA, self).__init__()
        # Trainable scaling parameters replacing the random scalars in JAYA
        self.r1 = nn.Parameter(torch.rand(1, in_channels, 1, 1))
        self.r2 = nn.Parameter(torch.rand(1, in_channels, 1, 1))
        self.r3 = nn.Parameter(torch.rand(1, in_channels, 1, 1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Flatten spatial dimensions to compute population stats per channel
        x_flat = x.view(B, C, -1)
        
        # Identify Best (max), Worst (min), and Mean "solutions" (features)
        best_x = torch.max(x_flat, dim=2)[0].view(B, C, 1, 1)
        worst_x = torch.min(x_flat, dim=2)[0].view(B, C, 1, 1)
        mean_x = torch.mean(x_flat, dim=2).view(B, C, 1, 1)
        
        abs_x = torch.abs(x)
        
        # Enhanced JAYA mathematical update
        # X_new = X + r1*(Best - |X|) - r2*(Worst - |X|) + r3*(Mean - |X|)
        x_new = x + self.r1 * (best_x - abs_x) - self.r2 * (worst_x - abs_x) + self.r3 * (mean_x - abs_x)
        
        return x_new

# ─────────────────────────────────────────────────────────────
# MODEL FUSION
# ─────────────────────────────────────────────────────────────
class FeatureFusionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(FeatureFusionModel, self).__init__()
        
        # --- 1. MobileNetV2 Backbone ---
        mobilenet = mobilenet_v2(weights=None)
        mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
        
        if os.path.exists(MOBILENET_WEIGHTS_PATH):
            ckpt = torch.load(MOBILENET_WEIGHTS_PATH, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
            mobilenet.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Successfully loaded custom MobileNetV2 weights.")
        else:
            print(f"[WARNING] Custom MobileNetV2 weights NOT found!")
            
        self.mobilenet_features = mobilenet.features
        
        # Apply Modules to MobileNetV2 spatial features
        self.mobilenet_rcbam = RCBAM(in_planes=1280)
        self.mobilenet_cseu = CSEU(in_channels=1280)
        self.mobilenet_ejaya = EnhancedJAYA(in_channels=1280)
        self.mobilenet_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- 2. ResNet50 Backbone ---
        resnet = resnet50(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
        if os.path.exists(RESNET_WEIGHTS_PATH):
            ckpt = torch.load(RESNET_WEIGHTS_PATH, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
            resnet.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Successfully loaded custom ResNet50 weights.")
        else:
            print(f"[WARNING] Custom ResNet50 weights NOT found!")
            
        self.resnet_features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        
        # Apply Modules to ResNet50 spatial features
        self.resnet_rcbam = RCBAM(in_planes=2048)
        self.resnet_cseu = CSEU(in_channels=2048)
        self.resnet_ejaya = EnhancedJAYA(in_channels=2048)
        self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1))
            
        # --- 3. Unified Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # MobileNetV2 Path 
        x1 = self.mobilenet_features(x)
        x1 = self.mobilenet_rcbam(x1)
        x1 = self.mobilenet_cseu(x1)
        x1 = self.mobilenet_ejaya(x1)
        x1 = self.mobilenet_pool(x1)
        x1 = torch.flatten(x1, 1)
        
        # ResNet50 Path 
        x2 = self.resnet_features(x)
        x2 = self.resnet_rcbam(x2)
        x2 = self.resnet_cseu(x2)
        x2 = self.resnet_ejaya(x2)
        x2 = self.resnet_pool(x2)
        x2 = torch.flatten(x2, 1)
        
        x_fused = torch.cat((x1, x2), dim=1)
        out = self.classifier(x_fused)
        return out

def build_model(num_classes=NUM_CLASSES):
    return FeatureFusionModel(num_classes=num_classes)

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
# TRAIN ONE EPOCH
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

        pbar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.4f}")

    pbar.close()
    return total_loss / n, correct / n

# ─────────────────────────────────────────────────────────────
# VALIDATE / EVALUATE
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
# TRAIN ONE FOLD
# ─────────────────────────────────────────────────────────────
def train_fold(fold, train_idx, val_idx, full_dataset, output_fold_dir):
    print(f"\n{'='*64}")
    print(f"  FOLD {fold+1}/{NUM_FOLDS}  |  "
          f"train={len(train_idx)}  val={len(val_idx)}")
    print(f"{'='*64}")
    os.makedirs(output_fold_dir, exist_ok=True)

    train_ds = Subset(_clone_dataset_with_transform(full_dataset, train_transform), train_idx)
    val_ds   = Subset(_clone_dataset_with_transform(full_dataset, val_transform),   val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model    = build_model().to(DEVICE)
    labels_tr = [full_dataset.samples[i][1] for i in train_idx]
    n_neg     = labels_tr.count(0)
    n_pos     = labels_tr.count(1)
    pos_w     = n_neg / (n_pos + 1e-9)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_w], device=DEVICE)
    )

    base_params = list(model.mobilenet_features.parameters()) + list(model.resnet_features.parameters())
    
    # Register all new attention and JAYA parameters to the normal learning rate
    head_params = list(model.classifier.parameters()) + \
                  list(model.mobilenet_rcbam.parameters()) + \
                  list(model.resnet_rcbam.parameters()) + \
                  list(model.mobilenet_cseu.parameters()) + \
                  list(model.resnet_cseu.parameters()) + \
                  list(model.mobilenet_ejaya.parameters()) + \
                  list(model.resnet_ejaya.parameters())
    
    optimizer = optim.AdamW([
        {"params": base_params, "lr": LR * 0.1},
        {"params": head_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history_rows   = []
    best_auc       = 0.0
    patience_count = 0

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
        lr_head = optimizer.param_groups[1]["lr"]   

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
            "lr_head"     : round(lr_head, 8),
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

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(os.path.join(output_fold_dir, "training_history.csv"), index=False)

    ckpt = torch.load(os.path.join(output_fold_dir, "best_model.pth"), map_location=DEVICE, weights_only=False)
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
    print(f"  Fusion (MobileNetV2 + ResNet50) — Leukemia Classification")
    print(f"  Ablation Step: + CLAHE Preprocessing + RCBAM + CSEU + Enhanced-JAYA")
    print(f"{'='*64}")
    print(f"  Device     : {DEVICE}")
    print(f"  DATA_DIR   : {DATA_DIR}")
    print(f"  OUTPUT_DIR : {OUTPUT_DIR}")
    print(f"  Folds      : {NUM_FOLDS}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR (head)  : {LR}")
    print(f"  LR (base)  : {LR * 0.1}")
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
    print(f"\n[Saved] All-folds metrics        → {metrics_path}")

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
        overall_best_path = os.path.join(OUTPUT_DIR, "fusion_mobilenet_resnet_full_proposed_best_model.pth")
        shutil.copy2(global_best_model_src, overall_best_path)
        print(f"[Saved] Overall best model (AUC: {global_best_auc:.4f}) → {overall_best_path}")

    # ── final summary ─────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  FINAL 5-FOLD SUMMARY — Fusion (MobileNetV2 + ResNet50 + CLAHE + RCBAM + CSEU + Enhanced-JAYA)")
    print(f"{'='*64}")
    key_metrics = ["accuracy", "f1", "auc", "sensitivity",
                   "specificity", "precision", "npv", "mcc", "kappa"]
    for m in key_metrics:
        if m in metrics_df.columns:
            vals = metrics_df[m].values
            print(f"  {m:<14}: {vals.mean():.4f} ± {vals.std():.4f}"
                  f"  [{vals.min():.4f} – {vals.max():.4f}]")
    print(f"{'='*64}")
    print(f"\n[Done] All outputs saved to {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()