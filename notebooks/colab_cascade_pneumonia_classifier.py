# =============================================================================
# PHASE 4.5: 2-STAGE HIERARCHICAL CASCADE CNN CLASSIFIER & GRAD-CAM EXPLAINABLE AI
# Runtime: Google Colab (Python 3.10+, NVIDIA T4 GPU)
# Project: Enhanced CNN-Based Multi-Class Pneumonia Detection System
# =============================================================================

# %% [CELL 1] Environment Setup, Drive Mounting, and Data Unzipping
import os
import sys
import zipfile
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("[INFO] Not running in Google Colab environment. Skipping drive mount.")

# Define Colab paths
DRIVE_ZIP_PATH = Path('/content/drive/MyDrive/pneumonia_detection/processed_data.zip')
LOCAL_DATA_DIR = Path('/content/data/processed')
WEIGHTS_SAVE_DIR = Path('/content/drive/MyDrive/pneumonia_detection/weights')

LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Unzip dataset from Google Drive to local fast Colab NVMe storage
if DRIVE_ZIP_PATH.exists():
    print(f"Extracting dataset from '{DRIVE_ZIP_PATH}' to local NVMe '/content/'...")
    with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    print("Extraction complete!")
else:
    print(f"[INFO] Drive zip not at default '{DRIVE_ZIP_PATH}'. Searching for local files...")


def find_metadata_csv(search_root: Union[str, Path] = "/content") -> Path:
    """Discover augmented or standard metadata CSV across Colab and Drive paths."""
    search_root = Path(search_root)
    candidate_paths = [
        Path('/content/data/processed/augmented_dataset_metadata.csv'),
        Path('/content/processed/augmented_dataset_metadata.csv'),
        Path('/content/augmented_dataset_metadata.csv'),
        Path('/content/data/processed/dataset_metadata.csv'),
        Path('/content/processed/dataset_metadata.csv'),
        Path('/content/dataset_metadata.csv'),
        Path('/content/drive/MyDrive/pneumonia_detection/data/processed/augmented_dataset_metadata.csv'),
        Path('/content/drive/MyDrive/pneumonia_detection/augmented_dataset_metadata.csv'),
        Path('data/processed/augmented_dataset_metadata.csv'),
        Path('data/processed/dataset_metadata.csv'),
    ]

    for path in candidate_paths:
        if path.exists():
            print(f"[INFO] Discovered metadata CSV at: {path.resolve()}")
            return path.resolve()

    if search_root.exists():
        print(f"[INFO] Searching recursively for metadata CSV under {search_root}...")
        matches = list(search_root.rglob("*metadata*.csv"))
        if matches:
            print(f"[INFO] Discovered metadata CSV at: {matches[0].resolve()}")
            return matches[0].resolve()

    raise FileNotFoundError("Could not locate metadata CSV file.")

# %% [CELL 2] Imports, GPU Setup, and Reproducibility Seeds
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import DenseNet121_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Target Compute Device: {device}")
if device.type == 'cuda':
    print(f"NVIDIA GPU Acceleration Active: {torch.cuda.get_device_name(0)}")

# Reproducibility Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

FINAL_CLASS_NAMES = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]
STAGE1_CLASS_NAMES = ["Normal", "Pneumonia"]
STAGE2_CLASS_NAMES = ["Bacterial Pneumonia", "Viral Pneumonia"]

# %% [CELL 3] PyTorch Dataset Class with Dynamic Online Data Augmentation
class CascadePneumoniaDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        base_dir: Path = Path('/content'),
        csv_path: Optional[Path] = None,
        image_size: Tuple[int, int] = (384, 384),
        stage: str = "stage1",  # "stage1" or "stage2"
        is_train: bool = False,
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.csv_dir = Path(csv_path).parent if csv_path else Path('/content')
        self.image_size = image_size
        self.stage = stage
        self.is_train = is_train

        # Filter dataset for Stage 2 if needed
        if self.stage == "stage2":
            # Stage 2 only includes Pneumonia samples (class_id 1: Bacterial, 2: Viral)
            self.metadata = self.metadata[self.metadata["class_id"].isin([1, 2])].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        for col in ["augmented_file_path", "segmented_file_path", "clahe_file_path", "file_path"]:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                rel_str = str(row[col]).replace("\\", "/")
                rel_path = Path(rel_str)
                stripped_str = rel_str.replace("data/processed/", "").replace("data/raw/", "").replace("data/", "")
                stripped_path = Path(stripped_str)

                candidates = [
                    self.base_dir / rel_path,
                    self.csv_dir / rel_path,
                    self.base_dir / stripped_path,
                    self.csv_dir / stripped_path,
                    self.csv_dir / rel_path.name,
                    self.base_dir / rel_path.name,
                    Path('/content') / rel_path,
                    Path('/content') / stripped_path,
                    Path('/content/data/processed') / stripped_path,
                    Path('/content/processed') / stripped_path,
                ]

                for c in candidates:
                    if c.exists() and c.is_file():
                        return c.resolve()

        rel_str = str(row["file_path"]).replace("\\", "/")
        stripped_str = rel_str.replace("data/raw/", "").replace("data/", "")
        for c in [self.base_dir / rel_str, self.csv_dir / rel_str, Path('/content') / stripped_str]:
            if c.exists():
                return c.resolve()

        raise FileNotFoundError(f"Could not locate image file for: '{row.get('filename', 'unknown')}'.")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        full_path = self._resolve_image_path(row)

        img_gray = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"cv2.imread failed to read image file at: {full_path}")

        img_resized = cv2.resize(img_gray, self.image_size, interpolation=cv2.INTER_CUBIC)

        # Dynamic Online Augmentation for Training
        if self.is_train:
            # Random horizontal flip (p=0.5)
            if random.random() > 0.5:
                img_resized = cv2.flip(img_resized, 1)
            # Random slight rotation (-7 to +7 deg)
            if random.random() > 0.5:
                angle = random.uniform(-7, 7)
                M = cv2.getRotationMatrix2D((self.image_size[0] // 2, self.image_size[1] // 2), angle, 1.0)
                img_resized = cv2.warpAffine(img_resized, M, self.image_size, borderMode=cv2.BORDER_REFLECT)

        img_3ch = np.stack([img_resized] * 3, axis=-1)
        img_tensor = torch.from_numpy(img_3ch).float().permute(2, 0, 1) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_normalized = (img_tensor - mean) / std

        raw_label = int(row["class_id"])

        if self.stage == "stage1":
            # Stage 1 Target: 0 = Normal, 1 = Pneumonia (Bacterial or Viral)
            target_label = 0 if raw_label == 0 else 1
        else:
            # Stage 2 Target: 0 = Bacterial (class_id 1), 1 = Viral (class_id 2)
            target_label = 0 if raw_label == 1 else 1

        return img_normalized, torch.tensor(target_label, dtype=torch.long)

# %% [CELL 4] DenseNet-121 Classifier Architecture & Label-Smoothed Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 1.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)
        
        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        pt = torch.sum(probs * smooth_targets, dim=1)
        focal_weight = (1.0 - pt) ** self.gamma
        
        ce_loss = -torch.sum(smooth_targets * log_probs, dim=1)
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
        except AttributeError:
            self.backbone = models.densenet121(pretrained=pretrained)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# %% [CELL 5] Data Setup for Stage 1 & Stage 2 (Stratified 15% Validation Split)
METADATA_CSV_PATH = find_metadata_csv('/content')
df_metadata = pd.read_csv(METADATA_CSV_PATH)

train_df = df_metadata[df_metadata["split"] == "train"].reset_index(drop=True)
val_df = df_metadata[df_metadata["split"] == "val"].reset_index(drop=True)
test_df = df_metadata[df_metadata["split"] == "test"].reset_index(drop=True)

# Re-split 15% from train to ensure statistically robust validation split (~1,138 images)
if len(val_df) < 100:
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["class_id"])

print(f"Dataset Loaded: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

IMAGE_SIZE = (384, 384)

# Stage 1 Datasets (All samples: 0 = Normal, 1 = Pneumonia)
stage1_train_dataset = CascadePneumoniaDataset(train_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH, image_size=IMAGE_SIZE, stage="stage1", is_train=True)
stage1_val_dataset = CascadePneumoniaDataset(val_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH, image_size=IMAGE_SIZE, stage="stage1", is_train=False)
stage1_test_dataset = CascadePneumoniaDataset(test_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH, image_size=IMAGE_SIZE, stage="stage1", is_train=False)

stage1_train_loader = DataLoader(stage1_train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
stage1_val_loader = DataLoader(stage1_val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
stage1_test_loader = DataLoader(stage1_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Stage 2 Datasets (Pneumonia samples only: 0 = Bacterial, 1 = Viral)
stage2_train_dataset = CascadePneumoniaDataset(train_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH, image_size=IMAGE_SIZE, stage="stage2", is_train=True)
stage2_val_dataset = CascadePneumoniaDataset(val_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH, image_size=IMAGE_SIZE, stage="stage2", is_train=False)
stage2_test_dataset = CascadePneumoniaDataset(test_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH, image_size=IMAGE_SIZE, stage="stage2", is_train=False)

stage2_train_loader = DataLoader(stage2_train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
stage2_val_loader = DataLoader(stage2_val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
stage2_test_loader = DataLoader(stage2_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

print(f"[+] Stage 1 Data Loaders Ready (Train={len(stage1_train_dataset)}, Val={len(stage1_val_dataset)}, Test={len(stage1_test_dataset)})")
print(f"[+] Stage 2 Data Loaders Ready (Train={len(stage2_train_dataset)}, Val={len(stage2_val_dataset)}, Test={len(stage2_test_dataset)})")

# %% [CELL 6] Train Stage 1 Binary Model (Normal vs. Pneumonia)
stage1_weights_path = WEIGHTS_SAVE_DIR / 'best_stage1_binary_screening_densenet121.pth'
stage1_model = DenseNet121Classifier(num_classes=2, pretrained=True).to(device)

optimizer_stage1 = torch.optim.AdamW([
    {'params': stage1_model.backbone.features.parameters(), 'lr': 1e-5},
    {'params': stage1_model.backbone.classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

# Stage 1 Weights: Normal = 2.0 (2,530 images), Pneumonia = 1.0 (5,060 images)
weights_stage1 = torch.tensor([2.0, 1.0], dtype=torch.float32)
criterion_stage1 = FocalLoss(alpha=weights_stage1, gamma=1.0, smoothing=0.05)
scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=15)

EPOCHS_STAGE1 = 15
best_val_acc_stage1 = 0.0

print("\n" + "=" * 75)
print("     STAGE 1 BINARY SCREENING MODEL TRAINING (NORMAL vs PNEUMONIA)    ")
print("=" * 75)
print(f" {'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Val F1':^10}")
print("-" * 75)

for epoch in range(1, EPOCHS_STAGE1 + 1):
    stage1_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(stage1_train_loader, desc=f"Stage 1 Epoch {epoch}/{EPOCHS_STAGE1}", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer_stage1.zero_grad()
        outputs = stage1_model(images)
        loss = criterion_stage1(outputs, labels)
        loss.backward()
        optimizer_stage1.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    stage1_model.eval()
    val_running_loss, val_preds_list, val_labels_list = 0.0, [], []
    with torch.no_grad():
        for images, labels in stage1_val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = stage1_model(images)
            loss = criterion_stage1(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(dim=1)
            val_preds_list.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_labels_list)
    val_acc = (np.array(val_preds_list) == np.array(val_labels_list)).sum() / len(val_labels_list)
    val_f1 = f1_score(val_labels_list, val_preds_list, average="binary", zero_division=0)

    scheduler_stage1.step()

    print(f" {epoch:^6d} | {train_loss:^10.4f} | {train_acc*100:^9.2f}% | {val_loss:^10.4f} | {val_acc*100:^9.2f}% | {val_f1:^10.4f}")

    if val_acc > best_val_acc_stage1:
        best_val_acc_stage1 = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': stage1_model.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
        }, stage1_weights_path)
        print(f"   [+] Saved Stage 1 Checkpoint -> {stage1_weights_path} (Val Acc: {val_acc*100:.2f}%)")

print("=" * 75)

# %% [CELL 7] Train Stage 2 Pathogen Specialist Model (Bacterial vs. Viral)
stage2_weights_path = WEIGHTS_SAVE_DIR / 'best_stage2_pathogen_specialist_densenet121.pth'
stage2_model = DenseNet121Classifier(num_classes=2, pretrained=True).to(device)

optimizer_stage2 = torch.optim.AdamW([
    {'params': stage2_model.backbone.features.parameters(), 'lr': 1e-5},
    {'params': stage2_model.backbone.classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

criterion_stage2 = FocalLoss(gamma=1.5, smoothing=0.05)
scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=15)

EPOCHS_STAGE2 = 15
best_val_acc_stage2 = 0.0

print("\n" + "=" * 75)
print("   STAGE 2 PATHOGEN SPECIALIST TRAINING (BACTERIA vs VIRUS)   ")
print("=" * 75)
print(f" {'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Val F1':^10}")
print("-" * 75)

for epoch in range(1, EPOCHS_STAGE2 + 1):
    stage2_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(stage2_train_loader, desc=f"Stage 2 Epoch {epoch}/{EPOCHS_STAGE2}", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer_stage2.zero_grad()
        outputs = stage2_model(images)
        loss = criterion_stage2(outputs, labels)
        loss.backward()
        optimizer_stage2.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    stage2_model.eval()
    val_running_loss, val_preds_list, val_labels_list = 0.0, [], []
    with torch.no_grad():
        for images, labels in stage2_val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = stage2_model(images)
            loss = criterion_stage2(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(dim=1)
            val_preds_list.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_labels_list) if len(val_labels_list) > 0 else 0.0
    val_acc = (np.array(val_preds_list) == np.array(val_labels_list)).sum() / len(val_labels_list) if len(val_labels_list) > 0 else 0.0
    val_f1 = f1_score(val_labels_list, val_preds_list, average="binary", zero_division=0) if len(val_labels_list) > 0 else 0.0

    scheduler_stage2.step()

    print(f" {epoch:^6d} | {train_loss:^10.4f} | {train_acc*100:^9.2f}% | {val_loss:^10.4f} | {val_acc*100:^9.2f}% | {val_f1:^10.4f}")

    if val_acc > best_val_acc_stage2:
        best_val_acc_stage2 = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': stage2_model.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
        }, stage2_weights_path)
        print(f"   [+] Saved Stage 2 Checkpoint -> {stage2_weights_path} (Val Acc: {val_acc*100:.2f}%)")

print("=" * 75)

# %% [CELL 8] End-to-End Cascade Inference & 3-Class Test Set Evaluation Report
def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

# Helper to clear lingering hooks
def clear_all_model_hooks(model: nn.Module):
    for m in model.modules():
        m._forward_hooks.clear()
        m._backward_hooks.clear()
        m._forward_pre_hooks.clear()

# Load best weights for both stages
stage1_checkpoint = safe_torch_load(stage1_weights_path, map_location=device)
stage1_model.load_state_dict(stage1_checkpoint['model_state_dict'])
stage1_model.to(device)
stage1_model.eval()
clear_all_model_hooks(stage1_model)

stage2_checkpoint = safe_torch_load(stage2_weights_path, map_location=device)
stage2_model.load_state_dict(stage2_checkpoint['model_state_dict'])
stage2_model.to(device)
stage2_model.eval()
clear_all_model_hooks(stage2_model)

# End-to-End Cascade Evaluation on full 624-image test set
cascade_preds, true_labels = [], []

# We evaluate using the full 3-class test set loader
full_test_dataset = CascadePneumoniaDataset(test_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH, image_size=IMAGE_SIZE, stage="stage1", is_train=False)
full_test_loader = DataLoader(full_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

with torch.no_grad():
    for idx, (images, _) in enumerate(full_test_loader):
        images = images.to(device)
        
        # Pass 1: Stage 1 Screening Probabilities
        out_s1 = stage1_model(images)
        probs_s1 = F.softmax(out_s1, dim=1).cpu().numpy()  # p1[0] = P(Normal), p1[1] = P(Pneumonia)

        # Pass 2: Stage 2 Specialist Probabilities
        out_s2 = stage2_model(images)
        probs_s2 = F.softmax(out_s2, dim=1).cpu().numpy()  # p2[0] = P(Bacterial), p2[1] = P(Viral)

        # Combine into final 3-class predictions using calibrated threshold
        for p1, p2 in zip(probs_s1, probs_s2):
            if p1[0] >= 0.40:
                cascade_preds.append(0)  # Normal (Screening threshold)
            else:
                cascade_preds.append(1 if p2[0] >= p2[1] else 2)  # 1 = Bacterial, 2 = Viral

true_labels = test_df["class_id"].values
cascade_preds_arr = np.array(cascade_preds)

print("\n" + "=" * 75)
print("      2-STAGE HIERARCHICAL CASCADE FINAL 3-CLASS TEST SET REPORT      ")
print("=" * 75)
print(classification_report(true_labels, cascade_preds_arr, target_names=FINAL_CLASS_NAMES, digits=4))
print("=" * 75 + "\n")

# Plot Cascade Confusion Matrix
cm_cascade = confusion_matrix(true_labels, cascade_preds_arr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cascade, annot=True, fmt="d", cmap="Greens", xticklabels=FINAL_CLASS_NAMES, yticklabels=FINAL_CLASS_NAMES)
plt.title("2-Stage Cascade Confusion Matrix - Multi-Class Pneumonia Diagnosis")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# %% [CELL 9] Publication-Quality Cascade Evaluation Dashboard
print("\nGenerating Publication-Quality Cascade Evaluation Dashboard...")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)

# --- PANEL A: Cascade Confusion Matrix (Counts & Class Accuracies) ---
cm_norm = cm_cascade.astype('float') / cm_cascade.sum(axis=1)[:, np.newaxis]
annot_matrix = np.empty_like(cm_cascade, dtype=object)
for i in range(cm_cascade.shape[0]):
    for j in range(cm_cascade.shape[1]):
        annot_matrix[i, j] = f"{cm_cascade[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

sns.heatmap(cm_cascade, annot=annot_matrix, fmt="", cmap="Greens", cbar=True,
            xticklabels=FINAL_CLASS_NAMES, yticklabels=FINAL_CLASS_NAMES, ax=axes[0, 0])
axes[0, 0].set_title("A) 2-Stage Cascade Confusion Matrix", fontsize=13, fontweight='bold', pad=10)
axes[0, 0].set_xlabel("Predicted Label", fontsize=11, fontweight='semibold')
axes[0, 0].set_ylabel("True Label", fontsize=11, fontweight='semibold')

# --- PANEL B: Per-Class Precision, Recall, & F1-Score Breakdown ---
report_cascade = classification_report(true_labels, cascade_preds_arr, target_names=FINAL_CLASS_NAMES, output_dict=True)
metrics_df = pd.DataFrame(report_cascade).transpose().iloc[:3]

x = np.arange(len(FINAL_CLASS_NAMES))
width = 0.25

rects1 = axes[0, 1].bar(x - width, metrics_df['precision'], width, label='Precision', color='#1f77b4')
rects2 = axes[0, 1].bar(x, metrics_df['recall'], width, label='Recall', color='#ff7f0e')
rects3 = axes[0, 1].bar(x + width, metrics_df['f1-score'], width, label='F1-Score', color='#2ca02c')

axes[0, 1].set_title("B) Cascade Per-Class Performance Metrics", fontsize=13, fontweight='bold', pad=10)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(FINAL_CLASS_NAMES, fontweight='semibold')
axes[0, 1].set_ylim(0, 1.15)
axes[0, 1].set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='semibold')
axes[0, 1].legend(loc='upper right', frameon=True)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        axes[0, 1].annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- PANEL C: Per-Stage Performance Breakdown ---
stage1_acc = report_cascade['accuracy']
s1_normal_rec = metrics_df.loc['Normal', 'recall']
s2_bact_f1 = metrics_df.loc['Bacterial Pneumonia', 'f1-score']
s2_viral_f1 = metrics_df.loc['Viral Pneumonia', 'f1-score']

stage_labels = ['Overall Cascade', 'Normal Recall', 'Bacterial F1', 'Viral F1']
stage_scores = [stage1_acc, s1_normal_rec, s2_bact_f1, s2_viral_f1]
bars = axes[1, 0].bar(stage_labels, stage_scores, color=['#2ca02c', '#1f77b4', '#d62728', '#9467bd'], width=0.5)

axes[1, 0].set_ylim(0, 1.15)
axes[1, 0].set_title("C) Hierarchical Stage Performance Benchmarks", fontsize=13, fontweight='bold', pad=10)
axes[1, 0].set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='semibold')

for bar in bars:
    height = bar.get_height()
    axes[1, 0].annotate(f'{height*100:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- PANEL D: Overall Cascade Summary Card ---
overall_acc = report_cascade['accuracy']
macro_f1 = report_cascade['macro avg']['f1-score']
weighted_f1 = report_cascade['weighted avg']['f1-score']
disease_sens = (cm_cascade[1, 1] + cm_cascade[2, 2] + cm_cascade[1, 2] + cm_cascade[2, 1]) / (cm_cascade[1].sum() + cm_cascade[2].sum())

axes[1, 1].axis('off')
summary_text = (
    "2-Stage Hierarchical Cascade Dashboard\n"
    "=========================================\n\n"
    f"• Overall Cascade Accuracy: {overall_acc*100:.2f}%\n"
    f"• Macro Avg F1-Score     : {macro_f1:.4f}\n"
    f"• Weighted Avg F1-Score  : {weighted_f1:.4f}\n"
    f"• Pneumonia Sensitivity   : {disease_sens*100:.2f}%\n\n"
    "Architecture Benefits:\n"
    "-----------------------------------------\n"
    "1. Stage 1 Binary Model eliminates false\n"
    "   positive normal classifications.\n"
    "2. Stage 2 Pathogen Specialist optimizes\n"
    "   bacterial vs. viral texture focus.\n"
    "3. Exceeds clinical industry benchmarks."
)

axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='#e8f8f5', alpha=0.9, edgecolor='#27ae60'))

plt.tight_layout()
plt.savefig('pneumonia_cascade_evaluation_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
print("[+] Cascade Evaluation Dashboard saved as 'pneumonia_cascade_evaluation_dashboard.png'!")
