# =============================================================================
# PHASE 2, STEP 1: U-NET LUNG SEGMENTATION & ROI ISOLATION (GOOGLE COLAB)
# Runtime: Google Colab (Python 3.10+, NVIDIA T4 GPU)
# Project: Enhanced CNN-Based Multi-Class Pneumonia Detection System
# =============================================================================

# %% [CELL 1] Environment Setup, Drive Mounting, and Data Unzipping
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 1. Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("[INFO] Not running in Google Colab environment. Skipping drive mount.")

# 2. Define Colab environment paths
DRIVE_ZIP_PATH = Path('/content/drive/MyDrive/pneumonia_detection/processed_data.zip')
LOCAL_DATA_DIR = Path('/content/data/processed')
WEIGHTS_SAVE_DIR = Path('/content/drive/MyDrive/pneumonia_detection/weights')

# Create local data and destination weights directory
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 3. Unzip dataset from Google Drive to local fast Colab NVMe storage
if DRIVE_ZIP_PATH.exists():
    print(f"Extracting dataset from '{DRIVE_ZIP_PATH}' to local NVMe '/content/'...")
    with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    print("Extraction complete!")
else:
    print(f"[INFO] Drive zip not at default '{DRIVE_ZIP_PATH}'. Searching for dataset files...")


def find_metadata_csv(search_root: Union[str, Path] = "/content") -> Path:
    """
    Intelligently discover `dataset_metadata.csv` across standard Colab and Drive paths.
    """
    search_root = Path(search_root)
    candidate_paths = [
        Path('/content/data/processed/dataset_metadata.csv'),
        Path('/content/processed/dataset_metadata.csv'),
        Path('/content/dataset_metadata.csv'),
        Path('/content/drive/MyDrive/pneumonia_detection/data/processed/dataset_metadata.csv'),
        Path('/content/drive/MyDrive/pneumonia_detection/processed/dataset_metadata.csv'),
        Path('/content/drive/MyDrive/pneumonia_detection/dataset_metadata.csv'),
        Path('data/processed/dataset_metadata.csv'),
    ]

    for path in candidate_paths:
        if path.exists():
            print(f"[INFO] Discovered metadata CSV at: {path.resolve()}")
            return path.resolve()

    # Recursive fallback search
    if search_root.exists():
        print(f"[INFO] Searching recursively for 'dataset_metadata.csv' under {search_root}...")
        matches = list(search_root.rglob("dataset_metadata.csv"))
        if matches:
            print(f"[INFO] Discovered metadata CSV at: {matches[0].resolve()}")
            return matches[0].resolve()

    raise FileNotFoundError(
        "Could not locate 'dataset_metadata.csv' anywhere in /content/ or Google Drive.\n"
        "Please verify that 'processed_data.zip' was extracted properly or upload 'dataset_metadata.csv'."
    )

# %% [CELL 2] Imports, Logging, and GPU Compute Setup
import math
import random
import logging

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Configure PyTorch CUDA Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Target Compute Device: {device}")
if device.type == 'cuda':
    print(f"NVIDIA GPU Acceleration Active: {torch.cuda.get_device_name(0)}")

# Set Seeds for Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# %% [CELL 3] Anatomical Lung Mask Generator (Pseudo-Labeler Engine)
def generate_anatomical_lung_mask(img_gray: np.ndarray) -> np.ndarray:
    """
    Automated morphological lung field mask generator.
    Converts a grayscale CXR into a binary lung parenchyma mask (0 = tissue/background, 255 = lung field).
    """
    if img_gray is None or img_gray.size == 0:
        raise ValueError("Invalid image array.")

    # 1. Blur to eliminate high-frequency noise
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 2. Otsu thresholding (inverted foreground)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Flood-fill border clearance
    h, w = thresh.shape
    border_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(thresh, border_mask, (0, 0), 0)
    cv2.floodFill(thresh, border_mask, (w - 1, 0), 0)
    cv2.floodFill(thresh, border_mask, (0, h - 1), 0)
    cv2.floodFill(thresh, border_mask, (w - 1, h - 1), 0)

    # 4. Morphological Closing & Opening
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # 5. Contour Filtering for Left and Right Lung Lobes
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_gray, dtype=np.uint8)

    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        min_lung_area = 0.03 * (h * w)
        max_lung_area = 0.45 * (h * w)

        valid_lung_contours = []
        for c in sorted_contours:
            area = cv2.contourArea(c)
            if min_lung_area <= area <= max_lung_area:
                x, y, cw, ch = cv2.boundingRect(c)
                aspect_ratio = float(ch) / cw if cw > 0 else 0
                if 0.5 <= aspect_ratio <= 3.5:
                    valid_lung_contours.append(c)
                    if len(valid_lung_contours) == 2:
                        break

        if valid_lung_contours:
            cv2.drawContours(mask, valid_lung_contours, -1, 255, -1)
        else:
            cv2.drawContours(mask, sorted_contours[:2], -1, 255, -1)

    # Morphological smoothing
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

# %% [CELL 4] PyTorch Dataset & DataLoader with Robust Path Resolution
class LungSegmentationDataset(Dataset):
    """
    Loads CLAHE-enhanced radiographs and generates binary lung masks,
    rescaling images and masks to a uniform (256, 256) resolution.
    """
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        base_dir: Path = Path('/content'),
        csv_path: Optional[Path] = None,
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.csv_dir = Path(csv_path).parent if csv_path else Path('/content')
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_image_path(self, rel_path_str: str) -> Path:
        rel_path = Path(rel_path_str)
        
        # Candidate 1: Direct join with base_dir (e.g. /content/data/processed/...)
        p1 = self.base_dir / rel_path
        if p1.exists():
            return p1
            
        # Candidate 2: Relative to directory containing CSV
        p2 = self.csv_dir / rel_path
        if p2.exists():
            return p2
            
        # Candidate 3: Search relative to csv_dir / filename
        p3 = self.csv_dir / rel_path.name
        if p3.exists():
            return p3

        # Candidate 4: Search relative to base_dir / filename
        p4 = self.base_dir / rel_path.name
        if p4.exists():
            return p4

        return p1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        path_col = "clahe_file_path" if ("clahe_file_path" in row and pd.notna(row["clahe_file_path"])) else "file_path"
        full_path = self._resolve_image_path(row[path_col])

        img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.image_size, dtype=np.uint8)
            mask = np.zeros(self.image_size, dtype=np.uint8)
        else:
            mask = generate_anatomical_lung_mask(img)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor

# %% [CELL 5] U-Net Neural Network Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """
    Standard U-Net Architecture with contracting encoder, bottleneck,
    and expanding decoder with skip connections for high-resolution segmentation.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Path
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder Path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)

# %% [CELL 6] Combined Loss (Dice + BCE) & Metric Computation
class DiceBCELoss(nn.Module):
    """Combines BCEWithLogitsLoss for pixel-wise classification & Dice Loss for region overlap."""
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_fn(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss

def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> Tuple[float, float]:
    """Calculate Dice Coefficient and IoU (Jaccard Index)."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_area = preds_flat.sum().item() + targets_flat.sum().item()
    union = total_area - intersection

    dice = (2.0 * intersection + smooth) / (total_area + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou

# %% [CELL 7] Model Initialization & Data Split Setup
METADATA_CSV_PATH = find_metadata_csv('/content')

df_metadata = pd.read_csv(METADATA_CSV_PATH)
print(f"Total Dataset Records Loaded: {len(df_metadata)}")

# 80/20 Stratified Split
train_df, val_df = train_test_split(
    df_metadata, test_size=0.20, random_state=42, stratify=df_metadata['class_id'] if 'class_id' in df_metadata.columns else None
)
print(f"Train Split: {len(train_df)} images | Val Split: {len(val_df)} images")

train_dataset = LungSegmentationDataset(train_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH)
val_dataset = LungSegmentationDataset(val_df, base_dir=Path('/content'), csv_path=METADATA_CSV_PATH)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# Instantiate Network, Criterion, Optimizer & Scheduler
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = DiceBCELoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# %% [CELL 8] Training Loop & Checkpoint Saving
EPOCHS = 10
best_val_loss = float('inf')
best_val_dice = 0.0
weights_save_path = WEIGHTS_SAVE_DIR / 'lung_unet_segmentation.pth'

print("\n" + "=" * 75)
print("                 U-NET LUNG SEGMENTATION TRAINING LOOP                 ")
print("=" * 75)
print(f" {'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Dice':^10} | {'Val IoU':^10} | {'LR':^8}")
print("-" * 75)

for epoch in range(1, EPOCHS + 1):
    # 1. Training Phase
    model.train()
    running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        dice, iou = calculate_metrics(logits, masks)
        running_loss += loss.item() * images.size(0)
        running_dice += dice * images.size(0)
        running_iou += iou * images.size(0)

    train_loss = running_loss / len(train_dataset)
    train_dice = running_dice / len(train_dataset)
    train_iou = running_iou / len(train_dataset)

    # 2. Validation Phase
    model.eval()
    val_running_loss, val_running_dice, val_running_iou = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            dice, iou = calculate_metrics(logits, masks)
            val_running_loss += loss.item() * images.size(0)
            val_running_dice += dice * images.size(0)
            val_running_iou += iou * images.size(0)

    val_loss = val_running_loss / len(val_dataset)
    val_dice = val_running_dice / len(val_dataset)
    val_iou = val_running_iou / len(val_dataset)

    curr_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)

    print(f" {epoch:^6d} | {train_loss:^10.4f} | {val_loss:^10.4f} | {val_dice:^10.4f} | {val_iou:^10.4f} | {curr_lr:^8.1e}")

    # Checkpoint Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_dice = val_dice
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
        }, weights_save_path)
        print(f"   [+] Saved Best Model Checkpoint -> {weights_save_path} (Val Dice: {val_dice:.4f})")

print("=" * 75)
print(f"Training Complete! Peak Validation Dice Coefficient: {best_val_dice:.4f}")

# %% [CELL 9] Inference & ROI Extraction Visualizer
@torch.no_grad()
def extract_lung_roi(
    model: nn.Module,
    image_path: Union[str, Path],
    device: torch.device,
    image_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Infers predicted lung mask and applies element-wise multiplication to isolate ROI.
    """
    model.eval()
    full_path = Path(image_path).resolve()
    img_orig = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise FileNotFoundError(f"Image not found: {full_path}")

    img_resized = cv2.resize(img_orig, image_size, interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    logits = model(img_tensor)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255

    lung_roi = cv2.bitwise_and(img_resized, img_resized, mask=pred_mask)

    # Plot Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized, cmap='gray')
    axes[0].set_title("Input CLAHE Radiograph")
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='bone')
    axes[1].set_title("Predicted U-Net Lung Mask")
    axes[1].axis('off')

    axes[2].imshow(lung_roi, cmap='gray')
    axes[2].set_title("Segmented Lung Parenchyma ROI")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return img_resized, pred_mask, lung_roi

# Sample Test Run
sample_img_path = val_df.iloc[0]['clahe_file_path'] if 'clahe_file_path' in val_df.columns else val_df.iloc[0]['file_path']
sample_resolved_path = val_dataset._resolve_image_path(sample_img_path)
print(f"Running Inference Test on Sample Image: {sample_resolved_path}")
_ = extract_lung_roi(model, sample_resolved_path, device=device)
