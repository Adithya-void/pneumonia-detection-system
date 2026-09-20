"""
Phase 2, Step 1: U-Net Semantic Segmentation Model for Anatomical Lung ROI Isolation.

This module provides a complete PyTorch workflow for training a U-Net model on chest
X-ray radiographs (CLAHE enhanced) to segment the lung parenchyma and extract the region of
interest (ROI), removing non-diagnostic background, ribs, clavicles, and mediastinal tissue.

Designed for local execution and Google Colab (NVIDIA T4 GPU environment).

Author: Senior AI/ML Engineer (Medical Image Analysis)
Project: Enhanced CNN-Based Multi-Class Pneumonia Detection System
"""

import os
import sys
import math
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. AUTOMATED ANATOMICAL LUNG MASK GENERATOR (PSEUDO-LABELER)
# =============================================================================

def generate_anatomical_lung_mask(img_gray: np.ndarray) -> np.ndarray:
    """
    Generate a clean binary lung mask from a grayscale chest X-ray using adaptive
    Otsu thresholding, morphological closing/opening, and contour filtering.

    Args:
        img_gray (np.ndarray): 2D uint8 grayscale chest radiograph array.

    Returns:
        np.ndarray: 2D uint8 binary mask array (0 = background/tissue, 255 = lung parenchyma).
    """
    if img_gray is None or img_gray.size == 0:
        raise ValueError("Invalid or empty image array passed to mask generator.")

    # 1. Normalize image contrast & blur to smooth high-frequency noise
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 2. Otsu thresholding (invert so dark lung regions become foreground)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Clear outer borders (remove collar bone / background edges touching borders)
    h, w = thresh.shape
    border_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(thresh, border_mask, (0, 0), 0)
    cv2.floodFill(thresh, border_mask, (w - 1, 0), 0)
    cv2.floodFill(thresh, border_mask, (0, h - 1), 0)
    cv2.floodFill(thresh, border_mask, (w - 1, h - 1), 0)

    # 4. Morphological Closing to fill interior pulmonary vessels and ribs
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # 5. Morphological Opening to break thin bridges between left and right lungs
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # 6. Find contours & select two largest valid lung lobes based on area and aspect ratio
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_gray, dtype=np.uint8)

    if not contours:
        # Fallback: simple center-crop mask if contour detection fails
        cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 255, -1)
        return mask

    # Sort contours by area in descending order
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

    # Draw selected lung contours or top 2 contours
    if valid_lung_contours:
        cv2.drawContours(mask, valid_lung_contours, -1, 255, -1)
    else:
        # Fallback to top 2 contours if area constraint too strict
        cv2.drawContours(mask, sorted_contours[:2], -1, 255, -1)

    # Final morphological smoothing
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


# =============================================================================
# 2. PYTORCH DATASET & DATALOADER
# =============================================================================

class LungSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Lung ROI Segmentation.

    Loads CLAHE-enhanced radiographs and generates/loads binary lung masks,
    rescaling images and masks to a uniform (256, 256) resolution.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        project_root: Path,
        image_size: Tuple[int, int] = (256, 256),
        use_clahe_path: bool = True,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing image metadata.
            project_root (Path): Base directory path for resolving relative file paths.
            image_size (Tuple[int, int]): Target image dimensions (width, height).
            use_clahe_path (bool): If True, uses 'clahe_file_path'; else uses 'file_path'.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.project_root = Path(project_root)
        self.image_size = image_size
        self.use_clahe_path = use_clahe_path

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        
        path_col = "clahe_file_path" if (self.use_clahe_path and "clahe_file_path" in row and pd.notna(row["clahe_file_path"])) else "file_path"
        rel_path = Path(row[path_col])
        full_path = self.project_root / rel_path if not rel_path.is_absolute() else rel_path

        # Read image in grayscale
        img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback zero tensor if file missing/corrupted
            logger.warning(f"Failed to read image at: {full_path}. Returning blank tensor.")
            img = np.zeros(self.image_size, dtype=np.uint8)
            mask = np.zeros(self.image_size, dtype=np.uint8)
        else:
            # Generate lung mask automatically
            mask = generate_anatomical_lung_mask(img)

            # Resize image and mask to target dimensions
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0.0, 1.0] and mask to binary {0.0, 1.0}
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor


# =============================================================================
# 3. U-NET ARCHITECTURE (PURE PYTORCH IMPLEMENTATION)
# =============================================================================

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""

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
    Standard U-Net Architecture for Binary Medical Image Segmentation.

    Input: (B, 1, H, W) grayscale radiograph tensor.
    Output: (B, 1, H, W) raw logits tensor (apply torch.sigmoid for probabilities).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Contracting Path)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Expanding Path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 Convolution layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder alignment
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Handle shape mismatch due to odd dimensions if any
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


# =============================================================================
# 4. LOSS FUNCTIONS & SEGMENTATION METRICS
# =============================================================================

class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss for Segmentation.

    BCE handles per-pixel classification while Dice Loss optimizes spatial overlap.
    """

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
    """
    Calculate Dice Coefficient and Intersection over Union (IoU / Jaccard Score).

    Args:
        logits (torch.Tensor): Model raw output logits (B, 1, H, W).
        targets (torch.Tensor): Ground truth binary mask (B, 1, H, W).
        threshold (float): Binarization probability threshold.
        smooth (float): Smoothing factor for numerical stability.

    Returns:
        Tuple[float, float]: (Dice Coefficient, IoU Score)
    """
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


# =============================================================================
# 5. TRAINING ENGINE & CHECKPOINTING
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train U-Net for one single epoch."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    pbar = tqdm(dataloader, desc="Training Batch", leave=False, unit="batch")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        dice, iou = calculate_metrics(logits, masks)

        running_loss += loss.item() * images.size(0)
        running_dice += dice * images.size(0)
        running_iou += iou * images.size(0)

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})

    total_samples = len(dataloader.dataset)
    return running_loss / total_samples, running_dice / total_samples, running_iou / total_samples


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validate U-Net for one single epoch."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        dice, iou = calculate_metrics(logits, masks)

        running_loss += loss.item() * images.size(0)
        running_dice += dice * images.size(0)
        running_iou += iou * images.size(0)

    total_samples = len(dataloader.dataset)
    return running_loss / total_samples, running_dice / total_samples, running_iou / total_samples


def train_unet_model(
    metadata_csv_path: Union[str, Path],
    weights_save_path: Union[str, Path],
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: Tuple[int, int] = (256, 256),
    device_str: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> nn.Module:
    """
    End-to-end U-Net segmentation training pipeline.

    Args:
        metadata_csv_path (Union[str, Path]): Path to dataset_metadata.csv.
        weights_save_path (Union[str, Path]): Destination path for model weights (.pth).
        epochs (int): Number of training epochs.
        batch_size (int): DataLoader batch size.
        lr (float): Initial learning rate for Adam optimizer.
        image_size (Tuple[int, int]): Input image resolution (width, height).
        device_str (Optional[str]): 'cuda' or 'cpu'. Auto-detects if None.
        project_root (Optional[Path]): Workspace root directory.

    Returns:
        nn.Module: Trained U-Net model.
    """
    csv_path = Path(metadata_csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at: {csv_path}")

    save_path = Path(weights_save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    # Device configuration
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Target Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load metadata and perform 80/20 train/val split
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["class_id"] if "class_id" in df.columns else None
    )

    logger.info(f"Dataset Split: Train={len(train_df)} images, Val={len(val_df)} images")

    # Datasets and DataLoaders
    train_dataset = LungSegmentationDataset(train_df, project_root=project_root, image_size=image_size)
    val_dataset = LungSegmentationDataset(val_df, project_root=project_root, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize U-Net, Loss, Optimizer & Scheduler
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_dice = 0.0

    print("\n" + "=" * 75)
    print("                 U-NET LUNG SEGMENTATION TRAINING LOOP                 ")
    print("=" * 75)
    print(f" {'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Dice':^10} | {'Val IoU':^10} | {'LR':^8}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)

        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        print(f" {epoch:^6d} | {train_loss:^10.4f} | {val_loss:^10.4f} | {val_dice:^10.4f} | {val_iou:^10.4f} | {curr_lr:^8.1e}")

        # Save model checkpoint on best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }, save_path)
            logger.info(f" Checkpoint saved at Epoch {epoch} -> {save_path} (Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f})")

    print("=" * 75)
    logger.info(f"Training Complete. Best Validation Dice Score: {best_dice:.4f}")
    return model


# =============================================================================
# 6. INFERENCE & LUNG ROI EXTRACTION HELPER FUNCTION
# =============================================================================

@torch.no_grad()
def extract_lung_roi(
    model: nn.Module,
    image_path: Union[str, Path],
    device: torch.device,
    image_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    visualize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Infer lung mask for an input radiograph and extract the lung parenchyma ROI.

    Args:
        model (nn.Module): Trained U-Net model.
        image_path (Union[str, Path]): Path to input radiograph image.
        device (torch.device): Torch device.
        image_size (Tuple[int, int]): Input resolution (width, height).
        threshold (float): Probability binarization threshold.
        visualize (bool): If True, plots side-by-side comparison figure.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Original Grayscale Image (256, 256) uint8
            - Predicted Binary Lung Mask (256, 256) uint8
            - Segmented Lung Parenchyma ROI (256, 256) uint8
    """
    model.eval()
    full_path = Path(image_path).resolve()

    img_orig = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise FileNotFoundError(f"Image file not found or invalid: {full_path}")

    # Resize image to model resolution
    img_resized = cv2.resize(img_orig, image_size, interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # Model Inference
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255

    # Element-wise multiplication to isolate lung parenchyma ROI
    lung_roi = cv2.bitwise_and(img_resized, img_resized, mask=pred_mask)

    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_resized, cmap="gray")
        axes[0].set_title("Input CLAHE Radiograph")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="bone")
        axes[1].set_title("Predicted U-Net Lung Mask")
        axes[1].axis("off")

        axes[2].imshow(lung_roi, cmap="gray")
        axes[2].set_title("Segmented Lung Parenchyma ROI")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return img_resized, pred_mask, lung_roi


# =============================================================================
# 7. CLI MAIN EXECUTION ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net for Lung Segmentation and ROI Isolation.")
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="data/processed/dataset_metadata.csv",
        help="Path to metadata CSV (default: data/processed/dataset_metadata.csv)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="weights/lung_unet_segmentation.pth",
        help="Destination path for trained model weights (default: weights/lung_unet_segmentation.pth)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")

    args = parser.parse_args()

    trained_model = train_unet_model(
        metadata_csv_path=args.metadata_csv,
        weights_save_path=args.weights_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
