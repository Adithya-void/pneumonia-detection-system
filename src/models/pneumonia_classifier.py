"""
Phase 4: Multi-Class Pneumonia CNN Classifier & Grad-CAM Explainability Engine.

This module provides a deep learning classification pipeline built on a ResNet-50 backbone,
trained on preprocessed/augmented chest radiographs to classify:
    - Class 0: Normal
    - Class 1: Bacterial Pneumonia
    - Class 2: Viral Pneumonia

Includes Focal Loss, Comprehensive Clinical Evaluation Metrics (Confusion Matrix, ROC-AUC),
and Grad-CAM (Gradient-Weighted Class Activation Mapping) for radiologist interpretability.

Author: Senior AI/ML Engineer (Medical Image Analysis)
Project: Enhanced CNN-Based Multi-Class Pneumonia Detection System
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from torchvision.models import ResNet50_Weights, DenseNet121_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]


# =============================================================================
# 1. FOCAL LOSS FOR MEDICAL IMAGE CLASSIFICATION
# =============================================================================

class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss to address hard-sample mining and residual class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# 2. PYTORCH DATASET & DATALOADER
# =============================================================================

class PneumoniaClassifierDataset(Dataset):
    """
    PyTorch Dataset for 3-Class Pneumonia Classification.
    Loads standardized (224, 224) or (256, 256) grayscale images converted to 3-channel tensors.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        project_root: Path,
        image_size: Tuple[int, int] = (384, 384),
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.project_root = Path(project_root)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_path(self, row: pd.Series) -> Path:
        for col in ["augmented_file_path", "segmented_file_path", "clahe_file_path", "file_path"]:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                rel = Path(row[col])
                full = self.project_root / rel if not rel.is_absolute() else rel
                if full.exists():
                    return full
        # Fallback to direct path join
        rel = Path(row["file_path"])
        return self.project_root / rel if not rel.is_absolute() else rel

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        full_path = self._resolve_path(row)

        img_gray = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            logger.warning(f"Failed to read image at: {full_path}. Returning zero tensor.")
            img_gray = np.zeros(self.image_size, dtype=np.uint8)

        img_resized = cv2.resize(img_gray, self.image_size, interpolation=cv2.INTER_AREA)

        # Convert 1-channel grayscale to 3-channel for backbone compatibility
        img_3ch = np.stack([img_resized] * 3, axis=-1)

        # Normalize to [0.0, 1.0] and apply ImageNet mean/std standardization
        img_tensor = torch.from_numpy(img_3ch).float().permute(2, 0, 1) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_normalized = (img_tensor - mean) / std

        label = int(row["class_id"])
        return img_normalized, torch.tensor(label, dtype=torch.long)


# =============================================================================
# 3. DEEP CNN CLASSIFIER ARCHITECTURE (RESNET-50 / DENSENET-121)
# =============================================================================

class ResNet50PneumoniaClassifier(nn.Module):
    """
    ResNet-50 Backbone with custom classification head for 3-class pneumonia diagnosis.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features

        # Replace default fc layer with custom medical classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNet121PneumoniaClassifier(nn.Module):
    """
    DenseNet-121 Backbone with custom classification head for 3-class pneumonia diagnosis.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_pneumonia_classifier(backbone: str = "densenet121", num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Factory function to build deep CNN classifier."""
    b_name = backbone.lower()
    if b_name == "densenet121":
        return DenseNet121PneumoniaClassifier(num_classes=num_classes, pretrained=pretrained)
    elif b_name == "resnet50":
        return ResNet50PneumoniaClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: '{backbone}'. Choose 'densenet121' or 'resnet50'.")


# =============================================================================
# 4. GRAD-CAM (GRADIENT-WEIGHTED CLASS ACTIVATION MAPPING) ENGINE
# =============================================================================

class GradCAM:
    """
    Grad-CAM Engine for generating clinical activation heatmaps overlaid on radiographs.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]) -> None:
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate 2D Grad-CAM heatmap array normalized to [0, 1].

        Args:
            input_tensor (torch.Tensor): Preprocessed input tensor (1, 3, H, W).
            target_class (Optional[int]): Target class index. Uses predicted class if None.

        Returns:
            np.ndarray: 2D float32 heatmap array resized to input spatial dimensions.
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients[0], dim=(1, 2), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations[0], dim=0)
        cam = F.relu(cam)  # Apply ReLU to focus on features that positively contribute

        cam_np = cam.cpu().numpy()
        cam_np = cv2.resize(cam_np, (input_tensor.shape[3], input_tensor.shape[2]))

        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        return cam_np

    def overlay_heatmap(
        self, img_orig: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on top of original radiograph image.

        Args:
            img_orig (np.ndarray): Original 2D uint8 radiograph image.
            heatmap (np.ndarray): 2D float32 heatmap in range [0, 1].
            alpha (float): Overlay transparency ratio (default: 0.4).

        Returns:
            np.ndarray: BGR image array with heatmap overlay.
        """
        h, w = img_orig.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        if len(img_orig.shape) == 2:
            img_bgr = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_orig

        overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, color_heatmap, alpha, 0)
        return overlay


# =============================================================================
# 5. TRAINING LOOP & CLINICAL EVALUATION
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = F.softmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    total = len(all_labels)
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / total

    return (
        running_loss / total,
        acc,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def train_pneumonia_classifier(
    metadata_csv_path: Union[str, Path],
    weights_save_path: Union[str, Path],
    backbone: str = "resnet50",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device_str: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> nn.Module:
    csv_path = Path(metadata_csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at: {csv_path}")

    save_path = Path(weights_save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Compute Device: {device}")

    df = pd.read_csv(csv_path)

    # Use train and val/test splits as defined in metadata
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(val_df) < 5:
        # If val split is tiny, split 15% from train
        train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["class_id"])

    logger.info(f"Dataset Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_dataset = PneumoniaClassifierDataset(train_df, project_root=project_root)
    val_dataset = PneumoniaClassifierDataset(val_df, project_root=project_root)
    test_dataset = PneumoniaClassifierDataset(test_df, project_root=project_root)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_pneumonia_classifier(backbone=backbone, num_classes=3, pretrained=True).to(device)

    # Compute class weights for Focal Loss
    class_counts = train_df["class_id"].value_counts().sort_index().values
    total_samples = sum(class_counts)
    weights = torch.tensor([total_samples / (len(class_counts) * c) for c in class_counts], dtype=torch.float32)

    criterion = FocalLoss(alpha=weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_f1 = 0.0

    print("\n" + "=" * 75)
    print(f"       TRAINING PNEUMONIA CNN CLASSIFIER ({backbone.upper()})       ")
    print("=" * 75)
    print(f" {'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Macro F1':^10}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_labels, val_preds, _ = evaluate_model(model, val_loader, criterion, device)

        macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        scheduler.step()

        print(f" {epoch:^6d} | {train_loss:^10.4f} | {train_acc*100:^9.2f}% | {val_loss:^10.4f} | {val_acc*100:^9.2f}% | {macro_f1:^10.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_f1 = macro_f1
            torch.save({
                "epoch": epoch,
                "backbone": backbone,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "macro_f1": macro_f1,
            }, save_path)
            logger.info(f" Checkpoint Saved -> {save_path} (Val Acc: {val_acc*100:.2f}%)")

    print("=" * 75)
    logger.info(f"Training Complete! Peak Validation Accuracy: {best_val_acc*100:.2f}% (Macro F1: {best_f1:.4f})")

    # Final Evaluation on Independent Test Set
    logger.info("Evaluating Best Model Checkpoint on Test Set...")
    try:
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_labels, test_preds, test_probs = evaluate_model(model, test_loader, criterion, device)

    print("\n" + "=" * 70)
    print("                FINAL TEST SET EVALUATION REPORT                ")
    print("=" * 70)
    print(f" Overall Test Accuracy : {test_acc*100:.2f}%")
    print(f" Overall Test Loss     : {test_loss:.4f}\n")

    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4))
    print("=" * 70 + "\n")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Multi-Class Pneumonia CNN Classifier with Grad-CAM.")
    parser.add_argument("--metadata-csv", type=str, default="data/processed/augmented_dataset_metadata.csv")
    parser.add_argument("--weights-path", type=str, default="weights/best_densenet121_pneumonia_classifier.pth")
    parser.add_argument("--backbone", type=str, default="densenet121", choices=["resnet50", "densenet121"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    # Fallback to dataset_metadata.csv if augmented CSV doesn't exist locally
    if not Path(args.metadata_csv).exists():
        args.metadata_csv = "data/processed/dataset_metadata.csv"

    trained_model = train_pneumonia_classifier(
        metadata_csv_path=args.metadata_csv,
        weights_save_path=args.weights_path,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
