"""
Training and validation utilities for surgical skill assessment models.

This module provides functions for training deep learning models with support for
mixed precision, gradient accumulation, and comprehensive metric tracking.

Author: Surgical Skill Assessment Team
Date: August 2025
"""

import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinarySpecificity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import rich for progress bars, fall back to basic logging if not available
try:
    from rich.progress import Progress

    RICH_AVAILABLE = True
    ProgressType = Progress
except ImportError:
    RICH_AVAILABLE = False
    ProgressType = Any  # Fallback type
    logger.warning("Rich not available. Using basic progress logging.")


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for video data with variable lengths.

    Args:
        batch: List of (video_tensor, label) tuples

    Returns:
        Tuple of (batched_videos, batched_labels)
    """
    videos, labels = zip(*batch)

    # Stack videos and labels
    videos = torch.stack(videos, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return videos, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: amp.GradScaler,
    device: torch.device,
    config: Dict,
    progress: Optional[Any] = None,
    task_id: Optional[int] = None,
) -> Dict:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model to train
        loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        scaler: Gradient scaler for mixed precision training
        device: Device to run training on (CPU/GPU)
        config: Configuration dictionary
        progress: Optional rich Progress object for progress bars
        task_id: Optional task ID for progress tracking

    Returns:
        Dict: Dictionary containing average metrics for the epoch
    """
    model.train()
    metrics = {name: 0.0 for name in config["metrics"]}

    for i, (videos, labels, _) in enumerate(loader):
        videos, labels = videos.to(device), labels.to(device)

        with amp.autocast(enabled=config["hardware"]["mixed_precision"]):
            outputs = model(videos)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        if (i + 1) % config["train"]["gradient_accumulation"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        metrics["loss"] += loss.item()

        # Assuming binary classification for other metrics
        acc_metric = Accuracy(task="multiclass", num_classes=2).to(device)
        prec_metric = Precision(task="multiclass", num_classes=2).to(device)
        rec_metric = Recall(task="multiclass", num_classes=2).to(device)
        f1_metric = F1Score(task="multiclass", num_classes=2).to(device)
        spec_metric = BinarySpecificity().to(device)
        auc_metric = AUROC(task="multiclass", num_classes=2).to(device)

        metrics["accuracy"] += acc_metric(preds, labels).item()
        metrics["precision"] += prec_metric(preds, labels).item()
        metrics["recall"] += rec_metric(preds, labels).item()
        metrics["f1"] += f1_metric(preds, labels).item()
        metrics["sensitivity"] += rec_metric(
            preds, labels
        ).item()  # Recall is sensitivity
        metrics["specificity"] += spec_metric(preds, labels).item()
        metrics["auc"] += auc_metric(outputs, labels).item()

        if progress and task_id:
            progress.update(task_id, advance=1)
        elif (i + 1) % config.get("logging", {}).get("print_freq", 10) == 0:
            logger.info(f"Batch {i+1}/{len(loader)}: Loss = {loss.item():.4f}")

    num_batches = len(loader)
    for name in metrics:
        metrics[name] /= num_batches

    return metrics


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict,
    progress: Optional[Any] = None,
    task_id: Optional[int] = None,
) -> Dict:
    """
    Validate the model for one epoch.

    Args:
        model: The neural network model to validate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on (CPU/GPU)
        config: Configuration dictionary
        progress: Optional rich Progress object for progress bars
        task_id: Optional task ID for progress tracking

    Returns:
        Dict: Dictionary containing average metrics for the epoch
    """
    model.eval()
    metrics = {name: 0.0 for name in config["metrics"]}

    with torch.no_grad():
        for videos, labels, _ in loader:
            videos, labels = videos.to(device), labels.to(device)

            with amp.autocast(enabled=config["hardware"]["mixed_precision"]):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            metrics["loss"] += loss.item()

            acc_metric = Accuracy(task="multiclass", num_classes=2).to(device)
            prec_metric = Precision(task="multiclass", num_classes=2).to(device)
            rec_metric = Recall(task="multiclass", num_classes=2).to(device)
            f1_metric = F1Score(task="multiclass", num_classes=2).to(device)
            spec_metric = BinarySpecificity().to(device)
            auc_metric = AUROC(task="multiclass", num_classes=2).to(device)

            metrics["accuracy"] += acc_metric(preds, labels).item()
            metrics["precision"] += prec_metric(preds, labels).item()
            metrics["recall"] += rec_metric(preds, labels).item()
            metrics["f1"] += f1_metric(preds, labels).item()
            metrics["sensitivity"] += rec_metric(preds, labels).item()
            metrics["specificity"] += spec_metric(preds, labels).item()
            metrics["auc"] += auc_metric(outputs, labels).item()

            if progress and task_id:
                progress.update(task_id, advance=1)

    num_batches = len(loader)
    for name in metrics:
        metrics[name] /= num_batches

    return metrics
