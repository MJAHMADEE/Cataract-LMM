"""
Mask R-CNN Trainer Implementation

This trainer exactly matches the training procedure from the reference maskRCNN.ipynb notebook
while providing additional production features and comprehensive logging.

The implementation follows the exact same:
- Model initialization and configuration
- Optimizer setup (AdamW with same parameters)
- Learning rate scheduler configuration
- Training loop structure and loss computation
- Model saving procedures
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset_utils import (
    SurgicalCocoDataset,
    create_data_loaders,
    create_data_splits,
)
from models.mask_rcnn.model import SurgicalMaskRCNN
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer


class MaskRCNNTrainer(BaseTrainer):
    """
    Mask R-CNN Trainer for Surgical Instance Segmentation

    This implementation exactly matches the training procedure from the reference notebook
    with the same model configuration, optimizer settings, and training loop structure.

    Key Features:
    - Exact same model architecture as notebook (ResNet50-FPN, 13 classes)
    - Same optimizer configuration (AdamW, lr=0.0005, weight_decay=0.0005)
    - Same learning rate scheduler (StepLR, step_size=3, gamma=0.1)
    - Same training loop structure with loss computation
    - Progress tracking with tqdm (same as notebook)
    - Model saving functionality
    """

    def __init__(
        self,
        num_classes: int = 13,
        backbone: str = "resnet50",
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Mask R-CNN trainer with notebook-compatible configuration

        Args:
            num_classes (int): Number of classes (13 to match notebook: 12 instruments + background)
            backbone (str): Backbone architecture (resnet50 to match notebook)
            pretrained (bool): Use pretrained weights (True to match notebook)
            trainable_backbone_layers (int): Number of trainable backbone layers (3 to match notebook)
            device (str): Device to use ('cuda' or 'cpu', auto-detect if None)
        """
        super().__init__()

        # Set device exactly as in notebook
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)

        # Initialize model exactly as in notebook
        self.model = SurgicalMaskRCNN(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )

        # Move model to device exactly as in notebook
        self.model.to(self.device)

        # Store configuration
        self.config = {
            "num_classes": num_classes,
            "backbone": backbone,
            "pretrained": pretrained,
            "trainable_backbone_layers": trainable_backbone_layers,
            "device": str(self.device),
        }

        # Initialize training components
        self.optimizer = None
        self.lr_scheduler = None
        self.training_history = defaultdict(list)

    def setup_optimizer(
        self,
        optimizer_type: str = "adamw",
        lr: float = 0.0005,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        **kwargs,
    ) -> None:
        """
        Setup optimizer exactly as in the reference notebook

        Default parameters match the notebook:
        - AdamW optimizer (as used in notebook instead of SGD)
        - Learning rate: 0.0005
        - Weight decay: 0.0005
        """
        # Parameters to optimize (only those requiring gradients) - exactly as in notebook
        params = [p for p in self.model.parameters() if p.requires_grad]

        if optimizer_type.lower() == "adamw":
            # AdamW optimizer exactly as configured in notebook
            self.optimizer = torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay, **kwargs
            )
        elif optimizer_type.lower() == "sgd":
            # SGD option (commented out in notebook but available)
            self.optimizer = torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        print(
            f"Optimizer configured: {optimizer_type.upper()} with lr={lr}, weight_decay={weight_decay}"
        )

    def setup_scheduler(
        self,
        scheduler_type: str = "step",
        step_size: int = 3,
        gamma: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Setup learning rate scheduler exactly as in the reference notebook

        Default parameters match the notebook:
        - StepLR scheduler
        - step_size: 3 (reduce LR every 3 epochs)
        - gamma: 0.1 (multiply LR by 0.1)
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be setup before scheduler")

        if scheduler_type.lower() == "step":
            # Learning rate scheduler exactly as in notebook
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,  # Reduce LR every 3 epochs (notebook default)
                gamma=gamma,  # Multiply LR by 0.1 (notebook default)
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        print(
            f"Scheduler configured: {scheduler_type.upper()} with step_size={step_size}, gamma={gamma}"
        )

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch following the exact same structure as the reference notebook

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dict containing epoch training metrics
        """
        # Set model to training mode - exactly as in notebook
        self.model.train()
        total_loss = 0

        # Create progress bar exactly as in notebook
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs if hasattr(self, 'num_epochs') else '?'}",
        )

        epoch_losses = defaultdict(float)
        num_batches = 0

        for images, targets in progress_bar:
            # Move data to the appropriate device exactly as in notebook
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass - exactly as in notebook
            loss_dict = self.model(images, targets)  # Returns a dictionary of losses
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization - exactly as in notebook
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # Accumulate losses
            total_loss += losses.item()
            for loss_name, loss_value in loss_dict.items():
                epoch_losses[loss_name] += loss_value.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{losses.item():.4f}",
                    "avg_loss": f"{total_loss/num_batches:.4f}",
                }
            )

        # Calculate average losses
        avg_total_loss = total_loss / len(train_loader)
        avg_losses = {
            name: loss_sum / len(train_loader)
            for name, loss_sum in epoch_losses.items()
        }

        # Print average loss for the epoch exactly as in notebook
        print(
            f"Epoch [{epoch+1}/{self.num_epochs if hasattr(self, 'num_epochs') else '?'}], Average Loss: {avg_total_loss:.4f}"
        )

        # Log individual loss components
        for loss_name, avg_loss in avg_losses.items():
            print(f"  {loss_name}: {avg_loss:.4f}")

        return {"total_loss": avg_total_loss, **avg_losses}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_path: Optional[str] = None,
        save_best: bool = True,
        validate_every: int = 1,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """
        Full training procedure following the exact same structure as the reference notebook

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs (default: 10 to match notebook)
            save_path: Path to save the model
            save_best: Whether to save the best model
            validate_every: Validation frequency

        Returns:
            Dict containing training history
        """
        self.num_epochs = num_epochs

        # Setup optimizer and scheduler if not already done
        if self.optimizer is None:
            self.setup_optimizer()
        if self.lr_scheduler is None:
            self.setup_scheduler()

        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Total training batches per epoch: {len(train_loader)}")

        best_val_loss = float("inf")
        start_time = time.time()

        # Training loop exactly as in notebook structure
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            # Update learning rate scheduler exactly as in notebook
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Learning rate updated to: {current_lr:.6f}")

            # Store training history
            for metric_name, metric_value in train_metrics.items():
                self.training_history[f"train_{metric_name}"].append(metric_value)

            # Validation
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                val_metrics = self.validate(val_loader)
                for metric_name, metric_value in val_metrics.items():
                    self.training_history[f"val_{metric_name}"].append(metric_value)

                # Save best model
                if (
                    save_best
                    and val_metrics.get("total_loss", float("inf")) < best_val_loss
                ):
                    best_val_loss = val_metrics["total_loss"]
                    if save_path:
                        best_path = save_path.replace(".pth", "_best.pth")
                        self.save_model(best_path)
                        print(f"New best model saved: {best_path}")

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            print("-" * 60)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.1f} minutes)")

        # Save final model exactly as in notebook
        if save_path:
            self.save_model(save_path)
            print(f"Final model saved: {save_path}")

        return dict(self.training_history)

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validation procedure

        Args:
            val_loader: Validation data loader

        Returns:
            Dict containing validation metrics
        """
        self.model.eval()
        total_loss = 0
        val_losses = defaultdict(float)

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                images = list(image.to(self.device) for image in images)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()
                for loss_name, loss_value in loss_dict.items():
                    val_losses[loss_name] += loss_value.item()

        avg_total_loss = total_loss / len(val_loader)
        avg_losses = {
            name: loss_sum / len(val_loader) for name, loss_sum in val_losses.items()
        }

        print(f"Validation Loss: {avg_total_loss:.4f}")
        for loss_name, avg_loss in avg_losses.items():
            print(f"  val_{loss_name}: {avg_loss:.4f}")

        return {"total_loss": avg_total_loss, **avg_losses}

    def save_model(self, save_path: str, save_optimizer: bool = False) -> None:
        """
        Save model exactly as in the reference notebook

        Args:
            save_path: Path to save the model
            save_optimizer: Whether to save optimizer state
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model state dictionary exactly as in notebook
        if save_optimizer and self.optimizer is not None:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "training_history": dict(self.training_history),
            }
            torch.save(checkpoint, save_path)
        else:
            # Save only model state dict exactly as in notebook
            torch.save(self.model.state_dict(), save_path)

        print(f"Model saved as '{save_path}'")

    def load_model(self, load_path: str, load_optimizer: bool = False) -> None:
        """
        Load model weights

        Args:
            load_path: Path to load the model from
            load_optimizer: Whether to load optimizer state
        """
        if load_optimizer:
            checkpoint = torch.load(
                load_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "training_history" in checkpoint:
                self.training_history = defaultdict(
                    list, checkpoint["training_history"]
                )
            print(f"Model and optimizer loaded from '{load_path}'")
        else:
            # Load only model state dict
            state_dict = torch.load(
                load_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from '{load_path}'")

    def get_model(self) -> SurgicalMaskRCNN:
        """Get the trained model"""
        return self.model

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return dict(self.training_history)


def train_mask_rcnn_from_notebook(
    data_root: str, num_epochs: int = 10, save_path: str = "maskrcnn_finetuned.pth"
) -> MaskRCNNTrainer:
    """
    Train Mask R-CNN following the exact same procedure as the reference notebook

    This function replicates the complete training procedure from the notebook:
    1. Load dataset exactly as in notebook
    2. Create data splits exactly as in notebook
    3. Initialize model exactly as in notebook
    4. Setup optimizer and scheduler exactly as in notebook
    5. Train with same configuration as notebook
    6. Save model exactly as in notebook

    Args:
        data_root (str): Path to dataset root (should contain 'train' folder and '_annotations.coco.json')
        num_epochs (int): Number of epochs (default: 10 to match notebook)
        save_path (str): Path to save the trained model

    Returns:
        MaskRCNNTrainer: Trained trainer instance
    """
    print("=" * 60)
    print("Training Mask R-CNN for Surgical Instance Segmentation")
    print("Following exact procedure from reference notebook")
    print("=" * 60)

    # Load dataset exactly as in notebook
    from data.dataset_utils import load_aras_dataset

    train_loader, val_loader, test_loader = load_aras_dataset(data_root)

    print(f"Dataset loaded:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Initialize trainer exactly as in notebook
    trainer = MaskRCNNTrainer(
        num_classes=13,  # 12 instruments + background (exact as notebook)
        backbone="resnet50",  # Same as notebook
        pretrained=True,  # Same as notebook
        trainable_backbone_layers=3,  # Same as notebook
    )

    # Setup training components exactly as in notebook
    trainer.setup_optimizer(
        optimizer_type="adamw",  # As used in notebook
        lr=0.0005,  # Exact same as notebook
        weight_decay=0.0005,  # Exact same as notebook
    )

    trainer.setup_scheduler(
        scheduler_type="step",  # Same as notebook
        step_size=3,  # Same as notebook
        gamma=0.1,  # Same as notebook
    )

    # Train exactly as in notebook
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,  # Same default as notebook
        save_path=save_path,  # Save path
        save_best=True,
    )

    print("=" * 60)
    print("Training completed successfully!")
    print(f"Model saved as: {save_path}")
    print("=" * 60)

    return trainer


# Example usage exactly matching the notebook workflow
if __name__ == "__main__":
    # Example path - update to match your dataset
    data_root = "/content/final_main_seg_dataset_just_ARAS-3/train/"

    # Train exactly as in notebook
    trainer = train_mask_rcnn_from_notebook(
        data_root=data_root,
        num_epochs=10,  # Same as notebook
        save_path="maskrcnn_finetuned.pth",  # Same as notebook
    )

    # Get training history
    history = trainer.get_training_history()
    print("Training history:", list(history.keys()))
