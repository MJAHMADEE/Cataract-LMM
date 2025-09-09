"""
Training Module for Surgical Instance Segmentation

This module provides training implementations exactly matching the reference notebooks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseTrainer(ABC):
    """Base trainer class matching the framework structure."""

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def train_epoch(self, dataloader, optimizer, criterion=None):
        """Train for one epoch."""
        pass

    @abstractmethod
    def validate(self, dataloader):
        """Validate the model."""
        pass


class MaskRCNNTrainer(BaseTrainer):
    """Mask R-CNN trainer exactly matching maskRCNN.ipynb."""

    def __init__(self, model, device: str = "cuda"):
        super().__init__(model, device)

    def train_epoch(self, dataloader, optimizer, criterion=None):
        """Train Mask R-CNN for one epoch exactly as in notebook."""
        self.model.train()
        total_loss = 0
        for images, targets in dataloader:
            optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """Validate Mask R-CNN model."""
        self.model.eval()
        with torch.no_grad():
            for images, targets in dataloader:
                predictions = self.model(images)
        return predictions


class YOLOTrainer(BaseTrainer):
    """YOLO trainer exactly matching train_yolo8.ipynb and train_yolo11.ipynb."""

    def __init__(self, model, device: str = "cuda"):
        super().__init__(model, device)

    def train_epoch(self, dataloader, optimizer, criterion=None):
        """YOLO training handled by ultralytics internally."""
        # YOLO uses internal training loop
        return self.model.train()

    def validate(self, dataloader):
        """YOLO validation handled by ultralytics internally."""
        return self.model.val()


class SAMTrainer(BaseTrainer):
    """SAM trainer (SAM is typically used for inference only)."""

    def __init__(self, model, device: str = "cuda"):
        super().__init__(model, device)

    def train_epoch(self, dataloader, optimizer, criterion=None):
        """SAM is pre-trained, typically used for inference only."""
        raise NotImplementedError("SAM is typically used for inference only")

    def validate(self, dataloader):
        """SAM validation through inference."""
        return self.model.predict()


class TrainingPipeline:
    """Complete training pipeline matching notebook workflows."""

    def __init__(self, model_type: str, model, device: str = "cuda"):
        self.model_type = model_type
        self.model = model
        self.device = device

        if model_type.lower() == "maskrcnn":
            self.trainer = MaskRCNNTrainer(model, device)
        elif model_type.lower() in ["yolo", "yolo8", "yolo11"]:
            self.trainer = YOLOTrainer(model, device)
        elif model_type.lower() == "sam":
            self.trainer = SAMTrainer(model, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, train_dataloader, val_dataloader=None, epochs: int = 10):
        """Train the model using the appropriate trainer."""
        if hasattr(self.trainer.model, "train") and callable(
            getattr(self.trainer.model, "train")
        ):
            # For YOLO models with built-in training
            return self.trainer.model.train(epochs=epochs)
        else:
            # For custom training loops
            for epoch in range(epochs):
                train_loss = self.trainer.train_epoch(train_dataloader, None)
                if val_dataloader:
                    val_results = self.trainer.validate(val_dataloader)
            return {"training_completed": True}


def create_trainer(model_type: str, model, device: str = "cuda"):
    """Create appropriate trainer for the model type."""
    return TrainingPipeline(model_type, model, device)


__all__ = [
    "BaseTrainer",
    "MaskRCNNTrainer",
    "YOLOTrainer",
    "SAMTrainer",
    "TrainingPipeline",
    "create_trainer",
]
