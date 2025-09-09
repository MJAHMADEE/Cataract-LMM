#!/usr/bin/env python3
"""
Training and Validation Framework for Surgical Phase Recognition

This module implements PyTorch Lightning training and validation framework
for surgical phase recognition models. Based on the validation notebook
pipeline with support for multiple architectures and training strategies.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from ..models import create_model, get_model_parameters
from ..models.multistage_models import TeCNOModel

logger = logging.getLogger(__name__)


class SurgicalPhaseClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for surgical phase recognition.

    This module provides a unified interface for training and validation
    of different model architectures for surgical phase recognition.

    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of surgical phases
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay for optimization
        optimizer (str): Optimizer type ('adam', 'adamw', 'sgd')
        scheduler (str): Learning rate scheduler type
        class_weights (torch.Tensor, optional): Class weights for imbalanced data
        use_wandb (bool): Whether to log metrics to Weights & Biases
        model_kwargs (dict): Additional model arguments
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 11,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        class_weights: Optional[torch.Tensor] = None,
        use_wandb: bool = False,
        **model_kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.use_wandb = use_wandb

        # Create model
        self.model = create_model(
            model_name=model_name, num_classes=num_classes, **model_kwargs
        )

        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # For TeCNO model, we also need temporal consistency loss
        self.is_tecno = isinstance(self.model, TeCNOModel)
        if self.is_tecno:
            self.consistency_weight = getattr(self.model, "consistency_weight", 0.1)

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Phase names for logging
        self.phase_names = [
            "Incision",
            "Viscous Agent Injection",
            "Rhexis",
            "Hydrodissection",
            "Phacoemulsification",
            "Irrigation and Aspiration",
            "Capsule Polishing",
            "Lens Implant Setting",
            "Viscous Agent Removal",
            "Suturing",
            "Tonifying Antibiotics",
        ]

        logger.info(f"Initialized {model_name} with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the model."""
        return self.model(x)

    def _compute_loss(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss based on model type."""
        if isinstance(outputs, dict):
            # Multi-output model (e.g., TeCNO, Hierarchical)
            main_loss = self.criterion(outputs["logits"], targets)

            losses = {"main_loss": main_loss}
            total_loss = main_loss

            # TeCNO temporal consistency loss
            if self.is_tecno and "consistency_logits" in outputs:
                consistency_loss = self.model.compute_temporal_consistency_loss(
                    outputs["consistency_logits"], targets
                )
                losses["consistency_loss"] = consistency_loss
                total_loss = total_loss + self.consistency_weight * consistency_loss

            # Hierarchical model losses
            if "coarse_logits" in outputs:
                # Map fine labels to coarse labels (simplified mapping)
                # In practice, you would have a proper fine-to-coarse mapping
                coarse_targets = targets // 3  # Simple grouping for example
                coarse_targets = torch.clamp(coarse_targets, 0, 3)  # Ensure valid range

                coarse_loss = self.criterion(outputs["coarse_logits"], coarse_targets)
                losses["coarse_loss"] = coarse_loss
                total_loss = total_loss + 0.3 * coarse_loss

            losses["total_loss"] = total_loss
            return total_loss, losses

        else:
            # Single output model
            loss = self.criterion(outputs, targets)
            return loss, {"loss": loss}

    def _compute_metrics(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        # Get predictions
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        preds = torch.argmax(logits, dim=1)

        # Convert to numpy for sklearn metrics
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Compute metrics
        accuracy = accuracy_score(targets_np, preds_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_np, preds_np, average="weighted", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        videos = batch["video"]
        labels = batch["label"]

        # Forward pass
        outputs = self(videos)

        # Compute loss
        loss, loss_dict = self._compute_loss(outputs, labels)

        # Compute metrics
        metrics = self._compute_metrics(outputs, labels)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", metrics["accuracy"], on_step=True, on_epoch=True, prog_bar=True
        )

        for loss_name, loss_value in loss_dict.items():
            if loss_name != "total_loss":
                self.log(f"train_{loss_name}", loss_value, on_step=True, on_epoch=True)

        # Store outputs for epoch-end processing
        self.training_step_outputs.append(
            {
                "loss": loss.detach(),
                "preds": torch.argmax(
                    outputs["logits"] if isinstance(outputs, dict) else outputs, dim=1
                ).detach(),
                "targets": labels.detach(),
            }
        )

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        videos = batch["video"]
        labels = batch["label"]

        # Forward pass
        outputs = self(videos)

        # Compute loss
        loss, loss_dict = self._compute_loss(outputs, labels)

        # Compute metrics
        metrics = self._compute_metrics(outputs, labels)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_acc", metrics["accuracy"], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val_f1", metrics["f1"], on_step=False, on_epoch=True, prog_bar=True)

        for loss_name, loss_value in loss_dict.items():
            if loss_name != "total_loss":
                self.log(f"val_{loss_name}", loss_value, on_step=False, on_epoch=True)

        # Store outputs for epoch-end processing
        self.validation_step_outputs.append(
            {
                "loss": loss.detach(),
                "preds": torch.argmax(
                    outputs["logits"] if isinstance(outputs, dict) else outputs, dim=1
                ).detach(),
                "targets": labels.detach(),
            }
        )

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        videos = batch["video"]
        labels = batch["label"]

        # Forward pass
        outputs = self(videos)

        # Compute loss
        loss, _ = self._compute_loss(outputs, labels)

        # Compute metrics
        metrics = self._compute_metrics(outputs, labels)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", metrics["accuracy"], on_step=False, on_epoch=True)
        self.log("test_f1", metrics["f1"], on_step=False, on_epoch=True)

        # Store outputs for final processing
        self.test_step_outputs.append(
            {
                "loss": loss.detach(),
                "preds": torch.argmax(
                    outputs["logits"] if isinstance(outputs, dict) else outputs, dim=1
                ).detach(),
                "targets": labels.detach(),
            }
        )

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        if self.training_step_outputs:
            # Compute epoch metrics
            all_preds = torch.cat([x["preds"] for x in self.training_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.training_step_outputs])

            # Compute confusion matrix
            cm = confusion_matrix(all_targets.cpu().numpy(), all_preds.cpu().numpy())

            # Log to wandb if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "train_confusion_matrix": wandb.Image(
                            self._plot_confusion_matrix(cm)
                        ),
                        "epoch": self.current_epoch,
                    }
                )

            # Clear outputs
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self.validation_step_outputs:
            # Compute epoch metrics
            all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
            all_targets = torch.cat(
                [x["targets"] for x in self.validation_step_outputs]
            )

            # Compute per-class metrics
            per_class_metrics = self._compute_per_class_metrics(all_preds, all_targets)

            # Log per-class metrics
            for i, phase_name in enumerate(self.phase_names[: self.num_classes]):
                if i in per_class_metrics:
                    self.log(
                        f"val_acc_{phase_name}",
                        per_class_metrics[i]["accuracy"],
                        on_epoch=True,
                    )
                    self.log(
                        f"val_f1_{phase_name}",
                        per_class_metrics[i]["f1"],
                        on_epoch=True,
                    )

            # Compute confusion matrix
            cm = confusion_matrix(all_targets.cpu().numpy(), all_preds.cpu().numpy())

            # Log to wandb if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "val_confusion_matrix": wandb.Image(
                            self._plot_confusion_matrix(cm)
                        ),
                        "epoch": self.current_epoch,
                    }
                )

            # Clear outputs
            self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        if self.test_step_outputs:
            # Compute final test metrics
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])

            # Compute detailed metrics
            per_class_metrics = self._compute_per_class_metrics(all_preds, all_targets)
            cm = confusion_matrix(all_targets.cpu().numpy(), all_preds.cpu().numpy())

            # Log final results
            logger.info("Test Results:")
            logger.info(
                f"Overall Accuracy: {accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy()):.4f}"
            )

            for i, phase_name in enumerate(self.phase_names[: self.num_classes]):
                if i in per_class_metrics:
                    metrics = per_class_metrics[i]
                    logger.info(
                        f"{phase_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}"
                    )

            # Clear outputs
            self.test_step_outputs.clear()

    def _compute_per_class_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-class metrics."""
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        per_class_metrics = {}

        for class_idx in range(self.num_classes):
            # Binary classification for this class
            class_preds = (preds_np == class_idx).astype(int)
            class_targets = (targets_np == class_idx).astype(int)

            if class_targets.sum() > 0:  # Only compute if class exists in targets
                precision, recall, f1, _ = precision_recall_fscore_support(
                    class_targets, class_preds, average="binary", zero_division=0
                )
                accuracy = accuracy_score(class_targets, class_preds)

                per_class_metrics[class_idx] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

        return per_class_metrics

    def _plot_confusion_matrix(self, cm: np.ndarray) -> Any:
        """Plot confusion matrix for logging."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.phase_names[: self.num_classes],
                yticklabels=self.phase_names[: self.num_classes],
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            return plt

        except ImportError:
            logger.warning(
                "matplotlib/seaborn not available for confusion matrix plotting"
            )
            return None

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Select optimizer
        if self.optimizer_name == "adam":
            optimizer = Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        # Select scheduler
        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        elif self.scheduler_name == "step":
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        elif self.scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", patience=10, factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }
        else:
            return optimizer


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.

    Provides detailed analysis of model performance including per-class metrics,
    confusion matrices, error analysis, and performance comparisons.

    Args:
        model (SurgicalPhaseClassifier): Trained model
        phase_names (List[str]): List of phase names
        device (torch.device): Device for evaluation
    """

    def __init__(
        self,
        model: SurgicalPhaseClassifier,
        phase_names: List[str],
        device: torch.device = None,
    ):
        self.model = model
        self.phase_names = phase_names
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.eval()
        self.model.to(self.device)

    def evaluate_dataloader(self, dataloader) -> Dict[str, Any]:
        """
        Evaluate model on a dataloader.

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                videos = batch["video"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(videos)

                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Compute metrics
        results = self._compute_comprehensive_metrics(all_preds, all_targets, all_probs)

        return results

    def _compute_comprehensive_metrics(
        self, preds: np.ndarray, targets: np.ndarray, probs: np.ndarray
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        # Overall metrics
        overall_acc = accuracy_score(targets, preds)
        overall_precision, overall_recall, overall_f1, _ = (
            precision_recall_fscore_support(
                targets, preds, average="weighted", zero_division=0
            )
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = (
            precision_recall_fscore_support(
                targets, preds, average=None, zero_division=0
            )
        )

        # Confusion matrix
        cm = confusion_matrix(targets, preds)

        # Per-class results
        per_class_results = {}
        for i, phase_name in enumerate(self.phase_names):
            if i < len(per_class_precision):
                per_class_results[phase_name] = {
                    "precision": per_class_precision[i],
                    "recall": per_class_recall[i],
                    "f1": per_class_f1[i],
                    "support": support[i],
                }

        # Class-wise accuracy from confusion matrix
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, phase_name in enumerate(self.phase_names):
            if i < len(class_accuracy) and phase_name in per_class_results:
                per_class_results[phase_name]["accuracy"] = class_accuracy[i]

        results = {
            "overall_metrics": {
                "accuracy": overall_acc,
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
            },
            "per_class_metrics": per_class_results,
            "confusion_matrix": cm,
            "predictions": preds,
            "targets": targets,
            "probabilities": probs,
        }

        return results

    def compare_models(
        self, models: Dict[str, SurgicalPhaseClassifier], dataloader
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset.

        Args:
            models (Dict[str, SurgicalPhaseClassifier]): Dictionary of models to compare
            dataloader: Evaluation dataloader

        Returns:
            Dict[str, Any]: Comparison results
        """
        comparison_results = {}

        for model_name, model in models.items():
            evaluator = ModelEvaluator(model, self.phase_names, self.device)
            results = evaluator.evaluate_dataloader(dataloader)
            comparison_results[model_name] = results

        # Create comparison summary
        summary = {"model_comparison": {}, "best_models": {}}

        metrics = ["accuracy", "precision", "recall", "f1"]

        for metric in metrics:
            scores = {
                name: results["overall_metrics"][metric]
                for name, results in comparison_results.items()
            }

            best_model = max(scores, key=scores.get)
            summary["best_models"][metric] = {
                "model": best_model,
                "score": scores[best_model],
            }

            summary["model_comparison"][metric] = scores

        return {"individual_results": comparison_results, "summary": summary}


if __name__ == "__main__":
    # Test the training framework
    logger.info("Testing training framework...")

    # Test model creation
    try:
        model = SurgicalPhaseClassifier(
            model_name="r3d_18", num_classes=11, learning_rate=1e-4
        )

        # Test forward pass
        dummy_input = torch.randn(2, 3, 16, 224, 224)
        output = model(dummy_input)

        print(f"Model output shape: {output.shape}")
        print("Training framework ready!")

    except Exception as e:
        print(f"Error testing framework: {e}")

    print("Use this framework with PyTorch Lightning Trainer for training.")
