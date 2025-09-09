#!/usr/bin/env python3
"""
Comprehensive Model Validation Module.

This module provides the exact validation functionality used in the reference
notebook phase_validation_comprehensive.ipynb for evaluating surgical phase
classification models.

Author: Surgical Phase Recognition Team
Date: August 29, 2025
"""

try:
    import pytorch_lightning as pl
    from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall

    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    PYTORCH_LIGHTNING_AVAILABLE = False
    print(
        "Warning: PyTorch Lightning not available. Some validation features may be limited."
    )

    # Create dummy base class
    class pl:
        class LightningModule:
            def __init__(self):
                pass


import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn


def strip_prefix(checkpoint_path: str, prefix: str = "model.") -> Dict[str, Any]:
    """
    Strip model prefix from checkpoint state dict.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        prefix (str): Prefix to strip from keys

    Returns:
        Dict[str, Any]: Cleaned state dict
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip the prefix from all keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


class ValidationModule(pl.LightningModule):
    """
    PyTorch Lightning module for comprehensive model validation.

    This class matches the exact validation approach used in the reference
    notebook for evaluating surgical phase recognition models.

    Args:
        model: The model to validate
        num_classes (int): Number of classes
        class_names (List[str], optional): Names of the classes
    """

    def __init__(
        self, model, num_classes: int, class_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.class_names = (
            class_names if class_names else [str(i) for i in range(num_classes)]
        )

        # Initialize metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision_macro = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall_macro = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1_macro = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_accuracy_per_class = Accuracy(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.val_precision_per_class = Precision(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.val_recall_per_class = Recall(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.val_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.val_conf_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )
        self.metrics = []

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        """Validation step for a single batch."""
        x, y, *z = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_accuracy.update(preds, y)
        self.val_precision_macro.update(preds, y)
        self.val_recall_macro.update(preds, y)
        self.val_f1_macro.update(preds, y)
        self.val_accuracy_per_class.update(preds, y)
        self.val_precision_per_class.update(preds, y)
        self.val_recall_per_class.update(preds, y)
        self.val_f1_per_class.update(preds, y)
        self.val_conf_matrix.update(preds, y)

    def _plot_confusion_matrix(
        self, conf_matrix_data, filename: str = "confusion_matrix.png"
    ):
        """Plot and save confusion matrix."""
        df_cm = pd.DataFrame(
            conf_matrix_data, index=self.class_names, columns=self.class_names
        )
        plt.figure(figsize=(12, 10))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="g").get_figure()
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        fig_.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig_)

    def on_validation_epoch_end(self):
        """Compute and log metrics at the end of validation epoch."""
        metrics = {
            "accuracy": self.val_accuracy.compute().item(),
            "precision_macro": self.val_precision_macro.compute().item(),
            "recall_macro": self.val_recall_macro.compute().item(),
            "f1_macro": self.val_f1_macro.compute().item(),
            "accuracy_per_class": self.val_accuracy_per_class.compute().cpu().numpy(),
            "precision_per_class": self.val_precision_per_class.compute().cpu().numpy(),
            "recall_per_class": self.val_recall_per_class.compute().cpu().numpy(),
            "f1_per_class": self.val_f1_per_class.compute().cpu().numpy(),
            "confusion_matrix": self.val_conf_matrix.compute().cpu().numpy(),
        }

        self.metrics.append(metrics)

        # Reset metrics
        self.val_accuracy.reset()
        self.val_precision_macro.reset()
        self.val_recall_macro.reset()
        self.val_f1_macro.reset()
        self.val_accuracy_per_class.reset()
        self.val_precision_per_class.reset()
        self.val_recall_per_class.reset()
        self.val_f1_per_class.reset()
        self.val_conf_matrix.reset()


def validate_model(
    model_name: str,
    checkpoint_paths: str,
    model,
    test_loader,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    output_dir: str = "metrics_full_farabi",
) -> Dict[str, Any]:
    """
    Validate a model using the checkpoint and test data.

    This function matches the exact validation approach used in the reference
    notebook for comprehensive model evaluation.

    Args:
        model_name (str): Name of the model
        checkpoint_paths (str): Path to the model checkpoint
        model: The model instance
        test_loader: DataLoader for test data
        num_classes (int): Number of classes
        class_names (List[str], optional): Names of the classes
        output_dir (str): Directory to save results

    Returns:
        Dict[str, Any]: Validation results
    """

    # Default class names if not provided
    if class_names is None:
        class_names = [
            "Incision",
            "Viscoelastic",
            "Capsulorhexis",
            "Hydrodissection",
            "Phacoemulsification",
            "IrrigationAspiration",
            "CapsulePolishing",
            "LensImplantation",
            "LensPositioning",
            "ViscoelasticSuction",
            "TonifyingAntibiotics",
        ][:num_classes]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Loading checkpoint from: {checkpoint_paths}")

        # Load and clean checkpoint
        cleaned_state_dict = strip_prefix(checkpoint_paths)
        model.load_state_dict(cleaned_state_dict, strict=False)

        # Create validation module
        val_module = ValidationModule(model, num_classes, class_names)

        # Create trainer
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

        # Run validation
        trainer.validate(val_module, test_loader, verbose=False)

        # Get results
        if val_module.metrics:
            metrics = val_module.metrics[-1]

            # Prepare results
            results = {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision_macro"],
                "recall": metrics["recall_macro"],
                "f1_score": metrics["f1_macro"],
            }

            # Add per-class metrics
            for i, class_name in enumerate(class_names):
                results[f"accuracy_{class_name}"] = metrics["accuracy_per_class"][
                    i
                ].item()
                results[f"precision_{class_name}"] = metrics["precision_per_class"][
                    i
                ].item()
                results[f"recall_{class_name}"] = metrics["recall_per_class"][i].item()
                results[f"f1_{class_name}"] = metrics["f1_per_class"][i].item()

            # Save metrics to CSV
            metrics_df = pd.DataFrame([results])
            metrics_file = os.path.join(output_dir, f"{model_name}_farabi_metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Metrics saved to {metrics_file}")

            # Plot and save confusion matrix
            confusion_matrix_file = os.path.join(
                output_dir, f"{model_name}_confusion_matrix_farabi.png"
            )
            val_module._plot_confusion_matrix(
                metrics["confusion_matrix"], confusion_matrix_file
            )
            print(f"Confusion matrix saved to {confusion_matrix_file}")

            # Print summary
            print(f"\nValidation Results for {model_name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision (macro): {results['precision']:.4f}")
            print(f"  Recall (macro): {results['recall']:.4f}")
            print(f"  F1-Score (macro): {results['f1_score']:.4f}")

            return results

        else:
            print(f"No metrics collected for {model_name}")
            return {"error": "No metrics collected"}

    except Exception as e:
        print(f"Error validating {model_name}: {e}")
        return {"error": str(e)}


def analyze_model_metrics(file_path: str):
    """
    Analyze and visualize model metrics from CSV files.

    This function provides comprehensive analysis of model performance
    as done in the reference notebook.

    Args:
        file_path (str): Path to the directory containing CSV files
    """

    # Get all CSV files
    csv_files = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith("_metrics.csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print(f"No metrics CSV files found in {file_path}")
        return

    # Load and combine all metrics
    all_metrics = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_metrics.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not all_metrics:
        print("No valid metrics files found")
        return

    # Combine all metrics
    combined_df = pd.concat(all_metrics, ignore_index=True)

    # Sort by F1 score
    combined_df = combined_df.sort_values("f1_score", ascending=False)

    print("Model Performance Summary:")
    print("=" * 80)
    print(
        f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
    )
    print("=" * 80)

    for _, row in combined_df.iterrows():
        print(
            f"{row['model']:<20} {row['accuracy']:<10.4f} {row['precision']:<10.4f} "
            f"{row['recall']:<10.4f} {row['f1_score']:<10.4f}"
        )

    # Create performance visualization
    plt.figure(figsize=(12, 8))

    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    x = range(len(combined_df))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        plt.bar(
            [xi + i * width for xi in x],
            combined_df[metric],
            width,
            label=metric.capitalize(),
        )

    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(
        [xi + width * 1.5 for xi in x], combined_df["model"], rotation=45, ha="right"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(file_path, "model_performance_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nPerformance comparison plot saved to {plot_file}")


# Default phase names used in the reference notebook
DEFAULT_PHASE_NAMES = [
    "Incision",
    "Viscoelastic",
    "Capsulorhexis",
    "Hydrodissection",
    "Phacoemulsification",
    "IrrigationAspiration",
    "CapsulePolishing",
    "LensImplantation",
    "LensPositioning",
    "ViscoelasticSuction",
    "TonifyingAntibiotics",
]

# Default label mapping used in the reference notebook
DEFAULT_LABEL_TO_IDX = {
    "Incision": 0,
    "Viscoelastic": 1,
    "Capsulorhexis": 2,
    "Hydrodissection": 3,
    "Phacoemulsification": 4,
    "IrrigationAspiration": 5,
    "CapsulePolishing": 6,
    "LensImplantation": 7,
    "LensPositioning": 8,
    "ViscoelasticSuction": 9,
    "AnteriorChamberFlushing": 1,  # Maps to Viscoelastic
    "TonifyingAntibiotics": 10,
}


if __name__ == "__main__":
    print("Comprehensive validation module loaded successfully!")
    print("Available functions:")
    print("  - validate_model(): Main validation function")
    print("  - analyze_model_metrics(): Analyze results from multiple models")
    print("  - strip_prefix(): Clean checkpoint state dicts")
    print("  - ValidationModule: PyTorch Lightning validation module")
