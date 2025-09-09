"""
Model evaluation utilities for surgical skill assessment.

This module provides comprehensive evaluation functionality including
classification reports, confusion matrices, and detailed prediction logging.

Author: Surgical Skill Assessment Team
Date: August 2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Dict,
    output_dirs: Dict[str, Path],
    class_names: List[str],
) -> Dict:
    """
    Comprehensive evaluation of a trained model on test data.

    Args:
        model: Trained neural network model
        loader: DataLoader for test data
        device: Device to run evaluation on (CPU/GPU)
        config: Configuration dictionary
        output_dirs: Dictionary of output directory paths
        class_names: List of class names for labeling

    Returns:
        Dict: Dictionary containing evaluation metrics and statistics

    Generates:
        - Classification report (text file)
        - Confusion matrix plot (PNG)
        - Detailed predictions (JSON file)
        - Performance summary (JSON file)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_metadata = []

    with torch.no_grad():
        for videos, labels, metadata in loader:
            videos, labels = videos.to(device), labels.to(device)

            with amp.autocast(enabled=config["hardware"]["mixed_precision"]):
                outputs = model(videos)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Transpose metadata
            for i in range(len(metadata["video_path"])):
                all_metadata.append({k: v[i] for k, v in metadata.items()})

    logger.info("Generating classification report...")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    logger.info(f"\nClassification Report:\n{report}")

    # Save classification report
    report_path = output_dirs["plots"] / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = output_dirs["plots"] / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

    # Detailed predictions
    if config["logging"]["save_detailed_predictions"]:
        predictions_data = []
        for i in range(len(all_preds)):
            predictions_data.append(
                {
                    "video_path": all_metadata[i]["video_path"],
                    "snippet_idx": all_metadata[i]["snippet_idx"],
                    "true_class": class_names[all_labels[i]],
                    "predicted_class": class_names[all_preds[i]],
                }
            )

        preds_path = output_dirs["predictions"] / "test_predictions.json"
        with open(preds_path, "w") as f:
            json.dump(predictions_data, f, indent=4)
        logger.info(f"Detailed predictions saved to {preds_path}")

    # Calculate and return summary metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    metrics_summary = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "num_samples": len(all_labels),
        "confusion_matrix": cm.tolist(),
    }

    # Save metrics summary
    metrics_path = output_dirs["predictions"] / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info(f"Evaluation metrics saved to {metrics_path}")

    return metrics_summary
