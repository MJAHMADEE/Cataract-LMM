"""
Visualization Utilities for Surgical Instance Segmentation

This module provides comprehensive visualization capabilities for all aspects
of the segmentation framework:
- Dataset visualization and exploration
- Model prediction visualization
- Training progress monitoring
- Evaluation result visualization
- Interactive analysis tools

Compatible with outputs from all model types (Mask R-CNN, YOLO, SAM).
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Polygon

    plotting_available = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Visualization will be limited.")
    plotting_available = False

try:
    import cv2

    cv2_available = True
except ImportError:
    print("Warning: OpenCV not installed. Some visualization features will be limited.")
    cv2_available = False

try:
    from PIL import Image, ImageDraw, ImageFont

    pil_available = True
except ImportError:
    print("Warning: PIL not installed. Image processing will be limited.")
    pil_available = False

try:
    from pycocotools import mask as coco_mask

    coco_available = True
except ImportError:
    coco_available = False

# Set style for consistent plots
if plotting_available:
    plt.style.use("default")
    sns.set_palette("husl")


class SegmentationVisualizer:
    """
    Comprehensive visualization toolkit for surgical instance segmentation

    This class provides visualization capabilities for:
    - Dataset exploration and analysis
    - Model prediction visualization
    - Training progress monitoring
    - Evaluation result presentation
    - Interactive analysis and comparison
    """

    def __init__(
        self,
        output_dir: str = "./visualizations",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
    ):
        """
        Initialize visualization toolkit

        Args:
            output_dir (str): Directory to save visualization outputs
            figsize (Tuple[int, int]): Default figure size
            dpi (int): DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figsize = figsize
        self.dpi = dpi

        # Color schemes for different visualization types
        self.class_colors = self._generate_class_colors()
        self.prediction_colors = {
            "true_positive": "#2ecc71",  # Green
            "false_positive": "#e74c3c",  # Red
            "false_negative": "#f39c12",  # Orange
            "ground_truth": "#3498db",  # Blue
        }

        print(f"Visualization toolkit initialized. Output directory: {self.output_dir}")

    def _generate_class_colors(self, num_classes: int = 20) -> List[str]:
        """Generate distinct colors for different classes"""
        if plotting_available:
            # Use seaborn color palette for consistent colors
            colors = sns.color_palette("husl", num_classes)
            return [
                f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                for r, g, b in colors
            ]
        else:
            # Fallback colors
            return [
                "#FF0000",
                "#00FF00",
                "#0000FF",
                "#FFFF00",
                "#FF00FF",
                "#00FFFF",
                "#800000",
                "#008000",
                "#000080",
                "#808000",
            ]

    def visualize_dataset_overview(
        self,
        annotations_file: str,
        images_dir: str,
        max_samples: int = 16,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create dataset overview visualization with sample images and statistics

        Args:
            annotations_file (str): Path to COCO annotations file
            images_dir (str): Directory containing images
            max_samples (int): Maximum number of sample images to show
            save_path (str): Path to save visualization (optional)
        """
        if not plotting_available:
            print("Matplotlib not available for dataset visualization")
            return

        print("Creating dataset overview visualization...")

        # Load annotations
        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        # Extract statistics
        stats = self._compute_dataset_statistics(coco_data)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Dataset statistics (top left)
        ax1 = plt.subplot(3, 4, (1, 2))
        self._plot_dataset_statistics(ax1, stats)

        # 2. Class distribution (top right)
        ax2 = plt.subplot(3, 4, (3, 4))
        self._plot_class_distribution(ax2, stats)

        # 3. Sample images with annotations (bottom 2 rows)
        sample_images = random.sample(
            coco_data["images"], min(max_samples, len(coco_data["images"]))
        )

        for i, img_info in enumerate(sample_images[:8]):
            ax = plt.subplot(3, 4, i + 5)
            self._plot_sample_image_with_annotations(
                ax, img_info, coco_data, images_dir
            )

        plt.tight_layout()

        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "dataset_overview.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"Dataset overview saved to: {save_path}")

    def visualize_predictions(
        self,
        image: Union[str, np.ndarray],
        predictions: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
        model_name: str = "Model",
        save_path: Optional[str] = None,
        show_confidence: bool = True,
        confidence_threshold: float = 0.5,
    ) -> None:
        """
        Visualize model predictions on an image

        Args:
            image: Input image (path or array)
            predictions: Model predictions
            ground_truth: Ground truth annotations (optional)
            model_name: Name of the model for title
            save_path: Path to save visualization
            show_confidence: Whether to show confidence scores
            confidence_threshold: Minimum confidence to display
        """
        if not plotting_available:
            print("Matplotlib not available for prediction visualization")
            return

        # Load image
        if isinstance(image, str):
            if cv2_available:
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.array(Image.open(image))
        else:
            img = image

        # Create figure
        num_plots = 2 if ground_truth else 1
        fig, axes = plt.subplots(
            1, num_plots, figsize=(self.figsize[0] * num_plots, self.figsize[1])
        )

        if num_plots == 1:
            axes = [axes]

        # Plot predictions
        self._plot_predictions_on_image(
            axes[0],
            img,
            predictions,
            f"{model_name} Predictions",
            show_confidence,
            confidence_threshold,
        )

        # Plot ground truth if available
        if ground_truth:
            self._plot_predictions_on_image(
                axes[1], img, ground_truth, "Ground Truth", False, 0.0
            )

        plt.tight_layout()

        # Save visualization
        if save_path is None:
            timestamp = int(time.time())
            save_path = self.output_dir / f"predictions_{timestamp}.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"Prediction visualization saved to: {save_path}")

    def visualize_training_progress(
        self,
        training_history: Dict[str, List[float]],
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize training progress with loss and metric curves

        Args:
            training_history: Dictionary containing training metrics over epochs
            model_name: Name of the model
            save_path: Path to save visualization
        """
        if not plotting_available:
            print("Matplotlib not available for training progress visualization")
            return

        # Determine number of subplots based on available metrics
        metrics = list(training_history.keys())
        num_metrics = len(metrics)

        if num_metrics == 0:
            print("No training history available for visualization")
            return

        # Create subplots
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot each metric
        for i, (metric_name, values) in enumerate(training_history.items()):
            if i >= len(axes):
                break

            ax = axes[i]
            epochs = range(1, len(values) + 1)

            ax.plot(epochs, values, "b-", linewidth=2, label=metric_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name.replace("_", " ").title())
            ax.set_title(f'{model_name} - {metric_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide unused subplots
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save visualization
        if save_path is None:
            save_path = self.output_dir / f"training_progress_{model_name.lower()}.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"Training progress visualization saved to: {save_path}")

    def visualize_evaluation_results(
        self,
        evaluation_results: Dict[str, Any],
        model_names: List[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize evaluation results with metrics comparison

        Args:
            evaluation_results: Dictionary containing evaluation metrics
            model_names: List of model names for comparison
            save_path: Path to save visualization
        """
        if not plotting_available:
            print("Matplotlib not available for evaluation visualization")
            return

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 1. COCO metrics comparison
        if "coco_evaluation" in evaluation_results:
            self._plot_coco_metrics(axes[0], evaluation_results["coco_evaluation"])

        # 2. Per-class performance
        if "per_class_metrics" in evaluation_results:
            self._plot_per_class_metrics(
                axes[1], evaluation_results["per_class_metrics"]
            )

        # 3. Confusion matrix
        if "confusion_matrix" in evaluation_results:
            self._plot_confusion_matrix(axes[2], evaluation_results["confusion_matrix"])

        # 4. Precision-recall curve
        if "precision_recall" in evaluation_results:
            self._plot_precision_recall_curve(
                axes[3], evaluation_results["precision_recall"]
            )

        plt.tight_layout()

        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "evaluation_results.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"Evaluation results visualization saved to: {save_path}")

    def create_model_comparison(
        self,
        model_results: Dict[str, Dict[str, Any]],
        metrics: List[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive comparison visualization between models

        Args:
            model_results: Dictionary with model names as keys and results as values
            metrics: List of metrics to compare
            save_path: Path to save visualization
        """
        if not plotting_available:
            print("Matplotlib not available for model comparison")
            return

        if metrics is None:
            metrics = ["AP", "AP_50", "AP_75", "precision", "recall", "f1_score"]

        # Extract available metrics from results
        available_metrics = []
        for metric in metrics:
            if any(metric in results for results in model_results.values()):
                available_metrics.append(metric)

        if not available_metrics:
            print("No common metrics found for comparison")
            return

        # Create comparison plots
        num_plots = len(available_metrics)
        cols = min(3, num_plots)
        rows = (num_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plot each metric comparison
        for i, metric in enumerate(available_metrics):
            if i >= len(axes):
                break

            ax = axes[i]
            model_names = []
            metric_values = []

            for model_name, results in model_results.items():
                if metric in results:
                    model_names.append(model_name)
                    metric_values.append(results[metric])

            if metric_values:
                bars = ax.bar(
                    model_names,
                    metric_values,
                    color=self.class_colors[: len(model_names)],
                )
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_ylabel(metric.replace("_", " ").title())

                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

                ax.set_ylim(0, max(metric_values) * 1.2)

        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "model_comparison.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"Model comparison visualization saved to: {save_path}")

    def _compute_dataset_statistics(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute dataset statistics from COCO annotations"""
        stats = {
            "num_images": len(coco_data["images"]),
            "num_annotations": len(coco_data["annotations"]),
            "num_categories": len(coco_data["categories"]),
            "category_counts": defaultdict(int),
            "image_sizes": [],
            "annotation_areas": [],
        }

        # Category name mapping
        category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

        # Count annotations per category
        for ann in coco_data["annotations"]:
            cat_name = category_map.get(ann["category_id"], "unknown")
            stats["category_counts"][cat_name] += 1
            stats["annotation_areas"].append(ann["area"])

        # Image size statistics
        for img in coco_data["images"]:
            stats["image_sizes"].append((img["width"], img["height"]))

        return stats

    def _plot_dataset_statistics(self, ax, stats: Dict[str, Any]) -> None:
        """Plot basic dataset statistics"""
        # Create text summary
        text_stats = [
            f"Total Images: {stats['num_images']:,}",
            f"Total Annotations: {stats['num_annotations']:,}",
            f"Categories: {stats['num_categories']}",
            f"Avg Annotations/Image: {stats['num_annotations']/stats['num_images']:.1f}",
        ]

        if stats["annotation_areas"]:
            avg_area = np.mean(stats["annotation_areas"])
            text_stats.append(f"Avg Annotation Area: {avg_area:.0f} px²")

        ax.text(
            0.1,
            0.9,
            "\n".join(text_stats),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Dataset Statistics", fontsize=16, fontweight="bold")
        ax.axis("off")

    def _plot_class_distribution(self, ax, stats: Dict[str, Any]) -> None:
        """Plot class distribution"""
        categories = list(stats["category_counts"].keys())
        counts = list(stats["category_counts"].values())

        if categories:
            bars = ax.bar(
                categories, counts, color=self.class_colors[: len(categories)]
            )
            ax.set_title("Class Distribution", fontsize=16, fontweight="bold")
            ax.set_xlabel("Categories")
            ax.set_ylabel("Number of Annotations")

            # Rotate x-axis labels if too many categories
            if len(categories) > 5:
                ax.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(counts) * 0.01,
                    f"{count}",
                    ha="center",
                    va="bottom",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No category data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Class Distribution")

    def _plot_sample_image_with_annotations(
        self, ax, img_info: Dict[str, Any], coco_data: Dict[str, Any], images_dir: str
    ) -> None:
        """Plot sample image with annotations"""
        # Load image
        img_path = os.path.join(images_dir, img_info["file_name"])

        try:
            if cv2_available:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.array(Image.open(img_path))
        except:
            # Create placeholder if image can't be loaded
            img = np.zeros((img_info["height"], img_info["width"], 3), dtype=np.uint8)

        ax.imshow(img)

        # Find annotations for this image
        image_annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]
        ]

        # Draw annotations
        for ann in image_annotations:
            if "bbox" in ann:
                x, y, w, h = ann["bbox"]
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)

        ax.set_title(f"{img_info['file_name']}\n{len(image_annotations)} annotations")
        ax.axis("off")

    def _plot_predictions_on_image(
        self,
        ax,
        img: np.ndarray,
        predictions: Dict[str, Any],
        title: str,
        show_confidence: bool,
        confidence_threshold: float,
    ) -> None:
        """Plot predictions on image"""
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

        # Draw bounding boxes
        if "boxes" in predictions and len(predictions["boxes"]) > 0:
            boxes = predictions["boxes"]
            scores = predictions.get("scores", [1.0] * len(boxes))
            labels = predictions.get("labels", [0] * len(boxes))

            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score < confidence_threshold:
                    continue

                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                # Choose color based on label
                color = self.class_colors[label % len(self.class_colors)]

                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add label and confidence
                if show_confidence:
                    label_text = f"{label}: {score:.2f}"
                else:
                    label_text = str(label)

                ax.text(
                    x1,
                    y1 - 5,
                    label_text,
                    color=color,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

        # Draw masks
        if "masks" in predictions and len(predictions["masks"]) > 0:
            masks = predictions["masks"]
            scores = predictions.get("scores", [1.0] * len(masks))

            for i, (mask, score) in enumerate(zip(masks, scores)):
                if score < confidence_threshold:
                    continue

                if mask.ndim == 3:
                    mask = mask[0]  # Take first mask if 3D

                # Create colored overlay
                color = np.array(self.class_colors[i % len(self.class_colors)][1:], "h")
                color = tuple(int(color[j : j + 2], 16) for j in (0, 2, 4))

                # Apply mask overlay
                overlay = np.zeros_like(img)
                overlay[mask > 0.5] = color

                # Blend with original image
                ax.imshow(overlay, alpha=0.3)

    def _plot_coco_metrics(self, ax, coco_results: Dict[str, Any]) -> None:
        """Plot COCO evaluation metrics"""
        if "metrics" not in coco_results:
            ax.text(
                0.5,
                0.5,
                "No COCO metrics available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("COCO Metrics")
            return

        metrics = coco_results["metrics"]
        metric_names = ["AP", "AP_50", "AP_75", "AR_1", "AR_10", "AR_100"]
        values = [metrics.get(name, 0) for name in metric_names]

        bars = ax.bar(
            metric_names, values, color=self.class_colors[: len(metric_names)]
        )
        ax.set_title("COCO Evaluation Metrics")
        ax.set_ylabel("Score")

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        ax.set_ylim(0, 1.0)

    def _plot_per_class_metrics(self, ax, per_class_results: Dict[str, Any]) -> None:
        """Plot per-class performance metrics"""
        # Placeholder implementation
        ax.text(
            0.5,
            0.5,
            "Per-class metrics\n(Implementation needed)",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title("Per-Class Performance")

    def _plot_confusion_matrix(self, ax, confusion_matrix: np.ndarray) -> None:
        """Plot confusion matrix"""
        if plotting_available:
            im = ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
            ax.set_title("Confusion Matrix")

            # Add colorbar
            plt.colorbar(im, ax=ax)

            # Add text annotations
            thresh = confusion_matrix.max() / 2.0
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{confusion_matrix[i, j]:.0f}",
                        ha="center",
                        va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black",
                    )
        else:
            ax.text(
                0.5,
                0.5,
                "Confusion Matrix\n(matplotlib required)",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Confusion Matrix")

    def _plot_precision_recall_curve(self, ax, pr_data: Dict[str, Any]) -> None:
        """Plot precision-recall curve"""
        if "precision" in pr_data and "recall" in pr_data:
            precision = pr_data["precision"]
            recall = pr_data["recall"]

            ax.plot(recall, precision, "b-", linewidth=2)
            ax.fill_between(recall, precision, alpha=0.3)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            # Add AP score if available
            if "ap" in pr_data:
                ax.text(
                    0.05,
                    0.95,
                    f'AP = {pr_data["ap"]:.3f}',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Precision-Recall data\nnot available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("Precision-Recall Curve")


def create_comprehensive_visualization_report(
    results_dir: str, output_dir: str = "./visualization_report"
) -> None:
    """
    Create comprehensive visualization report from results directory

    Args:
        results_dir (str): Directory containing model results
        output_dir (str): Directory to save visualization report
    """
    print("Creating comprehensive visualization report...")

    visualizer = SegmentationVisualizer(output_dir)

    # Find and process all result files
    results_path = Path(results_dir)

    # Look for training results
    training_files = list(results_path.glob("**/training_history.json"))
    for training_file in training_files:
        with open(training_file, "r") as f:
            training_data = json.load(f)

        model_name = training_file.parent.name
        visualizer.visualize_training_progress(training_data, model_name)

    # Look for evaluation results
    eval_files = list(results_path.glob("**/evaluation_results.json"))
    for eval_file in eval_files:
        with open(eval_file, "r") as f:
            eval_data = json.load(f)

        visualizer.visualize_evaluation_results(eval_data)

    print(f"Visualization report created in: {output_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = SegmentationVisualizer("./visualization_output")

    # Example dataset visualization
    # visualizer.visualize_dataset_overview(
    #     "/path/to/annotations.json",
    #     "/path/to/images",
    #     max_samples=16
    # )

    # Example prediction visualization
    # predictions = {
    #     'boxes': [[100, 100, 200, 200], [300, 150, 400, 250]],
    #     'scores': [0.9, 0.8],
    #     'labels': [1, 2],
    #     'masks': [np.random.rand(100, 100) > 0.5, np.random.rand(100, 100) > 0.5]
    # }
    #
    # visualizer.visualize_predictions(
    #     "/path/to/image.jpg",
    #     predictions,
    #     model_name="YOLO"
    # )

    print("Visualization utilities ready for use.")
