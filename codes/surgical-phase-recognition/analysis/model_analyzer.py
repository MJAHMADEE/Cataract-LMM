#!/usr/bin/env python3
"""
Model Analysis and Visualization Tools

This module provides comprehensive analysis tools for surgical phase recognition
models including performance analysis, feature visualization, error analysis,
and model comparison utilities.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

logger = logging.getLogger(__name__)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("plotly not available. Interactive plots will not work.")
    PLOTLY_AVAILABLE = False


class ModelAnalyzer:
    """
    Comprehensive model analysis and visualization toolkit.

    Provides tools for analyzing model performance, visualizing predictions,
    understanding feature representations, and comparing different models.

    Args:
        phase_names (List[str]): Names of surgical phases
        output_dir (str): Directory to save analysis outputs
    """

    def __init__(
        self, phase_names: List[str] = None, output_dir: str = "./analysis_outputs"
    ):
        if phase_names is None:
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
        else:
            self.phase_names = phase_names

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def analyze_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of model predictions.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray, optional): Prediction probabilities
            save_plots (bool): Whether to save plots to disk

        Returns:
            Dict[str, Any]: Analysis results
        """
        results = {}

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
        )

        results["overall_metrics"] = {
            "accuracy": accuracy,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        }

        # Per-class metrics
        results["per_class_metrics"] = {}
        for i, phase_name in enumerate(self.phase_names):
            if i < len(precision):
                results["per_class_metrics"][phase_name] = {
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1": f1[i],
                    "support": support[i],
                }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm

        if save_plots:
            # Plot confusion matrix
            self._plot_confusion_matrix(
                cm, save_path=self.output_dir / "confusion_matrix.png"
            )

            # Plot per-class metrics
            self._plot_per_class_metrics(
                results["per_class_metrics"],
                save_path=self.output_dir / "per_class_metrics.png",
            )

            # Plot probability distributions if available
            if y_prob is not None:
                self._plot_probability_distributions(
                    y_true,
                    y_prob,
                    save_path=self.output_dir / "probability_distributions.png",
                )

        # Classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.phase_names[: len(np.unique(y_true))],
            output_dict=True,
            zero_division=0,
        )
        results["classification_report"] = report

        logger.info(f"Analysis completed. Overall accuracy: {accuracy:.4f}")
        return results

    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Path = None):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(12, 10))

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.phase_names[: cm.shape[1]],
            yticklabels=self.phase_names[: cm.shape[0]],
            cbar_kws={"label": "Normalized Frequency"},
        )

        plt.title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Phase", fontsize=12)
        plt.ylabel("True Phase", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def _plot_per_class_metrics(
        self, per_class_metrics: Dict[str, Dict], save_path: Path = None
    ):
        """Plot per-class performance metrics."""
        phases = list(per_class_metrics.keys())
        metrics = ["precision", "recall", "f1"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, metric in enumerate(metrics):
            values = [per_class_metrics[phase][metric] for phase in phases]

            bars = axes[i].bar(range(len(phases)), values, alpha=0.7)
            axes[i].set_title(
                f"{metric.capitalize()} by Phase", fontsize=14, fontweight="bold"
            )
            axes[i].set_xlabel("Surgical Phase", fontsize=12)
            axes[i].set_ylabel(metric.capitalize(), fontsize=12)
            axes[i].set_xticks(range(len(phases)))
            axes[i].set_xticklabels(phases, rotation=45, ha="right")
            axes[i].set_ylim(0, 1)
            axes[i].grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Per-class metrics plot saved to {save_path}")

        plt.show()

    def _plot_probability_distributions(
        self, y_true: np.ndarray, y_prob: np.ndarray, save_path: Path = None
    ):
        """Plot prediction probability distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        # Plot 1: Max probability distribution
        max_probs = np.max(y_prob, axis=1)
        correct_predictions = y_true == np.argmax(y_prob, axis=1)

        axes[0].hist(
            max_probs[correct_predictions],
            bins=30,
            alpha=0.7,
            label="Correct",
            density=True,
        )
        axes[0].hist(
            max_probs[~correct_predictions],
            bins=30,
            alpha=0.7,
            label="Incorrect",
            density=True,
        )
        axes[0].set_title("Max Probability Distribution", fontweight="bold")
        axes[0].set_xlabel("Max Probability")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot 2: Confidence vs Accuracy
        confidence_bins = np.linspace(0, 1, 21)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []

        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (
                max_probs < confidence_bins[i + 1]
            )
            if mask.sum() > 0:
                bin_accuracies.append(correct_predictions[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        axes[1].plot(bin_centers, bin_accuracies, "o-", linewidth=2, markersize=6)
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
        axes[1].set_title("Reliability Diagram", fontweight="bold")
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Plot 3: Entropy distribution
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-8), axis=1)

        axes[2].hist(
            entropy[correct_predictions],
            bins=30,
            alpha=0.7,
            label="Correct",
            density=True,
        )
        axes[2].hist(
            entropy[~correct_predictions],
            bins=30,
            alpha=0.7,
            label="Incorrect",
            density=True,
        )
        axes[2].set_title("Prediction Entropy Distribution", fontweight="bold")
        axes[2].set_xlabel("Entropy")
        axes[2].set_ylabel("Density")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        # Plot 4: Phase-wise accuracy
        phase_accuracies = []
        for phase_idx in range(len(self.phase_names)):
            if phase_idx in y_true:
                phase_mask = y_true == phase_idx
                if phase_mask.sum() > 0:
                    phase_acc = correct_predictions[phase_mask].mean()
                    phase_accuracies.append(phase_acc)
                else:
                    phase_accuracies.append(0)
            else:
                phase_accuracies.append(0)

        bars = axes[3].bar(range(len(self.phase_names)), phase_accuracies, alpha=0.7)
        axes[3].set_title("Accuracy by Phase", fontweight="bold")
        axes[3].set_xlabel("Surgical Phase")
        axes[3].set_ylabel("Accuracy")
        axes[3].set_xticks(range(len(self.phase_names)))
        axes[3].set_xticklabels(self.phase_names, rotation=45, ha="right")
        axes[3].grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Probability distributions plot saved to {save_path}")

        plt.show()

    def analyze_feature_representations(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze and visualize feature representations.

        Args:
            features (np.ndarray): Feature representations
            labels (np.ndarray): Corresponding labels
            method (str): Dimensionality reduction method ('tsne', 'pca')
            save_plots (bool): Whether to save plots

        Returns:
            Dict[str, Any]: Analysis results
        """
        results = {}

        # Dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = reducer.fit_transform(features)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features)
            results["explained_variance_ratio"] = reducer.explained_variance_ratio_
        else:
            raise ValueError(f"Unsupported method: {method}")

        results["features_2d"] = features_2d
        results["method"] = method

        if save_plots:
            self._plot_feature_space(
                features_2d,
                labels,
                method,
                save_path=self.output_dir / f"feature_space_{method}.png",
            )

        # Compute feature statistics
        results["feature_stats"] = {
            "mean": np.mean(features, axis=0),
            "std": np.std(features, axis=0),
            "min": np.min(features, axis=0),
            "max": np.max(features, axis=0),
        }

        return results

    def _plot_feature_space(
        self,
        features_2d: np.ndarray,
        labels: np.ndarray,
        method: str,
        save_path: Path = None,
    ):
        """Plot feature space visualization."""
        plt.figure(figsize=(12, 8))

        # Create color map
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=(
                    self.phase_names[label]
                    if label < len(self.phase_names)
                    else f"Phase {label}"
                ),
                alpha=0.6,
                s=50,
            )

        plt.title(
            f"Feature Space Visualization ({method.upper()})",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
        plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature space plot saved to {save_path}")

        plt.tight_layout()
        plt.show()

    def compare_models(
        self, model_results: Dict[str, Dict[str, Any]], save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple models.

        Args:
            model_results (Dict): Results from different models
            save_plots (bool): Whether to save comparison plots

        Returns:
            Dict[str, Any]: Comparison analysis
        """
        comparison = {
            "model_names": list(model_results.keys()),
            "metric_comparison": {},
            "best_models": {},
        }

        # Extract metrics for comparison
        metrics = ["accuracy", "precision", "recall", "f1"]

        for metric in metrics:
            scores = {}
            for model_name, results in model_results.items():
                if (
                    "overall_metrics" in results
                    and metric in results["overall_metrics"]
                ):
                    scores[model_name] = results["overall_metrics"][metric]

            comparison["metric_comparison"][metric] = scores

            if scores:
                best_model = max(scores, key=scores.get)
                comparison["best_models"][metric] = {
                    "model": best_model,
                    "score": scores[best_model],
                }

        if save_plots:
            self._plot_model_comparison(
                comparison["metric_comparison"],
                save_path=self.output_dir / "model_comparison.png",
            )

        return comparison

    def _plot_model_comparison(
        self, metric_comparison: Dict[str, Dict[str, float]], save_path: Path = None
    ):
        """Plot model comparison chart."""
        metrics = list(metric_comparison.keys())
        model_names = list(next(iter(metric_comparison.values())).keys())

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        for i, model_name in enumerate(model_names):
            scores = [
                metric_comparison[metric].get(model_name, 0) for metric in metrics
            ]
            offset = (i - len(model_names) / 2) * width + width / 2

            bars = ax.bar(x + offset, scores, width, label=model_name, alpha=0.8)

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
        ax.set_xlabel("Metrics", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()

    def generate_analysis_report(
        self, analysis_results: Dict[str, Any], output_path: str = None
    ) -> str:
        """
        Generate a comprehensive analysis report.

        Args:
            analysis_results (Dict): Analysis results
            output_path (str, optional): Path to save the report

        Returns:
            str: Analysis report as markdown text
        """
        if output_path is None:
            output_path = self.output_dir / "analysis_report.md"

        report_lines = [
            "# Surgical Phase Recognition Analysis Report",
            "",
            f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]

        # Overall metrics
        if "overall_metrics" in analysis_results:
            metrics = analysis_results["overall_metrics"]
            report_lines.extend(
                [
                    "### Overall Performance",
                    "",
                    f"- **Accuracy:** {metrics['accuracy']:.4f}",
                    f"- **Precision:** {metrics['precision']:.4f}",
                    f"- **Recall:** {metrics['recall']:.4f}",
                    f"- **F1-Score:** {metrics['f1']:.4f}",
                    "",
                ]
            )

        # Per-class performance
        if "per_class_metrics" in analysis_results:
            report_lines.extend(
                [
                    "### Per-Class Performance",
                    "",
                    "| Phase | Precision | Recall | F1-Score | Support |",
                    "|-------|-----------|--------|----------|---------|",
                ]
            )

            for phase, metrics in analysis_results["per_class_metrics"].items():
                report_lines.append(
                    f"| {phase} | {metrics['precision']:.3f} | "
                    f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | "
                    f"{metrics['support']} |"
                )

            report_lines.append("")

        # Model recommendations
        report_lines.extend(
            [
                "## Recommendations",
                "",
                "### Strengths",
                "- High-performing phases and model capabilities",
                "",
                "### Areas for Improvement",
                "- Phases with lower performance that need attention",
                "",
                "### Next Steps",
                "- Suggested improvements and model optimizations",
                "",
            ]
        )

        # Save report
        report_text = "\n".join(report_lines)

        with open(output_path, "w") as f:
            f.write(report_text)

        logger.info(f"Analysis report saved to {output_path}")
        return report_text


if __name__ == "__main__":
    # Test the analyzer
    analyzer = ModelAnalyzer()

    # Generate dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    n_classes = 11

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)

    # Test analysis
    results = analyzer.analyze_predictions(y_true, y_pred, y_prob, save_plots=False)

    print("Analysis completed successfully!")
    print(f"Overall accuracy: {results['overall_metrics']['accuracy']:.4f}")

    # Test feature analysis
    features = np.random.randn(n_samples, 128)  # 128-dimensional features
    feature_results = analyzer.analyze_feature_representations(
        features, y_true, method="pca", save_plots=False
    )

    print("Feature analysis completed!")
    print("Analysis tools ready for use!")
