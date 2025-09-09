#!/usr/bin/env python3
"""
Error Analysis Tools for Surgical Phase Recognition

This module provides specialized tools for analyzing model errors,
identifying failure patterns, and suggesting improvements.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Specialized error analysis for surgical phase recognition.

    Provides tools for identifying error patterns, analyzing misclassifications,
    and understanding model failure modes.

    Args:
        phase_names (List[str]): Names of surgical phases
        output_dir (str): Directory to save analysis outputs
    """

    def __init__(
        self, phase_names: List[str] = None, output_dir: str = "./error_analysis"
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

    def analyze_misclassifications(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze misclassification patterns.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray, optional): Prediction probabilities
            sample_ids (List[str], optional): Sample identifiers

        Returns:
            Dict[str, Any]: Misclassification analysis results
        """
        results = {}

        # Identify misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]

        results["total_samples"] = len(y_true)
        results["misclassified_count"] = len(misclassified_indices)
        results["error_rate"] = len(misclassified_indices) / len(y_true)

        # Misclassification patterns
        misclass_patterns = defaultdict(list)
        for idx in misclassified_indices:
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            pattern = (true_label, pred_label)

            error_info = {
                "sample_index": idx,
                "true_phase": (
                    self.phase_names[true_label]
                    if true_label < len(self.phase_names)
                    else f"Phase {true_label}"
                ),
                "predicted_phase": (
                    self.phase_names[pred_label]
                    if pred_label < len(self.phase_names)
                    else f"Phase {pred_label}"
                ),
            }

            if sample_ids is not None and idx < len(sample_ids):
                error_info["sample_id"] = sample_ids[idx]

            if y_prob is not None:
                error_info["prediction_confidence"] = y_prob[idx, pred_label]
                error_info["true_class_probability"] = y_prob[idx, true_label]
                error_info["prediction_entropy"] = -np.sum(
                    y_prob[idx] * np.log(y_prob[idx] + 1e-8)
                )

            misclass_patterns[pattern].append(error_info)

        results["misclassification_patterns"] = dict(misclass_patterns)

        # Most common error patterns
        pattern_counts = Counter(misclass_patterns.keys())
        results["top_error_patterns"] = [
            {
                "true_phase": (
                    self.phase_names[true_idx]
                    if true_idx < len(self.phase_names)
                    else f"Phase {true_idx}"
                ),
                "predicted_phase": (
                    self.phase_names[pred_idx]
                    if pred_idx < len(self.phase_names)
                    else f"Phase {pred_idx}"
                ),
                "count": count,
                "percentage": count / results["misclassified_count"] * 100,
            }
            for (true_idx, pred_idx), count in pattern_counts.most_common(10)
        ]

        # Phase-wise error analysis
        results["phase_error_analysis"] = {}
        for phase_idx, phase_name in enumerate(self.phase_names):
            if phase_idx in y_true:
                phase_mask = y_true == phase_idx
                phase_total = phase_mask.sum()
                phase_errors = misclassified_mask[phase_mask].sum()

                if phase_total > 0:
                    results["phase_error_analysis"][phase_name] = {
                        "total_samples": phase_total,
                        "errors": phase_errors,
                        "error_rate": phase_errors / phase_total,
                        "accuracy": 1 - (phase_errors / phase_total),
                    }

        return results

    def analyze_confidence_errors(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze relationship between prediction confidence and errors.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Prediction probabilities

        Returns:
            Dict[str, Any]: Confidence-error analysis results
        """
        results = {}

        # Calculate confidence metrics
        max_probs = np.max(y_prob, axis=1)
        correct_predictions = y_true == y_pred

        # Confidence bins analysis
        confidence_bins = np.linspace(0, 1, 11)
        bin_analysis = []

        for i in range(len(confidence_bins) - 1):
            bin_start, bin_end = confidence_bins[i], confidence_bins[i + 1]
            bin_mask = (max_probs >= bin_start) & (max_probs < bin_end)

            if bin_mask.sum() > 0:
                bin_accuracy = correct_predictions[bin_mask].mean()
                bin_count = bin_mask.sum()

                bin_analysis.append(
                    {
                        "confidence_range": f"{bin_start:.1f}-{bin_end:.1f}",
                        "sample_count": bin_count,
                        "accuracy": bin_accuracy,
                        "error_rate": 1 - bin_accuracy,
                    }
                )

        results["confidence_bin_analysis"] = bin_analysis

        # High confidence errors (overconfident mistakes)
        high_conf_threshold = 0.8
        high_conf_errors = (max_probs >= high_conf_threshold) & (~correct_predictions)

        results["high_confidence_errors"] = {
            "count": high_conf_errors.sum(),
            "percentage_of_errors": high_conf_errors.sum()
            / (~correct_predictions).sum()
            * 100,
            "indices": np.where(high_conf_errors)[0].tolist(),
        }

        # Low confidence correct predictions
        low_conf_threshold = 0.5
        low_conf_correct = (max_probs <= low_conf_threshold) & correct_predictions

        results["low_confidence_correct"] = {
            "count": low_conf_correct.sum(),
            "percentage_of_correct": low_conf_correct.sum()
            / correct_predictions.sum()
            * 100,
            "indices": np.where(low_conf_correct)[0].tolist(),
        }

        return results

    def find_difficult_samples(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Identify the most difficult samples for the model.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Prediction probabilities
            sample_ids (List[str], optional): Sample identifiers
            top_k (int): Number of difficult samples to return

        Returns:
            Dict[str, Any]: Difficult samples analysis
        """
        results = {}

        # Calculate difficulty scores
        true_class_probs = y_prob[np.arange(len(y_true)), y_true]
        prediction_entropy = -np.sum(y_prob * np.log(y_prob + 1e-8), axis=1)

        # Different difficulty metrics
        difficulty_metrics = {
            "low_true_class_probability": -true_class_probs,  # Lower is more difficult
            "high_entropy": prediction_entropy,  # Higher is more difficult
            "misclassified_with_confidence": np.where(
                y_true != y_pred, np.max(y_prob, axis=1), 0  # High confidence mistakes
            ),
        }

        results["difficult_samples"] = {}

        for metric_name, scores in difficulty_metrics.items():
            # Get top difficult samples
            top_indices = np.argsort(scores)[-top_k:][::-1]

            difficult_samples = []
            for idx in top_indices:
                sample_info = {
                    "index": int(idx),
                    "difficulty_score": float(scores[idx]),
                    "true_phase": (
                        self.phase_names[y_true[idx]]
                        if y_true[idx] < len(self.phase_names)
                        else f"Phase {y_true[idx]}"
                    ),
                    "predicted_phase": (
                        self.phase_names[y_pred[idx]]
                        if y_pred[idx] < len(self.phase_names)
                        else f"Phase {y_pred[idx]}"
                    ),
                    "is_correct": bool(y_true[idx] == y_pred[idx]),
                    "prediction_confidence": float(np.max(y_prob[idx])),
                    "true_class_probability": float(true_class_probs[idx]),
                    "entropy": float(prediction_entropy[idx]),
                }

                if sample_ids is not None and idx < len(sample_ids):
                    sample_info["sample_id"] = sample_ids[idx]

                difficult_samples.append(sample_info)

            results["difficult_samples"][metric_name] = difficult_samples

        return results

    def analyze_temporal_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        temporal_info: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze errors in temporal context (if temporal information is available).

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            temporal_info (List[Dict], optional): Temporal information for each sample

        Returns:
            Dict[str, Any]: Temporal error analysis
        """
        results = {}

        if temporal_info is None:
            logger.warning(
                "No temporal information provided. Skipping temporal analysis."
            )
            return results

        # Group samples by video/sequence
        video_groups = defaultdict(list)
        for i, info in enumerate(temporal_info):
            if "video_id" in info:
                video_groups[info["video_id"]].append(i)

        # Analyze errors within videos
        video_error_analysis = {}
        for video_id, indices in video_groups.items():
            video_true = y_true[indices]
            video_pred = y_pred[indices]

            video_errors = video_true != video_pred
            error_count = video_errors.sum()
            total_count = len(indices)

            # Find error clusters (consecutive errors)
            error_clusters = []
            in_cluster = False
            cluster_start = None

            for i, is_error in enumerate(video_errors):
                if is_error and not in_cluster:
                    in_cluster = True
                    cluster_start = i
                elif not is_error and in_cluster:
                    in_cluster = False
                    error_clusters.append((cluster_start, i - 1))

            if in_cluster:  # Handle cluster that goes to the end
                error_clusters.append((cluster_start, len(video_errors) - 1))

            video_error_analysis[video_id] = {
                "total_frames": total_count,
                "error_frames": error_count,
                "error_rate": error_count / total_count if total_count > 0 else 0,
                "error_clusters": error_clusters,
                "max_cluster_length": (
                    max([end - start + 1 for start, end in error_clusters])
                    if error_clusters
                    else 0
                ),
            }

        results["video_error_analysis"] = video_error_analysis

        # Overall temporal statistics
        all_error_rates = [info["error_rate"] for info in video_error_analysis.values()]
        results["temporal_statistics"] = {
            "mean_video_error_rate": np.mean(all_error_rates),
            "std_video_error_rate": np.std(all_error_rates),
            "videos_with_high_errors": sum(1 for rate in all_error_rates if rate > 0.5),
            "total_videos": len(all_error_rates),
        }

        return results

    def generate_error_report(
        self, error_analysis: Dict[str, Any], output_path: str = None
    ) -> str:
        """
        Generate a comprehensive error analysis report.

        Args:
            error_analysis (Dict): Error analysis results
            output_path (str, optional): Path to save the report

        Returns:
            str: Error analysis report
        """
        if output_path is None:
            output_path = self.output_dir / "error_analysis_report.md"

        report_lines = [
            "# Error Analysis Report",
            "",
            f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Error Overview",
            "",
        ]

        # Basic error statistics
        if "error_rate" in error_analysis:
            report_lines.extend(
                [
                    f"- **Total Samples:** {error_analysis['total_samples']:,}",
                    f"- **Misclassified Samples:** {error_analysis['misclassified_count']:,}",
                    f"- **Overall Error Rate:** {error_analysis['error_rate']:.4f} ({error_analysis['error_rate']*100:.2f}%)",
                    "",
                ]
            )

        # Top error patterns
        if "top_error_patterns" in error_analysis:
            report_lines.extend(
                [
                    "## Most Common Error Patterns",
                    "",
                    "| True Phase | Predicted Phase | Count | Percentage |",
                    "|------------|-----------------|-------|------------|",
                ]
            )

            for pattern in error_analysis["top_error_patterns"][:5]:
                report_lines.append(
                    f"| {pattern['true_phase']} | {pattern['predicted_phase']} | "
                    f"{pattern['count']} | {pattern['percentage']:.1f}% |"
                )

            report_lines.append("")

        # Phase-wise error analysis
        if "phase_error_analysis" in error_analysis:
            report_lines.extend(
                [
                    "## Phase-wise Error Analysis",
                    "",
                    "| Phase | Total Samples | Errors | Error Rate | Accuracy |",
                    "|-------|---------------|--------|------------|----------|",
                ]
            )

            for phase, metrics in error_analysis["phase_error_analysis"].items():
                report_lines.append(
                    f"| {phase} | {metrics['total_samples']} | {metrics['errors']} | "
                    f"{metrics['error_rate']:.3f} | {metrics['accuracy']:.3f} |"
                )

            report_lines.append("")

        # Recommendations
        report_lines.extend(
            [
                "## Recommendations",
                "",
                "### Priority Actions",
                "1. Focus on phases with highest error rates",
                "2. Investigate common misclassification patterns",
                "3. Consider data augmentation for difficult phases",
                "",
                "### Model Improvements",
                "1. Analyze feature representations for confused phases",
                "2. Consider ensemble methods for difficult samples",
                "3. Implement confidence-based rejection for uncertain predictions",
                "",
            ]
        )

        # Save report
        report_text = "\n".join(report_lines)

        with open(output_path, "w") as f:
            f.write(report_text)

        logger.info(f"Error analysis report saved to {output_path}")
        return report_text

    def plot_error_patterns(
        self, error_analysis: Dict[str, Any], save_path: str = None
    ):
        """Plot error pattern visualizations."""
        if "top_error_patterns" not in error_analysis:
            logger.warning("No error patterns found to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Top error patterns
        patterns = error_analysis["top_error_patterns"][:10]
        pattern_labels = [
            f"{p['true_phase'][:10]}→{p['predicted_phase'][:10]}" for p in patterns
        ]
        pattern_counts = [p["count"] for p in patterns]

        axes[0, 0].barh(range(len(pattern_labels)), pattern_counts)
        axes[0, 0].set_yticks(range(len(pattern_labels)))
        axes[0, 0].set_yticklabels(pattern_labels)
        axes[0, 0].set_xlabel("Error Count")
        axes[0, 0].set_title("Top 10 Error Patterns")
        axes[0, 0].grid(axis="x", alpha=0.3)

        # Plot 2: Phase-wise error rates
        if "phase_error_analysis" in error_analysis:
            phase_analysis = error_analysis["phase_error_analysis"]
            phases = list(phase_analysis.keys())
            error_rates = [phase_analysis[phase]["error_rate"] for phase in phases]

            bars = axes[0, 1].bar(range(len(phases)), error_rates)
            axes[0, 1].set_xticks(range(len(phases)))
            axes[0, 1].set_xticklabels([p[:8] for p in phases], rotation=45, ha="right")
            axes[0, 1].set_ylabel("Error Rate")
            axes[0, 1].set_title("Error Rate by Phase")
            axes[0, 1].grid(axis="y", alpha=0.3)

            # Color bars by error rate
            max_error_rate = max(error_rates)
            for bar, rate in zip(bars, error_rates):
                bar.set_color(plt.cm.Reds(rate / max_error_rate))

        # Plot 3: Error distribution histogram
        all_errors = []
        for pattern in error_analysis["top_error_patterns"]:
            all_errors.extend([pattern["count"]] * 1)  # Simplified representation

        if all_errors:
            axes[1, 0].hist(all_errors, bins=20, alpha=0.7, edgecolor="black")
            axes[1, 0].set_xlabel("Error Count")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Error Count Distribution")
            axes[1, 0].grid(alpha=0.3)

        # Plot 4: Sample count vs error rate
        if "phase_error_analysis" in error_analysis:
            phase_analysis = error_analysis["phase_error_analysis"]
            sample_counts = [phase_analysis[phase]["total_samples"] for phase in phases]
            error_rates = [phase_analysis[phase]["error_rate"] for phase in phases]

            scatter = axes[1, 1].scatter(sample_counts, error_rates, alpha=0.7, s=100)
            axes[1, 1].set_xlabel("Sample Count")
            axes[1, 1].set_ylabel("Error Rate")
            axes[1, 1].set_title("Sample Count vs Error Rate")
            axes[1, 1].grid(alpha=0.3)

            # Add phase labels
            for i, phase in enumerate(phases):
                axes[1, 1].annotate(
                    phase[:6],
                    (sample_counts[i], error_rates[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Error pattern plots saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Test the error analyzer
    error_analyzer = ErrorAnalyzer()

    # Generate dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    n_classes = 11

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)

    # Test error analysis
    error_results = error_analyzer.analyze_misclassifications(y_true, y_pred, y_prob)

    print("Error analysis completed!")
    print(f"Error rate: {error_results['error_rate']:.4f}")
    print(f"Top error pattern: {error_results['top_error_patterns'][0]}")

    # Test confidence analysis
    conf_results = error_analyzer.analyze_confidence_errors(y_true, y_pred, y_prob)
    print(f"High confidence errors: {conf_results['high_confidence_errors']['count']}")

    print("Error analysis tools ready!")
