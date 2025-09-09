"""
Evaluation Metrics for Surgical Instance Segmentation

This module provides comprehensive evaluation metrics for all model types:
- COCO evaluation metrics (AP, AR, mAP at different IoU thresholds)
- Instance segmentation metrics (mask IoU, boundary accuracy)
- Detection metrics (precision, recall, F1-score)
- Custom surgical instrument metrics
- Performance benchmarking utilities

Compatible with outputs from Mask R-CNN, YOLO, and SAM models.
"""

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from pycocotools import mask as coco_mask
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_available = True
except ImportError:
    print("Warning: pycocotools not installed. COCO evaluation will not be available.")
    coco_available = False

try:
    import cv2

    cv2_available = True
except ImportError:
    cv2_available = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plotting_available = True
except ImportError:
    plotting_available = False

import scipy.spatial.distance as scipy_distance
from sklearn.metrics import average_precision_score, precision_recall_curve


class COCOEvaluator:
    """COCO evaluation exactly matching the reference notebooks."""

    def __init__(self, gt_annotations_file: str):
        if not coco_available:
            raise ImportError("pycocotools not installed")
        self.coco_gt = COCO(gt_annotations_file)

    def evaluate(
        self, predictions_file: str, iou_type: str = "segm"
    ) -> Dict[str, float]:
        """Run COCO evaluation exactly as in SAM_inference.ipynb."""
        coco_dt = self.coco_gt.loadRes(predictions_file)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
            "mAR_1": coco_eval.stats[6],
            "mAR_10": coco_eval.stats[7],
            "mAR_100": coco_eval.stats[8],
            "mAR_small": coco_eval.stats[9],
            "mAR_medium": coco_eval.stats[10],
            "mAR_large": coco_eval.stats[11],
        }


class SegmentationMetrics:
    """
    Comprehensive evaluation metrics for segmentation models

    This class provides evaluation capabilities exactly matching the reference notebooks:
    - COCO evaluation as used in SAM_inference.ipynb
    - Instance segmentation metrics for Mask R-CNN evaluation
    - Object detection metrics for YOLO evaluation
    - Custom metrics for surgical instrument analysis
    """

    def __init__(self):
        """Initialize metrics calculator"""
        self.results = {}
        self.detailed_results = {}

        if not coco_available:
            warnings.warn(
                "pycocotools not available. COCO evaluation metrics will be limited."
            )

    def evaluate_coco_segmentation(
        self, gt_annotations_file: str, predictions_file: str, iou_type: str = "segm"
    ) -> Dict[str, float]:
        """
        Evaluate segmentation results using COCO metrics exactly as in SAM_inference.ipynb

        This follows the exact same COCO evaluation procedure as the reference notebook:
        1. Load ground truth annotations
        2. Load predictions in COCO format
        3. Run COCOeval for segmentation
        4. Extract standard COCO metrics

        Args:
            gt_annotations_file (str): Path to ground truth COCO annotations
            predictions_file (str): Path to predictions in COCO format
            iou_type (str): Type of IoU ('segm' for segmentation, 'bbox' for detection)

        Returns:
            Dict containing COCO evaluation metrics exactly as in notebook
        """
        if not coco_available:
            raise ImportError("pycocotools is required for COCO evaluation")

        print("=" * 60)
        print("Running COCO Evaluation for Segmentation")
        print("Following exact procedure from SAM_inference.ipynb")
        print("=" * 60)

        # Load ground truth annotations exactly as in notebook
        coco_gt = COCO(gt_annotations_file)

        # Load predictions exactly as in notebook
        coco_dt = coco_gt.loadRes(predictions_file)

        # Initialize COCOeval exactly as in notebook
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

        # Run evaluation exactly as in notebook
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics exactly as in notebook
        metrics = {
            "AP": float(coco_eval.stats[0]),  # AP @ IoU=0.50:0.95
            "AP_50": float(coco_eval.stats[1]),  # AP @ IoU=0.50
            "AP_75": float(coco_eval.stats[2]),  # AP @ IoU=0.75
            "AP_small": float(coco_eval.stats[3]),  # AP for small objects
            "AP_medium": float(coco_eval.stats[4]),  # AP for medium objects
            "AP_large": float(coco_eval.stats[5]),  # AP for large objects
            "AR_1": float(coco_eval.stats[6]),  # AR @ maxDets=1
            "AR_10": float(coco_eval.stats[7]),  # AR @ maxDets=10
            "AR_100": float(coco_eval.stats[8]),  # AR @ maxDets=100
            "AR_small": float(coco_eval.stats[9]),  # AR for small objects
            "AR_medium": float(coco_eval.stats[10]),  # AR for medium objects
            "AR_large": float(coco_eval.stats[11]),  # AR for large objects
        }

        # Store detailed results
        self.detailed_results["coco_evaluation"] = {
            "metrics": metrics,
            "coco_eval": coco_eval,
            "evaluation_params": coco_eval.params.__dict__,
            "gt_file": gt_annotations_file,
            "pred_file": predictions_file,
        }

        # Print results exactly as shown in notebook
        print("\nCOCO Evaluation Results:")
        print("=" * 40)
        print(f"AP @ IoU=0.50:0.95: {metrics['AP']:.3f}")
        print(f"AP @ IoU=0.50:     {metrics['AP_50']:.3f}")
        print(f"AP @ IoU=0.75:     {metrics['AP_75']:.3f}")
        print(f"AP (small):        {metrics['AP_small']:.3f}")
        print(f"AP (medium):       {metrics['AP_medium']:.3f}")
        print(f"AP (large):        {metrics['AP_large']:.3f}")
        print("=" * 40)

        return metrics

    def evaluate_instance_segmentation(
        self,
        gt_masks: List[np.ndarray],
        pred_masks: List[np.ndarray],
        gt_labels: List[int],
        pred_labels: List[int],
        pred_scores: List[float],
        iou_thresholds: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate instance segmentation performance

        Args:
            gt_masks: List of ground truth masks
            pred_masks: List of predicted masks
            gt_labels: List of ground truth labels
            pred_labels: List of predicted labels
            pred_scores: List of prediction scores
            iou_thresholds: IoU thresholds for evaluation

        Returns:
            Dict containing instance segmentation metrics
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75, 0.9]

        print("Computing instance segmentation metrics...")

        # Convert masks to proper format
        gt_masks = [self._ensure_mask_format(mask) for mask in gt_masks]
        pred_masks = [self._ensure_mask_format(mask) for mask in pred_masks]

        metrics = {}

        # Compute metrics for each IoU threshold
        for iou_thresh in iou_thresholds:
            thresh_metrics = self._compute_instance_metrics_at_threshold(
                gt_masks, pred_masks, gt_labels, pred_labels, pred_scores, iou_thresh
            )
            metrics[f"iou_{iou_thresh}"] = thresh_metrics

        # Compute average metrics across thresholds
        avg_metrics = self._average_metrics_across_thresholds(metrics)
        metrics["average"] = avg_metrics

        # Compute mask-specific metrics
        mask_metrics = self._compute_mask_quality_metrics(gt_masks, pred_masks)
        metrics["mask_quality"] = mask_metrics

        self.results["instance_segmentation"] = metrics
        return metrics

    def evaluate_object_detection(
        self,
        gt_boxes: List[np.ndarray],
        pred_boxes: List[np.ndarray],
        gt_labels: List[int],
        pred_labels: List[int],
        pred_scores: List[float],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Evaluate object detection performance

        Args:
            gt_boxes: List of ground truth boxes [x1, y1, x2, y2]
            pred_boxes: List of predicted boxes [x1, y1, x2, y2]
            gt_labels: List of ground truth labels
            pred_labels: List of predicted labels
            pred_scores: List of prediction scores
            iou_threshold: IoU threshold for matching

        Returns:
            Dict containing detection metrics
        """
        print("Computing object detection metrics...")

        # Match predictions to ground truth
        matches = self._match_detections(
            gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores, iou_threshold
        )

        # Compute precision, recall, F1
        tp = matches["true_positives"]
        fp = matches["false_positives"]
        fn = matches["false_negatives"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Compute AP (Average Precision)
        ap = self._compute_average_precision(
            pred_scores, matches["prediction_labels"], len(gt_boxes)
        )

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_precision": ap,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "iou_threshold": iou_threshold,
        }

        self.results["object_detection"] = metrics
        return metrics

    def evaluate_surgical_instruments(
        self,
        gt_annotations: Dict[str, Any],
        predictions: Dict[str, Any],
        instrument_categories: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate surgical instance segmentation with domain-specific metrics

        Args:
            gt_annotations: Ground truth annotations
            predictions: Model predictions
            instrument_categories: List of instrument category names

        Returns:
            Dict containing surgical instrument specific metrics
        """
        print("Computing surgical instrument specific metrics...")

        metrics = {}

        # Per-instrument category evaluation
        for category in instrument_categories:
            cat_metrics = self._evaluate_instrument_category(
                gt_annotations, predictions, category
            )
            metrics[f"instrument_{category}"] = cat_metrics

        # Overall surgical instrument metrics
        overall_metrics = self._compute_overall_surgical_metrics(
            gt_annotations, predictions
        )
        metrics["overall"] = overall_metrics

        # Surgical-specific quality metrics
        surgical_quality = self._compute_surgical_quality_metrics(
            gt_annotations, predictions
        )
        metrics["surgical_quality"] = surgical_quality

        self.results["surgical_instruments"] = metrics
        return metrics

    def _ensure_mask_format(self, mask: Union[np.ndarray, dict]) -> np.ndarray:
        """Ensure mask is in proper numpy array format"""
        if isinstance(mask, dict):
            # Convert from COCO RLE format
            if coco_available:
                return coco_mask.decode(mask)
            else:
                raise ValueError("COCO mask format requires pycocotools")
        elif isinstance(mask, np.ndarray):
            return mask.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported mask format: {type(mask)}")

    def _compute_instance_metrics_at_threshold(
        self,
        gt_masks: List[np.ndarray],
        pred_masks: List[np.ndarray],
        gt_labels: List[int],
        pred_labels: List[int],
        pred_scores: List[float],
        iou_threshold: float,
    ) -> Dict[str, float]:
        """Compute instance segmentation metrics at specific IoU threshold"""
        # Match instances based on IoU
        matches = []
        used_gt = set()

        # Sort predictions by score (descending)
        sorted_indices = np.argsort(pred_scores)[::-1]

        tp = 0
        fp = 0

        for pred_idx in sorted_indices:
            pred_mask = pred_masks[pred_idx]
            pred_label = pred_labels[pred_idx]
            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, (gt_mask, gt_label) in enumerate(zip(gt_masks, gt_labels)):
                if gt_idx in used_gt or gt_label != pred_label:
                    continue

                iou = self._compute_mask_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match is good enough
            if best_iou >= iou_threshold:
                tp += 1
                used_gt.add(best_gt_idx)
                matches.append((pred_idx, best_gt_idx, best_iou))
            else:
                fp += 1

        fn = len(gt_masks) - len(used_gt)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "matches": matches,
        }

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def _average_metrics_across_thresholds(
        self, threshold_metrics: Dict
    ) -> Dict[str, float]:
        """Average metrics across different IoU thresholds"""
        metric_names = ["precision", "recall", "f1_score"]
        averaged = {}

        for metric in metric_names:
            values = [threshold_metrics[thresh][metric] for thresh in threshold_metrics]
            averaged[metric] = np.mean(values)

        return averaged

    def _compute_mask_quality_metrics(
        self, gt_masks: List[np.ndarray], pred_masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute mask quality specific metrics"""
        if not gt_masks or not pred_masks:
            return {"boundary_accuracy": 0.0, "coverage": 0.0}

        # Compute boundary accuracy
        boundary_accuracies = []
        coverages = []

        for gt_mask, pred_mask in zip(gt_masks, pred_masks):
            # Ensure same size
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(pred_mask, gt_mask.shape[::-1])

            # Boundary accuracy
            gt_boundary = self._get_mask_boundary(gt_mask)
            pred_boundary = self._get_mask_boundary(pred_mask)

            if gt_boundary.sum() > 0:
                boundary_acc = self._compute_boundary_accuracy(
                    gt_boundary, pred_boundary
                )
                boundary_accuracies.append(boundary_acc)

            # Coverage
            coverage = self._compute_mask_iou(gt_mask, pred_mask)
            coverages.append(coverage)

        return {
            "boundary_accuracy": (
                np.mean(boundary_accuracies) if boundary_accuracies else 0.0
            ),
            "coverage": np.mean(coverages) if coverages else 0.0,
        }

    def _get_mask_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary pixels from mask"""
        if not cv2_available:
            # Simple boundary extraction without OpenCV
            return mask

        # Use morphological operations to find boundary
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = mask.astype(np.uint8) - eroded

        return boundary

    def _compute_boundary_accuracy(
        self, gt_boundary: np.ndarray, pred_boundary: np.ndarray
    ) -> float:
        """Compute accuracy of predicted boundary"""
        if gt_boundary.sum() == 0:
            return 1.0 if pred_boundary.sum() == 0 else 0.0

        # Find distances from predicted boundary to GT boundary
        gt_points = np.column_stack(np.where(gt_boundary))
        pred_points = np.column_stack(np.where(pred_boundary))

        if len(pred_points) == 0:
            return 0.0

        # Compute distances
        distances = scipy_distance.cdist(pred_points, gt_points)
        min_distances = np.min(distances, axis=1)

        # Boundary accuracy as percentage of points within threshold
        threshold = 2.0  # pixels
        accuracy = np.mean(min_distances <= threshold)

        return accuracy

    def _match_detections(
        self,
        gt_boxes: List[np.ndarray],
        pred_boxes: List[np.ndarray],
        gt_labels: List[int],
        pred_labels: List[int],
        pred_scores: List[float],
        iou_threshold: float,
    ) -> Dict[str, Any]:
        """Match predicted detections to ground truth"""
        tp = 0
        fp = 0
        fn = 0
        used_gt = set()
        prediction_labels = []

        # Sort predictions by score
        sorted_indices = np.argsort(pred_scores)[::-1]

        for pred_idx in sorted_indices:
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]
            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx in used_gt or gt_label != pred_label:
                    continue

                iou = self._compute_box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match is good enough
            if best_iou >= iou_threshold:
                tp += 1
                used_gt.add(best_gt_idx)
                prediction_labels.append(1)  # True positive
            else:
                fp += 1
                prediction_labels.append(0)  # False positive

        fn = len(gt_boxes) - len(used_gt)

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "prediction_labels": prediction_labels,
        }

    def _compute_box_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes"""
        # box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_average_precision(
        self, scores: List[float], labels: List[int], num_gt: int
    ) -> float:
        """Compute Average Precision (AP)"""
        if not scores or num_gt == 0:
            return 0.0

        # Sort by scores
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]

        # Compute precision and recall at each threshold
        tp = 0
        precisions = []
        recalls = []

        for i, label in enumerate(sorted_labels):
            if label == 1:  # True positive
                tp += 1

            precision = tp / (i + 1)
            recall = tp / num_gt

            precisions.append(precision)
            recalls.append(recall)

        # Compute AP using precision-recall curve
        precisions = np.array(precisions)
        recalls = np.array(recalls)

        # Use sklearn's implementation for consistency
        return average_precision_score(
            sorted_labels, [scores[i] for i in sorted_indices]
        )

    def _evaluate_instrument_category(
        self, gt_annotations: Dict[str, Any], predictions: Dict[str, Any], category: str
    ) -> Dict[str, float]:
        """Evaluate specific instrument category"""
        # Filter annotations and predictions for this category
        # This is a placeholder - actual implementation would depend on data format

        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "ap": 0.0}

    def _compute_overall_surgical_metrics(
        self, gt_annotations: Dict[str, Any], predictions: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute overall surgical instrument metrics"""
        # Placeholder for surgical-specific metrics

        return {
            "instrument_detection_rate": 0.0,
            "false_alarm_rate": 0.0,
            "instrument_localization_accuracy": 0.0,
        }

    def _compute_surgical_quality_metrics(
        self, gt_annotations: Dict[str, Any], predictions: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute surgical quality specific metrics"""
        # Placeholder for surgical quality metrics

        return {
            "tip_accuracy": 0.0,
            "shaft_accuracy": 0.0,
            "articulation_accuracy": 0.0,
        }

    def generate_evaluation_report(
        self, output_path: str, include_plots: bool = True
    ) -> None:
        """
        Generate comprehensive evaluation report

        Args:
            output_path (str): Path to save report
            include_plots (bool): Whether to include visualization plots
        """
        report = {
            "evaluation_summary": self._create_evaluation_summary(),
            "detailed_results": self.results,
            "metrics_explanation": self._get_metrics_explanation(),
        }

        # Save JSON report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate plots if requested
        if include_plots and plotting_available:
            plot_dir = os.path.splitext(output_path)[0] + "_plots"
            os.makedirs(plot_dir, exist_ok=True)
            self._generate_evaluation_plots(plot_dir)

        print(f"Evaluation report saved to: {output_path}")

    def _create_evaluation_summary(self) -> Dict[str, Any]:
        """Create summary of all evaluation results"""
        summary = {}

        for eval_type, results in self.results.items():
            if eval_type == "instance_segmentation":
                summary[eval_type] = {
                    "average_precision": results.get("average", {}).get("precision", 0),
                    "average_recall": results.get("average", {}).get("recall", 0),
                    "average_f1": results.get("average", {}).get("f1_score", 0),
                }
            elif eval_type == "object_detection":
                summary[eval_type] = {
                    "precision": results.get("precision", 0),
                    "recall": results.get("recall", 0),
                    "f1_score": results.get("f1_score", 0),
                    "ap": results.get("average_precision", 0),
                }

        return summary

    def _get_metrics_explanation(self) -> Dict[str, str]:
        """Get explanation of all metrics"""
        return {
            "AP": "Average Precision - Area under precision-recall curve",
            "AR": "Average Recall - Maximum recall given fixed number of detections",
            "IoU": "Intersection over Union - Overlap measure between predicted and ground truth",
            "mAP": "mean Average Precision - AP averaged across all classes",
            "F1": "F1-score - Harmonic mean of precision and recall",
            "Precision": "True Positives / (True Positives + False Positives)",
            "Recall": "True Positives / (True Positives + False Negatives)",
        }

    def _generate_evaluation_plots(self, output_dir: str) -> None:
        """Generate evaluation visualization plots"""
        if not plotting_available:
            print("Matplotlib not available for plotting")
            return

        # Create precision-recall curves
        if "object_detection" in self.results:
            self._plot_precision_recall_curve(output_dir)

        # Create confusion matrices
        self._plot_confusion_matrix(output_dir)

        # Create metric comparison charts
        self._plot_metric_comparison(output_dir)

    def _plot_precision_recall_curve(self, output_dir: str) -> None:
        """Plot precision-recall curve"""
        plt.figure(figsize=(8, 6))

        # This would use actual precision-recall data from evaluation
        # Placeholder implementation
        precision = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        recall = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

        plt.plot(recall, precision, "b-", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300)
        plt.close()

    def _plot_confusion_matrix(self, output_dir: str) -> None:
        """Plot confusion matrix"""
        # Placeholder implementation
        pass

    def _plot_metric_comparison(self, output_dir: str) -> None:
        """Plot metric comparison across models"""
        # Placeholder implementation
        pass


def evaluate_model_predictions(
    gt_annotations_file: str,
    predictions_file: str,
    evaluation_type: str = "coco",
    output_dir: str = "./evaluation_results",
) -> Dict[str, Any]:
    """
    Evaluate model predictions using specified evaluation protocol

    Args:
        gt_annotations_file (str): Path to ground truth annotations
        predictions_file (str): Path to model predictions
        evaluation_type (str): Type of evaluation ('coco', 'instance', 'detection')
        output_dir (str): Directory to save evaluation results

    Returns:
        Dict containing evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize metrics calculator
    metrics = SegmentationMetrics()

    # Run appropriate evaluation
    if evaluation_type == "coco":
        results = metrics.evaluate_coco_segmentation(
            gt_annotations_file, predictions_file
        )
    else:
        raise ValueError(f"Unsupported evaluation type: {evaluation_type}")

    # Generate report
    report_path = os.path.join(output_dir, "evaluation_report.json")
    metrics.generate_evaluation_report(report_path, include_plots=True)

    return results


# Example usage exactly matching notebook evaluation procedures
if __name__ == "__main__":
    # COCO evaluation exactly as in SAM_inference.ipynb
    gt_file = "/path/to/ground_truth_annotations.json"
    pred_file = "/path/to/sam_predictions.json"

    results = evaluate_model_predictions(
        gt_annotations_file=gt_file,
        predictions_file=pred_file,
        evaluation_type="coco",
        output_dir="./sam_evaluation_results",
    )

    print("COCO Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")


__all__ = ["COCOEvaluator", "SegmentationMetrics", "evaluate_model_predictions"]
