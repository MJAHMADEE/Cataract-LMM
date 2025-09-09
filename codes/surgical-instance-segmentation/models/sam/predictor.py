"""
SAM Predictor for Surgical Instance Segmentation

This module provides the inference interface for SAM models using bounding box prompts
for surgical instance segmentation. It includes comprehensive prediction utilities,
evaluation on COCO datasets, and result visualization capabilities.

Features:
- COCO dataset evaluation with bbox prompts
- Batch processing capabilities
- Comprehensive visualization utilities
- Quality assessment and metrics
- COCO-style result formatting
- Performance benchmarking

Author: Research Team
Date: August 2025
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .model import SAMPostprocessor, SAMPreprocessor, SurgicalSAM

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    COCO_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: pycocotools not installed. COCO evaluation will be limited.")
    COCO_AVAILABLE = False


class SAMPredictor:
    """
    Comprehensive prediction interface for SAM-based surgical instance segmentation.

    This class provides a high-level interface for running inference with SAM models
    using bounding box prompts from COCO annotations, including preprocessing,
    prediction, post-processing, and comprehensive evaluation capabilities.

    Features:
    - COCO dataset processing and evaluation
    - Automatic bbox prompt extraction
    - Quality-based mask filtering
    - Comprehensive visualization options
    - Performance benchmarking
    - Batch processing capabilities
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: str = None,
        device: str = "auto",
    ):
        """
        Initialize the SAM predictor.

        Args:
            model_type (str): SAM model variant ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path (str): Path to SAM checkpoint file
            device (str): Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize SAM model
        self.sam_model = SurgicalSAM(
            model_type=model_type, checkpoint_path=checkpoint_path, device=device
        )

        # Initialize preprocessor and postprocessor
        self.preprocessor = SAMPreprocessor()
        self.postprocessor = SAMPostprocessor()

        # Class names for surgical instruments
        self.class_names = [
            "background",
            "forceps",
            "scissors",
            "needle_holder",
            "phacoemulsification_tip",
            "irrigation_aspiration",
            "iol_injector",
            "spatula",
            "cannula",
            "capsulorhexis_forceps",
            "chopper",
            "speculum",
            "other_instruments",
        ]

        print(f"✅ SAM Predictor initialized with {model_type} on {self.device}")

    def predict_image_with_bboxes(
        self,
        image_path: str,
        bboxes: List[List[float]],
        category_ids: Optional[List[int]] = None,
        refine_masks: bool = True,
    ) -> List[Dict[str, any]]:
        """
        Predict segmentation masks for an image with given bounding boxes.

        Args:
            image_path (str): Path to input image
            bboxes (List[List[float]]): List of bounding boxes [x1, y1, x2, y2]
            category_ids (Optional[List[int]]): Category IDs for each bbox
            refine_masks (bool): Whether to apply mask refinement

        Returns:
            List of prediction dictionaries with masks and metadata
        """
        # Load and preprocess image
        image = self.preprocessor.preprocess_image(image_path)

        # Predict with SAM
        predictions = self.sam_model.predict_with_multiple_bboxes(
            image, bboxes, category_ids
        )

        # Apply post-processing
        if refine_masks:
            for pred in predictions:
                if pred["mask"] is not None:
                    pred["mask"] = self.postprocessor.refine_mask(pred["mask"])
                    pred["quality"] = self.postprocessor.assess_mask_quality(
                        pred["mask"], pred["bbox"]
                    )

        return predictions

    def evaluate_on_coco_dataset(
        self,
        annotation_file: str,
        image_folder: str,
        output_dir: str = "./sam_evaluation",
        max_images: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate SAM on a COCO dataset using bounding box prompts.

        Args:
            annotation_file (str): Path to COCO annotation JSON file
            image_folder (str): Path to folder containing images
            output_dir (str): Directory to save evaluation results
            max_images (Optional[int]): Maximum number of images to evaluate

        Returns:
            Dict containing evaluation metrics
        """
        if not COCO_AVAILABLE:
            raise ImportError("COCO evaluation requires pycocotools")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load COCO annotations
        images_info, image_annotations = self.preprocessor.process_coco_annotations(
            annotation_file, image_folder
        )

        # Load COCO ground truth
        coco_gt = COCO(annotation_file)

        # Process images and generate predictions
        predictions = []
        processed_count = 0

        print(f"🔍 Evaluating SAM on COCO dataset...")
        print(f"📁 Images: {len(images_info)}, Annotations: {len(image_annotations)}")

        for image_id, img_info in images_info.items():
            if max_images and processed_count >= max_images:
                break

            image_file = os.path.join(image_folder, img_info["file_name"])
            if not os.path.exists(image_file):
                continue

            # Get annotations for this image
            anns = image_annotations.get(image_id, [])
            if not anns:
                continue

            # Extract bounding boxes and categories
            bboxes = []
            category_ids = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                bbox = [x, y, x + w, y + h]  # Convert to [x1, y1, x2, y2]
                bboxes.append(bbox)
                category_ids.append(ann["category_id"])

            # Predict with SAM
            image_predictions = self.predict_image_with_bboxes(
                image_file, bboxes, category_ids
            )

            # Format for COCO evaluation
            coco_preds = self.postprocessor.format_coco_results(
                image_predictions, image_id
            )
            predictions.extend(coco_preds)

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"✅ Processed {processed_count} images...")

        print(
            f"🎯 Generated {len(predictions)} predictions from {processed_count} images"
        )

        # Save predictions
        pred_file = os.path.join(output_dir, "sam_predictions.json")
        with open(pred_file, "w") as f:
            json.dump(predictions, f)

        print(f"💾 Predictions saved to: {pred_file}")

        # Evaluate using COCO metrics
        if predictions:
            coco_dt = coco_gt.loadRes(pred_file)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Extract metrics
            metrics = {
                "mAP": coco_eval.stats[0],
                "mAP_50": coco_eval.stats[1],
                "mAP_75": coco_eval.stats[2],
                "mAP_small": coco_eval.stats[3],
                "mAP_medium": coco_eval.stats[4],
                "mAP_large": coco_eval.stats[5],
                "AR_1": coco_eval.stats[6],
                "AR_10": coco_eval.stats[7],
                "AR_100": coco_eval.stats[8],
                "AR_small": coco_eval.stats[9],
                "AR_medium": coco_eval.stats[10],
                "AR_large": coco_eval.stats[11],
            }

            print(f"\n📊 SAM Evaluation Results:")
            print(f"   mAP: {metrics['mAP']:.4f}")
            print(f"   mAP@0.5: {metrics['mAP_50']:.4f}")
            print(f"   mAP@0.75: {metrics['mAP_75']:.4f}")

            # Save metrics
            metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            return metrics
        else:
            print("❌ No predictions generated for evaluation")
            return {}

    def visualize_predictions(
        self,
        image_path: str,
        predictions: List[Dict[str, any]],
        save_path: Optional[str] = None,
        show_bboxes: bool = True,
        show_masks: bool = True,
        show_scores: bool = True,
        figsize: Tuple[int, int] = (15, 10),
    ) -> np.ndarray:
        """
        Visualize SAM predictions with comprehensive annotation.

        Args:
            image_path (str): Path to input image
            predictions (List[Dict[str, any]]): SAM predictions
            save_path (Optional[str]): Path to save visualization
            show_bboxes (bool): Whether to show bounding boxes
            show_masks (bool): Whether to overlay masks
            show_scores (bool): Whether to show confidence scores
            figsize (Tuple[int, int]): Figure size for visualization

        Returns:
            np.ndarray: Visualization image as numpy array
        """
        # Load image
        image = self.preprocessor.preprocess_image(image_path)

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image)

        # Color map for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))

        # Draw predictions
        for i, pred in enumerate(predictions):
            bbox = pred["bbox"]
            category_id = pred.get("category_id", 1)
            score = pred.get("score", 0.0)
            mask = pred.get("mask")

            # Get color for this class
            color = colors[category_id] if category_id < len(colors) else colors[0]

            # Draw bounding box
            if show_bboxes:
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1

                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Draw mask
            if show_masks and mask is not None:
                masked_image = np.ma.masked_where(mask < 0.5, mask)
                ax.imshow(masked_image, alpha=0.5, cmap="jet")

            # Add label
            if show_bboxes or show_scores:
                class_name = (
                    self.class_names[category_id]
                    if category_id < len(self.class_names)
                    else "unknown"
                )
                label_text = f"{class_name}"
                if show_scores:
                    label_text += f": {score:.2f}"

                ax.text(
                    bbox[0],
                    bbox[1] - 5,
                    label_text,
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
                    fontsize=10,
                    color="white",
                    weight="bold",
                )

        ax.set_title(
            f"SAM Segmentation Results ({self.model_type})", fontsize=16, weight="bold"
        )
        ax.axis("off")

        # Add model info
        model_info = (
            f"Model: {self.model_type.upper()} | Predictions: {len(predictions)}"
        )
        ax.text(
            0.02,
            0.98,
            model_info,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )

        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"💾 Visualization saved to: {save_path}")

        # Convert to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return buf

    def benchmark_performance(
        self,
        test_images: List[str],
        test_bboxes: List[List[List[float]]],
        warmup_runs: int = 5,
        benchmark_runs: int = 20,
    ) -> Dict[str, float]:
        """
        Benchmark SAM inference performance.

        Args:
            test_images (List[str]): List of test image paths
            test_bboxes (List[List[List[float]]]): Bboxes for each image
            warmup_runs (int): Number of warmup iterations
            benchmark_runs (int): Number of benchmark iterations

        Returns:
            Dict containing performance metrics
        """
        import time

        print(f"🏃 Benchmarking SAM {self.model_type} performance...")

        # Warmup runs
        print(f"🔥 Warming up with {warmup_runs} runs...")
        for i in range(min(warmup_runs, len(test_images))):
            self.predict_image_with_bboxes(
                test_images[i % len(test_images)], test_bboxes[i % len(test_bboxes)]
            )

        # Benchmark runs
        times = []
        total_predictions = 0

        print(f"⏱️  Running {benchmark_runs} benchmark iterations...")
        for i in range(benchmark_runs):
            img_idx = i % len(test_images)
            bbox_idx = i % len(test_bboxes)

            start_time = time.time()
            predictions = self.predict_image_with_bboxes(
                test_images[img_idx], test_bboxes[bbox_idx]
            )
            end_time = time.time()

            times.append(end_time - start_time)
            total_predictions += len(predictions)

        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_predictions_per_image = total_predictions / benchmark_runs

        metrics = {
            "avg_inference_time_ms": avg_time * 1000,
            "std_inference_time_ms": std_time * 1000,
            "min_inference_time_ms": min_time * 1000,
            "max_inference_time_ms": max_time * 1000,
            "fps": fps,
            "avg_predictions_per_image": avg_predictions_per_image,
            "total_benchmark_runs": benchmark_runs,
            "model_type": self.model_type,
        }

        print(f"\n📊 Performance Results:")
        print(f"   Average time: {metrics['avg_inference_time_ms']:.2f}ms")
        print(f"   FPS: {metrics['fps']:.2f}")
        print(f"   Predictions/image: {metrics['avg_predictions_per_image']:.1f}")

        return metrics

    def get_model_info(self) -> Dict[str, any]:
        """Get comprehensive model and predictor information."""
        sam_info = self.sam_model.get_model_info()
        sam_info.update(
            {
                "predictor_type": "SAMPredictor",
                "supports_bbox_prompts": True,
                "supports_point_prompts": True,
                "supports_mask_prompts": True,
                "evaluation_datasets": ["COCO"],
                "output_formats": ["masks", "RLE", "polygons"],
            }
        )
        return sam_info


# Convenience function for quick SAM evaluation
def evaluate_sam_on_coco(
    annotation_file: str,
    image_folder: str,
    checkpoint_path: str,
    model_type: str = "vit_h",
    output_dir: str = "./sam_evaluation",
    max_images: Optional[int] = None,
) -> Dict[str, float]:
    """
    Convenience function for quick SAM evaluation on COCO dataset.

    Args:
        annotation_file (str): Path to COCO annotation file
        image_folder (str): Path to images folder
        checkpoint_path (str): Path to SAM checkpoint
        model_type (str): SAM model variant
        output_dir (str): Output directory for results
        max_images (Optional[int]): Maximum images to evaluate

    Returns:
        Dict containing evaluation metrics
    """
    # Initialize predictor
    predictor = SAMPredictor(model_type=model_type, checkpoint_path=checkpoint_path)

    # Run evaluation
    metrics = predictor.evaluate_on_coco_dataset(
        annotation_file=annotation_file,
        image_folder=image_folder,
        output_dir=output_dir,
        max_images=max_images,
    )

    print(f"✅ SAM evaluation completed!")
    print(f"📊 Results saved to: {output_dir}")

    return metrics


if __name__ == "__main__":
    # Example usage
    print("Testing SAM Predictor")
    print("=" * 40)

    # This would normally require a trained SAM checkpoint
    print("⚠️  Note: This example requires downloading SAM checkpoints")
    print("💡 Download commands:")
    print("   # For ViT-H (best accuracy)")
    print(
        "   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )
    print("   # For ViT-B (fastest)")
    print(
        "   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )

    print("\n🔧 SAM Predictor Features:")
    print("  ✅ COCO dataset evaluation")
    print("  ✅ Bbox prompt-based segmentation")
    print("  ✅ Quality assessment and filtering")
    print("  ✅ Performance benchmarking")
    print("  ✅ Comprehensive visualization")
    print("  ✅ Zero-shot capabilities")

    # Example of how to use the predictor:
    """
    predictor = SAMPredictor(
        model_type='vit_h',
        checkpoint_path='sam_vit_h_4b8939.pth'
    )
    
    metrics = predictor.evaluate_on_coco_dataset(
        annotation_file='test/_annotations.coco.json',
        image_folder='test/',
        output_dir='./sam_results'
    )
    """
