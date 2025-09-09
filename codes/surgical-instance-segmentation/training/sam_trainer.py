"""
SAM (Segment Anything Model) Trainer Implementation for Surgical Instance Segmentation

This trainer exactly matches the training and inference procedure from the reference
SAM_inference.ipynb notebook while providing additional production features.

The implementation follows the exact same:
- Model loading procedures
- Inference with bbox prompts
- COCO evaluation protocol
- Prediction pipeline and output handling
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    torch_available = True
except ImportError:
    print("Warning: PyTorch not installed")
    torch_available = False

try:
    from segment_anything import (
        SamAutomaticMaskGenerator,
        SamPredictor,
        sam_model_registry,
    )

    sam_available = True
except ImportError:
    print(
        "Warning: segment-anything not installed. Please install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
    )
    sam_available = False

try:
    from pycocotools import mask as coco_mask
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_available = True
except ImportError:
    print(
        "Warning: pycocotools not installed. Please install with: pip install pycocotools"
    )
    coco_available = False

from .base_trainer import BaseTrainer


class SAMTrainer(BaseTrainer):
    """
    SAM (Segment Anything Model) Trainer for Surgical Instance Segmentation

    This implementation exactly matches the inference procedure from SAM_inference.ipynb:
    - Same model loading (SAM checkpoint)
    - Same bbox prompt-based inference
    - Same COCO evaluation protocol
    - Same prediction and evaluation pipeline

    Note: SAM is typically used for inference rather than training from scratch.
    This class focuses on the inference pipeline as shown in the reference notebook.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize SAM trainer/predictor following notebook configuration

        Args:
            model_type (str): SAM model type ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path (str): Path to SAM checkpoint file
            device (str): Device to use ('cuda' or 'cpu')
        """
        super().__init__()

        if not torch_available:
            raise ImportError("PyTorch is required for SAM")

        if not sam_available:
            raise ImportError("segment-anything package is required")

        self.model_type = model_type
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load SAM model exactly as in notebook
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading SAM model from checkpoint: {checkpoint_path}")
            self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam_model.to(device)
        else:
            print(f"SAM checkpoint not found at: {checkpoint_path}")
            print("Please download SAM checkpoint and provide correct path")
            self.sam_model = None

        # Initialize predictor exactly as in notebook
        if self.sam_model is not None:
            self.predictor = SamPredictor(self.sam_model)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)
            print(f"SAM model '{model_type}' loaded successfully on {device}")
        else:
            self.predictor = None
            self.mask_generator = None

        # Store results
        self.evaluation_results = {}
        self.prediction_results = []

    def set_image(self, image: Union[str, np.ndarray, Image.Image]) -> None:
        """
        Set image for SAM prediction exactly as in notebook

        Args:
            image: Input image (path, numpy array, or PIL Image)
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")

        # Load image if path provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Set image for predictor exactly as in notebook
        self.predictor.set_image(image)
        self.current_image = image

    def predict_with_bbox(
        self, bbox: List[int], multimask_output: bool = True
    ) -> Dict[str, Any]:
        """
        Predict mask using bounding box prompt exactly as in SAM_inference.ipynb

        This follows the exact same procedure as the notebook:
        1. Use bbox as prompt for SAM prediction
        2. Get masks, scores, and logits
        3. Return results in same format as notebook

        Args:
            bbox (List[int]): Bounding box [x1, y1, x2, y2] (same format as notebook)
            multimask_output (bool): Whether to output multiple masks (same as notebook)

        Returns:
            Dict containing masks, scores, and logits (same as notebook output)
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded or image not set")

        # Convert bbox to numpy array exactly as in notebook
        input_box = np.array(bbox)

        # Predict with SAM exactly as in notebook
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=multimask_output,
        )

        # Return results in same format as notebook
        return {"masks": masks, "scores": scores, "logits": logits, "input_bbox": bbox}

    def predict_with_points(
        self,
        point_coords: List[List[int]],
        point_labels: List[int],
        multimask_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict mask using point prompts

        Args:
            point_coords: List of [x, y] coordinates
            point_labels: List of labels (1 for foreground, 0 for background)
            multimask_output (bool): Whether to output multiple masks

        Returns:
            Dict containing prediction results
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded or image not set")

        # Convert to numpy arrays
        input_points = np.array(point_coords)
        input_labels = np.array(point_labels)

        # Predict with SAM
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask_output,
        )

        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
            "input_points": point_coords,
            "input_labels": point_labels,
        }

    def generate_masks_automatic(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        Generate masks automatically using SAM's automatic mask generator

        Args:
            image: Input image

        Returns:
            List of mask dictionaries
        """
        if self.mask_generator is None:
            raise RuntimeError("SAM model not loaded")

        # Load image if path provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Generate masks automatically
        masks = self.mask_generator.generate(image)

        return masks

    def evaluate_coco_dataset(
        self,
        dataset_path: str,
        annotations_file: str,
        output_dir: str = "./sam_evaluation_results",
        bbox_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate SAM on COCO dataset exactly following the SAM_inference.ipynb notebook

        This replicates the exact evaluation procedure from the notebook:
        1. Load COCO dataset and annotations
        2. For each image, get ground truth bboxes
        3. Use bboxes as prompts for SAM prediction
        4. Convert SAM masks to COCO format
        5. Run COCO evaluation metrics

        Args:
            dataset_path (str): Path to dataset images
            annotations_file (str): Path to COCO annotations JSON file
            output_dir (str): Directory to save evaluation results
            bbox_threshold (float): Confidence threshold for bboxes
            mask_threshold (float): Threshold for mask binarization

        Returns:
            Dict containing evaluation metrics (same format as notebook)
        """
        if not coco_available:
            raise ImportError("pycocotools is required for COCO evaluation")

        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")

        print("=" * 60)
        print("Evaluating SAM on COCO Dataset")
        print("Following exact procedure from SAM_inference.ipynb notebook")
        print("=" * 60)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load COCO dataset exactly as in notebook
        coco_gt = COCO(annotations_file)
        image_ids = coco_gt.getImgIds()

        print(f"Found {len(image_ids)} images in dataset")

        # Prepare results for COCO evaluation
        results = []

        start_time = time.time()
        processed_count = 0

        for img_id in image_ids:
            try:
                # Get image info exactly as in notebook
                img_info = coco_gt.loadImgs(img_id)[0]
                image_path = os.path.join(dataset_path, img_info["file_name"])

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                # Load and set image exactly as in notebook
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.set_image(image_rgb)

                # Get ground truth annotations exactly as in notebook
                ann_ids = coco_gt.getAnnIds(imgIds=img_id)
                annotations = coco_gt.loadAnns(ann_ids)

                # Process each annotation exactly as in notebook
                for ann in annotations:
                    if ann["area"] < 100:  # Skip small annotations
                        continue

                    # Get bounding box exactly as in notebook
                    bbox = ann["bbox"]  # COCO format: [x, y, w, h]

                    # Convert to SAM format [x1, y1, x2, y2] exactly as in notebook
                    x, y, w, h = bbox
                    sam_bbox = [x, y, x + w, y + h]

                    # Predict with SAM using bbox prompt exactly as in notebook
                    pred_result = self.predict_with_bbox(
                        bbox=sam_bbox, multimask_output=True
                    )

                    # Get best mask (highest score) exactly as in notebook
                    masks = pred_result["masks"]
                    scores = pred_result["scores"]
                    best_mask_idx = np.argmax(scores)
                    best_mask = masks[best_mask_idx]
                    best_score = scores[best_mask_idx]

                    # Convert mask to COCO RLE format exactly as in notebook
                    mask_uint8 = (best_mask > mask_threshold).astype(np.uint8)
                    rle = coco_mask.encode(np.asfortranarray(mask_uint8))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    # Create result entry exactly as in notebook
                    result = {
                        "image_id": img_id,
                        "category_id": ann["category_id"],
                        "segmentation": rle,
                        "score": float(best_score),
                        "bbox": bbox,  # Keep original COCO format
                    }

                    results.append(result)

                processed_count += 1
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Processed {processed_count}/{len(image_ids)} images ({elapsed:.1f}s)"
                    )

            except Exception as e:
                print(f"Error processing image {img_id}: {str(e)}")
                continue

        # Save prediction results exactly as in notebook
        results_file = os.path.join(output_dir, "sam_predictions.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved {len(results)} predictions to {results_file}")

        # Run COCO evaluation exactly as in notebook
        if len(results) > 0:
            # Create COCO detection results object
            coco_dt = coco_gt.loadRes(results_file)

            # Run segmentation evaluation exactly as in notebook
            coco_eval = COCOeval(coco_gt, coco_dt, "segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Extract metrics exactly as in notebook
            evaluation_metrics = {
                "AP": coco_eval.stats[0],  # AP @ IoU=0.50:0.95
                "AP_50": coco_eval.stats[1],  # AP @ IoU=0.50
                "AP_75": coco_eval.stats[2],  # AP @ IoU=0.75
                "AP_small": coco_eval.stats[3],  # AP for small objects
                "AP_medium": coco_eval.stats[4],  # AP for medium objects
                "AP_large": coco_eval.stats[5],  # AP for large objects
                "AR_1": coco_eval.stats[6],  # AR @ maxDets=1
                "AR_10": coco_eval.stats[7],  # AR @ maxDets=10
                "AR_100": coco_eval.stats[8],  # AR @ maxDets=100
                "AR_small": coco_eval.stats[9],  # AR for small objects
                "AR_medium": coco_eval.stats[10],  # AR for medium objects
                "AR_large": coco_eval.stats[11],  # AR for large objects
            }

            # Save evaluation metrics
            metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(evaluation_metrics, f, indent=2)

            print(f"Saved evaluation metrics to {metrics_file}")

        else:
            evaluation_metrics = {}
            print("No valid predictions generated")

        total_time = time.time() - start_time
        print(
            f"Evaluation completed in {total_time:.2f}s ({total_time/60:.1f} minutes)"
        )

        # Store results
        self.evaluation_results = {
            "metrics": evaluation_metrics,
            "predictions": results,
            "evaluation_time": total_time,
            "processed_images": processed_count,
        }

        return self.evaluation_results

    def predict_on_dataset(
        self,
        dataset_path: str,
        annotations_file: str,
        output_dir: str = "./sam_predictions",
        save_visualizations: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run SAM predictions on entire dataset following notebook procedure

        Args:
            dataset_path (str): Path to dataset images
            annotations_file (str): Path to annotations JSON file
            output_dir (str): Directory to save predictions
            save_visualizations (bool): Whether to save visualization images

        Returns:
            List of prediction results
        """
        if not coco_available:
            raise ImportError("pycocotools is required for dataset processing")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        if save_visualizations:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

        # Load dataset
        coco = COCO(annotations_file)
        image_ids = coco.getImgIds()

        predictions = []

        for img_id in image_ids:
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            image_path = os.path.join(dataset_path, img_info["file_name"])

            if not os.path.exists(image_path):
                continue

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate automatic masks
            masks = self.generate_masks_automatic(image_rgb)

            # Store predictions
            prediction = {
                "image_id": img_id,
                "image_path": image_path,
                "masks": masks,
                "image_info": img_info,
            }

            predictions.append(prediction)

            # Save visualization if requested
            if save_visualizations:
                vis_path = os.path.join(vis_dir, f"{img_id}_prediction.jpg")
                self._save_prediction_visualization(image_rgb, masks, vis_path)

        # Save predictions
        pred_file = os.path.join(output_dir, "predictions.json")

        # Convert masks to serializable format
        serializable_predictions = []
        for pred in predictions:
            serializable_pred = pred.copy()
            # Convert masks to RLE format for serialization
            serializable_masks = []
            for mask_data in pred["masks"]:
                if "segmentation" in mask_data:
                    # Convert segmentation to RLE if needed
                    seg = mask_data["segmentation"]
                    if isinstance(seg, np.ndarray):
                        rle = coco_mask.encode(np.asfortranarray(seg.astype(np.uint8)))
                        rle["counts"] = rle["counts"].decode("utf-8")
                        mask_data["segmentation"] = rle
                    serializable_masks.append(mask_data)

            serializable_pred["masks"] = serializable_masks
            serializable_predictions.append(serializable_pred)

        with open(pred_file, "w") as f:
            json.dump(serializable_predictions, f, indent=2)

        print(f"Saved predictions for {len(predictions)} images to {pred_file}")

        return predictions

    def _save_prediction_visualization(
        self, image: np.ndarray, masks: List[Dict[str, Any]], save_path: str
    ) -> None:
        """
        Save visualization of predictions

        Args:
            image: Original image
            masks: List of mask predictions
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(image)

        # Overlay masks
        for i, mask_data in enumerate(masks[:10]):  # Show top 10 masks
            if "segmentation" in mask_data:
                mask = mask_data["segmentation"]
                if isinstance(mask, dict):  # RLE format
                    mask = coco_mask.decode(mask)

                # Create colored overlay
                color = np.random.rand(3)
                overlay = np.zeros_like(image, dtype=np.float32)
                overlay[mask > 0] = color

                plt.imshow(overlay, alpha=0.4)

        plt.axis("off")
        plt.title(f"SAM Predictions ({len(masks)} masks)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def get_model(self) -> Any:
        """Get the SAM model"""
        return self.sam_model

    def get_predictor(self) -> Any:
        """Get the SAM predictor"""
        return self.predictor

    def get_evaluation_results(self) -> Dict[str, Any]:
        """Get evaluation results"""
        return self.evaluation_results


def run_sam_inference_from_notebook(
    checkpoint_path: str,
    dataset_path: str,
    annotations_file: str,
    output_dir: str = "./sam_evaluation_results",
    model_type: str = "vit_h",
) -> SAMTrainer:
    """
    Run SAM inference following the exact same procedure as SAM_inference.ipynb notebook

    This function replicates the complete inference and evaluation procedure:
    1. Load SAM model exactly as in notebook
    2. Run inference with bbox prompts exactly as in notebook
    3. Evaluate on COCO dataset exactly as in notebook
    4. Generate same metrics and outputs as notebook

    Args:
        checkpoint_path (str): Path to SAM checkpoint file
        dataset_path (str): Path to dataset images
        annotations_file (str): Path to COCO annotations file
        output_dir (str): Directory to save results
        model_type (str): SAM model type ('vit_h', 'vit_l', 'vit_b')

    Returns:
        SAMTrainer: Trained/evaluated trainer instance
    """
    print("=" * 60)
    print("Running SAM Inference for Surgical Instance Segmentation")
    print("Following exact procedure from SAM_inference.ipynb notebook")
    print("=" * 60)

    # Initialize SAM trainer exactly as in notebook
    trainer = SAMTrainer(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if trainer.predictor is None:
        print("Failed to load SAM model. Please check checkpoint path.")
        return trainer

    # Run evaluation exactly as in notebook
    print("Running COCO evaluation...")
    evaluation_results = trainer.evaluate_coco_dataset(
        dataset_path=dataset_path,
        annotations_file=annotations_file,
        output_dir=output_dir,
    )

    # Print results exactly as shown in notebook
    if "metrics" in evaluation_results and evaluation_results["metrics"]:
        metrics = evaluation_results["metrics"]
        print("\n" + "=" * 60)
        print("SAM COCO Evaluation Results:")
        print("=" * 60)
        print(f"AP @ IoU=0.50:0.95: {metrics.get('AP', 0):.3f}")
        print(f"AP @ IoU=0.50:     {metrics.get('AP_50', 0):.3f}")
        print(f"AP @ IoU=0.75:     {metrics.get('AP_75', 0):.3f}")
        print(f"AP (small):        {metrics.get('AP_small', 0):.3f}")
        print(f"AP (medium):       {metrics.get('AP_medium', 0):.3f}")
        print(f"AP (large):        {metrics.get('AP_large', 0):.3f}")
        print("=" * 60)

    print("SAM inference and evaluation completed successfully!")

    return trainer


# Example usage exactly matching the notebook workflow
if __name__ == "__main__":
    # Example paths - update to match your setup
    checkpoint_path = "/path/to/sam_vit_h_4b8939.pth"
    dataset_path = "/path/to/coco/val2017"
    annotations_file = "/path/to/coco/annotations/instances_val2017.json"

    # Run SAM inference exactly as in notebook
    sam_trainer = run_sam_inference_from_notebook(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        annotations_file=annotations_file,
        output_dir="./sam_evaluation_results",
        model_type="vit_h",
    )
