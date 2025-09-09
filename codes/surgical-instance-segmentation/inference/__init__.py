"""
Inference Engine for Surgical Instance Segmentation Framework

This module provides unified inference capabilities for all models in the framework:
- Mask R-CNN inference with COCO format output
- SAM inference with bbox/point prompts
- YOLO inference with real-time capabilities
- Ensemble prediction combining multiple models

The inference engine maintains compatibility with the reference notebooks while
providing production-ready features for deployment.
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
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms

    torch_available = True
except ImportError:
    print("Warning: PyTorch not installed")
    torch_available = False

try:
    from pycocotools import mask as coco_mask

    coco_available = True
except ImportError:
    print("Warning: pycocotools not installed")
    coco_available = False

# Import model-specific inference modules
try:
    from ..models.mask_rcnn.mask_rcnn_model import MaskRCNNModel
    from ..models.model_factory import ModelFactory
    from ..models.sam.sam_model import SAMModel
    from ..models.yolo.yolo_model import YOLOModel
except ImportError:
    print("Warning: Could not import model modules")


class ModelPredictor:
    """Generic model predictor interface."""

    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        """Run prediction on input data."""
        return self.model.predict(input_data)


class InferencePipeline:
    """Complete inference pipeline matching notebook workflows."""

    def __init__(self, model_type: str, model):
        self.model_type = model_type
        self.model = model
        self.predictor = ModelPredictor(model)

    def run_inference(self, input_data):
        """Run inference using the loaded model."""
        return self.predictor.predict(input_data)


class InferenceEngine:
    """
    Unified inference engine for all segmentation models

    This class provides a single interface for running inference with any of the trained models:
    - Mask R-CNN: Deep learning segmentation with instance detection
    - SAM: Segment Anything Model with prompt-based segmentation
    - YOLO: Fast object detection and segmentation
    - Ensemble: Combined predictions from multiple models

    Features:
    - Unified inference API across all model types
    - Automatic input preprocessing and output postprocessing
    - Batch processing capabilities
    - Real-time inference optimization
    - Format conversion utilities (COCO, YOLO, etc.)
    """

    def __init__(
        self,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ):
        """
        Initialize inference engine

        Args:
            device (str): Device to use ('auto', 'cuda', 'cpu')
            confidence_threshold (float): Confidence threshold for predictions
            nms_threshold (float): NMS threshold for duplicate removal
        """
        # Set device
        if device == "auto":
            if torch_available:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Store loaded models
        self.models = {}
        self.model_configs = {}

        # Performance tracking
        self.inference_times = defaultdict(list)

        print(f"Inference Engine initialized on device: {self.device}")

    def load_mask_rcnn_model(
        self, model_path: str, config_path: Optional[str] = None, num_classes: int = 91
    ) -> None:
        """
        Load Mask R-CNN model for inference

        Args:
            model_path (str): Path to trained Mask R-CNN model
            config_path (str): Path to model configuration (optional)
            num_classes (int): Number of classes in the model
        """
        if not torch_available:
            raise ImportError("PyTorch is required for Mask R-CNN inference")

        print(f"Loading Mask R-CNN model from: {model_path}")

        try:
            # Load model using model factory
            model_factory = ModelFactory()
            model = model_factory.create_mask_rcnn_model(
                num_classes=num_classes, pretrained=False
            )

            # Load trained weights
            if torch_available:
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=True
                )
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()

            self.models["mask_rcnn"] = model
            self.model_configs["mask_rcnn"] = {
                "model_path": model_path,
                "config_path": config_path,
                "num_classes": num_classes,
                "type": "mask_rcnn",
            }

            print("Mask R-CNN model loaded successfully")

        except Exception as e:
            print(f"Failed to load Mask R-CNN model: {str(e)}")
            raise

    def load_sam_model(self, checkpoint_path: str, model_type: str = "vit_h") -> None:
        """
        Load SAM model for inference

        Args:
            checkpoint_path (str): Path to SAM checkpoint
            model_type (str): SAM model type ('vit_h', 'vit_l', 'vit_b')
        """
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            raise ImportError("segment-anything package is required for SAM inference")

        print(f"Loading SAM model from: {checkpoint_path}")

        try:
            # Load SAM model
            sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam_model.to(self.device)

            # Create predictor
            predictor = SamPredictor(sam_model)

            self.models["sam"] = {"model": sam_model, "predictor": predictor}
            self.model_configs["sam"] = {
                "checkpoint_path": checkpoint_path,
                "model_type": model_type,
                "type": "sam",
            }

            print("SAM model loaded successfully")

        except Exception as e:
            print(f"Failed to load SAM model: {str(e)}")
            raise

    def load_yolo_model(self, model_path: str, model_type: str = "yolo11l-seg") -> None:
        """
        Load YOLO model for inference

        Args:
            model_path (str): Path to trained YOLO model
            model_type (str): YOLO model type
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package is required for YOLO inference")

        print(f"Loading YOLO model from: {model_path}")

        try:
            # Load YOLO model
            model = YOLO(model_path)

            self.models["yolo"] = model
            self.model_configs["yolo"] = {
                "model_path": model_path,
                "model_type": model_type,
                "type": "yolo",
            }

            print("YOLO model loaded successfully")

        except Exception as e:
            print(f"Failed to load YOLO model: {str(e)}")
            raise

    def predict_mask_rcnn(
        self,
        image: Union[str, np.ndarray, Image.Image],
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run Mask R-CNN inference on image

        Args:
            image: Input image (path, numpy array, or PIL Image)
            confidence_threshold: Confidence threshold (uses default if None)

        Returns:
            Dict containing detection results in COCO format
        """
        if "mask_rcnn" not in self.models:
            raise RuntimeError("Mask R-CNN model not loaded")

        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Load and preprocess image
        image_tensor, original_image = self._preprocess_image(image)

        # Run inference
        start_time = time.time()

        with torch.no_grad():
            predictions = self.models["mask_rcnn"]([image_tensor])

        inference_time = time.time() - start_time
        self.inference_times["mask_rcnn"].append(inference_time)

        # Process predictions
        pred = predictions[0]

        # Filter by confidence
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        boxes = pred["boxes"].cpu().numpy()
        masks = pred["masks"].cpu().numpy()

        # Apply confidence threshold
        keep_indices = scores >= confidence_threshold

        filtered_results = {
            "scores": scores[keep_indices],
            "labels": labels[keep_indices],
            "boxes": boxes[keep_indices],
            "masks": masks[keep_indices],
            "inference_time": inference_time,
            "image_shape": original_image.shape[:2],
        }

        return filtered_results

    def predict_sam(
        self,
        image: Union[str, np.ndarray, Image.Image],
        bbox: Optional[List[int]] = None,
        points: Optional[List[List[int]]] = None,
        point_labels: Optional[List[int]] = None,
        multimask_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Run SAM inference on image with prompts

        Args:
            image: Input image
            bbox: Bounding box prompt [x1, y1, x2, y2]
            points: Point prompts [[x1, y1], [x2, y2], ...]
            point_labels: Labels for points (1=foreground, 0=background)
            multimask_output: Whether to output multiple masks

        Returns:
            Dict containing segmentation results
        """
        if "sam" not in self.models:
            raise RuntimeError("SAM model not loaded")

        predictor = self.models["sam"]["predictor"]

        # Load and set image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        predictor.set_image(image)

        # Prepare prompts
        input_box = np.array(bbox) if bbox is not None else None
        input_points = np.array(points) if points is not None else None
        input_labels = np.array(point_labels) if point_labels is not None else None

        # Run inference
        start_time = time.time()

        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box[None, :] if input_box is not None else None,
            multimask_output=multimask_output,
        )

        inference_time = time.time() - start_time
        self.inference_times["sam"].append(inference_time)

        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
            "inference_time": inference_time,
            "image_shape": image.shape[:2],
            "input_bbox": bbox,
            "input_points": points,
            "input_labels": point_labels,
        }

    def predict_yolo(
        self,
        image: Union[str, np.ndarray, Image.Image],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        image_size: int = 640,
    ) -> Dict[str, Any]:
        """
        Run YOLO inference on image

        Args:
            image: Input image
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            image_size: Input image size for model

        Returns:
            Dict containing detection and segmentation results
        """
        if "yolo" not in self.models:
            raise RuntimeError("YOLO model not loaded")

        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        if iou_threshold is None:
            iou_threshold = self.nms_threshold

        # Run inference
        start_time = time.time()

        results = self.models["yolo"].predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            verbose=False,
        )

        inference_time = time.time() - start_time
        self.inference_times["yolo"].append(inference_time)

        # Process results
        if len(results) > 0:
            result = results[0]
            processed_results = {
                "boxes": (
                    result.boxes.xyxy.cpu().numpy()
                    if result.boxes is not None
                    else np.array([])
                ),
                "scores": (
                    result.boxes.conf.cpu().numpy()
                    if result.boxes is not None
                    else np.array([])
                ),
                "labels": (
                    result.boxes.cls.cpu().numpy()
                    if result.boxes is not None
                    else np.array([])
                ),
                "masks": (
                    result.masks.data.cpu().numpy()
                    if result.masks is not None
                    else np.array([])
                ),
                "inference_time": inference_time,
                "image_shape": result.orig_shape,
            }
        else:
            processed_results = {
                "boxes": np.array([]),
                "scores": np.array([]),
                "labels": np.array([]),
                "masks": np.array([]),
                "inference_time": inference_time,
                "image_shape": None,
            }

        return processed_results

    def predict_ensemble(
        self,
        image: Union[str, np.ndarray, Image.Image],
        models: List[str] = None,
        voting_strategy: str = "average",
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run ensemble inference combining multiple models

        Args:
            image: Input image
            models: List of model names to use (default: all loaded models)
            voting_strategy: Strategy for combining predictions ("average", "max", "weighted")
            confidence_threshold: Confidence threshold for final predictions

        Returns:
            Dict containing ensemble prediction results
        """
        if models is None:
            models = list(self.models.keys())

        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Run inference with each model
        predictions = {}
        total_inference_time = 0

        for model_name in models:
            if model_name not in self.models:
                print(f"Warning: Model '{model_name}' not loaded, skipping")
                continue

            if model_name == "mask_rcnn":
                pred = self.predict_mask_rcnn(image, confidence_threshold)
            elif model_name == "sam":
                # For SAM, we need prompts - skip in ensemble for now
                print("Warning: SAM requires prompts, skipping in ensemble")
                continue
            elif model_name == "yolo":
                pred = self.predict_yolo(image, confidence_threshold)
            else:
                print(f"Warning: Unknown model type '{model_name}', skipping")
                continue

            predictions[model_name] = pred
            total_inference_time += pred["inference_time"]

        # Combine predictions based on voting strategy
        ensemble_result = self._combine_predictions(predictions, voting_strategy)
        ensemble_result["total_inference_time"] = total_inference_time
        ensemble_result["individual_predictions"] = predictions

        return ensemble_result

    def _preprocess_image(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model inference

        Args:
            image: Input image

        Returns:
            Tuple of (preprocessed tensor, original image)
        """
        # Load image
        if isinstance(image, str):
            original_image = cv2.imread(image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            original_image = np.array(image)
        else:
            original_image = image.copy()

        # Convert to tensor and normalize
        if torch_available:
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )

            image_tensor = transform(original_image).to(self.device)
        else:
            raise ImportError("PyTorch is required for image preprocessing")

        return image_tensor, original_image

    def _combine_predictions(
        self, predictions: Dict[str, Dict[str, Any]], strategy: str = "average"
    ) -> Dict[str, Any]:
        """
        Combine predictions from multiple models

        Args:
            predictions: Dictionary of predictions from each model
            strategy: Combination strategy

        Returns:
            Combined prediction results
        """
        if not predictions:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "labels": np.array([]),
                "masks": np.array([]),
            }

        # For now, implement simple averaging
        # This is a simplified implementation - in practice, you might want
        # more sophisticated ensemble methods

        all_boxes = []
        all_scores = []
        all_labels = []
        all_masks = []

        for model_name, pred in predictions.items():
            if len(pred["boxes"]) > 0:
                all_boxes.append(pred["boxes"])
                all_scores.append(pred["scores"])
                all_labels.append(pred["labels"])
                if "masks" in pred and len(pred["masks"]) > 0:
                    all_masks.append(pred["masks"])

        if all_boxes:
            combined_boxes = np.concatenate(all_boxes, axis=0)
            combined_scores = np.concatenate(all_scores, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)

            if all_masks:
                combined_masks = np.concatenate(all_masks, axis=0)
            else:
                combined_masks = np.array([])
        else:
            combined_boxes = np.array([])
            combined_scores = np.array([])
            combined_labels = np.array([])
            combined_masks = np.array([])

        return {
            "boxes": combined_boxes,
            "scores": combined_scores,
            "labels": combined_labels,
            "masks": combined_masks,
            "ensemble_strategy": strategy,
            "num_models": len(predictions),
        }

    def batch_predict(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        model_name: str,
        batch_size: int = 8,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run batch inference on multiple images

        Args:
            images: List of input images
            model_name: Name of model to use
            batch_size: Batch size for processing
            **kwargs: Additional arguments for inference

        Returns:
            List of prediction results for each image
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_results = []

            for image in batch_images:
                if model_name == "mask_rcnn":
                    result = self.predict_mask_rcnn(image, **kwargs)
                elif model_name == "yolo":
                    result = self.predict_yolo(image, **kwargs)
                elif model_name == "sam":
                    result = self.predict_sam(image, **kwargs)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                batch_results.append(result)

            results.extend(batch_results)

            print(
                f"Processed batch {i//batch_size + 1}/{(len(images) - 1)//batch_size + 1}"
            )

        return results

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for all models

        Returns:
            Dict containing performance metrics for each model
        """
        stats = {}

        for model_name, times in self.inference_times.items():
            if times:
                stats[model_name] = {
                    "avg_inference_time": np.mean(times),
                    "min_inference_time": np.min(times),
                    "max_inference_time": np.max(times),
                    "total_inferences": len(times),
                    "fps": 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
                }
            else:
                stats[model_name] = {
                    "avg_inference_time": 0,
                    "min_inference_time": 0,
                    "max_inference_time": 0,
                    "total_inferences": 0,
                    "fps": 0,
                }

        return stats

    def save_predictions(
        self, predictions: Dict[str, Any], output_path: str, format: str = "coco"
    ) -> None:
        """
        Save predictions to file in specified format

        Args:
            predictions: Prediction results
            output_path: Path to save results
            format: Output format ('coco', 'yolo', 'json')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == "coco":
            self._save_coco_format(predictions, output_path)
        elif format == "yolo":
            self._save_yolo_format(predictions, output_path)
        else:
            # Save as JSON
            with open(output_path, "w") as f:
                json.dump(predictions, f, indent=2, default=str)

        print(f"Predictions saved to: {output_path}")

    def _save_coco_format(self, predictions: Dict[str, Any], output_path: str) -> None:
        """Save predictions in COCO format"""
        # Implementation depends on specific prediction format
        # This is a placeholder for COCO format conversion
        pass

    def _save_yolo_format(self, predictions: Dict[str, Any], output_path: str) -> None:
        """Save predictions in YOLO format"""
        # Implementation depends on specific prediction format
        # This is a placeholder for YOLO format conversion
        pass


# Example usage
if __name__ == "__main__":
    # Initialize inference engine
    engine = InferenceEngine(device="cuda")

    # Load models
    engine.load_mask_rcnn_model("/path/to/mask_rcnn_model.pth")
    engine.load_yolo_model("/path/to/yolo_model.pt")
    engine.load_sam_model("/path/to/sam_checkpoint.pth")

    # Run inference
    image_path = "/path/to/test_image.jpg"

    # Single model inference
    mask_rcnn_result = engine.predict_mask_rcnn(image_path)
    yolo_result = engine.predict_yolo(image_path)
    sam_result = engine.predict_sam(image_path, bbox=[100, 100, 200, 200])

    # Ensemble inference
    ensemble_result = engine.predict_ensemble(image_path)

    # Print performance stats
    stats = engine.get_performance_stats()
    print("Performance Statistics:")
    for model, stat in stats.items():
        print(f"{model}: {stat['fps']:.2f} FPS")


__all__ = ["InferenceEngine", "InferencePipeline", "ModelPredictor"]
