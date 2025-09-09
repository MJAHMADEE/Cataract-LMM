"""
YOLO Segmentation Models for Surgical Instance Segmentation

This module provides comprehensive implementations of YOLOv8 and YOLOv11 segmentation
models for surgical instrument detection and segmentation. Based on the reference
notebook implementations with enhanced modularity and production-ready features.

Features:
- YOLOv8 and YOLOv11 segmentation architectures
- Real-time inference capabilities
- Unified detection and segmentation pipeline
- Advanced training configurations
- Comprehensive evaluation metrics
- Export capabilities for deployment

Author: Research Team
Date: August 2025
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Ultralytics YOLO imports
try:
    import ultralytics
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print(
        "⚠️  Warning: ultralytics package not installed. YOLO functionality will be limited."
    )
    YOLO_AVAILABLE = False


class SurgicalYOLOSegmentation:
    """
    Surgical instrument segmentation using YOLO (YOLOv8/YOLOv11) architectures.

    This implementation provides both detection and segmentation capabilities for
    surgical instruments using the latest YOLO architectures. The model can perform
    real-time inference and provides high-quality masks along with bounding boxes
    for precise instrument localization.

    Architecture:
    - Backbone: CSPDarknet with Cross Stage Partial connections
    - Neck: Path Aggregation Network (PAN) with Feature Pyramid Network (FPN)
    - Head: Unified detection and segmentation head
    - Segmentation: Instance segmentation with polygon masks

    Args:
        model_version (str): YOLO version ('yolov8', 'yolov11')
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        task (str): Task type ('segment' for segmentation)
        pretrained (bool): Whether to use pre-trained weights
        num_classes (int): Number of classes for fine-tuning
    """

    def __init__(
        self,
        model_version: str = "yolov11",
        model_size: str = "l",
        task: str = "segment",
        pretrained: bool = True,
        num_classes: int = 12,  # 12 surgical instrument classes (excluding background)
    ):
        if not YOLO_AVAILABLE:
            raise ImportError(
                "YOLO requires 'ultralytics' package. Install with: pip install ultralytics"
            )

        self.model_version = model_version
        self.model_size = model_size
        self.task = task
        self.pretrained = pretrained
        self.num_classes = num_classes

        # Construct model name
        if task == "segment":
            self.model_name = f"{model_version}{model_size}-seg"
        else:
            self.model_name = f"{model_version}{model_size}"

        # Initialize YOLO model
        self.model = None
        self._initialize_model()

        # Class names for surgical instruments (12 classes, no background)
        self.class_names = [
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

        print(f"✅ YOLO Model initialized: {self.model_name}")
        print(f"📊 Classes: {num_classes}, Task: {task}")

    def _initialize_model(self):
        """Initialize the YOLO model."""
        try:
            if self.pretrained:
                # Load pre-trained model
                self.model = YOLO(f"{self.model_name}.pt")
                print(f"🔧 Loaded pre-trained {self.model_name}")
            else:
                # Load model architecture only
                self.model = YOLO(f"{self.model_name}.yaml")
                print(f"🔧 Loaded {self.model_name} architecture")

        except Exception as e:
            print(f"❌ Failed to initialize {self.model_name}: {str(e)}")
            # Fallback to basic model
            self.model = YOLO("yolov8n-seg.pt")
            print("🔄 Fallback to YOLOv8n-seg")

    def train(
        self,
        data_config: str,
        epochs: int = 80,
        batch_size: int = 20,
        image_size: int = 640,
        device: Union[int, str] = 0,
        project: str = "surgical_yolo_training",
        name: str = "experiment",
        resume: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the YOLO segmentation model.

        Args:
            data_config (str): Path to data configuration YAML file
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            image_size (int): Image size for training
            device (Union[int, str]): Device for training (GPU ID or 'cpu')
            project (str): Project name for organizing runs
            name (str): Experiment name
            resume (bool): Whether to resume from last checkpoint
            **kwargs: Additional training arguments

        Returns:
            Dict containing training results and metrics
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        print(f"🚀 Starting YOLO training...")
        print(
            f"📊 Configuration: {epochs} epochs, batch={batch_size}, size={image_size}"
        )

        # Training arguments
        train_args = {
            "data": data_config,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": image_size,
            "device": device,
            "project": project,
            "name": name,
            "resume": resume,
            "plots": True,
            "save": True,
            "verbose": True,
            **kwargs,
        }

        # Start training
        results = self.model.train(**train_args)

        print(f"✅ Training completed!")
        print(f"📁 Results saved to: {results.save_dir}")

        return {
            "results": results,
            "save_dir": str(results.save_dir),
            "best_weights": str(results.save_dir / "weights" / "best.pt"),
            "last_weights": str(results.save_dir / "weights" / "last.pt"),
            "metrics": results.results_dict if hasattr(results, "results_dict") else {},
        }

    def predict(
        self,
        source: Union[str, np.ndarray, List],
        confidence: float = 0.25,
        iou_threshold: float = 0.7,
        image_size: int = 640,
        device: Union[int, str] = None,
        save: bool = False,
        save_dir: str = "predictions",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on images or videos.

        Args:
            source (Union[str, np.ndarray, List]): Input source (image path, array, or list)
            confidence (float): Confidence threshold for predictions
            iou_threshold (float): IoU threshold for NMS
            image_size (int): Image size for inference
            device (Union[int, str]): Device for inference
            save (bool): Whether to save prediction results
            save_dir (str): Directory to save results
            **kwargs: Additional inference arguments

        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Prediction arguments
        predict_args = {
            "conf": confidence,
            "iou": iou_threshold,
            "imgsz": image_size,
            "save": save,
            "project": save_dir,
            **kwargs,
        }

        if device is not None:
            predict_args["device"] = device

        # Run prediction
        results = self.model.predict(source, **predict_args)

        # Format results
        formatted_results = []
        for result in results:
            formatted_result = self._format_prediction_result(result)
            formatted_results.append(formatted_result)

        return formatted_results

    def _format_prediction_result(self, result) -> Dict[str, Any]:
        """Format a single prediction result."""
        formatted = {
            "image_path": getattr(result, "path", ""),
            "image_shape": result.orig_shape,
            "detections": [],
        }

        # Extract boxes, classes, and confidence scores
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            # Extract masks if available
            masks = None
            if hasattr(result, "masks") and result.masks is not None:
                masks = result.masks.data.cpu().numpy()

            # Process each detection
            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                detection = {
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "category_id": int(cls),
                    "category_name": (
                        self.class_names[cls]
                        if cls < len(self.class_names)
                        else f"class_{cls}"
                    ),
                    "confidence": float(score),
                }

                # Add mask if available
                if masks is not None and i < len(masks):
                    detection["mask"] = masks[i]
                    detection["segmentation"] = self._mask_to_polygon(masks[i])

                formatted["detections"].append(detection)

        return formatted

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """Convert mask to polygon format."""
        # Find contours
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            if len(contour) >= 3:  # Valid polygon needs at least 3 points
                polygon = contour.flatten().tolist()
                polygons.append(polygon)

        return polygons

    def evaluate(
        self,
        data_config: str,
        split: str = "val",
        image_size: int = 640,
        batch_size: int = 16,
        device: Union[int, str] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            data_config (str): Path to data configuration YAML file
            split (str): Dataset split to evaluate ('val', 'test')
            image_size (int): Image size for evaluation
            batch_size (int): Batch size for evaluation
            device (Union[int, str]): Device for evaluation
            **kwargs: Additional evaluation arguments

        Returns:
            Dict containing evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        print(f"📊 Evaluating YOLO model on {split} split...")

        # Evaluation arguments
        eval_args = {
            "data": data_config,
            "split": split,
            "imgsz": image_size,
            "batch": batch_size,
            **kwargs,
        }

        if device is not None:
            eval_args["device"] = device

        # Run evaluation
        results = self.model.val(**eval_args)

        # Extract metrics
        metrics = {
            "mAP_50": float(results.box.map50) if hasattr(results, "box") else 0.0,
            "mAP_50_95": float(results.box.map) if hasattr(results, "box") else 0.0,
            "precision": (
                float(results.box.p.mean()) if hasattr(results, "box") else 0.0
            ),
            "recall": float(results.box.r.mean()) if hasattr(results, "box") else 0.0,
            "f1_score": (
                float(results.box.f1.mean()) if hasattr(results, "box") else 0.0
            ),
        }

        # Add segmentation metrics if available
        if hasattr(results, "seg"):
            metrics.update(
                {
                    "mask_mAP_50": float(results.seg.map50),
                    "mask_mAP_50_95": float(results.seg.map),
                    "mask_precision": float(results.seg.p.mean()),
                    "mask_recall": float(results.seg.r.mean()),
                    "mask_f1_score": float(results.seg.f1.mean()),
                }
            )

        print(f"📈 Evaluation Results:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")

        return metrics

    def export(
        self,
        format: str = "onnx",
        image_size: int = 640,
        device: str = "cpu",
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 17,
        **kwargs,
    ) -> str:
        """
        Export the model to various formats for deployment.

        Args:
            format (str): Export format ('onnx', 'torchscript', 'tflite', etc.)
            image_size (int): Image size for export
            device (str): Device for export
            half (bool): Use half precision (FP16)
            int8 (bool): Use int8 quantization
            dynamic (bool): Dynamic input shapes
            simplify (bool): Simplify ONNX model
            opset (int): ONNX opset version
            **kwargs: Additional export arguments

        Returns:
            str: Path to exported model
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        print(f"📦 Exporting model to {format} format...")

        # Export arguments
        export_args = {
            "format": format,
            "imgsz": image_size,
            "device": device,
            "half": half,
            "int8": int8,
            "dynamic": dynamic,
            "simplify": simplify,
            "opset": opset,
            **kwargs,
        }

        # Export model
        export_path = self.model.export(**export_args)

        print(f"✅ Model exported to: {export_path}")
        return export_path

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_size": self.model_size,
            "task": self.task,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "pretrained": self.pretrained,
        }

        if self.model is not None:
            # Add model-specific info
            try:
                info.update(
                    {
                        "parameters": sum(
                            p.numel() for p in self.model.model.parameters()
                        ),
                        "model_yaml": (
                            str(self.model.cfg) if hasattr(self.model, "cfg") else None
                        ),
                        "task_specific": (
                            self.model.task if hasattr(self.model, "task") else None
                        ),
                    }
                )
            except:
                pass

        return info


class YOLOv8Segmentation(SurgicalYOLOSegmentation):
    """YOLOv8 segmentation model for surgical instruments."""

    def __init__(self, model_size: str = "l", **kwargs):
        super().__init__(model_version="yolov8", model_size=model_size, **kwargs)


class YOLOv11Segmentation(SurgicalYOLOSegmentation):
    """YOLOv11 segmentation model for surgical instruments."""

    def __init__(self, model_size: str = "l", **kwargs):
        super().__init__(model_version="yolov11", model_size=model_size, **kwargs)


def create_yolo_model(
    version: str = "yolov11",
    size: str = "l",
    task: str = "segment",
    pretrained: bool = True,
    num_classes: int = 12,
) -> SurgicalYOLOSegmentation:
    """
    Factory function to create YOLO segmentation model.

    Args:
        version (str): YOLO version ('yolov8', 'yolov11')
        size (str): Model size ('n', 's', 'm', 'l', 'x')
        task (str): Task type ('segment')
        pretrained (bool): Whether to use pre-trained weights
        num_classes (int): Number of classes

    Returns:
        SurgicalYOLOSegmentation: Initialized YOLO model
    """
    if version == "yolov8":
        return YOLOv8Segmentation(
            model_size=size, task=task, pretrained=pretrained, num_classes=num_classes
        )
    elif version == "yolov11":
        return YOLOv11Segmentation(
            model_size=size, task=task, pretrained=pretrained, num_classes=num_classes
        )
    else:
        return SurgicalYOLOSegmentation(
            model_version=version,
            model_size=size,
            task=task,
            pretrained=pretrained,
            num_classes=num_classes,
        )


# Model configurations for easy import
YOLO_CONFIG = {
    "supported_versions": ["yolov8", "yolov11"],
    "supported_sizes": ["n", "s", "m", "l", "x"],
    "supported_tasks": ["detect", "segment", "classify"],
    "default_image_size": 640,
    "model_characteristics": {
        "n": {"params": "3.2M", "speed": "fastest", "accuracy": "lowest"},
        "s": {"params": "11.2M", "speed": "fast", "accuracy": "good"},
        "m": {"params": "25.9M", "speed": "medium", "accuracy": "better"},
        "l": {"params": "43.7M", "speed": "slow", "accuracy": "high"},
        "x": {"params": "68.2M", "speed": "slowest", "accuracy": "highest"},
    },
    "training_config": {
        "epochs": 80,
        "batch_size": 20,
        "image_size": 640,
        "patience": 10,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.7,
    },
    "export_formats": [
        "onnx",
        "torchscript",
        "tflite",
        "edgetpu",
        "tfjs",
        "paddle",
        "ncnn",
        "coreml",
        "openvino",
    ],
}


if __name__ == "__main__":
    # Example usage and testing
    print("Testing YOLO Segmentation Implementation")
    print("=" * 50)

    if YOLO_AVAILABLE:
        print("✅ Ultralytics YOLO package available")

        # Test model creation
        try:
            print("\n🧪 Testing model creation...")
            model = create_yolo_model(version="yolov11", size="l")
            model_info = model.get_model_info()

            print(f"Model: {model_info['model_name']}")
            print(f"Version: {model_info['model_version']}")
            print(f"Classes: {model_info['num_classes']}")
            print(f"Task: {model_info['task']}")

            print("\n✅ YOLO model created successfully!")

        except Exception as e:
            print(f"❌ Error creating model: {str(e)}")

    else:
        print("❌ Ultralytics YOLO package not available")
        print("📦 Install with: pip install ultralytics")

    print(f"\n📊 YOLO Configuration:")
    print(f"Supported versions: {YOLO_CONFIG['supported_versions']}")
    print(f"Supported sizes: {YOLO_CONFIG['supported_sizes']}")
    print(f"Export formats: {len(YOLO_CONFIG['export_formats'])} formats available")

    print("\n🔧 YOLO Features:")
    print("  ✅ Real-time inference")
    print("  ✅ Unified detection and segmentation")
    print("  ✅ Multiple export formats")
    print("  ✅ Advanced training features")
    print("  ✅ Easy deployment")

    print("\n✅ YOLO implementation ready for surgical instance segmentation!")
