"""
YOLO Predictor for Surgical Instance Segmentation

This module provides the inference interface for YOLO segmentation models (YOLOv8/YOLOv11)
for surgical instance segmentation. It includes comprehensive prediction utilities,
evaluation capabilities, and result visualization.

Features:
- Real-time inference with YOLO models
- Batch processing capabilities
- Performance benchmarking
- Comprehensive visualization utilities
- Export and deployment support
- Advanced post-processing

Author: Research Team
Date: August 2025
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .yolo_models import SurgicalYOLOSegmentation, create_yolo_model

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print(
        "⚠️  Warning: ultralytics package not installed. YOLO functionality will be limited."
    )
    YOLO_AVAILABLE = False


class YOLOPredictor:
    """
    Comprehensive prediction interface for YOLO-based surgical instance segmentation.

    This class provides a high-level interface for running inference with trained
    YOLO segmentation models, including preprocessing, prediction, post-processing,
    and comprehensive visualization capabilities.

    Features:
    - Real-time inference capabilities
    - Batch processing for multiple images/videos
    - Performance benchmarking and optimization
    - Comprehensive result visualization
    - Export capabilities for deployment
    - Advanced post-processing and filtering
    """

    def __init__(
        self,
        model_path: str,
        model_version: str = "yolov11",
        device: str = "auto",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        image_size: int = 640,
    ):
        """
        Initialize the YOLO predictor.

        Args:
            model_path (str): Path to the trained YOLO model weights
            model_version (str): YOLO version ('yolov8', 'yolov11')
            device (str): Device to run inference on ('auto', 'cuda', 'cpu')
            confidence_threshold (float): Minimum confidence for predictions
            iou_threshold (float): IoU threshold for NMS
            image_size (int): Image size for inference
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "YOLO requires 'ultralytics' package. Install with: pip install ultralytics"
            )

        self.model_path = model_path
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model = self._load_model()

        # Class names for surgical instruments
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

        print(f"✅ YOLO Predictor initialized: {model_version} on {self.device}")
        print(
            f"📊 Confidence: {confidence_threshold}, IoU: {iou_threshold}, Size: {image_size}"
        )

    def _load_model(self) -> YOLO:
        """Load the trained YOLO model."""
        try:
            if os.path.exists(self.model_path):
                model = YOLO(self.model_path)
                print(f"🔧 Loaded model from: {self.model_path}")
            else:
                # Load pre-trained model if path doesn't exist
                model = YOLO(self.model_path)  # Ultralytics will handle downloading
                print(f"🔧 Loaded pre-trained model: {self.model_path}")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def predict_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        save_results: bool = False,
        output_dir: str = "./predictions",
        return_format: str = "dict",
    ) -> Dict[str, Any]:
        """
        Predict surgical instruments in a single image.

        Args:
            image_input (Union[str, np.ndarray, Image.Image]): Input image
            save_results (bool): Whether to save prediction results
            output_dir (str): Directory to save results
            return_format (str): Format for results ('dict', 'ultralytics')

        Returns:
            Dict containing predictions with boxes, labels, scores, and masks
        """
        # Run inference
        results = self.model.predict(
            source=image_input,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            save=save_results,
            project=output_dir,
            verbose=False,
        )

        if return_format == "ultralytics":
            return results

        # Format results
        formatted_results = []
        for result in results:
            formatted_result = self._format_prediction_result(result)
            formatted_results.append(formatted_result)

        return (
            formatted_results[0] if len(formatted_results) == 1 else formatted_results
        )

    def predict_batch(
        self,
        image_inputs: List[Union[str, np.ndarray, Image.Image]],
        batch_size: int = 16,
        save_results: bool = False,
        output_dir: str = "./batch_predictions",
    ) -> List[Dict[str, Any]]:
        """
        Predict surgical instruments in a batch of images.

        Args:
            image_inputs (List): List of input images
            batch_size (int): Batch size for processing
            save_results (bool): Whether to save results
            output_dir (str): Directory to save results

        Returns:
            List of prediction dictionaries
        """
        all_results = []

        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i : i + batch_size]

            # Run batch inference
            results = self.model.predict(
                source=batch,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                save=save_results,
                project=output_dir,
                verbose=False,
            )

            # Format results
            for result in results:
                formatted_result = self._format_prediction_result(result)
                all_results.append(formatted_result)

        return all_results

    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        save_frames: bool = False,
        frame_skip: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Predict surgical instruments in a video.

        Args:
            video_path (str): Path to input video
            output_path (Optional[str]): Path to save annotated video
            save_frames (bool): Whether to save individual frames
            frame_skip (int): Number of frames to skip between predictions

        Returns:
            List of prediction dictionaries for each frame
        """
        # Run video inference
        results = self.model.predict(
            source=video_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            save=output_path is not None,
            save_frames=save_frames,
            verbose=True,
        )

        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            if i % (frame_skip + 1) == 0:  # Apply frame skipping
                formatted_result = self._format_prediction_result(result)
                formatted_result["frame_index"] = i
                formatted_results.append(formatted_result)

        return formatted_results

    def _format_prediction_result(self, result) -> Dict[str, Any]:
        """Format a single prediction result."""
        formatted = {
            "image_path": getattr(result, "path", ""),
            "image_shape": result.orig_shape,
            "inference_time_ms": getattr(result, "speed", {}).get("inference", 0.0),
            "num_detections": 0,
            "detections": [],
        }

        # Extract predictions if available
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            formatted["num_detections"] = len(boxes)

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
                    "area": float((box[2] - box[0]) * (box[3] - box[1])),
                }

                # Add mask if available
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    detection["mask"] = mask
                    detection["mask_area"] = float(np.sum(mask > 0.5))
                    detection["segmentation"] = self._mask_to_polygon(mask)

                formatted["detections"].append(detection)

        return formatted

    def _mask_to_polygon(
        self, mask: np.ndarray, simplify_tolerance: float = 1.0
    ) -> List[List[float]]:
        """Convert mask to polygon format with optional simplification."""
        # Find contours
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            if len(contour) >= 3:  # Valid polygon needs at least 3 points
                # Simplify contour to reduce polygon complexity
                epsilon = simplify_tolerance * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) >= 3:
                    polygon = approx.flatten().tolist()
                    polygons.append(polygon)

        return polygons

    def visualize_predictions(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        predictions: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        show_confidence: bool = True,
        show_masks: bool = True,
        mask_alpha: float = 0.5,
        figsize: Tuple[int, int] = (15, 10),
    ) -> np.ndarray:
        """
        Visualize YOLO predictions with comprehensive annotation.

        Args:
            image_input (Union[str, np.ndarray, Image.Image]): Input image
            predictions (Optional[Dict[str, Any]]): Predictions (if None, will predict)
            save_path (Optional[str]): Path to save visualization
            show_confidence (bool): Whether to show confidence scores
            show_masks (bool): Whether to overlay masks
            mask_alpha (float): Transparency for mask overlay
            figsize (Tuple[int, int]): Figure size for visualization

        Returns:
            np.ndarray: Visualization image as numpy array
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
        else:
            image = image_input.copy()

        # Get predictions if not provided
        if predictions is None:
            predictions = self.predict_image(image_input)

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image)

        # Color map for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))

        # Draw predictions
        for detection in predictions["detections"]:
            bbox = detection["bbox"]
            category_id = detection["category_id"]
            confidence = detection["confidence"]

            # Get color for this class
            color = colors[category_id] if category_id < len(colors) else colors[0]

            # Draw bounding box
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1

            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Draw mask if available
            if show_masks and "mask" in detection:
                mask = detection["mask"]
                masked_image = np.ma.masked_where(mask < 0.5, mask)
                ax.imshow(masked_image, alpha=mask_alpha, cmap="jet")

            # Add label
            class_name = detection["category_name"]
            label_text = f"{class_name}"
            if show_confidence:
                label_text += f": {confidence:.2f}"

            ax.text(
                x1,
                y1 - 5,
                label_text,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
                fontsize=10,
                color="white",
                weight="bold",
            )

        # Add title and info
        title = f"YOLO Segmentation Results ({self.model_version.upper()})"
        ax.set_title(title, fontsize=16, weight="bold")
        ax.axis("off")

        # Add performance info
        num_detections = predictions["num_detections"]
        inference_time = predictions.get("inference_time_ms", 0)
        info_text = f"Detections: {num_detections} | Time: {inference_time:.1f}ms"
        ax.text(
            0.02,
            0.98,
            info_text,
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
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        batch_sizes: List[int] = [1, 4, 8, 16],
    ) -> Dict[str, Any]:
        """
        Benchmark YOLO inference performance.

        Args:
            test_images (List[str]): List of test image paths
            warmup_runs (int): Number of warmup iterations
            benchmark_runs (int): Number of benchmark iterations
            batch_sizes (List[int]): Batch sizes to test

        Returns:
            Dict containing performance metrics
        """
        print(f"🏃 Benchmarking YOLO {self.model_version} performance...")

        results = {
            "model_version": self.model_version,
            "device": self.device,
            "image_size": self.image_size,
            "batch_results": {},
        }

        for batch_size in batch_sizes:
            print(f"🔥 Testing batch size: {batch_size}")

            # Warmup
            for i in range(warmup_runs):
                batch = test_images[:batch_size] * (batch_size // len(test_images) + 1)
                batch = batch[:batch_size]
                self.predict_batch(batch, batch_size=batch_size)

            # Benchmark
            times = []
            total_detections = 0

            for i in range(benchmark_runs):
                batch = test_images[:batch_size] * (batch_size // len(test_images) + 1)
                batch = batch[:batch_size]

                start_time = time.time()
                batch_results = self.predict_batch(batch, batch_size=batch_size)
                end_time = time.time()

                times.append(end_time - start_time)
                total_detections += sum(r["num_detections"] for r in batch_results)

            # Calculate metrics
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            avg_detections = total_detections / (benchmark_runs * batch_size)

            results["batch_results"][batch_size] = {
                "avg_time_s": avg_time,
                "std_time_s": np.std(times),
                "throughput_fps": throughput,
                "avg_detections_per_image": avg_detections,
                "total_benchmark_runs": benchmark_runs,
            }

            print(f"   Throughput: {throughput:.2f} FPS")
            print(f"   Avg time: {avg_time*1000:.2f}ms")

        return results

    def export_model(
        self,
        export_format: str = "onnx",
        output_path: Optional[str] = None,
        **export_kwargs,
    ) -> str:
        """
        Export the YOLO model for deployment.

        Args:
            export_format (str): Export format ('onnx', 'torchscript', etc.)
            output_path (Optional[str]): Path for exported model
            **export_kwargs: Additional export arguments

        Returns:
            str: Path to exported model
        """
        export_path = self.model.export(
            format=export_format, imgsz=self.image_size, **export_kwargs
        )

        if output_path and export_path != output_path:
            import shutil

            shutil.move(export_path, output_path)
            export_path = output_path

        print(f"📦 Model exported to: {export_path}")
        return export_path

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model and predictor information."""
        info = {
            "predictor_type": "YOLOPredictor",
            "model_version": self.model_version,
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "image_size": self.image_size,
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
        }

        # Add model-specific info if available
        if hasattr(self.model, "model"):
            try:
                info["model_parameters"] = sum(
                    p.numel() for p in self.model.model.parameters()
                )
            except:
                pass

        return info


# Convenience function for quick YOLO prediction
def predict_with_yolo(
    image_path: str,
    model_path: str,
    model_version: str = "yolov11",
    confidence_threshold: float = 0.25,
    save_visualization: bool = True,
    output_dir: str = "./yolo_predictions",
) -> Dict[str, Any]:
    """
    Convenience function for quick YOLO prediction.

    Args:
        image_path (str): Path to input image
        model_path (str): Path to YOLO model weights
        model_version (str): YOLO version ('yolov8', 'yolov11')
        confidence_threshold (float): Confidence threshold
        save_visualization (bool): Whether to save visualization
        output_dir (str): Output directory for results

    Returns:
        Dict containing predictions and metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize predictor
    predictor = YOLOPredictor(
        model_path=model_path,
        model_version=model_version,
        confidence_threshold=confidence_threshold,
    )

    # Run prediction
    results = predictor.predict_image(image_path)

    # Save visualization if requested
    if save_visualization:
        image_name = Path(image_path).stem
        viz_path = os.path.join(output_dir, f"{image_name}_yolo_segmentation.png")
        predictor.visualize_predictions(image_path, results, save_path=viz_path)

    # Save results
    results_path = os.path.join(output_dir, f"{Path(image_path).stem}_results.json")
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = results.copy()
        for detection in json_results["detections"]:
            if "mask" in detection and isinstance(detection["mask"], np.ndarray):
                detection["mask"] = detection["mask"].tolist()
        json.dump(json_results, f, indent=2)

    print(f"✅ YOLO prediction completed!")
    print(f"📊 Found {results['num_detections']} instruments")
    print(f"💾 Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    # Example usage
    print("Testing YOLO Predictor")
    print("=" * 40)

    if YOLO_AVAILABLE:
        print("✅ Ultralytics YOLO package available")
        print("\n🔧 YOLO Predictor Features:")
        print("  ✅ Real-time inference")
        print("  ✅ Batch processing")
        print("  ✅ Video processing")
        print("  ✅ Performance benchmarking")
        print("  ✅ Model export capabilities")
        print("  ✅ Comprehensive visualization")

        # Example of how to use the predictor:
        """
        predictor = YOLOPredictor(
            model_path='path/to/trained/yolo_model.pt',
            model_version='yolov11'
        )
        
        results = predictor.predict_image('surgical_image.jpg')
        visualization = predictor.visualize_predictions('surgical_image.jpg', results)
        """

    else:
        print("❌ Ultralytics YOLO package not available")
        print("📦 Install with: pip install ultralytics")

    print("\n⚠️  Note: This example requires trained YOLO model weights")
    print("📚 See training scripts to train a model first")
