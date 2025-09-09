"""
Mask R-CNN Predictor for Surgical Instance Segmentation

This module provides the inference interface for Mask R-CNN models trained on surgical
instrument segmentation. It includes comprehensive prediction utilities, post-processing,
and result visualization capabilities.

Features:
- Single image and batch prediction
- Confidence-based filtering
- Non-maximum suppression
- Mask refinement and post-processing
- COCO-style result formatting
- Comprehensive visualization utilities

Author: Research Team
Date: August 2025
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .model import MaskRCNNPostprocessor, MaskRCNNPreprocessor, SurgicalMaskRCNN


class MaskRCNNPredictor:
    """
    Comprehensive prediction interface for Mask R-CNN surgical instance segmentation.

    This class provides a high-level interface for running inference with pre-trained
    Mask R-CNN models, including preprocessing, prediction, post-processing, and
    visualization capabilities.

    Features:
    - Automatic model loading and device management
    - Flexible input handling (file paths, arrays, PIL images)
    - Confidence and NMS filtering
    - COCO-style result formatting
    - Comprehensive visualization options
    - Batch processing capabilities
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int = 13,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ):
        """
        Initialize the Mask R-CNN predictor.

        Args:
            model_path (str): Path to the trained model checkpoint
            num_classes (int): Number of classes in the model
            device (str): Device to run inference on ('auto', 'cuda', 'cpu')
            confidence_threshold (float): Minimum confidence for predictions
            nms_threshold (float): IoU threshold for non-maximum suppression
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = self._load_model()

        # Initialize preprocessor and postprocessor
        self.preprocessor = MaskRCNNPreprocessor()
        self.postprocessor = MaskRCNNPostprocessor()

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

        print(f"✅ Mask R-CNN Predictor initialized on {self.device}")
        print(
            f"📊 Model: {num_classes} classes, Confidence: {confidence_threshold}, NMS: {nms_threshold}"
        )

    def _load_model(self) -> SurgicalMaskRCNN:
        """
        Load the trained Mask R-CNN model from checkpoint.

        Returns:
            SurgicalMaskRCNN: Loaded and initialized model
        """
        try:
            # Create model architecture
            model = SurgicalMaskRCNN(num_classes=self.num_classes)

            # Load checkpoint
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            # Move to device and set to evaluation mode
            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")

    def predict_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        return_format: str = "dict",
    ) -> Dict[str, any]:
        """
        Predict surgical instruments in a single image.

        Args:
            image_input (Union[str, np.ndarray, Image.Image]): Input image
            return_format (str): Format for results ('dict', 'coco', 'numpy')

        Returns:
            Dict containing predictions with boxes, labels, scores, and masks
        """
        # Load and preprocess image
        image_array = self._load_image(image_input)
        image_tensor = self.preprocessor.preprocess_image(image_array)

        # Run inference
        with torch.no_grad():
            self.model.eval()
            predictions = self.model([image_tensor.to(self.device)])
            pred = predictions[0]

        # Apply confidence filtering
        keep = pred["scores"] > self.confidence_threshold
        filtered_pred = {
            "boxes": pred["boxes"][keep],
            "labels": pred["labels"][keep],
            "scores": pred["scores"][keep],
            "masks": pred["masks"][keep],
        }

        # Apply NMS
        final_pred = self.postprocessor.apply_nms(filtered_pred, self.nms_threshold)

        # Format results
        if return_format == "dict":
            return self._format_dict_results(final_pred, image_array.shape)
        elif return_format == "coco":
            return self._format_coco_results(final_pred, image_array.shape)
        elif return_format == "numpy":
            return self._format_numpy_results(final_pred, image_array.shape)
        else:
            raise ValueError(f"Unknown return format: {return_format}")

    def predict_batch(
        self,
        image_inputs: List[Union[str, np.ndarray, Image.Image]],
        batch_size: int = 4,
    ) -> List[Dict[str, any]]:
        """
        Predict surgical instruments in a batch of images.

        Args:
            image_inputs (List): List of input images
            batch_size (int): Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i : i + batch_size]
            batch_results = []

            for image_input in batch:
                result = self.predict_image(image_input)
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def visualize_predictions(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        save_path: Optional[str] = None,
        show_confidence: bool = True,
        show_masks: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ) -> np.ndarray:
        """
        Visualize predictions on an image with comprehensive annotation.

        Args:
            image_input (Union[str, np.ndarray, Image.Image]): Input image
            save_path (Optional[str]): Path to save the visualization
            show_confidence (bool): Whether to show confidence scores
            show_masks (bool): Whether to overlay segmentation masks
            figsize (Tuple[int, int]): Figure size for visualization

        Returns:
            np.ndarray: Visualization image as numpy array
        """
        # Load image and get predictions
        image_array = self._load_image(image_input)
        predictions = self.predict_image(image_array)

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image_array)

        # Color map for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))

        # Draw predictions
        for i, detection in enumerate(predictions["detections"]):
            box = detection["bbox"]
            label = detection["category_id"]
            score = detection["score"]

            # Draw bounding box
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1

            color = colors[label]
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Add label
            class_name = self.class_names[label]
            label_text = f"{class_name}"
            if show_confidence:
                label_text += f": {score:.2f}"

            ax.text(
                x1,
                y1 - 5,
                label_text,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
                fontsize=10,
                color="white",
                weight="bold",
            )

            # Draw mask if available
            if show_masks and "segmentation" in detection:
                mask = detection["segmentation"]
                if isinstance(mask, np.ndarray):
                    masked_image = np.ma.masked_where(mask < 0.5, mask)
                    ax.imshow(masked_image, alpha=0.4, cmap="jet")

        ax.set_title(
            "Surgical Instance Segmentation Results", fontsize=16, weight="bold"
        )
        ax.axis("off")

        # Add legend
        if len(predictions["detections"]) > 0:
            unique_labels = list(
                set(det["category_id"] for det in predictions["detections"])
            )
            legend_elements = [
                patches.Patch(color=colors[label], label=self.class_names[label])
                for label in unique_labels
            ]
            ax.legend(
                handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1)
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

    def evaluate_on_dataset(
        self,
        dataset_path: str,
        annotation_file: str,
        output_dir: str = "./evaluation_results",
    ) -> Dict[str, float]:
        """
        Evaluate the model on a complete dataset using COCO metrics.

        Args:
            dataset_path (str): Path to the dataset images
            annotation_file (str): Path to COCO annotation file
            output_dir (str): Directory to save evaluation results

        Returns:
            Dict containing evaluation metrics
        """
        import os

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load annotations
        coco_gt = COCO(annotation_file)
        image_ids = list(coco_gt.imgs.keys())

        # Generate predictions
        predictions = []
        print(f"🔍 Evaluating on {len(image_ids)} images...")

        for img_id in image_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(dataset_path, img_info["file_name"])

            if os.path.exists(img_path):
                pred = self.predict_image(img_path, return_format="coco")
                predictions.extend(pred)

        # Save predictions
        pred_file = os.path.join(output_dir, "predictions.json")
        with open(pred_file, "w") as f:
            json.dump(predictions, f)

        # Evaluate using COCO metrics
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
        }

        print(f"📊 Evaluation Results:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")

        return metrics

    def _load_image(
        self, image_input: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Load image from various input formats."""
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        return image

    def _format_dict_results(
        self, predictions: Dict[str, torch.Tensor], image_shape: Tuple[int, int, int]
    ) -> Dict[str, any]:
        """Format predictions as dictionary with detailed information."""
        detections = []

        for i in range(len(predictions["boxes"])):
            box = predictions["boxes"][i].cpu().numpy()
            label = predictions["labels"][i].cpu().item()
            score = predictions["scores"][i].cpu().item()
            mask = predictions["masks"][i].squeeze().cpu().numpy()

            detection = {
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "category_id": int(label),
                "category_name": self.class_names[label],
                "score": float(score),
                "segmentation": mask,
                "area": float(np.sum(mask > 0.5)),
            }
            detections.append(detection)

        return {
            "image_shape": image_shape,
            "num_detections": len(detections),
            "detections": detections,
            "model_info": {
                "model_type": "Mask R-CNN",
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold,
            },
        }

    def _format_coco_results(
        self, predictions: Dict[str, torch.Tensor], image_shape: Tuple[int, int, int]
    ) -> List[Dict[str, any]]:
        """Format predictions in COCO evaluation format."""
        results = []

        for i in range(len(predictions["boxes"])):
            box = predictions["boxes"][i].cpu().numpy()
            label = predictions["labels"][i].cpu().item()
            score = predictions["scores"][i].cpu().item()
            mask = predictions["masks"][i].squeeze().cpu().numpy()

            # Convert mask to RLE format
            from pycocotools import mask as maskUtils

            rle = maskUtils.encode(np.asfortranarray((mask > 0.5).astype(np.uint8)))
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("utf-8")

            result = {
                "category_id": int(label),
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1]),
                ],
                "segmentation": rle,
                "score": float(score),
            }
            results.append(result)

        return results

    def _format_numpy_results(
        self, predictions: Dict[str, torch.Tensor], image_shape: Tuple[int, int, int]
    ) -> Dict[str, np.ndarray]:
        """Format predictions as numpy arrays."""
        return {
            "boxes": predictions["boxes"].cpu().numpy(),
            "labels": predictions["labels"].cpu().numpy(),
            "scores": predictions["scores"].cpu().numpy(),
            "masks": predictions["masks"].cpu().numpy(),
        }


# Convenience function for quick inference
def predict_surgical_instruments(
    image_path: str,
    model_path: str,
    confidence_threshold: float = 0.5,
    save_visualization: bool = True,
    output_dir: str = "./predictions",
) -> Dict[str, any]:
    """
    Convenience function for quick surgical instrument prediction.

    Args:
        image_path (str): Path to input image
        model_path (str): Path to trained model
        confidence_threshold (float): Confidence threshold for predictions
        save_visualization (bool): Whether to save visualization
        output_dir (str): Output directory for results

    Returns:
        Dict containing predictions and metadata
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize predictor
    predictor = MaskRCNNPredictor(
        model_path=model_path, confidence_threshold=confidence_threshold
    )

    # Run prediction
    results = predictor.predict_image(image_path)

    # Save visualization if requested
    if save_visualization:
        image_name = Path(image_path).stem
        viz_path = os.path.join(output_dir, f"{image_name}_segmentation.png")
        predictor.visualize_predictions(image_path, save_path=viz_path)

    # Save results
    results_path = os.path.join(output_dir, f"{Path(image_path).stem}_results.json")
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = results.copy()
        for detection in json_results["detections"]:
            if isinstance(detection["segmentation"], np.ndarray):
                detection["segmentation"] = detection["segmentation"].tolist()
        json.dump(json_results, f, indent=2)

    print(f"✅ Prediction completed!")
    print(f"📊 Found {results['num_detections']} instruments")
    print(f"💾 Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    # Example usage
    print("Testing Mask R-CNN Predictor")
    print("=" * 40)

    # This would normally require a trained model
    print("⚠️  Note: This example requires a trained model checkpoint")
    print("📚 See training scripts to train a model first")

    # Example of how to use the predictor:
    """
    predictor = MaskRCNNPredictor(
        model_path='path/to/trained/model.pth',
        confidence_threshold=0.5
    )
    
    results = predictor.predict_image('surgical_image.jpg')
    visualization = predictor.visualize_predictions('surgical_image.jpg')
    """
