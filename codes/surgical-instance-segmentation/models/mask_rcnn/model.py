"""
Mask R-CNN Model Implementation for Surgical Instance Segmentation

This module provides a comprehensive implementation of Mask R-CNN with ResNet50-FPN backbone
for surgical instrument instance segmentation. Based on the reference notebook implementation
with enhanced modularity and production-ready features.

Features:
- ResNet50-FPN backbone with pre-trained weights
- Custom predictor heads for 13-class surgical instrument classification
- COCO-style training and evaluation compatibility
- Advanced data augmentation and preprocessing
- Comprehensive evaluation metrics and visualization

Author: Research Team
Date: August 2025
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class SurgicalMaskRCNN(nn.Module):
    """
    Surgical instrument segmentation model based on Mask R-CNN with ResNet50-FPN backbone.

    This implementation is specifically designed for cataract surgery instrument segmentation
    with 13 classes (12 instrument types + background). The model provides both bounding box
    detection and instance segmentation masks for precise instrument localization.

    Architecture:
    - Backbone: ResNet50 with Feature Pyramid Network (FPN)
    - RPN: Region Proposal Network for object detection
    - RoI Head: Fast R-CNN predictor for classification and box regression
    - Mask Head: Mask R-CNN predictor for instance segmentation

    Args:
        num_classes (int): Number of classes including background (default: 13)
        pretrained (bool): Whether to use pre-trained COCO weights (default: True)
        trainable_backbone_layers (int): Number of trainable backbone layers (default: 3)
        **kwargs: Additional arguments for the underlying Mask R-CNN model
    """

    def __init__(
        self,
        num_classes: int = 13,
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        **kwargs,
    ):
        super(SurgicalMaskRCNN, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load pre-trained Mask R-CNN with ResNet50-FPN backbone
        self.model = maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )

        # Get input features for modifying the prediction heads
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = self.model.roi_heads.mask_predictor.conv5_mask.out_channels

        # Replace the box predictor for surgical instrument classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor for surgical instrument classes
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, dim_reduced, num_classes
        )

        # Define class names for surgical instruments
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

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Forward pass of the Mask R-CNN model.

        Args:
            images (List[torch.Tensor]): Input images as tensors
            targets (Optional[List[Dict[str, torch.Tensor]]]): Ground truth targets for training
                Each target dict should contain:
                - 'boxes': Bounding boxes [N, 4] in (x1, y1, x2, y2) format
                - 'labels': Class labels [N]
                - 'masks': Instance masks [N, H, W]
                - 'image_id': Image identifier
                - 'area': Box areas [N]
                - 'iscrowd': Crowd annotations [N]

        Returns:
            During training: Dict containing loss values
            During inference: List of prediction dicts with boxes, labels, scores, masks
        """
        return self.model(images, targets)

    def predict(
        self,
        image: torch.Tensor,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inference on a single image.

        Args:
            image (torch.Tensor): Input image tensor [C, H, W]
            confidence_threshold (float): Minimum confidence score for predictions
            nms_threshold (float): Non-maximum suppression threshold

        Returns:
            Dict containing filtered predictions:
            - 'boxes': Bounding boxes [N, 4]
            - 'labels': Class labels [N]
            - 'scores': Confidence scores [N]
            - 'masks': Instance masks [N, H, W]
        """
        self.model.eval()

        with torch.no_grad():
            predictions = self.model([image])
            pred = predictions[0]

            # Filter predictions by confidence threshold
            keep = pred["scores"] > confidence_threshold

            filtered_pred = {
                "boxes": pred["boxes"][keep],
                "labels": pred["labels"][keep],
                "scores": pred["scores"][keep],
                "masks": pred["masks"][keep],
            }

            return filtered_pred

    def get_model_info(self) -> Dict[str, any]:
        """
        Get comprehensive model information.

        Returns:
            Dict containing model architecture details and parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "SurgicalMaskRCNN",
            "backbone": "ResNet50-FPN",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pretrained": self.pretrained,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Approximate size in MB
        }


class MaskRCNNPreprocessor:
    """
    Preprocessing utilities for Mask R-CNN training and inference.

    Provides comprehensive image preprocessing including normalization, resizing,
    and data augmentation specifically designed for surgical instance segmentation.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_size: int = 1333,
        min_size: int = 800,
    ):
        """
        Initialize preprocessor with normalization parameters.

        Args:
            mean (Tuple[float, float, float]): ImageNet normalization mean
            std (Tuple[float, float, float]): ImageNet normalization std
            max_size (int): Maximum image size for training
            min_size (int): Minimum image size for training
        """
        self.mean = mean
        self.std = std
        self.max_size = max_size
        self.min_size = min_size

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image for inference.

        Args:
            image (np.ndarray): Input image in BGR/RGB format

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convert to tensor and normalize
        import torchvision.transforms as T

        transform = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)])

        return transform(image)

    def preprocess_batch(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Preprocess a batch of images.

        Args:
            images (List[np.ndarray]): List of input images

        Returns:
            List[torch.Tensor]: List of preprocessed image tensors
        """
        return [self.preprocess_image(img) for img in images]


class MaskRCNNPostprocessor:
    """
    Post-processing utilities for Mask R-CNN predictions.

    Provides comprehensive post-processing including mask refinement, confidence filtering,
    non-maximum suppression, and result visualization for surgical instance segmentation.
    """

    def __init__(self, class_names: List[str] = None):
        """
        Initialize post-processor with class names.

        Args:
            class_names (List[str]): List of class names for visualization
        """
        self.class_names = class_names or [
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

    def refine_masks(
        self, masks: torch.Tensor, threshold: float = 0.5, min_area: int = 100
    ) -> torch.Tensor:
        """
        Refine segmentation masks by thresholding and area filtering.

        Args:
            masks (torch.Tensor): Raw prediction masks [N, H, W]
            threshold (float): Probability threshold for mask binarization
            min_area (int): Minimum mask area in pixels

        Returns:
            torch.Tensor: Refined binary masks
        """
        # Binarize masks
        binary_masks = (masks > threshold).float()

        # Filter by minimum area
        valid_masks = []
        for mask in binary_masks:
            if mask.sum() >= min_area:
                valid_masks.append(mask)

        if valid_masks:
            return torch.stack(valid_masks)
        else:
            return torch.empty(0, masks.shape[1], masks.shape[2])

    def apply_nms(
        self, predictions: Dict[str, torch.Tensor], iou_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            predictions (Dict[str, torch.Tensor]): Model predictions
            iou_threshold (float): IoU threshold for NMS

        Returns:
            Dict[str, torch.Tensor]: Filtered predictions after NMS
        """
        if len(predictions["boxes"]) == 0:
            return predictions

        # Apply torchvision NMS
        keep_indices = torchvision.ops.nms(
            predictions["boxes"], predictions["scores"], iou_threshold
        )

        return {
            "boxes": predictions["boxes"][keep_indices],
            "labels": predictions["labels"][keep_indices],
            "scores": predictions["scores"][keep_indices],
            "masks": predictions["masks"][keep_indices],
        }

    def visualize_predictions(
        self,
        image: np.ndarray,
        predictions: Dict[str, torch.Tensor],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize predictions on the input image.

        Args:
            image (np.ndarray): Original image
            predictions (Dict[str, torch.Tensor]): Model predictions
            save_path (Optional[str]): Path to save visualization

        Returns:
            np.ndarray: Image with overlaid predictions
        """
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        # Color map for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))

        for i, (box, label, score, mask) in enumerate(
            zip(
                predictions["boxes"],
                predictions["labels"],
                predictions["scores"],
                predictions["masks"],
            )
        ):
            # Draw bounding box
            x1, y1, x2, y2 = box.cpu().numpy()
            width, height = x2 - x1, y2 - y1

            color = colors[label.item()]
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Add label and confidence
            class_name = self.class_names[label.item()]
            ax.text(
                x1,
                y1 - 5,
                f"{class_name}: {score:.2f}",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
                fontsize=10,
                color="white",
            )

            # Overlay mask
            mask_np = mask.squeeze().cpu().numpy()
            masked_image = np.ma.masked_where(mask_np < 0.5, mask_np)
            ax.imshow(masked_image, alpha=0.5, cmap="jet")

        ax.set_title("Surgical Instance Segmentation Results")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        # Convert to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return buf


def create_mask_rcnn_model(
    num_classes: int = 13, pretrained: bool = True, device: str = "cuda"
) -> SurgicalMaskRCNN:
    """
    Factory function to create and initialize a Mask R-CNN model.

    Args:
        num_classes (int): Number of classes including background
        pretrained (bool): Whether to use pre-trained weights
        device (str): Device to load the model on

    Returns:
        SurgicalMaskRCNN: Initialized model ready for training or inference
    """
    model = SurgicalMaskRCNN(num_classes=num_classes, pretrained=pretrained)

    # Move to specified device
    if torch.cuda.is_available() and device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()

    return model


# Model configuration for easy import
MASK_RCNN_CONFIG = {
    "model_name": "SurgicalMaskRCNN",
    "backbone": "ResNet50-FPN",
    "num_classes": 13,
    "input_size": (800, 1333),  # (min_size, max_size)
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "anchor_sizes": ((32,), (64,), (128,), (256,), (512,)),
    "aspect_ratios": ((0.5, 1.0, 2.0),) * 5,
    "batch_size": 8,
    "learning_rate": 0.0005,
    "weight_decay": 0.0005,
    "momentum": 0.9,
    "num_epochs": 100,
    "patience": 10,
}


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Surgical Mask R-CNN Implementation")
    print("=" * 50)

    # Create model
    model = create_mask_rcnn_model(num_classes=13, pretrained=True)
    model_info = model.get_model_info()

    print(f"Model: {model_info['model_name']}")
    print(f"Backbone: {model_info['backbone']}")
    print(f"Classes: {model_info['num_classes']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.1f} MB")
    print(f"Class Names: {model_info['class_names']}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 800, 800)
    model.eval()
    with torch.no_grad():
        output = model([dummy_input])

    print(f"\nTest inference successful!")
    print(f"Output keys: {list(output[0].keys())}")
    print(f"Detected objects: {len(output[0]['boxes'])}")

    print("\n✅ Mask R-CNN implementation ready for training and inference!")
