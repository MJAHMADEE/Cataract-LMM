"""
Surgical Image Preprocessing Module

This module provides image preprocessing functionality specifically designed for
surgical video analysis, with optimizations for cataract surgery instrument segmentation.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class SurgicalImagePreprocessor:
    """
    Image preprocessor specifically designed for surgical video analysis.

    This class handles preprocessing operations that are optimized for
    surgical instrumentation and lighting conditions commonly found in
    cataract surgery videos.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the surgical image preprocessor.

        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size

    def resize_image(
        self, image: np.ndarray, maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize image while optionally maintaining aspect ratio.

        Args:
            image: Input image array
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Resized image array
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            scale = min(self.target_size[0] / h, self.target_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)

            resized = cv2.resize(image, (new_w, new_h))

            # Pad to target size
            result = np.zeros(
                (self.target_size[0], self.target_size[1], 3), dtype=image.dtype
            )
            y_offset = (self.target_size[0] - new_h) // 2
            x_offset = (self.target_size[1] - new_w) // 2
            result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

            return result
        else:
            return cv2.resize(image, self.target_size[::-1])

    def normalize_surgical_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply surgical-specific normalization.

        Args:
            image: Input image array

        Returns:
            Normalized image array
        """
        # Convert to float32
        image = image.astype(np.float32) / 255.0

        # Apply surgical lighting normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        return (image - mean) / std

    def enhance_surgical_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast for better surgical instrument visibility.

        Args:
            image: Input image array

        Returns:
            Contrast-enhanced image array
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def preprocess(
        self, image: np.ndarray, enhance_contrast: bool = True
    ) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for surgical images.

        Args:
            image: Input image array
            enhance_contrast: Whether to apply contrast enhancement

        Returns:
            Dictionary containing processed image and metadata
        """
        original_shape = image.shape[:2]

        # Enhance contrast if requested
        if enhance_contrast:
            image = self.enhance_surgical_contrast(image)

        # Resize image
        resized_image = self.resize_image(image)

        # Normalize
        normalized_image = self.normalize_surgical_image(resized_image)

        return {
            "image": normalized_image,
            "original_shape": original_shape,
            "processed_shape": resized_image.shape[:2],
            "scale_factor": min(
                self.target_size[0] / original_shape[0],
                self.target_size[1] / original_shape[1],
            ),
        }


def get_surgical_transforms(
    mode: str = "train", target_size: Tuple[int, int] = (640, 640)
) -> transforms.Compose:
    """
    Get PyTorch transforms for surgical image processing.

    Args:
        mode: 'train', 'val', or 'test'
        target_size: Target image size

    Returns:
        Composed transforms
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def get_yolo_transforms(img_size: int = 640) -> Dict[str, Any]:
    """
    Get YOLO-specific transforms configuration.

    Args:
        img_size: Input image size for YOLO

    Returns:
        Transform configuration dictionary
    """
    return {
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic_prob": 1.0,
        "mixup_prob": 0.0,
        "copy_paste_prob": 0.0,
    }
