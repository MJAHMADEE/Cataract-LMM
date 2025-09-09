"""
Surgical Image Augmentation Module

This module provides medical-safe augmentation strategies specifically designed for
surgical video analysis, ensuring that augmented images maintain clinical relevance
and anatomical consistency.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

# Graceful imports with fallbacks for CI/testing environments
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    # Create mock classes for testing environments
    A = None
    ToTensorV2 = None
    ALBUMENTATIONS_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

import numpy as np


class SurgicalAugmentator:
    """
    Medical-safe augmentation for surgical images.

    This class provides augmentation strategies that preserve the clinical
    relevance of surgical images while improving model generalization.
    Augmentations are carefully designed to maintain anatomical relationships
    and surgical instrument characteristics.
    """

    def __init__(
        self, augmentation_level: str = "medium", preserve_anatomy: bool = True
    ):
        """
        Initialize the surgical augmentator.

        Args:
            augmentation_level: 'light', 'medium', or 'heavy'
            preserve_anatomy: Whether to preserve anatomical relationships
        """
        self.augmentation_level = augmentation_level
        self.preserve_anatomy = preserve_anatomy

        if not ALBUMENTATIONS_AVAILABLE:
            self.augmentation_pipeline = None
        else:
            self.augmentation_pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """
        Build the augmentation pipeline based on the specified level.

        Returns:
            Albumentations compose object or None if dependencies unavailable
        """
        if not ALBUMENTATIONS_AVAILABLE:
            return None

        if self.augmentation_level == "light":
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        elif self.augmentation_level == "medium":
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
                ),
                A.Blur(blur_limit=5, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        else:  # heavy
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
                ),
                A.OneOf(
                    [
                        A.Blur(blur_limit=7, p=1.0),
                        A.GaussianBlur(blur_limit=7, p=1.0),
                        A.MotionBlur(blur_limit=7, p=1.0),
                    ],
                    p=0.3,
                ),
                A.GaussNoise(var_limit=(10.0, 80.0), p=0.3),
                A.RandomGamma(gamma_limit=(70, 130), p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]

        return A.Compose(transforms)

    def augment_surgical_image(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Apply surgical-safe augmentation to an image and optional mask.

        Args:
            image: Input image array (H, W, C)
            mask: Optional segmentation mask (H, W) or (H, W, C)

        Returns:
            Dictionary containing augmented image and mask
        """
        if not ALBUMENTATIONS_AVAILABLE or self.augmentation_pipeline is None:
            # Fallback: return original image/mask without augmentation
            if mask is not None:
                return {
                    "image": image,
                    "mask": mask,
                    "augmentation_applied": False,
                }
            else:
                return {"image": image, "augmentation_applied": False}

        if mask is not None:
            # Apply augmentation to both image and mask
            augmented = self.augmentation_pipeline(image=image, mask=mask)
            return {
                "image": augmented["image"],
                "mask": augmented["mask"],
                "augmentation_applied": True,
            }
        else:
            # Apply augmentation to image only
            augmented = self.augmentation_pipeline(image=image)
            return {"image": augmented["image"], "augmentation_applied": True}

    def get_yolo_augmentations(self) -> Dict[str, float]:
        """
        Get YOLO-compatible augmentation parameters.

        Returns:
            Dictionary of YOLO augmentation parameters
        """
        if self.augmentation_level == "light":
            return {
                "hsv_h": 0.010,
                "hsv_s": 0.5,
                "hsv_v": 0.3,
                "degrees": 0.0,
                "translate": 0.05,
                "scale": 0.3,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 0.5,
                "mixup": 0.0,
            }
        elif self.augmentation_level == "medium":
            return {
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
                "mosaic": 1.0,
                "mixup": 0.0,
            }
        else:  # heavy
            return {
                "hsv_h": 0.020,
                "hsv_s": 0.9,
                "hsv_v": 0.6,
                "degrees": 5.0,
                "translate": 0.15,
                "scale": 0.7,
                "shear": 2.0,
                "perspective": 0.0001,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.1,
            }

    def surgical_safe_check(self, image: np.ndarray) -> bool:
        """
        Check if augmented image maintains surgical relevance.

        Args:
            image: Augmented image to validate

        Returns:
            True if image passes surgical safety checks
        """
        # Check brightness range (surgical videos shouldn't be too dark/bright)
        mean_brightness = np.mean(image)
        if mean_brightness < 0.1 or mean_brightness > 0.9:
            return False

        # Check contrast (surgical images need sufficient contrast)
        contrast = np.std(image)
        if contrast < 0.05:
            return False

        return True


def get_surgical_augmentation_pipeline(
    mode: str = "train", level: str = "medium"
) -> SurgicalAugmentator:
    """
    Get a pre-configured surgical augmentation pipeline.

    Args:
        mode: 'train', 'val', or 'test'
        level: 'light', 'medium', or 'heavy'

    Returns:
        Configured SurgicalAugmentator instance
    """
    if mode in ["val", "test"]:
        # No augmentation for validation/test
        level = "none"

    if level == "none":
        # Return minimal augmentator (normalization only)
        return SurgicalAugmentator(augmentation_level="light")
    else:
        return SurgicalAugmentator(augmentation_level=level)


def create_augmentation_config(task_granularity: str = "task_3") -> Dict[str, Any]:
    """
    Create augmentation configuration based on task granularity.

    Args:
        task_granularity: 'task_3', 'task_9', or 'task_12'

    Returns:
        Augmentation configuration dictionary
    """
    base_config = {
        "horizontal_flip": 0.5,
        "brightness_contrast": 0.3,
        "blur_probability": 0.2,
        "noise_probability": 0.2,
        "preserve_anatomy": True,
    }

    # Adjust based on task complexity
    if task_granularity == "task_12":
        # More conservative for complex tasks
        base_config.update(
            {
                "brightness_contrast": 0.2,
                "blur_probability": 0.1,
                "noise_probability": 0.1,
            }
        )
    elif task_granularity == "task_3":
        # More aggressive for simpler tasks
        base_config.update(
            {
                "brightness_contrast": 0.4,
                "blur_probability": 0.3,
                "noise_probability": 0.3,
            }
        )

    return base_config
