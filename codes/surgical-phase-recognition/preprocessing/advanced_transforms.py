#!/usr/bin/env python3
"""
Transform utilities for surgical phase recognition.

This module provides specialized transformation classes for video sequences
used in surgical phase recognition, including data augmentation and preprocessing.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import random
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None


class SequenceTransform:
    """
    Transform class for video sequences with optional augmentation.

    Applies consistent transformations across all frames in a sequence
    while supporting temporal data augmentation techniques.

    Args:
        base_transform: Base transformation to apply to all frames
        apply_augmentation (bool): Whether to apply data augmentation
        sequence_length (int): Expected sequence length for temporal augmentation
        augmentation_prob (float): Probability of applying each augmentation
    """

    def __init__(
        self,
        base_transform=None,
        apply_augmentation=False,
        sequence_length=10,
        augmentation_prob=0.5,
    ):
        self.apply_augmentation = apply_augmentation
        self.sequence_length = sequence_length
        self.augmentation_prob = augmentation_prob

        # Base transformations applied to all frames
        if base_transform is None:
            self.base_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.base_transform = base_transform

        # Augmentation transformations
        self.augment_transform = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=5),
                transforms.RandomResizedCrop(
                    size=224, scale=(0.9, 1.0), ratio=(0.9, 1.1)
                ),
            ]
        )

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        """
        Apply transformations to a sequence of frames.

        Args:
            frames (List[Image.Image]): List of PIL Images

        Returns:
            List[torch.Tensor]: List of transformed tensors
        """
        transformed_frames = []

        # Determine if augmentation should be applied
        apply_aug = self.apply_augmentation and random.random() < self.augmentation_prob

        # Apply temporal augmentation (affects all frames in sequence)
        if apply_aug:
            frames = self._apply_temporal_augmentation(frames)

        # Apply frame-level transformations
        for frame in frames:
            if apply_aug:
                # Apply augmentation before base transform
                frame = self._apply_spatial_augmentation(frame)

            # Apply base transformation
            transformed_frame = self.base_transform(frame)
            transformed_frames.append(transformed_frame)

        return transformed_frames

    def _apply_temporal_augmentation(
        self, frames: List[Image.Image]
    ) -> List[Image.Image]:
        """
        Apply temporal-level augmentations that affect the entire sequence.

        Args:
            frames (List[Image.Image]): Input frames

        Returns:
            List[Image.Image]: Augmented frames
        """
        augmented_frames = frames.copy()

        # Temporal drop (randomly drop some frames and duplicate others)
        if random.random() < self.augmentation_prob * 0.3:
            augmented_frames = self._temporal_drop(augmented_frames)

        # Temporal jittering (random frame order perturbation)
        if random.random() < self.augmentation_prob * 0.2:
            augmented_frames = self._temporal_jitter(augmented_frames)

        # Temporal speed variation (frame skipping/duplication)
        if random.random() < self.augmentation_prob * 0.3:
            augmented_frames = self._temporal_speed_variation(augmented_frames)

        return augmented_frames

    def _apply_spatial_augmentation(self, frame: Image.Image) -> Image.Image:
        """
        Apply spatial augmentations to individual frames.

        Args:
            frame (Image.Image): Input frame

        Returns:
            Image.Image: Augmented frame
        """
        # Random brightness adjustment
        if random.random() < self.augmentation_prob * 0.5:
            enhancer = ImageEnhance.Brightness(frame)
            frame = enhancer.enhance(random.uniform(0.8, 1.2))

        # Random contrast adjustment
        if random.random() < self.augmentation_prob * 0.5:
            enhancer = ImageEnhance.Contrast(frame)
            frame = enhancer.enhance(random.uniform(0.8, 1.2))

        # Random gaussian blur (simulating motion blur)
        if random.random() < self.augmentation_prob * 0.3:
            frame = frame.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))
            )

        # Surgical-specific augmentations
        frame = self._apply_surgical_augmentation(frame)

        return frame

    def _apply_surgical_augmentation(self, frame: Image.Image) -> Image.Image:
        """
        Apply surgical video-specific augmentations.

        Args:
            frame (Image.Image): Input frame

        Returns:
            Image.Image: Augmented frame
        """
        # Simulate surgical lighting variations
        if random.random() < self.augmentation_prob * 0.4:
            frame = self._simulate_lighting_variation(frame)

        # Simulate surgical instrument occlusion
        if random.random() < self.augmentation_prob * 0.2:
            frame = self._simulate_instrument_occlusion(frame)

        return frame

    def _simulate_lighting_variation(self, frame: Image.Image) -> Image.Image:
        """Simulate surgical lighting variations."""
        # Convert to numpy for easier manipulation
        frame_np = np.array(frame)

        # Create gradient lighting effect
        height, width = frame_np.shape[:2]

        # Random center for lighting
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)

        # Create distance-based lighting mask
        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        max_distance = np.sqrt(width**2 + height**2)

        # Normalize distances and create lighting mask
        lighting_mask = 1.0 - (distances / max_distance) * random.uniform(0.1, 0.3)
        lighting_mask = np.clip(lighting_mask, 0.7, 1.0)

        # Apply lighting effect
        frame_np = frame_np * lighting_mask[..., np.newaxis]
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        return Image.fromarray(frame_np)

    def _simulate_instrument_occlusion(self, frame: Image.Image) -> Image.Image:
        """Simulate surgical instrument occlusion."""
        frame_np = np.array(frame)
        height, width = frame_np.shape[:2]

        # Random rectangular occlusion
        occlusion_width = random.randint(width // 20, width // 8)
        occlusion_height = random.randint(height // 20, height // 8)

        start_x = random.randint(0, width - occlusion_width)
        start_y = random.randint(0, height - occlusion_height)

        # Create darker region (instrument shadow)
        frame_np[
            start_y : start_y + occlusion_height, start_x : start_x + occlusion_width
        ] *= random.uniform(0.3, 0.7)

        return Image.fromarray(frame_np)

    def _temporal_drop(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Randomly drop and duplicate frames."""
        if len(frames) <= 2:
            return frames

        # Randomly select frames to drop
        drop_count = random.randint(1, min(2, len(frames) // 3))
        indices_to_drop = random.sample(range(len(frames)), drop_count)

        # Remove dropped frames
        kept_frames = [
            frame for i, frame in enumerate(frames) if i not in indices_to_drop
        ]

        # Duplicate random frames to maintain sequence length
        while len(kept_frames) < len(frames):
            duplicate_idx = random.randint(0, len(kept_frames) - 1)
            kept_frames.insert(duplicate_idx + 1, kept_frames[duplicate_idx].copy())

        return kept_frames[: len(frames)]

    def _temporal_jitter(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Apply small temporal perturbations."""
        if len(frames) <= 2:
            return frames

        # Randomly swap adjacent frames
        jittered_frames = frames.copy()
        swap_count = random.randint(1, len(frames) // 4)

        for _ in range(swap_count):
            idx = random.randint(0, len(jittered_frames) - 2)
            jittered_frames[idx], jittered_frames[idx + 1] = (
                jittered_frames[idx + 1],
                jittered_frames[idx],
            )

        return jittered_frames

    def _temporal_speed_variation(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Simulate speed variations by frame skipping/duplication."""
        if len(frames) <= 3:
            return frames

        # Randomly choose speed variation type
        if random.random() < 0.5:
            # Speed up (skip frames)
            skip_count = random.randint(1, len(frames) // 4)
            indices_to_skip = sorted(
                random.sample(range(1, len(frames) - 1), skip_count)
            )

            varied_frames = [
                frame for i, frame in enumerate(frames) if i not in indices_to_skip
            ]

            # Pad with last frame if needed
            while len(varied_frames) < len(frames):
                varied_frames.append(varied_frames[-1].copy())

        else:
            # Slow down (duplicate frames)
            duplicate_count = random.randint(1, len(frames) // 4)
            varied_frames = frames.copy()

            for _ in range(duplicate_count):
                if len(varied_frames) >= len(frames):
                    break
                insert_idx = random.randint(1, len(varied_frames) - 1)
                varied_frames.insert(insert_idx, varied_frames[insert_idx - 1].copy())

        return varied_frames[: len(frames)]


class SurgicalVideoTransform:
    """
    Specialized transforms for surgical videos with medical imaging considerations.

    Args:
        image_size (tuple): Target image size (height, width)
        normalize (bool): Whether to apply ImageNet normalization
        surgical_specific (bool): Apply surgical video-specific augmentations
    """

    def __init__(self, image_size=(224, 224), normalize=True, surgical_specific=True):
        self.image_size = image_size
        self.normalize = normalize
        self.surgical_specific = surgical_specific

        # Build transform pipeline
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]

        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )

        self.base_transform = transforms.Compose(transform_list)

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Apply transforms to a single image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.surgical_specific:
            image = self._enhance_surgical_features(image)

        return self.base_transform(image)

    def _enhance_surgical_features(self, image: Image.Image) -> Image.Image:
        """Enhance surgical video features for better analysis."""
        # Enhance contrast for better tissue visibility
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)

        # Slight sharpening for instrument details
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)

        return image


# Convenience functions for common transform configurations
def get_training_transforms(
    image_size=(224, 224), augmentation_prob=0.5, sequence_length=10
) -> SequenceTransform:
    """Get training transforms with augmentation."""
    return SequenceTransform(
        apply_augmentation=True,
        sequence_length=sequence_length,
        augmentation_prob=augmentation_prob,
    )


def get_validation_transforms(image_size=(224, 224)) -> SequenceTransform:
    """Get validation transforms without augmentation."""
    return SequenceTransform(apply_augmentation=False)


def get_surgical_transforms(
    image_size=(224, 224), normalize=True
) -> SurgicalVideoTransform:
    """Get surgical video-specific transforms."""
    return SurgicalVideoTransform(
        image_size=image_size, normalize=normalize, surgical_specific=True
    )


if __name__ == "__main__":
    # Test the transforms
    import numpy as np
    from PIL import Image

    # Create dummy frames
    dummy_frames = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(10)
    ]

    # Test training transforms
    train_transform = get_training_transforms()
    transformed_frames = train_transform(dummy_frames)

    print(f"Input frames: {len(dummy_frames)}")
    print(f"Transformed frames: {len(transformed_frames)}")
    print(f"Frame shape: {transformed_frames[0].shape}")

    # Test validation transforms
    val_transform = get_validation_transforms()
    val_frames = val_transform(dummy_frames)

    print(f"Validation frames: {len(val_frames)}")
    print("Transform testing completed!")
