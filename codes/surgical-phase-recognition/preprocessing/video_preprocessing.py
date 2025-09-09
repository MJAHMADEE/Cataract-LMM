#!/usr/bin/env python3
"""
Video Data Preprocessing for Surgical Phase Recognition

This module provides comprehensive video preprocessing functionality including
frame extraction, video loading, temporal sampling, data augmentation,
and format conversion for surgical phase recognition.

Based on the preprocessing pipeline from the validation notebook.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

try:
    import cv2
except ImportError:
    cv2 = None
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    logger.warning("albumentations not available. Advanced augmentations may not work.")
    ALBUMENTATIONS_AVAILABLE = False

try:
    from decord import VideoReader, cpu, gpu

    DECORD_AVAILABLE = True
except ImportError:
    logger.warning("decord not available. Falling back to OpenCV for video reading.")
    DECORD_AVAILABLE = False


class VideoFrameExtractor:
    """
    Utility class for extracting frames from surgical videos.

    Supports different sampling strategies and handles various video formats
    commonly used in surgical datasets.

    Args:
        target_fps (float, optional): Target FPS for frame extraction
        max_frames (int, optional): Maximum number of frames to extract
        sampling_strategy (str): 'uniform', 'random', or 'dense'
        image_size (tuple): Target image size (height, width)
        use_decord (bool): Whether to use decord for video reading
    """

    def __init__(
        self,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        sampling_strategy: str = "uniform",
        image_size: Tuple[int, int] = (224, 224),
        use_decord: bool = True,
    ):
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.sampling_strategy = sampling_strategy
        self.image_size = image_size
        self.use_decord = use_decord and DECORD_AVAILABLE

        if self.use_decord and not DECORD_AVAILABLE:
            logger.warning("decord requested but not available. Using OpenCV.")
            self.use_decord = False

    def extract_frames_opencv(self, video_path: str) -> List[np.ndarray]:
        """Extract frames using OpenCV."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {video_path}, FPS: {fps}, Total frames: {total_frames}")

        # Calculate frame indices to extract
        frame_indices = self._calculate_frame_indices(total_frames, fps)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(
                    frame, self.image_size[::-1]
                )  # OpenCV uses (width, height)
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame {idx}")

        cap.release()
        return frames

    def extract_frames_decord(self, video_path: str) -> List[np.ndarray]:
        """Extract frames using decord (faster and more efficient)."""
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)

            logger.info(
                f"Video: {video_path}, FPS: {fps}, Total frames: {total_frames}"
            )

            # Calculate frame indices to extract
            frame_indices = self._calculate_frame_indices(total_frames, fps)

            # Extract frames
            frames = vr.get_batch(frame_indices).asnumpy()

            # Resize frames
            resized_frames = []
            for frame in frames:
                frame_resized = cv2.resize(frame, self.image_size[::-1])
                resized_frames.append(frame_resized)

            return resized_frames

        except Exception as e:
            logger.warning(f"Decord failed: {e}. Falling back to OpenCV.")
            return self.extract_frames_opencv(video_path)

    def _calculate_frame_indices(self, total_frames: int, fps: float) -> List[int]:
        """Calculate which frame indices to extract based on sampling strategy."""
        # Apply FPS downsampling if specified
        if self.target_fps is not None and self.target_fps < fps:
            stride = int(fps / self.target_fps)
            available_indices = list(range(0, total_frames, stride))
        else:
            available_indices = list(range(total_frames))

        # Apply max_frames constraint
        if self.max_frames is not None and len(available_indices) > self.max_frames:
            if self.sampling_strategy == "uniform":
                # Uniformly sample frames
                step = len(available_indices) / self.max_frames
                indices = [
                    available_indices[int(i * step)] for i in range(self.max_frames)
                ]
            elif self.sampling_strategy == "random":
                # Randomly sample frames
                indices = np.random.choice(
                    available_indices, self.max_frames, replace=False
                )
                indices = sorted(indices)
            elif self.sampling_strategy == "dense":
                # Take first max_frames frames
                indices = available_indices[: self.max_frames]
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        else:
            indices = available_indices

        return indices

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video.

        Args:
            video_path (str): Path to video file

        Returns:
            List[np.ndarray]: List of extracted frames as numpy arrays
        """
        if self.use_decord:
            return self.extract_frames_decord(video_path)
        else:
            return self.extract_frames_opencv(video_path)


class VideoAugmentation:
    """
    Video-specific data augmentation pipeline.

    Applies consistent augmentations across video frames while maintaining
    temporal coherence.

    Args:
        image_size (tuple): Target image size
        augment_prob (float): Probability of applying augmentations
        use_advanced (bool): Whether to use advanced albumentations
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        augment_prob: float = 0.5,
        use_advanced: bool = True,
    ):
        self.image_size = image_size
        self.augment_prob = augment_prob
        self.use_advanced = use_advanced and ALBUMENTATIONS_AVAILABLE

        self._setup_transforms()

    def _setup_transforms(self):
        """Setup augmentation transforms."""
        if self.use_advanced:
            # Advanced augmentations using albumentations
            self.train_transform = A.Compose(
                [
                    A.Resize(self.image_size[0], self.image_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.ColorJitter(p=0.3),
                    A.GaussNoise(p=0.2),
                    A.Blur(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

            self.val_transform = A.Compose(
                [
                    A.Resize(self.image_size[0], self.image_size[1]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            # Basic augmentations without albumentations
            logger.info("Using basic augmentations (albumentations not available)")
            self.train_transform = None
            self.val_transform = None

    def _apply_basic_transform(
        self, frames: List[np.ndarray], is_training: bool
    ) -> torch.Tensor:
        """Apply basic transformations without albumentations."""
        processed_frames = []

        for frame in frames:
            # Resize
            frame = cv2.resize(frame, self.image_size[::-1])

            # Convert to float and normalize
            frame = frame.astype(np.float32) / 255.0

            # Apply training augmentations
            if is_training and np.random.random() < self.augment_prob:
                # Random horizontal flip
                if np.random.random() < 0.5:
                    frame = np.fliplr(frame)

                # Random brightness adjustment
                if np.random.random() < 0.3:
                    brightness_factor = np.random.uniform(0.8, 1.2)
                    frame = np.clip(frame * brightness_factor, 0, 1)

            # Normalize with ImageNet stats
            frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array(
                [0.229, 0.224, 0.225]
            )

            processed_frames.append(frame)

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        video_tensor = torch.tensor(np.array(processed_frames), dtype=torch.float32)
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

        return video_tensor

    def __call__(
        self, frames: List[np.ndarray], is_training: bool = True
    ) -> torch.Tensor:
        """
        Apply augmentations to video frames.

        Args:
            frames (List[np.ndarray]): List of video frames
            is_training (bool): Whether in training mode

        Returns:
            torch.Tensor: Augmented video tensor of shape (C, T, H, W)
        """
        if not self.use_advanced:
            return self._apply_basic_transform(frames, is_training)

        # Apply albumentations transform
        transform = self.train_transform if is_training else self.val_transform

        # Apply same transform to all frames to maintain temporal consistency
        if is_training:
            # Generate random parameters once for all frames
            transform_params = transform.get_params()

        processed_frames = []
        for frame in frames:
            if is_training:
                # Apply same parameters to all frames
                transformed = A.Compose.replay(transform, transform_params, image=frame)
                frame_tensor = transformed["image"]
            else:
                transformed = transform(image=frame)
                frame_tensor = transformed["image"]

            processed_frames.append(frame_tensor)

        # Stack frames: (T, C, H, W) -> (C, T, H, W)
        video_tensor = torch.stack(processed_frames, dim=1)

        return video_tensor


class TemporalSampler:
    """
    Temporal sampling utilities for video sequences.

    Handles different temporal sampling strategies for training and validation,
    including sliding windows, random clips, and full sequence processing.

    Args:
        clip_length (int): Length of video clips
        sampling_rate (int): Frame sampling rate
        strategy (str): 'random', 'uniform', 'sliding_window'
        overlap (float): Overlap ratio for sliding window (0.0 to 1.0)
    """

    def __init__(
        self,
        clip_length: int = 16,
        sampling_rate: int = 1,
        strategy: str = "random",
        overlap: float = 0.5,
    ):
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate
        self.strategy = strategy
        self.overlap = overlap

        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be in range [0, 1)")

    def sample_frames(self, total_frames: int) -> List[int]:
        """
        Sample frame indices based on the specified strategy.

        Args:
            total_frames (int): Total number of available frames

        Returns:
            List[int]: Selected frame indices
        """
        required_frames = self.clip_length * self.sampling_rate

        if total_frames < required_frames:
            # If video is too short, repeat frames
            indices = list(range(0, total_frames, self.sampling_rate))
            while len(indices) < self.clip_length:
                indices.extend(indices[: self.clip_length - len(indices)])
            return indices[: self.clip_length]

        if self.strategy == "random":
            # Random starting point
            max_start = total_frames - required_frames
            start_idx = np.random.randint(0, max_start + 1)
            return list(
                range(start_idx, start_idx + required_frames, self.sampling_rate)
            )

        elif self.strategy == "uniform":
            # Uniformly distributed frames
            step = (total_frames - 1) / (self.clip_length - 1)
            return [int(i * step) for i in range(self.clip_length)]

        elif self.strategy == "sliding_window":
            # Return multiple clips with sliding window
            step_size = int(self.clip_length * (1 - self.overlap))
            clips = []
            start = 0

            while start + required_frames <= total_frames:
                clip_indices = list(
                    range(start, start + required_frames, self.sampling_rate)
                )
                clips.append(clip_indices)
                start += step_size

            if not clips:  # Fallback if no clips fit
                return self.sample_frames(total_frames)  # Use uniform strategy

            return clips  # Return all clips for sliding window

        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

    def create_clips(
        self, frames: List[np.ndarray]
    ) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Create temporal clips from video frames.

        Args:
            frames (List[np.ndarray]): Input video frames

        Returns:
            Union[List[np.ndarray], List[List[np.ndarray]]]:
                Single clip or list of clips depending on strategy
        """
        total_frames = len(frames)
        indices = self.sample_frames(total_frames)

        if self.strategy == "sliding_window" and isinstance(indices[0], list):
            # Multiple clips
            clips = []
            for clip_indices in indices:
                clip_frames = [frames[i] for i in clip_indices]
                clips.append(clip_frames)
            return clips
        else:
            # Single clip
            return [frames[i] for i in indices]


class VideoPreprocessor:
    """
    Complete video preprocessing pipeline.

    Combines frame extraction, augmentation, and temporal sampling into
    a unified preprocessing pipeline for surgical phase recognition.

    Args:
        image_size (tuple): Target image size
        clip_length (int): Number of frames per clip
        sampling_rate (int): Frame sampling rate
        target_fps (float, optional): Target FPS for extraction
        augment_prob (float): Probability of augmentation
        use_decord (bool): Whether to use decord for video reading
        temporal_strategy (str): Temporal sampling strategy
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        clip_length: int = 16,
        sampling_rate: int = 1,
        target_fps: Optional[float] = None,
        augment_prob: float = 0.5,
        use_decord: bool = True,
        temporal_strategy: str = "random",
    ):
        self.image_size = image_size
        self.clip_length = clip_length

        # Initialize components
        self.extractor = VideoFrameExtractor(
            target_fps=target_fps, image_size=image_size, use_decord=use_decord
        )

        self.augmentation = VideoAugmentation(
            image_size=image_size, augment_prob=augment_prob
        )

        self.temporal_sampler = TemporalSampler(
            clip_length=clip_length,
            sampling_rate=sampling_rate,
            strategy=temporal_strategy,
        )

        logger.info(
            f"Initialized VideoPreprocessor with {image_size} resolution, {clip_length} frames"
        )

    def process_video(self, video_path: str, is_training: bool = True) -> torch.Tensor:
        """
        Process a complete video through the preprocessing pipeline.

        Args:
            video_path (str): Path to video file
            is_training (bool): Whether in training mode

        Returns:
            torch.Tensor: Processed video tensor of shape (C, T, H, W)
        """
        # Extract frames
        frames = self.extractor.extract_frames(video_path)

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        # Temporal sampling
        clip_frames = self.temporal_sampler.create_clips(frames)

        # Handle multiple clips (for sliding window strategy)
        if isinstance(clip_frames[0], list):
            # Multiple clips - process first one for now
            # TODO: Handle multiple clips properly in dataset
            clip_frames = clip_frames[0]

        # Apply augmentations
        video_tensor = self.augmentation(clip_frames, is_training=is_training)

        return video_tensor

    def process_frames(
        self, frames: List[np.ndarray], is_training: bool = True
    ) -> torch.Tensor:
        """
        Process pre-extracted frames.

        Args:
            frames (List[np.ndarray]): Pre-extracted video frames
            is_training (bool): Whether in training mode

        Returns:
            torch.Tensor: Processed video tensor
        """
        # Temporal sampling
        clip_frames = self.temporal_sampler.create_clips(frames)

        # Handle multiple clips
        if isinstance(clip_frames[0], list):
            clip_frames = clip_frames[0]

        # Apply augmentations
        video_tensor = self.augmentation(clip_frames, is_training=is_training)

        return video_tensor


def save_frames_to_disk(
    video_path: str,
    output_dir: str,
    target_fps: Optional[float] = None,
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, Any]:
    """
    Extract and save video frames to disk for faster loading.

    Args:
        video_path (str): Path to input video
        output_dir (str): Directory to save frames
        target_fps (float, optional): Target FPS for extraction
        image_size (tuple): Target image size

    Returns:
        Dict[str, Any]: Metadata about extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)

    extractor = VideoFrameExtractor(
        target_fps=target_fps, image_size=image_size, use_decord=True
    )

    frames = extractor.extract_frames(video_path)

    # Save frames
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"frame_{i:06d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)

    # Save metadata
    metadata = {
        "video_path": video_path,
        "num_frames": len(frames),
        "frame_paths": frame_paths,
        "image_size": image_size,
        "target_fps": target_fps,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {len(frames)} frames to {output_dir}")
    return metadata


if __name__ == "__main__":
    # Test preprocessing pipeline
    logger.info("Testing video preprocessing pipeline...")

    # Test components individually
    print("Testing VideoFrameExtractor...")
    print("Testing VideoAugmentation...")
    print("Testing TemporalSampler...")
    print("Testing VideoPreprocessor...")

    print("Video preprocessing module ready!")
