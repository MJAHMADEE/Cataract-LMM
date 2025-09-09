#!/usr/bin/env python3
"""
Dataset Classes for Surgical Phase Recognition

This module implements PyTorch dataset classes for loading and processing
surgical video data for phase classification. Supports multiple data formats
and annotation styles based on the validation notebook.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .video_preprocessing import VideoFrameExtractor, VideoPreprocessor
except ImportError:
    # Create dummy classes for components not yet implemented
    class VideoPreprocessor:
        pass

    class VideoFrameExtractor:
        pass


logger = logging.getLogger(__name__)


class SurgicalPhaseDataset(Dataset):
    """
    Dataset for surgical phase recognition from videos.

    This dataset handles video loading, preprocessing, and phase label mapping
    for surgical procedure videos. Supports both frame-level and video-level
    annotations.

    Args:
        data_file (str): Path to CSV/JSON file with video paths and labels
        video_root (str): Root directory containing video files
        preprocessor (VideoPreprocessor): Video preprocessing pipeline
        phase_mapping (dict, optional): Mapping from phase names to class indices
        is_training (bool): Whether dataset is for training
        annotation_type (str): 'video_level' or 'frame_level'
        cache_frames (bool): Whether to cache extracted frames in memory
    """

    def __init__(
        self,
        data_file: str,
        video_root: str,
        preprocessor: VideoPreprocessor,
        phase_mapping: Optional[Dict[str, int]] = None,
        is_training: bool = True,
        annotation_type: str = "video_level",
        cache_frames: bool = False,
    ):
        self.data_file = data_file
        self.video_root = video_root
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.annotation_type = annotation_type
        self.cache_frames = cache_frames

        # Default phase mapping for cataract surgery
        if phase_mapping is None:
            self.phase_mapping = {
                "Incision": 0,
                "Viscous Agent Injection": 1,
                "Rhexis": 2,
                "Hydrodissection": 3,
                "Phacoemulsification": 4,
                "Irrigation and Aspiration": 5,
                "Capsule Polishing": 6,
                "Lens Implant Setting": 7,
                "Viscous Agent Removal": 8,
                "Suturing": 9,
                "Tonifying Antibiotics": 10,
            }
        else:
            self.phase_mapping = phase_mapping

        self.num_classes = len(self.phase_mapping)
        self.reverse_mapping = {v: k for k, v in self.phase_mapping.items()}

        # Load data
        self.data = self._load_data()

        # Frame cache for performance
        if self.cache_frames:
            self.frame_cache = {}

        logger.info(
            f"Loaded {len(self.data)} samples for {'training' if is_training else 'validation'}"
        )

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from annotation file."""
        if self.data_file.endswith(".csv"):
            return self._load_csv_data()
        elif self.data_file.endswith(".json"):
            return self._load_json_data()
        else:
            raise ValueError(f"Unsupported data file format: {self.data_file}")

    def _load_csv_data(self) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        df = pd.read_csv(self.data_file)

        required_columns = ["video_path", "phase"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")

        data = []
        for _, row in df.iterrows():
            video_path = os.path.join(self.video_root, row["video_path"])

            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}")
                continue

            phase_name = row["phase"]
            if phase_name not in self.phase_mapping:
                logger.warning(f"Unknown phase: {phase_name}")
                continue

            sample = {
                "video_path": video_path,
                "phase": phase_name,
                "phase_id": self.phase_mapping[phase_name],
                "metadata": {
                    col: row[col] for col in df.columns if col not in required_columns
                },
            }
            data.append(sample)

        return data

    def _load_json_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(self.data_file, "r") as f:
            json_data = json.load(f)

        data = []

        if isinstance(json_data, list):
            # List format
            for item in json_data:
                if "video_path" not in item or "phase" not in item:
                    continue

                video_path = os.path.join(self.video_root, item["video_path"])

                if not os.path.exists(video_path):
                    logger.warning(f"Video file not found: {video_path}")
                    continue

                phase_name = item["phase"]
                if phase_name not in self.phase_mapping:
                    logger.warning(f"Unknown phase: {phase_name}")
                    continue

                sample = {
                    "video_path": video_path,
                    "phase": phase_name,
                    "phase_id": self.phase_mapping[phase_name],
                    "metadata": {
                        k: v
                        for k, v in item.items()
                        if k not in ["video_path", "phase"]
                    },
                }
                data.append(sample)

        elif isinstance(json_data, dict):
            # Dictionary format with video names as keys
            for video_name, annotations in json_data.items():
                video_path = os.path.join(self.video_root, video_name)

                if not os.path.exists(video_path):
                    logger.warning(f"Video file not found: {video_path}")
                    continue

                if isinstance(annotations, str):
                    # Single phase annotation
                    phase_name = annotations
                    if phase_name in self.phase_mapping:
                        sample = {
                            "video_path": video_path,
                            "phase": phase_name,
                            "phase_id": self.phase_mapping[phase_name],
                            "metadata": {},
                        }
                        data.append(sample)

                elif isinstance(annotations, dict):
                    # Complex annotation structure
                    if "phase" in annotations:
                        phase_name = annotations["phase"]
                        if phase_name in self.phase_mapping:
                            sample = {
                                "video_path": video_path,
                                "phase": phase_name,
                                "phase_id": self.phase_mapping[phase_name],
                                "metadata": {
                                    k: v for k, v in annotations.items() if k != "phase"
                                },
                            }
                            data.append(sample)

        return data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            Dict containing:
                - 'video': Video tensor of shape (C, T, H, W)
                - 'label': Phase class index
                - 'phase_name': Phase name string
                - 'video_path': Path to video file
                - 'metadata': Additional metadata
        """
        sample = self.data[idx]
        video_path = sample["video_path"]
        phase_id = sample["phase_id"]
        phase_name = sample["phase"]

        try:
            # Load and process video
            if self.cache_frames and video_path in self.frame_cache:
                video_tensor = self.frame_cache[video_path]
            else:
                video_tensor = self.preprocessor.process_video(
                    video_path, is_training=self.is_training
                )

                if self.cache_frames:
                    self.frame_cache[video_path] = video_tensor

            return {
                "video": video_tensor,
                "label": phase_id,
                "phase_name": phase_name,
                "video_path": video_path,
                "metadata": sample["metadata"],
            }

        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            # Return a dummy sample to avoid breaking the batch
            dummy_video = torch.zeros(
                3, self.preprocessor.clip_length, *self.preprocessor.image_size
            )
            return {
                "video": dummy_video,
                "label": 0,  # Default to first class
                "phase_name": "unknown",
                "video_path": video_path,
                "metadata": {},
            }

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {phase: 0 for phase in self.phase_mapping.keys()}

        for sample in self.data:
            distribution[sample["phase"]] += 1

        return distribution

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        distribution = self.get_class_distribution()
        total_samples = sum(distribution.values())

        weights = []
        for phase_id in range(self.num_classes):
            phase_name = self.reverse_mapping[phase_id]
            count = distribution[phase_name]
            weight = total_samples / (self.num_classes * count) if count > 0 else 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


class FrameLevelDataset(Dataset):
    """
    Dataset for frame-level surgical phase recognition.

    This dataset provides frame-by-frame labels for temporal modeling
    and detailed phase boundary detection.

    Args:
        data_file (str): Path to frame-level annotation file
        video_root (str): Root directory containing video files
        preprocessor (VideoPreprocessor): Video preprocessing pipeline
        phase_mapping (dict, optional): Mapping from phase names to class indices
        sequence_length (int): Length of frame sequences to return
        stride (int): Stride between sequences
        is_training (bool): Whether dataset is for training
    """

    def __init__(
        self,
        data_file: str,
        video_root: str,
        preprocessor: VideoPreprocessor,
        phase_mapping: Optional[Dict[str, int]] = None,
        sequence_length: int = 16,
        stride: int = 8,
        is_training: bool = True,
    ):
        self.data_file = data_file
        self.video_root = video_root
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length
        self.stride = stride
        self.is_training = is_training

        # Default phase mapping
        if phase_mapping is None:
            self.phase_mapping = {
                "Incision": 0,
                "Viscous Agent Injection": 1,
                "Rhexis": 2,
                "Hydrodissection": 3,
                "Phacoemulsification": 4,
                "Irrigation and Aspiration": 5,
                "Capsule Polishing": 6,
                "Lens Implant Setting": 7,
                "Viscous Agent Removal": 8,
                "Suturing": 9,
                "Tonifying Antibiotics": 10,
            }
        else:
            self.phase_mapping = phase_mapping

        self.num_classes = len(self.phase_mapping)

        # Load frame-level annotations
        self.sequences = self._load_frame_annotations()

        logger.info(f"Loaded {len(self.sequences)} frame sequences")

    def _load_frame_annotations(self) -> List[Dict[str, Any]]:
        """Load frame-level annotations and create sequences."""
        # Load annotations
        if self.data_file.endswith(".csv"):
            df = pd.read_csv(self.data_file)
        else:
            raise ValueError("Frame-level dataset currently supports CSV format only")

        required_columns = ["video_path", "frame_idx", "phase"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")

        # Group by video
        sequences = []
        for video_path, group in df.groupby("video_path"):
            full_video_path = os.path.join(self.video_root, video_path)

            if not os.path.exists(full_video_path):
                logger.warning(f"Video file not found: {full_video_path}")
                continue

            # Sort by frame index
            group = group.sort_values("frame_idx")

            # Create sequences
            frame_indices = group["frame_idx"].tolist()
            phase_labels = group["phase"].tolist()

            # Convert phase names to indices
            label_indices = []
            for phase in phase_labels:
                if phase in self.phase_mapping:
                    label_indices.append(self.phase_mapping[phase])
                else:
                    logger.warning(f"Unknown phase: {phase}")
                    label_indices.append(0)  # Default to first class

            # Create overlapping sequences
            for start_idx in range(
                0, len(frame_indices) - self.sequence_length + 1, self.stride
            ):
                end_idx = start_idx + self.sequence_length

                sequence = {
                    "video_path": full_video_path,
                    "frame_indices": frame_indices[start_idx:end_idx],
                    "labels": label_indices[start_idx:end_idx],
                    "start_frame": frame_indices[start_idx],
                    "end_frame": frame_indices[end_idx - 1],
                }
                sequences.append(sequence)

        return sequences

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a frame sequence from the dataset.

        Args:
            idx (int): Sequence index

        Returns:
            Dict containing:
                - 'video': Video tensor of shape (C, T, H, W)
                - 'labels': Frame-level labels of shape (T,)
                - 'video_path': Path to video file
        """
        sequence = self.sequences[idx]
        video_path = sequence["video_path"]
        frame_indices = sequence["frame_indices"]
        labels = sequence["labels"]

        try:
            # Extract specific frames
            extractor = VideoFrameExtractor(
                image_size=self.preprocessor.image_size, use_decord=True
            )

            # Load all frames and select required ones
            all_frames = extractor.extract_frames(video_path)

            # Select frames by index (with bounds checking)
            selected_frames = []
            for frame_idx in frame_indices:
                if frame_idx < len(all_frames):
                    selected_frames.append(all_frames[frame_idx])
                else:
                    # Use last available frame if index is out of bounds
                    selected_frames.append(all_frames[-1])

            # Process frames
            video_tensor = self.preprocessor.process_frames(
                selected_frames, is_training=self.is_training
            )

            return {
                "video": video_tensor,
                "labels": torch.tensor(labels, dtype=torch.long),
                "video_path": video_path,
            }

        except Exception as e:
            logger.error(f"Error loading sequence from {video_path}: {e}")
            # Return dummy data
            dummy_video = torch.zeros(
                3, self.sequence_length, *self.preprocessor.image_size
            )
            dummy_labels = torch.zeros(self.sequence_length, dtype=torch.long)

            return {
                "video": dummy_video,
                "labels": dummy_labels,
                "video_path": video_path,
            }


def create_surgical_phase_dataloaders(
    train_data_file: str,
    val_data_file: str,
    video_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    clip_length: int = 16,
    phase_mapping: Optional[Dict[str, int]] = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for surgical phase recognition.

    Args:
        train_data_file (str): Path to training annotation file
        val_data_file (str): Path to validation annotation file
        video_root (str): Root directory containing videos
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        image_size (tuple): Target image size
        clip_length (int): Number of frames per clip
        phase_mapping (dict, optional): Custom phase mapping
        **kwargs: Additional arguments for preprocessing

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders
    """
    # Create preprocessing pipelines
    train_preprocessor = VideoPreprocessor(
        image_size=image_size,
        clip_length=clip_length,
        temporal_strategy="random",
        augment_prob=0.5,
        **kwargs,
    )

    val_preprocessor = VideoPreprocessor(
        image_size=image_size,
        clip_length=clip_length,
        temporal_strategy="uniform",
        augment_prob=0.0,  # No augmentation for validation
        **kwargs,
    )

    # Create datasets
    train_dataset = SurgicalPhaseDataset(
        data_file=train_data_file,
        video_root=video_root,
        preprocessor=train_preprocessor,
        phase_mapping=phase_mapping,
        is_training=True,
    )

    val_dataset = SurgicalPhaseDataset(
        data_file=val_data_file,
        video_root=video_root,
        preprocessor=val_preprocessor,
        phase_mapping=phase_mapping,
        is_training=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"Created dataloaders: Train {len(train_dataset)} samples, Val {len(val_dataset)} samples"
    )

    return train_loader, val_loader


def collate_fn_surgical_phase(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for surgical phase data.

    Args:
        batch (List[Dict]): List of samples from dataset

    Returns:
        Dict[str, Any]: Batched data
    """
    videos = torch.stack([item["video"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    phase_names = [item["phase_name"] for item in batch]
    video_paths = [item["video_path"] for item in batch]

    return {
        "video": videos,
        "label": labels,
        "phase_name": phase_names,
        "video_path": video_paths,
    }


if __name__ == "__main__":
    # Test dataset creation
    logger.info("Testing dataset classes...")

    # Test with dummy data
    dummy_data = [
        {"video_path": "video1.mp4", "phase": "Incision"},
        {"video_path": "video2.mp4", "phase": "Rhexis"},
    ]

    print("Dataset classes ready for use!")
    print("Remember to prepare your annotation files in CSV or JSON format.")
    print("CSV format should have columns: video_path, phase")
    print("JSON format should map video paths to phase labels.")
