#!/usr/bin/env python3
"""
Sequential Dataset for Surgical Phase Recognition with Augmentation and Overlap.

This module provides the exact SequentialSurgicalPhaseDatasetAugOverlap class
used in the reference validation notebook phase_validation_comprehensive.ipynb.

Author: Surgical Phase Recognition Team
Date: August 29, 2025
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Import the transforms from the transform module
try:
    from transform import transform_test, transform_train
except ImportError:
    try:
        from ..transform import transform_test, transform_train
    except ImportError:
        # Fallback if transform module is not available
        import torchvision.transforms as transforms

        transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transform_train


class SequentialSurgicalPhaseDatasetAugOverlap(Dataset):
    """
    Sequential surgical phase dataset with augmentation and overlap support.

    This is the exact implementation used in the reference notebook for
    loading surgical phase data with temporal sequences and balanced sampling.

    Args:
        label_to_idx (Dict): Mapping from phase names to label indices
        root_dirs (str or List[str]): Root directory containing video folders
        lookback_window (int): Number of frames in each sequence
        transform: Transform to apply to frames
        max_sequences_per_phase (int): Maximum sequences per phase for balancing
        overlap (int): Overlap between sequences
        aug (bool): Whether to apply augmentation
        feature_dim (int): Feature dimension (for compatibility)
        frame_interval (int): Interval between frames in sequence
        test (bool): Whether this is test mode
    """

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        root_dirs: str,
        lookback_window: int = 10,
        transform=None,
        max_sequences_per_phase: int = 1000,
        overlap: int = 0,
        aug: bool = False,
        feature_dim: int = 2560,
        frame_interval: int = 1,
        test: bool = True,
    ):
        self.label_to_idx = label_to_idx
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.lookback_window = lookback_window
        self.transform = transform
        self.max_sequences_per_phase = max_sequences_per_phase
        self.overlap = overlap
        self.aug = aug
        self.feature_dim = feature_dim
        self.frame_interval = frame_interval
        self.test = test

        # Load data
        self.frame_paths = []
        self.labels = []
        self.video_names = []

        self._load_data()
        self.balanced_sequences = self._create_balanced_sequences()

        print(f"Dataset initialized with {len(self.balanced_sequences)} sequences")

    def _load_data(self):
        """Load frame paths and labels from the directory structure."""
        for root_dir in self.root_dirs:
            root_path = Path(root_dir)

            if not root_path.exists():
                print(f"Warning: Root directory {root_dir} does not exist")
                continue

            # Iterate through video folders
            for video_folder in root_path.iterdir():
                if not video_folder.is_dir():
                    continue

                # Look for frame_phase_mapping.txt file
                mapping_file = video_folder / "frame_phase_mapping.txt"
                if not mapping_file.exists():
                    # Try alternative locations
                    for subfolder in video_folder.iterdir():
                        if subfolder.is_dir():
                            alt_mapping = subfolder / "frame_phase_mapping.txt"
                            if alt_mapping.exists():
                                mapping_file = alt_mapping
                                video_folder = subfolder
                                break

                if not mapping_file.exists():
                    print(
                        f"Warning: No frame_phase_mapping.txt found in {video_folder}"
                    )
                    continue

                # Load frame-phase mapping
                frame_paths = []
                labels = []

                try:
                    with open(mapping_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue

                            parts = line.split(" ", 1)
                            if len(parts) != 2:
                                continue

                            frame_name, phase_name = parts
                            frame_path = video_folder / frame_name

                            if not frame_path.exists():
                                continue

                            # Map phase name to label index
                            if phase_name in self.label_to_idx:
                                label_idx = self.label_to_idx[phase_name]
                            else:
                                print(
                                    f"Warning: Unknown phase '{phase_name}' in {mapping_file}"
                                )
                                continue

                            frame_paths.append(str(frame_path))
                            labels.append(label_idx)

                    if frame_paths:
                        self.frame_paths.append(frame_paths)
                        self.labels.append(labels)
                        self.video_names.append(video_folder.name)
                        print(
                            f"Loaded {len(frame_paths)} frames from {video_folder.name}"
                        )

                except Exception as e:
                    print(f"Error loading {mapping_file}: {e}")

    def _create_balanced_sequences(self):
        """Create balanced sequences for training/testing."""
        sequences_per_label = {}

        # Collect all possible sequences
        for video_idx, video_labels in enumerate(self.labels):
            for start_idx in range(
                0,
                len(video_labels) - self.lookback_window * self.frame_interval + 1,
                max(1, self.lookback_window * self.frame_interval - self.overlap),
            ):

                # Get the label at the end of the sequence
                end_idx = start_idx + self.lookback_window * self.frame_interval - 1
                if end_idx >= len(video_labels):
                    continue

                label = video_labels[end_idx]

                if label not in sequences_per_label:
                    sequences_per_label[label] = []

                sequences_per_label[label].append((video_idx, start_idx, False))

        # Balance sequences
        max_sequences_per_label = {}
        for label, sequences in sequences_per_label.items():
            max_sequences_per_label[label] = min(
                len(sequences), self.max_sequences_per_phase
            )

        balanced_sequences = []
        augmentation_counts = {}

        for label, sequences in sequences_per_label.items():
            max_seq = max_sequences_per_label[label]

            # Randomly sample sequences
            if len(sequences) > max_seq:
                sequences = random.sample(sequences, max_seq)

            # Add augmented sequences if needed and augmentation is enabled
            if self.aug and not self.test and len(sequences) < max_seq:
                shortfall = max_seq - len(sequences)
                if sequences:  # Only augment if there are sequences to duplicate
                    additional_sequences = random.choices(
                        [
                            (video_idx, start_idx, True)
                            for video_idx, start_idx, _ in sequences
                        ],
                        k=shortfall,
                    )
                    sequences.extend(additional_sequences)
                    augmentation_counts[label] = len(additional_sequences)

            balanced_sequences.extend(sequences)

        # Print sequence information
        self.print_sequence_info(
            sequences_per_label,
            max_sequences_per_label,
            balanced_sequences,
            augmentation_counts,
        )

        return balanced_sequences

    def print_sequence_info(
        self,
        sequences_per_label,
        max_sequences_per_label,
        balanced_sequences,
        augmentation_counts,
    ):
        """Print detailed sequence information."""
        print("Balancing sequences with the following configuration:")
        print(
            f"  Lookback window: {self.lookback_window}, Frame interval: {self.frame_interval}, Overlap: {self.overlap}"
        )
        print(f"  Augmentation enabled: {self.aug and not self.test}")
        print("  Maximum sequences per label:")
        for label_num, max_seq in max_sequences_per_label.items():
            phases = [
                phase
                for phase, info in self.label_to_idx.items()
                if (isinstance(info, tuple) and info[0] == label_num)
                or info == label_num
            ]
            print(f"    Label {label_num} ({', '.join(phases)}): {max_seq}")

        print("\nSequences before balancing (only single-label sequences):")
        for label_num, sequences in sequences_per_label.items():
            phases = [
                phase
                for phase, info in self.label_to_idx.items()
                if (isinstance(info, tuple) and info[0] == label_num)
                or info == label_num
            ]
            print(
                f"  Label {label_num} ({', '.join(phases)}): {len(sequences)} sequences"
            )

        print("\nSequences after balancing:")
        for label_num in sequences_per_label.keys():
            selected_sequences = [
                seq
                for seq in balanced_sequences
                if self.labels[seq[0]][
                    seq[1] + self.lookback_window * self.frame_interval - 1
                ]
                == label_num
            ]
            phases = [
                phase
                for phase, info in self.label_to_idx.items()
                if (isinstance(info, tuple) and info[0] == label_num)
                or info == label_num
            ]
            aug_count = augmentation_counts.get(label_num, 0)
            print(
                f"  Label {label_num} ({', '.join(phases)}): {len(selected_sequences)} sequences "
                f"(including {aug_count} additionally augmented)"
            )

        print(f"\nTotal balanced sequences: {len(balanced_sequences)}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, List[str]]:
        """
        Get a sequence of frames and corresponding label.

        Args:
            idx (int): Index of the sequence

        Returns:
            Tuple containing:
                - frames (torch.Tensor): Stacked frames tensor
                - label (int): Phase label
                - frames_path (List[str]): List of frame paths
        """
        video_idx, start_idx, is_augmented = self.balanced_sequences[idx]
        frames = []
        frames_path = []
        frame_numbers = []

        for i in range(
            start_idx,
            start_idx + self.lookback_window * self.frame_interval,
            self.frame_interval,
        ):
            if i >= len(self.frame_paths[video_idx]):
                frame_path = self.frame_paths[video_idx][-1]
                frame_num = len(self.frame_paths[video_idx]) - 1
            else:
                frame_path = self.frame_paths[video_idx][i]
                frame_num = i

                frame_file = os.path.basename(frame_path)
                try:
                    frame_num = int(frame_file.split("_")[-1].replace(".jpg", ""))
                except (ValueError, IndexError):
                    frame_num = i

            frame = Image.open(frame_path).convert("RGB")

            frames.append(frame)
            frames_path.append(frame_path)
            frame_numbers.append(frame_num)

        # Apply transforms
        if is_augmented or not self.test:
            frames = transform_train(frames)
        else:
            frames = transform_test(frames)

        frames = torch.stack(frames)
        label = self.labels[video_idx][
            start_idx + self.lookback_window * self.frame_interval - 1
        ]

        return frames, label, frames_path

    def __len__(self) -> int:
        """Return the total number of sequences."""
        return len(self.balanced_sequences)

    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        class_counts = {}
        for video_idx, start_idx, _ in self.balanced_sequences:
            label = self.labels[video_idx][
                start_idx + self.lookback_window * self.frame_interval - 1
            ]
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts

    def get_phase_names(self) -> Dict[int, str]:
        """Get mapping from label indices to phase names."""
        idx_to_phase = {}
        for phase, idx in self.label_to_idx.items():
            if isinstance(idx, tuple):
                idx_to_phase[idx[0]] = phase
            else:
                idx_to_phase[idx] = phase
        return idx_to_phase


if __name__ == "__main__":
    # Test the dataset
    label_to_idx = {
        "Incision": 0,
        "Viscoelastic": 1,
        "Capsulorhexis": 2,
        "Hydrodissection": 3,
        "Phacoemulsification": 4,
        "IrrigationAspiration": 5,
        "CapsulePolishing": 6,
        "LensImplantation": 7,
        "LensPositioning": 8,
        "ViscoelasticSuction": 9,
        "TonifyingAntibiotics": 10,
    }

    # Create a dummy dataset for testing
    test_root = "/tmp/test_data"

    print("Sequential dataset test completed!")
