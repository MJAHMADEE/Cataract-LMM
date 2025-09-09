#!/usr/bin/env python3
"""
Transform module for surgical phase recognition.

This module provides the exact transforms and dataset classes used in the
Cataract-LMM dataset phase recognition task as described in the academic paper.

The transforms implement the preprocessing pipeline for the 13-phase surgical
classification as outlined in the "Phase Recognition Dataset Description" section.

Reference: Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark
           for Deep Learning in Surgical Video Analysis

Author: Surgical Phase Recognition Team
Date: August 31, 2025
"""

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Surgical Phase Recognition Constants
# Based on the Cataract-LMM dataset taxonomy of 13 distinct surgical phases
# as described in the academic paper methodology section

CATARACT_LMM_PHASES_13 = {
    "Incision": 0,                   # Initial corneal incision
    "Viscoelastic": 1,               # Viscoelastic agent injection
    "Capsulorhexis": 2,              # Opening of the anterior capsule
    "Hydrodissection": 3,            # Separation of lens nucleus from cortex
    "Phacoemulsification": 4,        # Ultrasonic lens fragmentation and removal
    "Irrigation Aspiration": 5,      # Cortex removal using irrigation/aspiration
    "Capsule Polishing": 6,          # Posterior capsule cleaning
    "Lens Implantation": 7,          # Intraocular lens implantation
    "Lens Positioning": 8,           # Adjustment of lens position in capsule
    "Viscoelastic Suction": 9,       # Removal of viscoelastic material
    "Anterior Chamber Flushing": 10, # Final irrigation of anterior chamber
    "Tonifying Antibiotics": 11,     # Instillation of antibiotics/medication
    "Idle": 12                       # Surgical inactivity or instrument exchange
}

# Reduced 11-phase mapping used in the current implementation
# (Compatible with existing notebooks and validation pipeline)
# Merges "Viscoelastic" and "Anterior Chamber Flushing" as mentioned in paper
CATARACT_LMM_PHASES_11 = {
    "Incision": 0,
    "Viscoelastic_Anterior_Chamber_Flushing": 1,  # Merged phases as per paper
    "Rhexis": 2,
    "Hydrodissection": 3,
    "Phacoemulsification": 4,
    "Aspiration": 5,
    "Capsule Polishing": 6,
    "Lens Insertion": 7,
    "Viscoelastic Removal": 8,
    "Tonifying-Antibiotics": 9,
    "Idle": 10,
}

# Default phase mapping (11 phases for backward compatibility)
DEFAULT_PHASE_MAPPING = CATARACT_LMM_PHASES_11


class SequenceTransform:
    """
    Transform class for video sequences used in Cataract-LMM phase recognition.

    This class implements the exact transform functionality used in the
    Cataract-LMM dataset for the phase recognition task. The transforms follow
    the preprocessing pipeline described in the academic paper methodology.

    Key features:
    - ImageNet-compatible normalization for pre-trained backbone compatibility
    - Data augmentation for training (horizontal flip, color jitter, rotation)
    - Consistent preprocessing for test/validation
    - Batch processing for video sequences

    Args:
        is_training (bool): Whether this is for training (with augmentation) or testing
        image_size (tuple): Target image size (height, width), default (224, 224)

    Reference:
        Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark
        for Deep Learning in Surgical Video Analysis
    """

    def __init__(self, is_training=True, image_size=(224, 224)):
        self.is_training = is_training
        self.image_size = image_size

        if is_training:
            # Training transforms with augmentation
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            # Test transforms without augmentation
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        """
        Apply transforms to a sequence of frames.

        Args:
            frames (List[Image.Image]): List of PIL Images

        Returns:
            List[torch.Tensor]: List of transformed tensors
        """
        transformed_frames = []

        for frame in frames:
            transformed_frame = self.transform(frame)
            transformed_frames.append(transformed_frame)

        return transformed_frames


# Global transform instances as used in the notebook
transform_train = SequenceTransform(is_training=True)
transform_test = SequenceTransform(is_training=False)


def get_training_transforms():
    """Get training transforms with augmentation."""
    return SequenceTransform(is_training=True)


def get_test_transforms():
    """Get test transforms without augmentation."""
    return SequenceTransform(is_training=False)


def create_transforms(is_training=True):
    """Create transforms compatible with the reference notebook."""
    return SequenceTransform(is_training=is_training)


def get_phase_mapping(variant="11_phase"):
    """
    Get the phase mapping for surgical phase recognition.

    Args:
        variant (str): Phase mapping variant ("11_phase" or "13_phase")

    Returns:
        Dict[str, int]: Phase name to index mapping

    Raises:
        ValueError: If variant is not supported
    """
    if variant == "11_phase":
        return CATARACT_LMM_PHASES_11.copy()
    elif variant == "13_phase":
        return CATARACT_LMM_PHASES_13.copy()
    else:
        raise ValueError(
            f"Unsupported phase mapping variant: {variant}. "
            "Use '11_phase' or '13_phase'"
        )


def get_reverse_phase_mapping(variant="11_phase"):
    """
    Get the reverse phase mapping (index to phase name).

    Args:
        variant (str): Phase mapping variant ("11_phase" or "13_phase")

    Returns:
        Dict[int, str]: Index to phase name mapping
    """
    phase_mapping = get_phase_mapping(variant)
    return {v: k for k, v in phase_mapping.items()}


def validate_phase_names(phase_names: List[str], variant="11_phase") -> bool:
    """
    Validate that phase names are compatible with the chosen mapping.

    Args:
        phase_names (List[str]): List of phase names to validate
        variant (str): Phase mapping variant to validate against

    Returns:
        bool: True if all phase names are valid, False otherwise
    """
    valid_phases = set(get_phase_mapping(variant).keys())
    return all(phase in valid_phases for phase in phase_names)


if __name__ == "__main__":
    # Test the transforms and phase mappings
    import numpy as np
    from PIL import Image

    print("Testing Cataract-LMM Phase Recognition Transforms")
    print("=" * 50)

    # Test phase mappings
    print("Available phase mappings:")
    print(f"11-phase mapping: {len(get_phase_mapping('11_phase'))} phases")
    for phase, idx in get_phase_mapping("11_phase").items():
        print(f"  {idx}: {phase}")

    print(f"\n13-phase mapping: {len(get_phase_mapping('13_phase'))} phases")
    for phase, idx in get_phase_mapping("13_phase").items():
        print(f"  {idx}: {phase}")

    # Test phase validation
    test_phases = ["Incision", "Capsulorhexis", "Phacoemulsification"]
    is_valid_11 = validate_phase_names(test_phases, "11_phase")
    is_valid_13 = validate_phase_names(test_phases, "13_phase")
    print(f"\nPhase validation test for {test_phases}:")
    print(f"  Valid for 11-phase: {is_valid_11}")
    print(f"  Valid for 13-phase: {is_valid_13}")

    # Create dummy frames for transform testing
    dummy_frames = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(5)
    ]

    # Test training transforms
    print(f"\nTesting transforms:")
    train_transform = get_training_transforms()
    train_tensors = train_transform(dummy_frames)

    print(f"Training transform test:")
    print(f"  Input frames: {len(dummy_frames)}")
    print(f"  Output tensors: {len(train_tensors)}")
    print(f"  Tensor shape: {train_tensors[0].shape}")
    print(
        f"  Tensor range: [{train_tensors[0].min():.3f}, {train_tensors[0].max():.3f}]"
    )

    # Test validation transforms
    test_transform = get_test_transforms()
    test_tensors = test_transform(dummy_frames)

    print(f"\nTest transform test:")
    print(f"  Output tensors: {len(test_tensors)}")
    print(f"  Tensor shape: {test_tensors[0].shape}")
    print(f"  Tensor range: [{test_tensors[0].min():.3f}, {test_tensors[0].max():.3f}]")

    print("\n✅ All transform tests completed successfully!")
    print("\nReady for Cataract-LMM phase recognition tasks!")
