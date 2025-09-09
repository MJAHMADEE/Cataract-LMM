#!/usr/bin/env python3
"""
Data Loading Utilities for Surgical Phase Recognition

This module provides data loading utilities, dataset creation helpers,
and data management functions for surgical phase recognition.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Some image processing functions may not work.")
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class DataManager:
    """
    Data management utilities for surgical phase recognition.

    Handles data organization, validation, splitting, and format conversion
    for surgical video datasets.

    Args:
        data_root (str): Root directory containing all data
        phase_mapping (dict, optional): Custom phase name to ID mapping
    """

    def __init__(self, data_root: str, phase_mapping: Optional[Dict[str, int]] = None):
        self.data_root = Path(data_root)

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

        self.reverse_mapping = {v: k for k, v in self.phase_mapping.items()}
        self.num_classes = len(self.phase_mapping)

        logger.info(f"DataManager initialized with {self.num_classes} surgical phases")

    def validate_video_files(self, video_dir: str) -> Dict[str, Any]:
        """
        Validate video files in a directory.

        Args:
            video_dir (str): Directory containing video files

        Returns:
            Dict[str, Any]: Validation results
        """
        video_dir = Path(video_dir)

        if not video_dir.exists():
            raise ValueError(f"Video directory does not exist: {video_dir}")

        # Supported video formats
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

        valid_videos = []
        invalid_videos = []
        video_info = []

        for video_path in video_dir.rglob("*"):
            if video_path.suffix.lower() in video_extensions:
                try:
                    if not CV2_AVAILABLE:
                        raise ImportError("cv2 not available")

                    # Test video reading
                    cap = cv2.VideoCapture(str(video_path))

                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        duration = frame_count / fps if fps > 0 else 0

                        valid_videos.append(str(video_path))
                        video_info.append(
                            {
                                "path": str(video_path),
                                "fps": fps,
                                "frames": frame_count,
                                "width": width,
                                "height": height,
                                "duration": duration,
                                "size_mb": video_path.stat().st_size / (1024 * 1024),
                            }
                        )

                        cap.release()
                    else:
                        invalid_videos.append(str(video_path))

                except Exception as e:
                    logger.error(f"Error validating video {video_path}: {e}")
                    invalid_videos.append(str(video_path))

        return {
            "total_videos": len(valid_videos) + len(invalid_videos),
            "valid_videos": len(valid_videos),
            "invalid_videos": len(invalid_videos),
            "valid_paths": valid_videos,
            "invalid_paths": invalid_videos,
            "video_info": video_info,
        }

    def create_annotation_template(
        self, video_paths: List[str], output_path: str, format_type: str = "csv"
    ) -> str:
        """
        Create an annotation template file for labeling videos.

        Args:
            video_paths (List[str]): List of video file paths
            output_path (str): Output file path
            format_type (str): 'csv' or 'json'

        Returns:
            str: Path to created template file
        """
        if format_type == "csv":
            # Create CSV template
            data = []
            for video_path in video_paths:
                rel_path = os.path.relpath(video_path, self.data_root)
                data.append(
                    {
                        "video_path": rel_path,
                        "phase": "",  # To be filled
                        "notes": "",  # Optional notes field
                    }
                )

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

            # Add phase options as comment
            with open(output_path, "a") as f:
                f.write(f"\n# Available phases: {list(self.phase_mapping.keys())}\n")

        elif format_type == "json":
            # Create JSON template
            template = {
                "metadata": {
                    "dataset_name": "surgical_phase_dataset",
                    "phases": list(self.phase_mapping.keys()),
                    "num_classes": self.num_classes,
                },
                "annotations": {},
            }

            for video_path in video_paths:
                rel_path = os.path.relpath(video_path, self.data_root)
                template["annotations"][rel_path] = {
                    "phase": "",  # To be filled
                    "notes": "",
                }

            with open(output_path, "w") as f:
                json.dump(template, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.info(f"Created annotation template: {output_path}")
        return output_path

    def validate_annotations(self, annotation_file: str) -> Dict[str, Any]:
        """
        Validate annotation file for consistency and completeness.

        Args:
            annotation_file (str): Path to annotation file

        Returns:
            Dict[str, Any]: Validation results
        """
        results = {"valid": True, "errors": [], "warnings": [], "statistics": {}}

        try:
            if annotation_file.endswith(".csv"):
                df = pd.read_csv(annotation_file)

                # Check required columns
                required_cols = ["video_path", "phase"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    results["errors"].append(f"Missing columns: {missing_cols}")
                    results["valid"] = False
                    return results

                # Check for missing values
                missing_videos = df["video_path"].isna().sum()
                missing_phases = df["phase"].isna().sum()

                if missing_videos > 0:
                    results["errors"].append(
                        f"{missing_videos} rows with missing video_path"
                    )
                    results["valid"] = False

                if missing_phases > 0:
                    results["errors"].append(
                        f"{missing_phases} rows with missing phase"
                    )
                    results["valid"] = False

                # Check video file existence
                missing_files = []
                for video_path in df["video_path"]:
                    if pd.notna(video_path):
                        full_path = self.data_root / video_path
                        if not full_path.exists():
                            missing_files.append(video_path)

                if missing_files:
                    results["warnings"].append(
                        f"{len(missing_files)} video files not found"
                    )

                # Check phase validity
                invalid_phases = []
                for phase in df["phase"]:
                    if pd.notna(phase) and phase not in self.phase_mapping:
                        invalid_phases.append(phase)

                if invalid_phases:
                    unique_invalid = set(invalid_phases)
                    results["errors"].append(f"Invalid phases: {list(unique_invalid)}")
                    results["valid"] = False

                # Statistics
                phase_counts = df["phase"].value_counts().to_dict()
                results["statistics"] = {
                    "total_samples": len(df),
                    "valid_samples": len(df) - missing_videos - missing_phases,
                    "phase_distribution": phase_counts,
                    "missing_files": len(missing_files),
                }

            elif annotation_file.endswith(".json"):
                with open(annotation_file, "r") as f:
                    data = json.load(f)

                # Validate JSON structure
                if "annotations" not in data:
                    results["errors"].append("Missing 'annotations' key in JSON")
                    results["valid"] = False
                    return results

                annotations = data["annotations"]

                # Check annotations
                missing_files = []
                invalid_phases = []
                phase_counts = defaultdict(int)

                for video_path, annotation in annotations.items():
                    # Check file existence
                    full_path = self.data_root / video_path
                    if not full_path.exists():
                        missing_files.append(video_path)

                    # Check phase validity
                    if isinstance(annotation, str):
                        phase = annotation
                    elif isinstance(annotation, dict) and "phase" in annotation:
                        phase = annotation["phase"]
                    else:
                        results["errors"].append(
                            f"Invalid annotation format for {video_path}"
                        )
                        results["valid"] = False
                        continue

                    if phase and phase not in self.phase_mapping:
                        invalid_phases.append(phase)
                    else:
                        phase_counts[phase] += 1

                if invalid_phases:
                    unique_invalid = set(invalid_phases)
                    results["errors"].append(f"Invalid phases: {list(unique_invalid)}")
                    results["valid"] = False

                if missing_files:
                    results["warnings"].append(
                        f"{len(missing_files)} video files not found"
                    )

                # Statistics
                results["statistics"] = {
                    "total_samples": len(annotations),
                    "phase_distribution": dict(phase_counts),
                    "missing_files": len(missing_files),
                }

        except Exception as e:
            results["errors"].append(f"Error reading annotation file: {e}")
            results["valid"] = False

        return results

    def split_dataset(
        self,
        annotation_file: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
        random_seed: int = 42,
    ) -> Dict[str, str]:
        """
        Split dataset into train/validation/test sets.

        Args:
            annotation_file (str): Path to annotation file
            train_ratio (float): Proportion of data for training
            val_ratio (float): Proportion of data for validation
            test_ratio (float): Proportion of data for testing
            stratify (bool): Whether to stratify splits by phase
            random_seed (int): Random seed for reproducibility

        Returns:
            Dict[str, str]: Paths to split annotation files
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        np.random.seed(random_seed)

        # Load annotations
        if annotation_file.endswith(".csv"):
            df = pd.read_csv(annotation_file)

            if stratify:
                # Stratified split by phase
                train_data = []
                val_data = []
                test_data = []

                for phase in df["phase"].unique():
                    if pd.isna(phase):
                        continue

                    phase_data = df[df["phase"] == phase].copy()
                    n_samples = len(phase_data)

                    # Calculate split sizes
                    n_train = int(n_samples * train_ratio)
                    n_val = int(n_samples * val_ratio)
                    n_test = n_samples - n_train - n_val

                    # Shuffle and split
                    phase_data = phase_data.sample(frac=1, random_state=random_seed)

                    train_data.append(phase_data.iloc[:n_train])
                    val_data.append(phase_data.iloc[n_train : n_train + n_val])
                    test_data.append(phase_data.iloc[n_train + n_val :])

                # Combine splits
                train_df = pd.concat(train_data, ignore_index=True)
                val_df = pd.concat(val_data, ignore_index=True)
                test_df = pd.concat(test_data, ignore_index=True)

            else:
                # Random split
                df = df.sample(frac=1, random_state=random_seed)
                n_total = len(df)

                n_train = int(n_total * train_ratio)
                n_val = int(n_total * val_ratio)

                train_df = df.iloc[:n_train]
                val_df = df.iloc[n_train : n_train + n_val]
                test_df = df.iloc[n_train + n_val :]

            # Save split files
            base_path = Path(annotation_file).parent
            base_name = Path(annotation_file).stem

            train_path = base_path / f"{base_name}_train.csv"
            val_path = base_path / f"{base_name}_val.csv"
            test_path = base_path / f"{base_name}_test.csv"

            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info(
                f"Dataset split: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}"
            )

            return {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            }

        else:
            raise ValueError("Dataset splitting currently supports CSV format only")

    def convert_annotation_format(
        self,
        input_file: str,
        output_file: str,
        input_format: str = "auto",
        output_format: str = "auto",
    ) -> str:
        """
        Convert annotation file between different formats.

        Args:
            input_file (str): Input annotation file
            output_file (str): Output annotation file
            input_format (str): Input format ('csv', 'json', or 'auto')
            output_format (str): Output format ('csv', 'json', or 'auto')

        Returns:
            str: Path to converted file
        """
        # Auto-detect formats
        if input_format == "auto":
            input_format = "csv" if input_file.endswith(".csv") else "json"

        if output_format == "auto":
            output_format = "csv" if output_file.endswith(".csv") else "json"

        # Load input data
        if input_format == "csv":
            df = pd.read_csv(input_file)

            if output_format == "json":
                # Convert CSV to JSON
                annotations = {}
                for _, row in df.iterrows():
                    annotations[row["video_path"]] = {"phase": row["phase"]}
                    # Add other columns as metadata
                    for col in df.columns:
                        if col not in ["video_path", "phase"]:
                            annotations[row["video_path"]][col] = row[col]

                output_data = {
                    "metadata": {
                        "dataset_name": "surgical_phase_dataset",
                        "phases": list(self.phase_mapping.keys()),
                        "num_classes": self.num_classes,
                    },
                    "annotations": annotations,
                }

                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=2)

        elif input_format == "json":
            with open(input_file, "r") as f:
                data = json.load(f)

            if output_format == "csv":
                # Convert JSON to CSV
                rows = []
                annotations = data.get("annotations", data)  # Handle both formats

                for video_path, annotation in annotations.items():
                    if isinstance(annotation, str):
                        row = {"video_path": video_path, "phase": annotation}
                    elif isinstance(annotation, dict):
                        row = {"video_path": video_path}
                        row.update(annotation)

                    rows.append(row)

                df = pd.DataFrame(rows)
                df.to_csv(output_file, index=False)

        logger.info(f"Converted {input_file} to {output_file}")
        return output_file

    def get_dataset_statistics(self, annotation_file: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.

        Args:
            annotation_file (str): Path to annotation file

        Returns:
            Dict[str, Any]: Dataset statistics
        """
        stats = {
            "total_samples": 0,
            "phase_distribution": {},
            "class_balance": {},
            "video_statistics": {},
            "recommendations": [],
        }

        try:
            if annotation_file.endswith(".csv"):
                df = pd.read_csv(annotation_file)

                # Basic statistics
                stats["total_samples"] = len(df)

                # Phase distribution
                phase_counts = df["phase"].value_counts().to_dict()
                stats["phase_distribution"] = phase_counts

                # Class balance analysis
                max_count = max(phase_counts.values()) if phase_counts else 0
                min_count = min(phase_counts.values()) if phase_counts else 0

                stats["class_balance"] = {
                    "max_samples": max_count,
                    "min_samples": min_count,
                    "balance_ratio": (
                        max_count / min_count if min_count > 0 else float("inf")
                    ),
                    "is_balanced": (
                        max_count / min_count < 2.0 if min_count > 0 else False
                    ),
                }

                # Video statistics (if available)
                video_validation = self.validate_video_files(str(self.data_root))
                if "video_info" in video_validation:
                    video_info = video_validation["video_info"]
                    if video_info:
                        durations = [v["duration"] for v in video_info]
                        fps_values = [v["fps"] for v in video_info]
                        resolutions = [(v["width"], v["height"]) for v in video_info]

                        stats["video_statistics"] = {
                            "total_videos": len(video_info),
                            "avg_duration": np.mean(durations),
                            "min_duration": np.min(durations),
                            "max_duration": np.max(durations),
                            "avg_fps": np.mean(fps_values),
                            "common_resolutions": Counter(resolutions).most_common(5),
                        }

                # Recommendations
                if not stats["class_balance"]["is_balanced"]:
                    stats["recommendations"].append(
                        "Consider using class weights or data augmentation for imbalanced classes"
                    )

                if stats["total_samples"] < 1000:
                    stats["recommendations"].append(
                        "Small dataset - consider data augmentation or transfer learning"
                    )

                if len(phase_counts) != self.num_classes:
                    missing_phases = set(self.phase_mapping.keys()) - set(
                        phase_counts.keys()
                    )
                    if missing_phases:
                        stats["recommendations"].append(
                            f"Missing phases in dataset: {missing_phases}"
                        )

        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            stats["error"] = str(e)

        return stats


def create_data_splits_from_directory(
    video_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy_files: bool = False,
) -> Dict[str, str]:
    """
    Create train/val/test splits from a directory of videos organized by phase.

    Expected directory structure:
    video_dir/
        Incision/
            video1.mp4
            video2.mp4
        Rhexis/
            video3.mp4
            ...

    Args:
        video_dir (str): Directory with videos organized by phase
        output_dir (str): Output directory for splits
        train_ratio (float): Training split ratio
        val_ratio (float): Validation split ratio
        test_ratio (float): Test split ratio
        copy_files (bool): Whether to copy video files or just create annotations

    Returns:
        Dict[str, str]: Paths to created annotation files
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find phase directories
    phase_dirs = [d for d in video_dir.iterdir() if d.is_dir()]

    if not phase_dirs:
        raise ValueError(f"No phase directories found in {video_dir}")

    # Collect all videos by phase
    videos_by_phase = defaultdict(list)

    for phase_dir in phase_dirs:
        phase_name = phase_dir.name
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

        for video_file in phase_dir.iterdir():
            if video_file.suffix.lower() in video_extensions:
                videos_by_phase[phase_name].append(video_file)

    # Create splits
    np.random.seed(42)

    train_data = []
    val_data = []
    test_data = []

    for phase_name, videos in videos_by_phase.items():
        n_videos = len(videos)

        # Shuffle videos
        np.random.shuffle(videos)

        # Calculate split sizes
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)

        # Split videos
        train_videos = videos[:n_train]
        val_videos = videos[n_train : n_train + n_val]
        test_videos = videos[n_train + n_val :]

        # Add to splits
        for video in train_videos:
            train_data.append({"video_path": str(video), "phase": phase_name})

        for video in val_videos:
            val_data.append({"video_path": str(video), "phase": phase_name})

        for video in test_videos:
            test_data.append({"video_path": str(video), "phase": phase_name})

    # Create annotation files
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    train_file = output_dir / "train_annotations.csv"
    val_file = output_dir / "val_annotations.csv"
    test_file = output_dir / "test_annotations.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(
        f"Created data splits: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}"
    )

    return {"train": str(train_file), "val": str(val_file), "test": str(test_file)}


if __name__ == "__main__":
    # Test data management utilities
    logger.info("Testing data management utilities...")

    print("DataManager class ready!")
    print("Use this module to:")
    print("1. Validate video files")
    print("2. Create annotation templates")
    print("3. Split datasets")
    print("4. Convert annotation formats")
    print("5. Get dataset statistics")
