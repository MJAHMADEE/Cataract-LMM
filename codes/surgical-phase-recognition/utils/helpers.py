#!/usr/bin/env python3
"""
Utility functions and helper modules for surgical phase recognition.

This module provides common utilities, helper functions, and convenience tools
used across the surgical phase recognition framework.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import hashlib
import json
import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    logger_name: str = "surgical_phase_recognition",
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file
        logger_name (str): Name of the logger

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        dir_path (Union[str, Path]): Directory path

    Returns:
        Path: Path object of the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.

    Args:
        data (Dict): Data to save
        file_path (Union[str, Path]): Output file path
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        file_path (Union[str, Path]): Input file path

    Returns:
        Dict[str, Any]: Loaded data
    """
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to pickle file.

    Args:
        data (Any): Data to save
        file_path (Union[str, Path]): Output file path
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load data from pickle file.

    Args:
        file_path (Union[str, Path]): Input file path

    Returns:
        Any: Loaded data
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.

    Args:
        file_path (Union[str, Path]): Path to the file
        algorithm (str): Hashing algorithm ('md5', 'sha1', 'sha256')

    Returns:
        str: File hash
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information.

    Returns:
        Dict[str, Any]: GPU memory statistics
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    info = {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": [],
    }

    for i in range(torch.cuda.device_count()):
        device_info = {
            "device_id": i,
            "name": torch.cuda.get_device_name(i),
            "memory_allocated": torch.cuda.memory_allocated(i),
            "memory_reserved": torch.cuda.memory_reserved(i),
            "memory_total": torch.cuda.get_device_properties(i).total_memory,
        }
        info["devices"].append(device_info)

    return info


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def format_bytes(bytes_val: int) -> str:
    """
    Format bytes in human-readable format.

    Args:
        bytes_val (int): Number of bytes

    Returns:
        str: Formatted bytes string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


class Timer:
    """Context manager for timing operations."""

    def __init__(
        self, name: str = "Operation", logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"{self.name} completed in {format_time(duration)}")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


class ConfigValidator:
    """Utility for validating configuration dictionaries."""

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """
        Validate model configuration.

        Args:
            config (Dict): Model configuration

        Returns:
            bool: True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["type", "num_classes"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in model config: {field}")

        # Validate model type
        valid_types = [
            "swin_video_3d",
            "mvit",
            "swin3d_transformer",
            "r3d_18",
            "mc3",
            "r2plus1d",
            "slow_r50",
            "x3d",
            "resnet_lstm",
            "efficientnet_gru",
            "tecno",
        ]

        if config["type"] not in valid_types:
            raise ValueError(
                f"Invalid model type: {config['type']}. Must be one of {valid_types}"
            )

        # Validate num_classes
        if not isinstance(config["num_classes"], int) or config["num_classes"] < 1:
            raise ValueError("num_classes must be a positive integer")

        return True

    @staticmethod
    def validate_data_config(config: Dict[str, Any]) -> bool:
        """
        Validate data configuration.

        Args:
            config (Dict): Data configuration

        Returns:
            bool: True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["batch_size"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in data config: {field}")

        # Validate batch_size
        if not isinstance(config["batch_size"], int) or config["batch_size"] < 1:
            raise ValueError("batch_size must be a positive integer")

        # Validate optional fields
        if "num_workers" in config:
            if not isinstance(config["num_workers"], int) or config["num_workers"] < 0:
                raise ValueError("num_workers must be a non-negative integer")

        if "sequence_length" in config:
            if (
                not isinstance(config["sequence_length"], int)
                or config["sequence_length"] < 1
            ):
                raise ValueError("sequence_length must be a positive integer")

        return True

    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> bool:
        """
        Validate training configuration.

        Args:
            config (Dict): Training configuration

        Returns:
            bool: True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["learning_rate", "max_epochs"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in training config: {field}")

        # Validate learning_rate
        if (
            not isinstance(config["learning_rate"], (int, float))
            or config["learning_rate"] <= 0
        ):
            raise ValueError("learning_rate must be a positive number")

        # Validate max_epochs
        if not isinstance(config["max_epochs"], int) or config["max_epochs"] < 1:
            raise ValueError("max_epochs must be a positive integer")

        return True


class MetricsTracker:
    """Track and manage training/validation metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_metrics = {}

    def update(self, metric_dict: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics.

        Args:
            metric_dict (Dict[str, float]): Dictionary of metrics
            step (int, optional): Step number
        """
        for name, value in metric_dict.items():
            self.metrics[name].append(
                {"value": value, "step": step or len(self.metrics[name])}
            )

            # Track best metrics
            if (
                name not in self.best_metrics
                or value > self.best_metrics[name]["value"]
            ):
                self.best_metrics[name] = {
                    "value": value,
                    "step": step or len(self.metrics[name]) - 1,
                }

    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get history of a specific metric."""
        return self.metrics.get(metric_name, [])

    def get_best_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get best value of a specific metric."""
        return self.best_metrics.get(metric_name)

    def plot_metrics(self, metric_names: List[str], save_path: Optional[str] = None):
        """
        Plot metric histories.

        Args:
            metric_names (List[str]): Names of metrics to plot
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(
            len(metric_names), 1, figsize=(10, 4 * len(metric_names))
        )

        if len(metric_names) == 1:
            axes = [axes]

        for i, metric_name in enumerate(metric_names):
            history = self.get_metric_history(metric_name)
            if history:
                steps = [h["step"] for h in history]
                values = [h["value"] for h in history]

                axes[i].plot(steps, values, marker="o", linewidth=2, markersize=4)
                axes[i].set_title(f"{metric_name} History")
                axes[i].set_xlabel("Step")
                axes[i].set_ylabel(metric_name)
                axes[i].grid(True, alpha=0.3)

                # Mark best value
                best = self.get_best_metric(metric_name)
                if best:
                    axes[i].axhline(
                        y=best["value"], color="red", linestyle="--", alpha=0.7
                    )
                    axes[i].text(
                        0.02,
                        0.98,
                        f"Best: {best['value']:.4f} (Step {best['step']})",
                        transform=axes[i].transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        data = []
        for metric_name, history in self.metrics.items():
            for entry in history:
                data.append(
                    {
                        "metric": metric_name,
                        "step": entry["step"],
                        "value": entry["value"],
                    }
                )

        return pd.DataFrame(data)

    def save(self, file_path: Union[str, Path]):
        """Save metrics to file."""
        data = {"metrics": dict(self.metrics), "best_metrics": self.best_metrics}
        save_json(data, file_path)

    def load(self, file_path: Union[str, Path]):
        """Load metrics from file."""
        data = load_json(file_path)
        self.metrics = defaultdict(list, data["metrics"])
        self.best_metrics = data["best_metrics"]


def create_video_summary_stats(video_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Create summary statistics for video dataset.

    Args:
        video_dir (Union[str, Path]): Directory containing videos

    Returns:
        Dict[str, Any]: Summary statistics
    """
    video_dir = Path(video_dir)

    if not video_dir.exists():
        return {"error": "Video directory does not exist"}

    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))

    stats = {
        "total_videos": len(video_files),
        "video_formats": {},
        "total_size_bytes": 0,
        "videos": [],
    }

    for video_file in video_files:
        file_size = video_file.stat().st_size
        file_ext = video_file.suffix.lower()

        stats["total_size_bytes"] += file_size
        stats["video_formats"][file_ext] = stats["video_formats"].get(file_ext, 0) + 1

        stats["videos"].append(
            {
                "name": video_file.name,
                "size_bytes": file_size,
                "size_formatted": format_bytes(file_size),
                "format": file_ext,
            }
        )

    stats["total_size_formatted"] = format_bytes(stats["total_size_bytes"])
    stats["average_size_bytes"] = stats["total_size_bytes"] / max(len(video_files), 1)
    stats["average_size_formatted"] = format_bytes(stats["average_size_bytes"])

    return stats


def validate_phase_annotations(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate phase annotations for consistency and completeness.

    Args:
        annotations (List[Dict]): List of annotation dictionaries

    Returns:
        Dict[str, Any]: Validation results
    """
    results = {
        "total_annotations": len(annotations),
        "valid_annotations": 0,
        "errors": [],
        "warnings": [],
        "phase_distribution": defaultdict(int),
        "video_coverage": defaultdict(list),
    }

    required_fields = ["video_id", "start_frame", "end_frame", "phase"]

    for i, annotation in enumerate(annotations):
        # Check required fields
        missing_fields = [field for field in required_fields if field not in annotation]
        if missing_fields:
            results["errors"].append(f"Annotation {i}: Missing fields {missing_fields}")
            continue

        # Check frame order
        if annotation["start_frame"] >= annotation["end_frame"]:
            results["errors"].append(f"Annotation {i}: start_frame >= end_frame")
            continue

        # Check non-negative frames
        if annotation["start_frame"] < 0 or annotation["end_frame"] < 0:
            results["errors"].append(f"Annotation {i}: Negative frame numbers")
            continue

        results["valid_annotations"] += 1
        results["phase_distribution"][annotation["phase"]] += 1
        results["video_coverage"][annotation["video_id"]].append(
            {
                "start_frame": annotation["start_frame"],
                "end_frame": annotation["end_frame"],
                "phase": annotation["phase"],
            }
        )

    # Check for overlapping annotations within videos
    for video_id, video_annotations in results["video_coverage"].items():
        sorted_annotations = sorted(video_annotations, key=lambda x: x["start_frame"])

        for i in range(len(sorted_annotations) - 1):
            current = sorted_annotations[i]
            next_ann = sorted_annotations[i + 1]

            if current["end_frame"] > next_ann["start_frame"]:
                results["warnings"].append(
                    f"Video {video_id}: Overlapping annotations at frames "
                    f"{current['end_frame']} and {next_ann['start_frame']}"
                )

    results["validation_summary"] = {
        "success_rate": results["valid_annotations"]
        / max(results["total_annotations"], 1),
        "error_count": len(results["errors"]),
        "warning_count": len(results["warnings"]),
        "unique_phases": len(results["phase_distribution"]),
        "unique_videos": len(results["video_coverage"]),
    }

    return results


if __name__ == "__main__":
    # Test utilities
    logger = setup_logging("DEBUG")
    logger.info("Testing surgical phase recognition utilities")

    # Test timer
    with Timer("Test operation", logger):
        import time

        time.sleep(1)

    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update({"accuracy": 0.85, "loss": 0.5}, step=1)
    tracker.update({"accuracy": 0.87, "loss": 0.45}, step=2)
    tracker.update({"accuracy": 0.90, "loss": 0.40}, step=3)

    print("Best accuracy:", tracker.get_best_metric("accuracy"))

    # Test GPU info
    gpu_info = get_gpu_memory_info()
    print("GPU Info:", gpu_info)

    logger.info("Utilities test completed!")
