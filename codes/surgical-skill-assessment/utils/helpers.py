"""
Helper utilities for the surgical skill assessment project.

This module provides common utility functions for reproducibility, hardware detection,
logging, and directory management throughout the project.

Author: Surgical Skill Assessment Team
Date: August 2025
"""

import gc
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import rich for better console output, fall back to logging if not available
try:
    from rich.console import Console

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False
    logger.warning("Rich not available. Using basic logging for console output.")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string (e.g., "1h 30m 45s", "2m 30s", "45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def seed_everything(seed: int) -> None:
    """
    Set seeds for all random number generators to ensure reproducibility.

    Args:
        seed (int): Random seed value

    Note:
        This function sets seeds for:
        - Python's built-in random module
        - NumPy's random number generator
        - PyTorch's random number generators (CPU and CUDA)
        - PyTorch's deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")


def check_gpu_memory() -> Tuple[bool, float]:
    """
    Check GPU availability and memory information.

    Returns:
        Tuple[bool, float]: (gpu_available, total_memory_gb)
            - gpu_available: Whether CUDA GPU is available
            - total_memory_gb: Total GPU memory in gigabytes (0.0 if no GPU)
    """
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name} with {gpu_mem:.2f} GB memory")
        return True, gpu_mem
    else:
        logger.info("No GPU available, using CPU")
        return False, 0.0


def print_section(title: str, emoji: str = "🔹", use_emojis: bool = True) -> None:
    """
    Print a formatted section header to the console.

    Args:
        title (str): The section title to display
        emoji (str, optional): Emoji to use in the header. Defaults to "🔹".
        use_emojis (bool, optional): Whether to include emojis. Defaults to True.
    """
    if RICH_AVAILABLE and console:
        if use_emojis:
            console.print(f"\n{emoji} ───── {title.upper()} ─────", style="bold cyan")
        else:
            console.print(f"\n───── {title.upper()} ─────", style="bold cyan")
    else:
        if use_emojis:
            logger.info(f"\n{emoji} ───── {title.upper()} ─────")
        else:
            logger.info(f"\n───── {title.upper()} ─────")


def setup_output_dirs(output_root: str) -> Dict[str, Path]:
    """
    Create organized output directory structure for experiment results.

    Args:
        output_root (str): Base directory for all outputs

    Returns:
        Dict[str, Path]: Dictionary mapping directory names to Path objects:
            - 'root': Main experiment directory (timestamped)
            - 'checkpoints': Model checkpoint storage
            - 'logs': Training and evaluation logs
            - 'plots': Visualization outputs
            - 'predictions': Prediction results and analysis

    Note:
        Creates a timestamped subdirectory to prevent overwriting previous runs.
        Directory structure: output_root/run_YYYYMMDD_HHMMSS/
    """
    output_path = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"run_{timestamp}"

    dirs = {
        "root": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "plots": run_dir / "plots",
        "predictions": run_dir / "predictions",
    }

    for dir_name, dir_path in dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")

    logger.info(f"Output directories created in: {run_dir}")
    return dirs


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available computing device.

    Args:
        prefer_cuda (bool, optional): Whether to prefer CUDA over CPU. Defaults to True.

    Returns:
        torch.device: PyTorch device object (cuda or cpu)
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        Tuple[int, int]: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    )
    return total_params, trainable_params


def tune_batch_size(
    model, device, initial_batch_size: float, clip_len: int, model_name: str
) -> Tuple[int, int]:
    """
    Dynamically tune batch size and clip length to fit in GPU memory.

    This function is essential for production deployment as it automatically
    optimizes memory usage based on available GPU resources.

    Args:
        model: PyTorch model
        device: Device (cuda/cpu)
        initial_batch_size: Starting batch size
        clip_len: Number of frames in clips
        model_name: Name of the model (affects memory requirements)

    Returns:
        Tuple of (optimal_batch_size, optimal_clip_len)
    """
    import gc

    model.eval()
    batch_size = initial_batch_size
    current_clip_len = clip_len

    # Adjust initial batch size based on model type
    if model_name in ["timesformer", "mvit", "videomae", "vivit"]:
        batch_size = max(1, batch_size // 2)  # Transformers need more memory

    # SlowFast and Slow models need at least 32 frames
    if model_name in ["slowfast_r50", "slow_r50"]:
        min_clip_len = 32
    else:
        min_clip_len = 8

    while batch_size >= 1 and current_clip_len >= min_clip_len:
        try:
            # Test forward pass
            dummy_input = torch.randn(batch_size, 3, current_clip_len, 224, 224).to(
                device
            )
            with torch.no_grad():
                if model_name.startswith("slowfast"):
                    alpha = 4
                    slow_pathway = dummy_input[:, :, ::alpha, :, :]
                    fast_pathway = dummy_input
                    model_inputs = [slow_pathway, fast_pathway]
                    _ = model(model_inputs)
                else:
                    _ = model(dummy_input)

            logger.info(
                f"Optimal batch_size: {batch_size}, clip_len: {current_clip_len}"
            )
            return batch_size, current_clip_len

        except RuntimeError as e:
            if "out of memory" in str(e) or "smaller than kernel size" in str(e):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                if "smaller than kernel size" in str(e) and model_name in [
                    "slowfast_r50",
                    "slow_r50",
                ]:
                    # SlowFast/Slow models need more frames, double the clip length
                    current_clip_len *= 2
                    logger.warning(
                        f"{model_name} needs more frames, increasing clip_len to {current_clip_len}"
                    )
                elif batch_size > 1:
                    batch_size //= 2
                    logger.warning(f"OOM detected, reducing batch_size to {batch_size}")
                else:
                    current_clip_len //= 2
                    batch_size = initial_batch_size
                    logger.warning(
                        f"OOM detected, reducing clip_len to {current_clip_len}"
                    )
            else:
                raise e

    logger.error("Could not find suitable batch_size/clip_len combination")
    return 1, min_clip_len


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for minimizing metric, 'max' for maximizing
        restore_best_weights: Whether to restore best weights on early stop
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

        if mode == "min":
            self.monitor_op = lambda x, y: x < (y - min_delta)
        elif mode == "max":
            self.monitor_op = lambda x, y: x > (y + min_delta)
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'")

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped.

        Args:
            score: Current validation score
            model: Model to save weights from

        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        elif self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                # Restore best weights
                model.load_state_dict(
                    {
                        k: v.to(next(model.parameters()).device)
                        for k, v in self.best_weights.items()
                    }
                )

        return self.early_stop


def create_plots(
    train_losses: list,
    val_losses: list,
    val_accuracies: list,
    output_dir: Path,
    title: str = "Training Progress",
) -> None:
    """
    Create training progress plots.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        output_dir: Directory to save plots
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, val_accuracies, "g-", label="Validation Accuracy", linewidth=2)
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        plot_path = output_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Training plots saved to {plot_path}")

    except ImportError:
        logger.warning("Matplotlib not available. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error creating plots: {e}")


def create_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    output_dir: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Create and save confusion matrix plot.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save plot
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to {cm_path}")

    except ImportError:
        logger.warning(
            "Required packages (matplotlib, seaborn, sklearn) not available. Skipping confusion matrix."
        )
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e}")


def print_tree(
    path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0
) -> None:
    """
    Print directory structure as a tree.

    Args:
        path: Root path to print
        prefix: Current line prefix
        max_depth: Maximum depth to traverse
        current_depth: Current traversal depth
    """
    if current_depth > max_depth:
        return

    if not path.is_dir():
        return

    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "

        if console and RICH_AVAILABLE:
            icon = "📁" if item.is_dir() else "📄"
            console.print(f"{prefix}{current_prefix}{icon} {item.name}")
        else:
            print(f"{prefix}{current_prefix}{item.name}")

        if item.is_dir() and current_depth < max_depth:
            extension_prefix = "    " if is_last else "│   "
            print_tree(item, prefix + extension_prefix, max_depth, current_depth + 1)
