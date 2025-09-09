"""
YOLO Trainer Implementation for Surgical Instance Segmentation

This trainer exactly matches the training procedure from the reference YOLO notebooks
(train_yolo8.ipynb and train_yolo11.ipynb) while providing additional production features.

The implementation follows the exact same:
- Model loading (yolo8l-seg, yolo11l-seg)
- Training configuration (data.yaml, epochs=80, imgsz=640, batch=20, etc.)
- Training parameters and settings
- Model saving procedures
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print(
        "Warning: ultralytics not installed. Please install with: pip install ultralytics"
    )
    YOLO = None

from .base_trainer import BaseTrainer


class YOLOTrainer(BaseTrainer):
    """
    YOLO Trainer for Surgical Instance Segmentation

    This implementation exactly matches the training procedure from the reference notebooks:
    - train_yolo8.ipynb: YOLOv8 segmentation training
    - train_yolo11.ipynb: YOLOv11 segmentation training

    Key Features:
    - Exact same model loading (yolo8l-seg, yolo11l-seg)
    - Same training configuration (epochs=80, imgsz=640, batch=20, etc.)
    - Same data.yaml configuration
    - Same training parameters and device settings
    - Automatic model saving and checkpointing
    """

    def __init__(
        self, model_name: str = "yolo11l-seg", device: Union[int, str] = 0, **kwargs
    ):
        """
        Initialize YOLO trainer with notebook-compatible configuration

        Args:
            model_name (str): YOLO model name ('yolo8l-seg' or 'yolo11l-seg' to match notebooks)
            device (int or str): Device to use (0 for GPU, 'cpu' for CPU, matches notebooks)
        """
        super().__init__()

        if YOLO is None:
            raise ImportError(
                "ultralytics package is required. Install with: pip install ultralytics"
            )

        self.model_name = model_name
        self.device = device

        # Load pretrained YOLO model exactly as in notebooks
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)

        # Store configuration
        self.config = {"model_name": model_name, "device": device, "task": "segment"}

        # Training history
        self.training_history = defaultdict(list)

        print(f"YOLO model '{model_name}' loaded successfully")

        # Run ultralytics checks exactly as in notebooks
        try:
            ultralytics.checks()
        except:
            print("Warning: ultralytics checks failed")

    def train(
        self,
        data_yaml: str = "./data.yaml",
        epochs: int = 80,
        imgsz: int = 640,
        batch: int = 20,
        device: Optional[Union[int, str]] = None,
        plots: bool = True,
        resume: bool = True,
        save_dir: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train YOLO model following the exact same configuration as the reference notebooks

        Default parameters match the notebooks exactly:
        - data: "./data.yaml" (same as notebooks)
        - epochs: 80 (same as notebooks)
        - imgsz: 640 (same as notebooks)
        - batch: 20 (same as notebooks)
        - device: 0 (same as notebooks, GPU)
        - plots: True (same as notebooks)
        - resume: True (same as notebooks)

        Args:
            data_yaml (str): Path to dataset YAML file (default: "./data.yaml" to match notebooks)
            epochs (int): Number of epochs (default: 80 to match notebooks)
            imgsz (int): Image size for training (default: 640 to match notebooks)
            batch (int): Batch size (default: 20 to match notebooks)
            device (int or str): Device to use (default: 0 for GPU to match notebooks)
            plots (bool): Generate training plots (default: True to match notebooks)
            resume (bool): Resume training if interrupted (default: True to match notebooks)
            save_dir (str): Directory to save results
            project (str): Project name for organization
            name (str): Experiment name
            **kwargs: Additional arguments for YOLO training

        Returns:
            Dict containing training results and metrics
        """
        # Use instance device if not specified
        if device is None:
            device = self.device

        print("=" * 60)
        print(f"Training {self.model_name.upper()} for Surgical Instance Segmentation")
        print("Following exact configuration from reference notebooks")
        print("=" * 60)

        # Validate data.yaml exists
        if not os.path.exists(data_yaml):
            print(
                f"Warning: {data_yaml} not found. Please ensure data.yaml is properly configured."
            )

        # Print training configuration exactly as shown in notebooks
        print("Training Configuration:")
        print(f"  Data: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Image Size: {imgsz}")
        print(f"  Batch Size: {batch}")
        print(f"  Device: {device}")
        print(f"  Plots: {plots}")
        print(f"  Resume: {resume}")

        start_time = time.time()

        try:
            # Train model exactly as in notebooks
            results = self.model.train(
                data=data_yaml,  # Path to dataset YAML (same as notebooks)
                epochs=epochs,  # Number of epochs (same as notebooks)
                imgsz=imgsz,  # Image size for training (same as notebooks)
                batch=batch,  # Batch size (same as notebooks)
                device=device,  # Use GPU (0) or 'CPU' (same as notebooks)
                plots=plots,  # Generate plots (same as notebooks)
                resume=resume,  # Resume training (same as notebooks)
                save_dir=save_dir,  # Save directory
                project=project,  # Project name
                name=name,  # Experiment name
                **kwargs,  # Additional arguments
            )

            training_time = time.time() - start_time
            print(
                f"Training completed successfully in {training_time:.2f}s ({training_time/60:.1f} minutes)"
            )

            # Store training results
            self.training_results = results

            # Extract and store training history
            if hasattr(results, "results_dict"):
                for key, value in results.results_dict.items():
                    self.training_history[key].append(value)

            return {
                "success": True,
                "results": results,
                "training_time": training_time,
                "model_path": (
                    results.save_dir if hasattr(results, "save_dir") else None
                ),
            }

        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "training_time": time.time() - start_time,
            }

    def validate(
        self,
        data_yaml: Optional[str] = None,
        imgsz: int = 640,
        batch: int = 1,
        device: Optional[Union[int, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate YOLO model

        Args:
            data_yaml (str): Path to dataset YAML file
            imgsz (int): Image size for validation
            batch (int): Batch size for validation
            device (int or str): Device to use
            **kwargs: Additional arguments for YOLO validation

        Returns:
            Dict containing validation results
        """
        if device is None:
            device = self.device

        print("Running validation...")

        try:
            # Run validation
            results = self.model.val(
                data=data_yaml, imgsz=imgsz, batch=batch, device=device, **kwargs
            )

            print("Validation completed successfully")
            return {"success": True, "results": results}

        except Exception as e:
            print(f"Validation failed with error: {str(e)}")
            return {"success": False, "error": str(e)}

    def predict(
        self,
        source: Union[str, List[str]],
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.7,
        device: Optional[Union[int, str]] = None,
        save: bool = False,
        **kwargs,
    ) -> Any:
        """
        Run prediction with YOLO model

        Args:
            source: Source for prediction (image path, video, etc.)
            imgsz (int): Image size for prediction
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            device (int or str): Device to use
            save (bool): Save prediction results
            **kwargs: Additional arguments for YOLO prediction

        Returns:
            Prediction results
        """
        if device is None:
            device = self.device

        try:
            results = self.model.predict(
                source=source,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                save=save,
                **kwargs,
            )

            return results

        except Exception as e:
            print(f"Prediction failed with error: {str(e)}")
            return None

    def export(
        self, format: str = "onnx", imgsz: int = 640, optimize: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Export YOLO model to different formats

        Args:
            format (str): Export format ('onnx', 'tensorrt', 'coreml', etc.)
            imgsz (int): Image size for export
            optimize (bool): Optimize for deployment
            **kwargs: Additional export arguments

        Returns:
            Dict containing export results
        """
        try:
            print(f"Exporting model to {format.upper()} format...")

            exported_model = self.model.export(
                format=format, imgsz=imgsz, optimize=optimize, **kwargs
            )

            print(f"Model exported successfully to {format.upper()} format")
            return {"success": True, "exported_model": exported_model, "format": format}

        except Exception as e:
            print(f"Export failed with error: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_model(self, save_path: str) -> None:
        """
        Save YOLO model

        Args:
            save_path (str): Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save model
            self.model.save(save_path)
            print(f"Model saved as '{save_path}'")

        except Exception as e:
            print(f"Failed to save model: {str(e)}")

    def load_model(self, load_path: str) -> None:
        """
        Load YOLO model weights

        Args:
            load_path (str): Path to load the model from
        """
        try:
            self.model = YOLO(load_path)
            print(f"Model loaded from '{load_path}'")

        except Exception as e:
            print(f"Failed to load model: {str(e)}")

    def get_model(self) -> Any:
        """Get the YOLO model"""
        return self.model

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return dict(self.training_history)


def train_yolo8_from_notebook(
    data_root: str, epochs: int = 80, save_path: Optional[str] = None
) -> YOLOTrainer:
    """
    Train YOLOv8 following the exact same procedure as train_yolo8.ipynb notebook

    This function replicates the complete training procedure from the notebook:
    1. Load YOLOv8l-seg model exactly as in notebook
    2. Train with same configuration as notebook
    3. Use same parameters (epochs=80, imgsz=640, batch=20, etc.)

    Args:
        data_root (str): Path to dataset root containing data.yaml
        epochs (int): Number of epochs (default: 80 to match notebook)
        save_path (str): Path to save the trained model (optional)

    Returns:
        YOLOTrainer: Trained trainer instance
    """
    print("=" * 60)
    print("Training YOLOv8 for Surgical Instance Segmentation")
    print("Following exact procedure from train_yolo8.ipynb notebook")
    print("=" * 60)

    # Change to data directory if needed
    original_cwd = os.getcwd()
    if os.path.exists(data_root):
        os.chdir(data_root)
        print(f"Changed directory to: {data_root}")

    try:
        # Initialize trainer exactly as in notebook
        trainer = YOLOTrainer(
            model_name="yolo8l-seg",  # Exact same as notebook
            device=0,  # Same as notebook (GPU)
        )

        # Train exactly as in notebook
        results = trainer.train(
            data="./data.yaml",  # Same as notebook
            epochs=epochs,  # Same as notebook (80)
            imgsz=640,  # Same as notebook
            batch=20,  # Same as notebook
            device=0,  # Same as notebook
            plots=True,  # Same as notebook
            resume=True,  # Same as notebook
        )

        if save_path:
            trainer.save_model(save_path)

        print("=" * 60)
        print("YOLOv8 training completed successfully!")
        print("=" * 60)

        return trainer

    finally:
        # Restore original directory
        os.chdir(original_cwd)


def train_yolo11_from_notebook(
    data_root: str, epochs: int = 80, save_path: Optional[str] = None
) -> YOLOTrainer:
    """
    Train YOLOv11 following the exact same procedure as train_yolo11.ipynb notebook

    This function replicates the complete training procedure from the notebook:
    1. Load YOLOv11l-seg model exactly as in notebook
    2. Train with same configuration as notebook
    3. Use same parameters (epochs=80, imgsz=640, batch=20, etc.)

    Args:
        data_root (str): Path to dataset root containing data.yaml
        epochs (int): Number of epochs (default: 80 to match notebook)
        save_path (str): Path to save the trained model (optional)

    Returns:
        YOLOTrainer: Trained trainer instance
    """
    print("=" * 60)
    print("Training YOLOv11 for Surgical Instance Segmentation")
    print("Following exact procedure from train_yolo11.ipynb notebook")
    print("=" * 60)

    # Change to data directory if needed
    original_cwd = os.getcwd()
    if os.path.exists(data_root):
        os.chdir(data_root)
        print(f"Changed directory to: {data_root}")

    try:
        # Initialize trainer exactly as in notebook
        trainer = YOLOTrainer(
            model_name="yolo11l-seg",  # Exact same as notebook
            device=0,  # Same as notebook (GPU)
        )

        # Train exactly as in notebook
        results = trainer.train(
            data="./data.yaml",  # Same as notebook
            epochs=epochs,  # Same as notebook (80)
            imgsz=640,  # Same as notebook
            batch=20,  # Same as notebook
            device=0,  # Same as notebook
            plots=True,  # Same as notebook
            resume=True,  # Same as notebook
        )

        if save_path:
            trainer.save_model(save_path)

        print("=" * 60)
        print("YOLOv11 training completed successfully!")
        print("=" * 60)

        return trainer

    finally:
        # Restore original directory
        os.chdir(original_cwd)


def create_data_yaml(
    dataset_path: str, class_names: List[str], save_path: str = "./data.yaml"
) -> str:
    """
    Create data.yaml file for YOLO training

    Args:
        dataset_path (str): Path to dataset root
        class_names (List[str]): List of class names
        save_path (str): Path to save data.yaml

    Returns:
        str: Path to created data.yaml file
    """
    # Create data.yaml structure matching YOLO format
    data_config = {
        "path": dataset_path,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    # Save data.yaml
    with open(save_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"Created data.yaml at: {save_path}")
    return save_path


# Example usage exactly matching the notebook workflows
if __name__ == "__main__":
    # Example paths - update to match your dataset
    data_root = "/content/content/drive/MyDrive/final_main_seg_dataset_just_ARAS"

    # Train YOLOv8 exactly as in notebook
    print("Training YOLOv8...")
    yolo8_trainer = train_yolo8_from_notebook(
        data_root=data_root, epochs=80  # Same as notebook
    )

    # Train YOLOv11 exactly as in notebook
    print("Training YOLOv11...")
    yolo11_trainer = train_yolo11_from_notebook(
        data_root=data_root, epochs=80  # Same as notebook
    )
