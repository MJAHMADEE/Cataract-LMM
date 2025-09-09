"""
Training Manager for Surgical Instance Segmentation Framework

This module provides a unified interface for training all models in the framework:
- Mask R-CNN (following maskRCNN.ipynb)
- SAM inference (following SAM_inference.ipynb)
- YOLOv8 (following train_yolo8.ipynb)
- YOLOv11 (following train_yolo11.ipynb)

The manager coordinates training across different model types while maintaining
compatibility with the reference notebooks.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import trainers
from .base_trainer import BaseTrainer
from .mask_rcnn_trainer import MaskRCNNTrainer, train_mask_rcnn_from_notebook
from .sam_trainer import SAMTrainer, run_sam_inference_from_notebook
from .yolo_trainer import (
    YOLOTrainer,
    train_yolo8_from_notebook,
    train_yolo11_from_notebook,
)

# Import data utilities
try:
    from data.dataset_utils import SurgicalCocoDataset, create_data_splits
except ImportError:
    print("Warning: Could not import dataset utilities")

# Import model factory
try:
    from models.model_factory import ModelFactory
except ImportError:
    print("Warning: Could not import model factory")


class TrainingManager:
    """
    Unified training manager for all segmentation models

    This class provides a single interface for training and managing all model types:
    - Mask R-CNN: Deep learning segmentation following maskRCNN.ipynb
    - SAM: Segment Anything Model inference following SAM_inference.ipynb
    - YOLO: YOLOv8/v11 segmentation following train_yolo8.ipynb and train_yolo11.ipynb

    Features:
    - Unified training interface
    - Experiment tracking and logging
    - Model comparison and evaluation
    - Automatic result organization
    - Notebook compatibility mode
    """

    def __init__(
        self,
        experiment_name: str = None,
        output_dir: str = "./training_experiments",
        device: str = "auto",
    ):
        """
        Initialize training manager

        Args:
            experiment_name (str): Name for this experiment (auto-generated if None)
            output_dir (str): Base directory for saving results
            device (str): Device to use ('auto', 'cuda', 'cpu', or device id)
        """
        # Set experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"surgical_segmentation_{timestamp}"

        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.experiment_dir = self.output_dir / experiment_name

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Initialize logging
        self._setup_logging()

        # Store trainers and results
        self.trainers = {}
        self.training_results = {}

        self.logger.info(
            f"Training Manager initialized for experiment: {experiment_name}"
        )
        self.logger.info(f"Output directory: {self.experiment_dir}")
        self.logger.info(f"Device: {self.device}")

    def _setup_logging(self) -> None:
        """Setup logging for the training manager"""
        log_file = self.experiment_dir / "training.log"

        # Create logger
        self.logger = logging.getLogger(f"TrainingManager_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def train_mask_rcnn(
        self,
        data_root: str,
        num_epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 0.0005,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train Mask R-CNN following exact notebook procedure

        Args:
            data_root (str): Path to dataset root
            num_epochs (int): Number of epochs (default: 10 as in notebook)
            batch_size (int): Batch size (default: 4 as in notebook)
            learning_rate (float): Learning rate (default: 0.0005 as in notebook)
            **kwargs: Additional arguments for training

        Returns:
            Dict containing training results
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Mask R-CNN Training")
        self.logger.info("Following procedure from maskRCNN.ipynb notebook")
        self.logger.info("=" * 60)

        # Create output directory for Mask R-CNN
        mask_rcnn_dir = self.experiment_dir / "mask_rcnn"
        mask_rcnn_dir.mkdir(exist_ok=True)

        try:
            # Train using notebook procedure
            trainer = train_mask_rcnn_from_notebook(
                data_root=data_root,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=self.device,
                save_dir=str(mask_rcnn_dir),
                **kwargs,
            )

            # Store trainer and results
            self.trainers["mask_rcnn"] = trainer
            self.training_results["mask_rcnn"] = {
                "status": "completed",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "output_dir": str(mask_rcnn_dir),
                "device": self.device,
            }

            self.logger.info("Mask R-CNN training completed successfully")
            return self.training_results["mask_rcnn"]

        except Exception as e:
            error_msg = f"Mask R-CNN training failed: {str(e)}"
            self.logger.error(error_msg)
            self.training_results["mask_rcnn"] = {"status": "failed", "error": str(e)}
            return self.training_results["mask_rcnn"]

    def train_yolo8(self, data_root: str, epochs: int = 80, **kwargs) -> Dict[str, Any]:
        """
        Train YOLOv8 following exact notebook procedure

        Args:
            data_root (str): Path to dataset root containing data.yaml
            epochs (int): Number of epochs (default: 80 as in notebook)
            **kwargs: Additional arguments for training

        Returns:
            Dict containing training results
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting YOLOv8 Training")
        self.logger.info("Following procedure from train_yolo8.ipynb notebook")
        self.logger.info("=" * 60)

        # Create output directory for YOLOv8
        yolo8_dir = self.experiment_dir / "yolo8"
        yolo8_dir.mkdir(exist_ok=True)

        try:
            # Train using notebook procedure
            trainer = train_yolo8_from_notebook(
                data_root=data_root,
                epochs=epochs,
                save_path=str(yolo8_dir / "yolo8_trained.pt"),
                **kwargs,
            )

            # Store trainer and results
            self.trainers["yolo8"] = trainer
            self.training_results["yolo8"] = {
                "status": "completed",
                "epochs": epochs,
                "model_name": "yolo8l-seg",
                "output_dir": str(yolo8_dir),
                "device": self.device,
            }

            self.logger.info("YOLOv8 training completed successfully")
            return self.training_results["yolo8"]

        except Exception as e:
            error_msg = f"YOLOv8 training failed: {str(e)}"
            self.logger.error(error_msg)
            self.training_results["yolo8"] = {"status": "failed", "error": str(e)}
            return self.training_results["yolo8"]

    def train_yolo11(
        self, data_root: str, epochs: int = 80, **kwargs
    ) -> Dict[str, Any]:
        """
        Train YOLOv11 following exact notebook procedure

        Args:
            data_root (str): Path to dataset root containing data.yaml
            epochs (int): Number of epochs (default: 80 as in notebook)
            **kwargs: Additional arguments for training

        Returns:
            Dict containing training results
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting YOLOv11 Training")
        self.logger.info("Following procedure from train_yolo11.ipynb notebook")
        self.logger.info("=" * 60)

        # Create output directory for YOLOv11
        yolo11_dir = self.experiment_dir / "yolo11"
        yolo11_dir.mkdir(exist_ok=True)

        try:
            # Train using notebook procedure
            trainer = train_yolo11_from_notebook(
                data_root=data_root,
                epochs=epochs,
                save_path=str(yolo11_dir / "yolo11_trained.pt"),
                **kwargs,
            )

            # Store trainer and results
            self.trainers["yolo11"] = trainer
            self.training_results["yolo11"] = {
                "status": "completed",
                "epochs": epochs,
                "model_name": "yolo11l-seg",
                "output_dir": str(yolo11_dir),
                "device": self.device,
            }

            self.logger.info("YOLOv11 training completed successfully")
            return self.training_results["yolo11"]

        except Exception as e:
            error_msg = f"YOLOv11 training failed: {str(e)}"
            self.logger.error(error_msg)
            self.training_results["yolo11"] = {"status": "failed", "error": str(e)}
            return self.training_results["yolo11"]

    def run_sam_inference(
        self,
        checkpoint_path: str,
        dataset_path: str,
        annotations_file: str,
        model_type: str = "vit_h",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run SAM inference following exact notebook procedure

        Args:
            checkpoint_path (str): Path to SAM checkpoint
            dataset_path (str): Path to dataset images
            annotations_file (str): Path to COCO annotations
            model_type (str): SAM model type ('vit_h', 'vit_l', 'vit_b')
            **kwargs: Additional arguments for inference

        Returns:
            Dict containing inference results
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting SAM Inference")
        self.logger.info("Following procedure from SAM_inference.ipynb notebook")
        self.logger.info("=" * 60)

        # Create output directory for SAM
        sam_dir = self.experiment_dir / "sam"
        sam_dir.mkdir(exist_ok=True)

        try:
            # Run inference using notebook procedure
            trainer = run_sam_inference_from_notebook(
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                annotations_file=annotations_file,
                output_dir=str(sam_dir),
                model_type=model_type,
                **kwargs,
            )

            # Store trainer and results
            self.trainers["sam"] = trainer
            evaluation_results = trainer.get_evaluation_results()

            self.training_results["sam"] = {
                "status": "completed",
                "model_type": model_type,
                "checkpoint_path": checkpoint_path,
                "output_dir": str(sam_dir),
                "evaluation_results": evaluation_results,
                "device": self.device,
            }

            self.logger.info("SAM inference completed successfully")
            return self.training_results["sam"]

        except Exception as e:
            error_msg = f"SAM inference failed: {str(e)}"
            self.logger.error(error_msg)
            self.training_results["sam"] = {"status": "failed", "error": str(e)}
            return self.training_results["sam"]

    def train_all_models(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train all models according to configuration

        Args:
            config (Dict): Configuration dictionary with settings for each model
                Expected structure:
                {
                    'mask_rcnn': {
                        'data_root': str,
                        'num_epochs': int,
                        'batch_size': int,
                        'learning_rate': float
                    },
                    'yolo8': {
                        'data_root': str,
                        'epochs': int
                    },
                    'yolo11': {
                        'data_root': str,
                        'epochs': int
                    },
                    'sam': {
                        'checkpoint_path': str,
                        'dataset_path': str,
                        'annotations_file': str,
                        'model_type': str
                    }
                }

        Returns:
            Dict containing all training results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Complete Model Training Pipeline")
        self.logger.info("Training all models: Mask R-CNN, YOLOv8, YOLOv11, SAM")
        self.logger.info("=" * 80)

        total_start_time = time.time()

        # Train Mask R-CNN
        if "mask_rcnn" in config:
            self.logger.info("Step 1/4: Training Mask R-CNN...")
            mask_rcnn_config = config["mask_rcnn"]
            self.train_mask_rcnn(**mask_rcnn_config)

        # Train YOLOv8
        if "yolo8" in config:
            self.logger.info("Step 2/4: Training YOLOv8...")
            yolo8_config = config["yolo8"]
            self.train_yolo8(**yolo8_config)

        # Train YOLOv11
        if "yolo11" in config:
            self.logger.info("Step 3/4: Training YOLOv11...")
            yolo11_config = config["yolo11"]
            self.train_yolo11(**yolo11_config)

        # Run SAM inference
        if "sam" in config:
            self.logger.info("Step 4/4: Running SAM inference...")
            sam_config = config["sam"]
            self.run_sam_inference(**sam_config)

        total_time = time.time() - total_start_time

        # Save complete results
        self.save_experiment_results()

        self.logger.info("=" * 80)
        self.logger.info(
            f"Complete training pipeline finished in {total_time:.2f}s ({total_time/60:.1f} minutes)"
        )
        self.logger.info(f"Results saved to: {self.experiment_dir}")
        self.logger.info("=" * 80)

        return self.training_results

    def save_experiment_results(self) -> None:
        """Save complete experiment results to JSON file"""
        results_file = self.experiment_dir / "experiment_results.json"

        # Prepare serializable results
        serializable_results = {}
        for model_name, results in self.training_results.items():
            serializable_results[model_name] = results.copy()

            # Remove non-serializable objects
            if "evaluation_results" in serializable_results[model_name]:
                eval_results = serializable_results[model_name]["evaluation_results"]
                if "predictions" in eval_results:
                    # Keep only summary of predictions
                    predictions = eval_results["predictions"]
                    eval_results["predictions_summary"] = {
                        "count": len(predictions),
                        "sample": predictions[:3] if len(predictions) > 0 else [],
                    }
                    del eval_results["predictions"]

        # Add experiment metadata
        experiment_summary = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "output_dir": str(self.experiment_dir),
            "models_trained": list(self.training_results.keys()),
            "status": {
                model: results.get("status", "unknown")
                for model, results in self.training_results.items()
            },
        }

        final_results = {
            "experiment_summary": experiment_summary,
            "model_results": serializable_results,
        }

        # Save to JSON
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        self.logger.info(f"Experiment results saved to: {results_file}")

    def get_trainer(self, model_name: str) -> Optional[BaseTrainer]:
        """
        Get trainer for specific model

        Args:
            model_name (str): Name of model ('mask_rcnn', 'yolo8', 'yolo11', 'sam')

        Returns:
            BaseTrainer: The requested trainer or None if not found
        """
        return self.trainers.get(model_name)

    def get_results(
        self, model_name: Optional[str] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        Get training results

        Args:
            model_name (str): Specific model name or None for all results

        Returns:
            Training results for specified model or all results
        """
        if model_name is None:
            return self.training_results
        return self.training_results.get(model_name)

    def compare_models(self) -> Dict[str, Any]:
        """
        Compare performance across all trained models

        Returns:
            Dict containing model comparison metrics
        """
        comparison = {
            "models_completed": [],
            "models_failed": [],
            "metrics_comparison": {},
        }

        for model_name, results in self.training_results.items():
            if results.get("status") == "completed":
                comparison["models_completed"].append(model_name)

                # Extract relevant metrics for comparison
                if model_name == "sam" and "evaluation_results" in results:
                    eval_results = results["evaluation_results"]
                    if "metrics" in eval_results:
                        metrics = eval_results["metrics"]
                        comparison["metrics_comparison"][model_name] = {
                            "AP": metrics.get("AP", 0),
                            "AP_50": metrics.get("AP_50", 0),
                            "AP_75": metrics.get("AP_75", 0),
                        }
            else:
                comparison["models_failed"].append(model_name)

        return comparison


def create_training_config_from_notebooks(
    mask_rcnn_data_root: str,
    yolo_data_root: str,
    sam_checkpoint: str,
    sam_dataset_path: str,
    sam_annotations: str,
) -> Dict[str, Any]:
    """
    Create training configuration that exactly matches the notebook procedures

    Args:
        mask_rcnn_data_root (str): Data root for Mask R-CNN (COCO format)
        yolo_data_root (str): Data root for YOLO (contains data.yaml)
        sam_checkpoint (str): Path to SAM checkpoint
        sam_dataset_path (str): Dataset path for SAM evaluation
        sam_annotations (str): Annotations file for SAM evaluation

    Returns:
        Dict: Complete training configuration
    """
    return {
        "mask_rcnn": {
            "data_root": mask_rcnn_data_root,
            "num_epochs": 10,  # Same as notebook
            "batch_size": 4,  # Same as notebook
            "learning_rate": 0.0005,  # Same as notebook
        },
        "yolo8": {"data_root": yolo_data_root, "epochs": 80},  # Same as notebook
        "yolo11": {"data_root": yolo_data_root, "epochs": 80},  # Same as notebook
        "sam": {
            "checkpoint_path": sam_checkpoint,
            "dataset_path": sam_dataset_path,
            "annotations_file": sam_annotations,
            "model_type": "vit_h",  # Same as typical notebook usage
        },
    }


# Example usage exactly matching the notebook workflows
if __name__ == "__main__":
    # Initialize training manager
    manager = TrainingManager(
        experiment_name="surgical_segmentation_complete",
        output_dir="./training_experiments",
    )

    # Create configuration exactly matching notebooks
    config = create_training_config_from_notebooks(
        mask_rcnn_data_root="/path/to/coco/dataset",
        yolo_data_root="/path/to/yolo/dataset",
        sam_checkpoint="/path/to/sam_vit_h_4b8939.pth",
        sam_dataset_path="/path/to/evaluation/images",
        sam_annotations="/path/to/annotations.json",
    )

    # Train all models following notebook procedures
    results = manager.train_all_models(config)

    # Compare results
    comparison = manager.compare_models()
    print("Model Comparison:")
    print(f"Completed: {comparison['models_completed']}")
    print(f"Failed: {comparison['models_failed']}")
