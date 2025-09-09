"""
Base Trainer Abstract Class for Surgical Instance Segmentation

This module provides the abstract base class for all model trainers in the framework,
ensuring consistent training interfaces and behavior across different model architectures.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers.

    This class defines the standard interface that all trainer implementations
    must follow, ensuring consistency across different model architectures
    (YOLO, Mask R-CNN, SAM, etc.).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, data_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            data_config: Dataset configuration
            **kwargs: Additional training arguments

        Returns:
            Training results and metrics
        """
        pass

    @abstractmethod
    def validate(self, data_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Validate the model.

        Args:
            data_config: Dataset configuration
            **kwargs: Additional validation arguments

        Returns:
            Validation results and metrics
        """
        pass

    @abstractmethod
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save the trained model.

        Args:
            save_path: Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a pre-trained model.

        Args:
            model_path: Path to the model file
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Model information dictionary
        """
        return {
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "config": self.config,
        }
