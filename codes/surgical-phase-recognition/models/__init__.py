#!/usr/bin/env python3
"""
Model Factory and Registry for Surgical Phase Recognition

This module provides a unified interface for creating and managing all model
architectures used in surgical phase recognition. It includes model factory
functions, model registry, and utilities for model selection and configuration.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .cnn_3d_models import CNN_3D_MODELS, create_3d_cnn_model
from .cnn_rnn_hybrids import CNN_RNN_MODELS, create_cnn_rnn_model
from .multistage_models import HierarchicalPhaseModel, TeCNOModel
from .tecno import MultiStageModel
from .tecno import TeCNOModel as TeCNOModelAlias

# Import all model modules
from .video_transformers import VIDEO_TRANSFORMER_MODELS, create_video_transformer

# Define multistage models dictionary
MULTISTAGE_MODELS = {
    "tecno": TeCNOModel,
    "multistage": MultiStageModel,
    "hierarchical": HierarchicalPhaseModel,
}


def create_multistage_model(model_name: str, **kwargs):
    """Create a multistage model."""
    if model_name in MULTISTAGE_MODELS:
        return MULTISTAGE_MODELS[model_name](**kwargs)
    else:
        raise ValueError(f"Unknown multistage model: {model_name}")


# Import model components for framework compatibility
from .cnn_3d_models import CNN_3D_MODELS, create_3d_cnn_model
from .cnn_rnn_hybrids import CNN_RNN_MODELS, create_cnn_rnn_model
from .multistage_models import MULTISTAGE_MODELS, create_multistage_model
from .tecno import MultiStageModel
from .video_transformers import VIDEO_TRANSFORMER_MODELS, create_video_transformer


# Notebook compatibility imports
def get_notebook_compatible_imports():
    """Return imports compatible with notebook usage."""
    return {
        "cnn_3d_models": CNN_3D_MODELS,
        "cnn_rnn_models": CNN_RNN_MODELS,
        "multistage_models": MULTISTAGE_MODELS,
        "video_transformers": VIDEO_TRANSFORMER_MODELS,
        "tecno_model": MultiStageModel,
    }


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for all available models in the surgical phase recognition system.

    This class maintains a registry of all available model architectures,
    their configurations, and provides utilities for model creation and management.
    """

    def __init__(self):
        self.models = {}
        self._register_all_models()

    def _register_all_models(self):
        """Register all available models from different categories."""
        # Video Transformers
        for name, model_class in VIDEO_TRANSFORMER_MODELS.items():
            self.models[f"transformer_{name}"] = {
                "class": model_class,
                "category": "transformer",
                "factory": create_video_transformer,
                "description": f"Video transformer model: {name}",
            }

        # 3D CNNs
        for name, model_class in CNN_3D_MODELS.items():
            self.models[f"3dcnn_{name}"] = {
                "class": model_class,
                "category": "3d_cnn",
                "factory": create_3d_cnn_model,
                "description": f"3D CNN model: {name}",
            }

        # CNN-RNN Hybrids
        for name, model_class in CNN_RNN_MODELS.items():
            self.models[f"hybrid_{name}"] = {
                "class": model_class,
                "category": "cnn_rnn",
                "factory": create_cnn_rnn_model,
                "description": f"CNN-RNN hybrid model: {name}",
            }

        # Multi-stage Models
        for name, model_class in MULTISTAGE_MODELS.items():
            self.models[f"multistage_{name}"] = {
                "class": model_class,
                "category": "multistage",
                "factory": create_multistage_model,
                "description": f"Multi-stage model: {name}",
            }

        # Video Transformer Models
        for name, model_class in VIDEO_TRANSFORMER_MODELS.items():
            self.models[f"video_transformer_{name}"] = {
                "class": model_class,
                "category": "video_transformer",
                "factory": create_video_transformer,
                "description": f"Video transformer model: {name}",
            }

    def list_models(self, category: Optional[str] = None) -> List[str]:
        """
        List all available models, optionally filtered by category.

        Args:
            category (str, optional): Filter by model category

        Returns:
            List[str]: List of model names
        """
        if category is None:
            return list(self.models.keys())

        return [
            name for name, info in self.models.items() if info["category"] == category
        ]

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_name (str): Name of the model

        Returns:
            Dict[str, Any]: Model information

        Raises:
            ValueError: If model is not found
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry")

        return self.models[model_name]

    def create_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        Create a model instance using the registry.

        Args:
            model_name (str): Name of the model
            **kwargs: Model-specific arguments

        Returns:
            nn.Module: Created model instance

        Raises:
            ValueError: If model is not found
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry")

        model_info = self.models[model_name]
        factory_func = model_info["factory"]

        # Extract the original model name (remove prefix)
        original_name = model_name.split("_", 1)[1]

        return factory_func(original_name, **kwargs)


# Global model registry instance
model_registry = ModelRegistry()


def create_model(model_name: str, num_classes: int = 11, **kwargs) -> nn.Module:
    """
    Unified factory function to create any model in the system.

    This is the main entry point for model creation. It automatically
    determines the model category and uses the appropriate factory function.

    Args:
        model_name (str): Name of the model to create
        num_classes (int): Number of surgical phases to classify
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: The created model

    Raises:
        ValueError: If model_name is not supported

    Examples:
        >>> model = create_model('swin3d', num_classes=11)
        >>> model = create_model('resnet50_lstm', num_classes=11, hidden_size=256)
        >>> model = create_model('tecno', num_classes=11, backbone='resnet50')
    """
    # Normalize model name
    model_name = model_name.lower()

    # Check if it's a prefixed name from registry
    if model_name in model_registry.models:
        return model_registry.create_model(
            model_name, num_classes=num_classes, **kwargs
        )

    # Try to match with known model patterns
    # Video Transformers
    if model_name in VIDEO_TRANSFORMER_MODELS:
        return create_video_transformer(model_name, num_classes=num_classes, **kwargs)

    # 3D CNNs
    if model_name in CNN_3D_MODELS:
        return create_3d_cnn_model(model_name, num_classes=num_classes, **kwargs)

    # CNN-RNN Hybrids
    if model_name in CNN_RNN_MODELS:
        return create_cnn_rnn_model(model_name, num_classes=num_classes, **kwargs)

    # Multi-stage Models
    if model_name in MULTISTAGE_MODELS:
        return create_multistage_model(model_name, num_classes=num_classes, **kwargs)

    # Try common aliases and variations
    model_aliases = {
        "swin3d_t": "swin3d",
        "swin3d_small": "swin3d",
        "mvit_base": "mvit",
        "mvit_v2": "mvit",
        "r3d18": "r3d_18",
        "mc318": "mc3_18",
        "r2plus1d18": "r2plus1d_18",
        "slow_r50": "slow_r50",
        "x3d_xs": "x3d_xs",
        "resnet_lstm": "resnet50_lstm",
        "resnet_gru": "resnet50_gru",
        "efficientnet_lstm": "efficientnetb5_lstm",
        "efficientnet_gru": "efficientnetb5_gru",
    }

    if model_name in model_aliases:
        return create_model(
            model_aliases[model_name], num_classes=num_classes, **kwargs
        )

    raise ValueError(f"Unsupported model: {model_name}")


def get_model_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed information about model parameters.

    Args:
        model (nn.Module): The model to analyze

    Returns:
        Dict[str, Any]: Parameter information including counts and memory usage
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory usage (in MB)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (
        1024 * 1024
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameter_memory_mb": param_memory,
        "model_size_mb": param_memory,  # Approximate model size
    }


def get_model_summary(model_name: str, **kwargs) -> Dict[str, Any]:
    """
    Get a comprehensive summary of a model without creating it.

    Args:
        model_name (str): Name of the model
        **kwargs: Model arguments for parameter estimation

    Returns:
        Dict[str, Any]: Model summary information
    """
    try:
        # Create model for analysis
        model = create_model(model_name, **kwargs)

        # Get parameter information
        param_info = get_model_parameters(model)

        # Get model info from registry if available
        try:
            registry_info = model_registry.get_model_info(model_name)
            category = registry_info["category"]
            description = registry_info["description"]
        except ValueError:
            category = "unknown"
            description = f"Model: {model_name}"

        return {
            "model_name": model_name,
            "category": category,
            "description": description,
            **param_info,
            "input_shape": "Expected: (B, C, T, H, W) or (B, T, C, H, W)",
            "output_shape": f'(B, {kwargs.get("num_classes", 11)})',
        }

    except Exception as e:
        return {"model_name": model_name, "error": str(e), "status": "failed"}


def list_all_models() -> Dict[str, List[str]]:
    """
    List all available models organized by category.

    Returns:
        Dict[str, List[str]]: Dictionary mapping categories to model lists
    """
    categories = ["transformer", "3d_cnn", "cnn_rnn", "multistage"]
    result = {}

    for category in categories:
        result[category] = model_registry.list_models(category)

    # Also include direct model names (without prefixes)
    result["direct_access"] = {
        "transformers": list(VIDEO_TRANSFORMER_MODELS.keys()),
        "3d_cnns": list(CNN_3D_MODELS.keys()),
        "cnn_rnn_hybrids": list(CNN_RNN_MODELS.keys()),
        "multistage": list(MULTISTAGE_MODELS.keys()),
    }

    return result


# Recommended model configurations for different scenarios
RECOMMENDED_CONFIGS = {
    "fast_inference": {
        "model": "x3d_xs",
        "description": "Lightweight 3D CNN for fast inference",
        "config": {"num_classes": 11},
    },
    "high_accuracy": {
        "model": "swin3d",
        "description": "State-of-the-art transformer for highest accuracy",
        "config": {"num_classes": 11, "embed_dim": 96},
    },
    "balanced": {
        "model": "r3d_18",
        "description": "Good balance of speed and accuracy",
        "config": {"num_classes": 11, "pretrained": True},
    },
    "temporal_modeling": {
        "model": "resnet50_lstm",
        "description": "Strong temporal modeling with CNN-RNN hybrid",
        "config": {"num_classes": 11, "hidden_size": 256, "bidirectional": True},
    },
    "multi_stage": {
        "model": "tecno",
        "description": "Advanced multi-stage model with temporal consistency",
        "config": {"num_classes": 11, "backbone": "resnet50", "use_attention": True},
    },
}


def get_recommended_model(scenario: str) -> Dict[str, Any]:
    """
    Get a recommended model configuration for a specific scenario.

    Args:
        scenario (str): One of 'fast_inference', 'high_accuracy', 'balanced',
                       'temporal_modeling', 'multi_stage'

    Returns:
        Dict[str, Any]: Recommended model configuration

    Raises:
        ValueError: If scenario is not recognized
    """
    if scenario not in RECOMMENDED_CONFIGS:
        available = list(RECOMMENDED_CONFIGS.keys())
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {available}")

    return RECOMMENDED_CONFIGS[scenario].copy()


if __name__ == "__main__":
    # Test model factory
    print("Testing Model Factory...")

    # List all models
    all_models = list_all_models()
    print("\nAvailable models by category:")
    for category, models in all_models.items():
        if isinstance(models, list):
            print(f"  {category}: {len(models)} models")
        else:
            print(f"  {category}:")
            for subcat, submodels in models.items():
                print(f"    {subcat}: {submodels}")

    # Test model creation
    test_models = ["swin3d", "r3d_18", "resnet50_lstm", "tecno"]

    for model_name in test_models:
        try:
            print(f"\nTesting {model_name}:")
            summary = get_model_summary(model_name, num_classes=11)

            for key, value in summary.items():
                if key != "error":
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"  Error: {e}")

    # Test recommendations
    print("\nRecommended configurations:")
    for scenario in RECOMMENDED_CONFIGS:
        config = get_recommended_model(scenario)
        print(f"  {scenario}: {config['model']} - {config['description']}")


# Export individual models for direct import (notebook compatibility)
notebook_models = get_notebook_compatible_imports()
globals().update(notebook_models)

# Export all important functions and classes
__all__ = [
    # Core functions
    "create_model",
    "list_all_models",
    "get_model_summary",
    "get_recommended_model",
    "model_registry",
    "ModelRegistry",
    # Individual models (notebook compatible)
    "Resnet50_LSTM",
    "Resnet50_GRU",
    "EfficientNetB5_GRU",
    "EfficientNetB5_LSTM",
    "MultiStageModel",
    "SingleStageModel",
    "DilatedResidualLayer",
    # Factory functions
    "create_video_transformer",
    "create_3d_cnn_model",
    "create_cnn_rnn_model",
    "create_multistage_model",
]
