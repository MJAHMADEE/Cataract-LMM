#!/usr/bin/env python3
"""
Individual Model Implementations for Surgical Phase Recognition

This module contains individual model implementations that match the exact
structure and naming conventions used in the reference validation notebook.

These models are designed to be directly compatible with the checkpoint
loading and validation procedures in the notebook.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

# Advanced TeCNO models (extended implementations)
from .advanced_tecno_models import (
    DilatedResidualLayer,
)
from .advanced_tecno_models import MultiStageModel as AdvancedMultiStageModel
from .advanced_tecno_models import (
    SingleStageModel,
    TeCNOFeatureExtractor,
    create_multi_stage_tecno,
    create_single_stage_tecno,
)

# EfficientNet-B5-based models
from .efficientnet_b5_gru import EfficientNetB5_GRU, create_efficientnetb5_gru
from .efficientnet_b5_lstm import EfficientNetB5_LSTM, create_efficientnetb5_lstm
from .resnet50_gru import Resnet50_GRU, create_resnet50_gru

# ResNet50-based models
from .resnet50_lstm import Resnet50_LSTM, create_resnet50_lstm

__all__ = [
    # ResNet50 models
    "Resnet50_LSTM",
    "Resnet50_GRU",
    "create_resnet50_lstm",
    "create_resnet50_gru",
    # EfficientNet-B5 models
    "EfficientNetB5_GRU",
    "EfficientNetB5_LSTM",
    "create_efficientnetb5_gru",
    "create_efficientnetb5_lstm",
    # Advanced TeCNO models
    "SingleStageModel",
    "AdvancedMultiStageModel",
    "DilatedResidualLayer",
    "TeCNOFeatureExtractor",
    "create_single_stage_tecno",
    "create_multi_stage_tecno",
]

# Model registry for easy access
INDIVIDUAL_MODELS = {
    # ResNet50-based
    "resnet50_lstm": Resnet50_LSTM,
    "resnet50_gru": Resnet50_GRU,
    # EfficientNet-B5-based
    "efficientnetb5_lstm": EfficientNetB5_LSTM,
    "efficientnetb5_gru": EfficientNetB5_GRU,
    # Advanced TeCNO models
    "tecno_single": SingleStageModel,
    "tecno_multi": AdvancedMultiStageModel,
}

# Factory functions registry
MODEL_FACTORIES = {
    "resnet50_lstm": create_resnet50_lstm,
    "resnet50_gru": create_resnet50_gru,
    "efficientnetb5_lstm": create_efficientnetb5_lstm,
    "efficientnetb5_gru": create_efficientnetb5_gru,
    "tecno_single": create_single_stage_tecno,
    "tecno_multi": create_multi_stage_tecno,
}


def get_model_class(model_name: str):
    """
    Get model class by name.

    Args:
        model_name (str): Name of the model

    Returns:
        type: Model class

    Raises:
        KeyError: If model name is not found
    """
    if model_name not in INDIVIDUAL_MODELS:
        available_models = list(INDIVIDUAL_MODELS.keys())
        raise KeyError(
            f"Model '{model_name}' not found. Available models: {available_models}"
        )

    return INDIVIDUAL_MODELS[model_name]


def create_model(model_name: str, **kwargs):
    """
    Create model instance by name.

    Args:
        model_name (str): Name of the model
        **kwargs: Model-specific arguments

    Returns:
        nn.Module: Model instance

    Raises:
        KeyError: If model name is not found
    """
    if model_name not in MODEL_FACTORIES:
        available_models = list(MODEL_FACTORIES.keys())
        raise KeyError(
            f"Model factory for '{model_name}' not found. Available models: {available_models}"
        )

    factory_func = MODEL_FACTORIES[model_name]
    return factory_func(**kwargs)


def list_available_models():
    """
    List all available individual models.

    Returns:
        List[str]: List of available model names
    """
    return list(INDIVIDUAL_MODELS.keys())


# Compatibility aliases for exact notebook imports
# These allow the notebook imports to work without modification
def get_notebook_compatible_imports():
    """
    Get imports in the format expected by the validation notebook.

    Returns:
        dict: Dictionary of model classes with notebook-compatible names
    """
    return {
        "Resnet50_LSTM": Resnet50_LSTM,
        "Resnet50_GRU": Resnet50_GRU,
        "EfficientNetB5_GRU": EfficientNetB5_GRU,
        "EfficientNetB5_LSTM": EfficientNetB5_LSTM,
        "AdvancedMultiStageModel": AdvancedMultiStageModel,
        "SingleStageModel": SingleStageModel,
        "DilatedResidualLayer": DilatedResidualLayer,
    }
