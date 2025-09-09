"""
Training and evaluation engine for surgical skill assessment models.

This module provides the core functionality for training deep learning models,
evaluating their performance, and running inference on new data.
"""

from .evaluator import evaluate_model
from .predictor import run_inference
from .trainer import train_one_epoch, validate_one_epoch

__all__ = ["train_one_epoch", "validate_one_epoch", "evaluate_model", "run_inference"]
