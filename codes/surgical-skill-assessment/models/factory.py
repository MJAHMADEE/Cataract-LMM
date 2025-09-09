"""
Model factory for creating video classification models.

This module provides a centralized factory function for instantiating
various deep learning architectures for surgical skill assessment.

Author: Surgical Skill Assessment Team
Date: August 2025
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model architectures
from .cnn_rnn import CNN_GRU, CNN_LSTM
from .transformers import MViT, VideoMAE, ViViT

# Try to import optional dependencies
try:
    from timesformer.models.vit import TimeSformer

    TIMESFORMER_AVAILABLE = True
except ImportError:
    TIMESFORMER_AVAILABLE = False
    logger.warning(
        "TimeSformer not available. Install timesformer-pytorch to use TimeSformer models."
    )


def create_model(
    model_name: str,
    num_classes: int,
    clip_len: int,
    freeze_backbone: bool = False,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Create model based on model_name - FULLY ALIGNED WITH NOTEBOOK.

    Available models (exact keywords):

    CNN-based models:
    - "x3d_m": X3D-Medium (efficient 3D CNN)
    - "slow_r50": Slow pathway ResNet-50
    - "slowfast_r50": SlowFast ResNet-50 (dual pathway)
    - "r2plus1d": R(2+1)D-18 (decomposed 3D convolutions)
    - "r3d_18": ResNet 3D-18 (full 3D convolutions)

    Hybrid CNN-RNN models:
    - "cnn_lstm": CNN backbone + Bidirectional LSTM
    - "cnn_gru": CNN backbone + Bidirectional GRU

    Transformer-based models:
    - "timesformer": TimeSformer (divided space-time attention)
    - "mvit": Multiscale Vision Transformer
    - "videomae": Video Masked Autoencoder V2
    - "vivit": Video Vision Transformer

    Args:
        model_name: Name of the model to create
        num_classes: Number of output classes
        clip_len: Number of frames in input clips
        freeze_backbone: Whether to freeze pretrained weights
        dropout: Dropout rate for classifier

    Returns:
        PyTorch model
    """
    logger.info(f"Creating model '{model_name}' with {num_classes} classes")

    try:
        # CNN-based models
        if model_name == "x3d_m":
            try:
                model = torch.hub.load(
                    "facebookresearch/pytorchvideo", "x3d_m", pretrained=True
                )
                in_features = model.blocks[-1].proj.in_features
                model.blocks[-1].proj = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(in_features, num_classes)
                )
            except Exception as e:
                logger.error(f"Failed to load X3D model: {e}")
                raise RuntimeError(
                    "pytorchvideo required for X3D models. Install with: pip install pytorchvideo"
                )

        elif model_name == "r2plus1d":
            try:
                model = torch.hub.load("pytorch/vision", "r2plus1d_18", pretrained=True)
                in_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(in_features, num_classes)
                )
            except Exception as e:
                logger.error(f"Failed to load R(2+1)D model: {e}")
                raise

        elif model_name == "r3d_18":
            try:
                model = torch.hub.load("pytorch/vision", "r3d_18", pretrained=True)
                in_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(in_features, num_classes)
                )
            except Exception as e:
                logger.error(f"Failed to load R3D model: {e}")
                raise

        elif model_name == "slow_r50":
            try:
                model = torch.hub.load(
                    "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
                )
                in_features = model.blocks[-1].proj.in_features
                model.blocks[-1].proj = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(in_features, num_classes)
                )
            except Exception as e:
                logger.error(f"Failed to load Slow R50 model: {e}")
                raise RuntimeError(
                    "pytorchvideo required for Slow models. Install with: pip install pytorchvideo"
                )

        elif model_name == "slowfast_r50":
            try:
                model = torch.hub.load(
                    "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True
                )
                in_features = model.blocks[-1].proj.in_features
                model.blocks[-1].proj = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(in_features, num_classes)
                )
            except Exception as e:
                logger.error(f"Failed to load SlowFast model: {e}")
                raise RuntimeError(
                    "pytorchvideo required for SlowFast models. Install with: pip install pytorchvideo"
                )

        # CNN-RNN models (bidirectional as per notebook)
        elif model_name == "cnn_lstm":
            from .cnn_rnn import CNN_LSTM

            model = CNN_LSTM(
                num_classes=num_classes, hidden_dim=512, num_layers=2, dropout=dropout
            )

        elif model_name == "cnn_gru":
            from .cnn_rnn import CNN_GRU

            model = CNN_GRU(
                num_classes=num_classes, hidden_dim=512, num_layers=2, dropout=dropout
            )

        # Transformer models
        elif model_name == "timesformer":
            try:
                from timesformer_pytorch import TimeSformer

                class TimeSformerWrapper(nn.Module):
                    """Wrapper to permute input tensor dimensions for TimeSformer."""

                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, x):
                        # Permute from (B, C, T, H, W) to (B, T, C, H, W)
                        return self.model(x.permute(0, 2, 1, 3, 4))

                timesformer_model = TimeSformer(
                    dim=512,
                    image_size=224,
                    patch_size=16,
                    num_frames=clip_len,
                    num_classes=num_classes,
                    depth=12,
                    heads=8,
                    dim_head=64,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                )
                model = TimeSformerWrapper(timesformer_model)
            except ImportError:
                logger.warning(
                    "TimeSformer not available. Install timesformer-pytorch to use TimeSformer models."
                )
                raise RuntimeError(
                    "timesformer-pytorch required. Install with: pip install timesformer-pytorch"
                )

        elif model_name == "mvit":
            from .transformers import MViT

            model = MViT(
                num_classes=num_classes,
                img_size=224,
                patch_size=16,
                num_frames=clip_len,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                dropout=dropout,
            )

        elif model_name == "videomae":
            from .transformers import VideoMAE

            model = VideoMAE(
                num_classes=num_classes,
                img_size=224,
                patch_size=16,
                num_frames=clip_len,
                embed_dim=768,
                depth=12,
                num_heads=12,
                decoder_embed_dim=512,
                decoder_depth=4,
                decoder_num_heads=8,
                mlp_ratio=4.0,
                dropout=dropout,
            )

        elif model_name == "vivit":
            from .transformers import ViViT

            model = ViViT(
                num_classes=num_classes,
                img_size=224,
                patch_size=16,
                num_frames=clip_len,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                dropout=dropout,
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if requested (except for transformer models and CNN-RNN)
        if freeze_backbone and model_name not in [
            "timesformer",
            "mvit",
            "videomae",
            "vivit",
            "cnn_lstm",
            "cnn_gru",
        ]:
            for name, param in model.named_parameters():
                if "proj" not in name and "fc" not in name and "classifier" not in name:
                    param.requires_grad = False
            logger.info("Backbone frozen - only training classifier head")

        logger.info(
            f"Model '{model_name}' created successfully with {num_classes} classes"
        )
        return model

    except Exception as e:
        logger.error(f"Failed to create model '{model_name}': {e}")
        raise
