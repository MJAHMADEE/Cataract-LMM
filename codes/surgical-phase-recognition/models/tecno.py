#!/usr/bin/env python3
"""
TeCNO (Temporal Consistency Network for Operating) Model Implementation.

This module implements the MultiStageModel class that matches the exact
usage pattern from the reference notebook phase_validation_comprehensive.ipynb.

The MultiStageModel is designed for surgical workflow analysis with temporal
consistency constraints.

Author: Surgical Phase Recognition Team
Date: August 29, 2025
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import torchvision.models as models

    TORCHVISION_AVAILABLE = True
except ImportError:
    logger.warning("torchvision not available. CNN backbones may not work.")
    TORCHVISION_AVAILABLE = False


class MultiStageModel(nn.Module):
    """
    Multi-Stage Model for Surgical Phase Recognition (TeCNO Architecture).

    This model implements the TeCNO (Temporal Consistency Network for Operating)
    architecture for surgical workflow analysis. It uses a multi-stage approach
    with temporal consistency constraints.

    Args:
        backbone: Pre-trained CNN backbone model
        num_layers (int): Number of TCN layers
        num_stages (int): Number of refinement stages
        num_f_maps (int): Number of feature maps
        dim (int): Input feature dimension
        num_classes (int): Number of surgical phase classes
    """

    def __init__(
        self,
        backbone,
        num_layers: int = 6,
        num_stages: int = 1,
        num_f_maps: int = 256,
        dim: int = 2048,
        num_classes: int = 11,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_f_maps = num_f_maps
        self.dim = dim

        # Remove the final classification layer from backbone
        if hasattr(backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            if hasattr(backbone.classifier, "in_features"):
                # EfficientNet case
                self.backbone.classifier = nn.Identity()
            else:
                # Other cases
                self.backbone.classifier[-1] = nn.Identity()
        elif hasattr(backbone, "head"):
            self.backbone.head = nn.Identity()

        # Feature projection layer
        self.feature_projection = nn.Linear(dim, num_f_maps)

        # Multi-stage TCN
        self.stages = nn.ModuleList()
        for s in range(num_stages):
            self.stages.append(
                SingleStageModel(num_layers, num_f_maps, num_f_maps, num_classes)
            )

        logger.info(
            f"Initialized MultiStageModel with {num_stages} stages, "
            f"{num_layers} layers, {num_f_maps} feature maps"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-stage model.

        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, C, T, H, W)

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes, T)
        """
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # (B, C, T, H, W)
                B, C, T, H, W = x.shape
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            else:  # (B, T, C, H, W)
                B, T, C, H, W = x.shape
        else:
            raise ValueError(f"Expected 5D input, got {x.dim()}D")

        # Extract features using backbone
        # Reshape to process all frames through backbone
        x = x.reshape(B * T, C, H, W)

        # Extract features
        features = self.backbone(x)  # (B*T, dim)

        # Handle different backbone output shapes
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.squeeze(-1).squeeze(-1)

        # Project features
        features = self.feature_projection(features)  # (B*T, num_f_maps)

        # Reshape back to temporal sequence
        features = features.reshape(B, T, self.num_f_maps)
        features = features.transpose(1, 2)  # (B, num_f_maps, T)

        # Multi-stage processing
        outputs = []
        for stage in self.stages:
            features = stage(features)
            outputs.append(features)

        # Return the final stage output
        return outputs[-1]  # (B, num_classes, T)


class SingleStageModel(nn.Module):
    """
    Single stage of the multi-stage model with dilated convolutions.
    """

    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int):
        super().__init__()

        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(2**i, num_f_maps, num_f_maps)
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    """
    Dilated residual layer for temporal modeling.
    """

    def __init__(self, dilation: int, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# For backward compatibility, also expose TeCNOModel
TeCNOModel = MultiStageModel
