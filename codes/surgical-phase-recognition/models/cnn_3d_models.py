#!/usr/bin/env python3
"""
3D CNN Models for Surgical Phase Recognition

This module implements various 3D convolutional neural network architectures
for surgical phase recognition. These models are designed to capture
spatial-temporal patterns in surgical videos.

Models included:
- R3D-18: 3D ResNet
- MC3-18: Mixed 3D CNN
- R2Plus1D: (2+1)D factorized convolutions
- Slow R50: Slow pathway ResNet
- X3D: Efficient 3D networks

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18

    TORCHVISION_VIDEO_AVAILABLE = True
except ImportError:
    logger.warning("torchvision video models not available")
    TORCHVISION_VIDEO_AVAILABLE = False

try:
    from pytorchvideo.models.hub import slow_r50, x3d_xs

    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    logger.warning("pytorchvideo not available. Some models may not work.")
    PYTORCHVIDEO_AVAILABLE = False


class R3D18(nn.Module):
    """
    3D ResNet-18 for surgical phase recognition.

    This model extends 2D ResNet to 3D convolutions, enabling it to capture
    temporal dynamics in addition to spatial features.

    Args:
        num_classes (int): Number of surgical phases to classify
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self, num_classes: int = 11, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()

        if not TORCHVISION_VIDEO_AVAILABLE:
            raise ImportError("torchvision video models required for R3D-18")

        self.num_classes = num_classes
        weights = "DEFAULT" if pretrained else None
        self.backbone = r3d_18(weights=weights)

        # Replace the classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

        logger.info(
            f"Initialized R3D-18 with {num_classes} classes, pretrained: {pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through R3D-18.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        return self.backbone(x)


class MC318(nn.Module):
    """
    Mixed Convolution 3D CNN (MC3-18) for surgical phase recognition.

    MC3 uses a mix of 3D and (2+1)D convolutions to balance computational
    efficiency with temporal modeling capability.

    Args:
        num_classes (int): Number of surgical phases to classify
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self, num_classes: int = 11, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()

        if not TORCHVISION_VIDEO_AVAILABLE:
            raise ImportError("torchvision video models required for MC3-18")

        self.num_classes = num_classes
        weights = "DEFAULT" if pretrained else None
        self.backbone = mc3_18(weights=weights)

        # Replace the classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

        logger.info(
            f"Initialized MC3-18 with {num_classes} classes, pretrained: {pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MC3-18.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        return self.backbone(x)


class R2Plus1D18(nn.Module):
    """
    (2+1)D ResNet-18 for surgical phase recognition.

    R2Plus1D factorizes 3D convolutions into separate 2D spatial and 1D temporal
    convolutions, reducing parameters while maintaining expressiveness.

    Args:
        num_classes (int): Number of surgical phases to classify
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self, num_classes: int = 11, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()

        if not TORCHVISION_VIDEO_AVAILABLE:
            raise ImportError("torchvision video models required for R2Plus1D")

        self.num_classes = num_classes
        weights = "DEFAULT" if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)

        # Replace the classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

        logger.info(
            f"Initialized R2Plus1D-18 with {num_classes} classes, pretrained: {pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through R2Plus1D-18.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        return self.backbone(x)


class SlowR50(nn.Module):
    """
    Slow pathway ResNet-50 for surgical phase recognition.

    This model uses a slow pathway that operates on sparsely sampled frames
    to capture semantic information and spatial details.

    Args:
        num_classes (int): Number of surgical phases to classify
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self, num_classes: int = 11, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()

        if not PYTORCHVIDEO_AVAILABLE:
            raise ImportError("pytorchvideo required for Slow R50")

        self.num_classes = num_classes
        self.backbone = slow_r50(pretrained=pretrained)

        # Replace the classification head
        # Slow R50 structure: backbone.blocks[-1].proj
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

        logger.info(
            f"Initialized Slow R50 with {num_classes} classes, pretrained: {pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Slow R50.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        return self.backbone(x)


class X3DXS(nn.Module):
    """
    X3D-XS for surgical phase recognition.

    X3D is designed for efficient video understanding with progressive
    expansion from 2D to 3D along multiple axes.

    Args:
        num_classes (int): Number of surgical phases to classify
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self, num_classes: int = 11, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()

        if not PYTORCHVIDEO_AVAILABLE:
            raise ImportError("pytorchvideo required for X3D")

        self.num_classes = num_classes
        self.backbone = x3d_xs(pretrained=pretrained)

        # Replace the classification head
        # X3D structure: backbone.blocks[-1].proj
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

        logger.info(
            f"Initialized X3D-XS with {num_classes} classes, pretrained: {pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through X3D-XS.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        return self.backbone(x)


class Simple3DCNN(nn.Module):
    """
    Simple 3D CNN for surgical phase recognition.

    A lightweight 3D CNN architecture for when pretrained models are not available
    or when a simpler model is needed for experimentation.

    Args:
        num_classes (int): Number of surgical phases to classify
        input_channels (int): Number of input channels (typically 3 for RGB)
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self, num_classes: int = 11, input_channels: int = 3, dropout: float = 0.2
    ):
        super().__init__()

        self.num_classes = num_classes

        # 3D CNN backbone
        self.features = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(
                input_channels,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            # Second 3D conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Third 3D conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Fourth 3D conv block
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        logger.info(f"Initialized Simple3DCNN with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Simple3DCNN.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        logits = self.classifier(features)
        return logits


def create_3d_cnn_model(
    model_name: str, num_classes: int = 11, pretrained: bool = True, **kwargs
) -> nn.Module:
    """
    Factory function to create 3D CNN models.

    Args:
        model_name (str): Name of the model
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained weights
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: The requested model

    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()

    if model_name == "r3d_18":
        return R3D18(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_name == "mc3_18":
        return MC318(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_name == "r2plus1d_18":
        return R2Plus1D18(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_name == "slow_r50":
        return SlowR50(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_name == "x3d_xs":
        return X3DXS(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_name == "simple3d":
        return Simple3DCNN(
            num_classes=num_classes,
            input_channels=kwargs.get("input_channels", 3),
            dropout=kwargs.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unsupported 3D CNN model: {model_name}")


# Model registry for easy access
CNN_3D_MODELS = {
    "r3d_18": R3D18,
    "mc3_18": MC318,
    "r2plus1d_18": R2Plus1D18,
    "slow_r50": SlowR50,
    "x3d_xs": X3DXS,
    "simple3d": Simple3DCNN,
}


if __name__ == "__main__":
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test each available model
    test_models = []

    if TORCHVISION_VIDEO_AVAILABLE:
        test_models.extend(["r3d_18", "mc3_18", "r2plus1d_18"])

    if PYTORCHVIDEO_AVAILABLE:
        test_models.extend(["slow_r50", "x3d_xs"])

    # Always test simple model
    test_models.append("simple3d")

    for model_name in test_models:
        try:
            model = create_3d_cnn_model(model_name, num_classes=11, pretrained=False)
            model = model.to(device)

            # Test forward pass
            x = torch.randn(2, 3, 16, 224, 224).to(device)
            output = model(x)
            print(f"{model_name} output shape: {output.shape}")

        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
