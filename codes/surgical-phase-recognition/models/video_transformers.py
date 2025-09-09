#!/usr/bin/env python3
"""
Video Transformer Models for Surgical Phase Recognition

This module implements various transformer-based architectures for video understanding
specifically adapted for surgical phase recognition. These models leverage self-attention
mechanisms to capture long-range temporal dependencies in surgical videos.

Models included:
- Swin3D Transformer
- Multiscale Vision Transformer (MViT)

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from torchvision.models.video import mvit_v1_b, swin3d_t

    TORCHVISION_AVAILABLE = True
except ImportError:
    logger.warning(
        "torchvision video models not available. Some transformer models may not work."
    )
    TORCHVISION_AVAILABLE = False


class Swin3DTransformer(nn.Module):
    """
    Swin3D Transformer for surgical phase recognition.

    This model uses a hierarchical transformer architecture with shifted windows
    for efficient video understanding. It's particularly effective for capturing
    both spatial and temporal patterns in surgical videos.

    Args:
        num_classes (int): Number of surgical phases to classify
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self, num_classes: int = 11, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision video models required for Swin3D")

        self.num_classes = num_classes
        self.backbone = swin3d_t(weights="DEFAULT" if pretrained else None)

        # Replace the classification head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

        logger.info(
            f"Initialized Swin3D with {num_classes} classes, pretrained: {pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Swin3D model.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)
                             where B=batch_size, C=channels, T=time, H=height, W=width

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        return self.backbone(x)

    def get_feature_dim(self) -> int:
        """Get the feature dimension before the classification head."""
        return (
            self.backbone.head[1].in_features
            if isinstance(self.backbone.head, nn.Sequential)
            else self.backbone.head.in_features
        )


class MViTTransformer(nn.Module):
    """
    Multiscale Vision Transformer (MViT) for surgical phase recognition.

    MViT uses a hierarchical approach with multiscale feature representations,
    making it effective for understanding temporal dynamics at different scales
    in surgical procedures.

    Args:
        num_classes (int): Number of surgical phases to classify
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self, num_classes: int = 11, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision video models required for MViT")

        self.num_classes = num_classes
        self.backbone = mvit_v1_b(weights="DEFAULT" if pretrained else None)

        # Replace the classification head
        # MViT has a head structure: [norm, linear]
        if hasattr(self.backbone, "head") and len(self.backbone.head) > 1:
            in_features = self.backbone.head[1].in_features
            self.backbone.head[1] = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        else:
            # Fallback if structure is different
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )

        logger.info(
            f"Initialized MViT with {num_classes} classes, pretrained: {pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MViT model.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        return self.backbone(x)

    def get_feature_dim(self) -> int:
        """Get the feature dimension before the classification head."""
        if hasattr(self.backbone, "head") and len(self.backbone.head) > 1:
            head_layer = self.backbone.head[1]
            if isinstance(head_layer, nn.Sequential):
                return head_layer[1].in_features
            else:
                return head_layer.in_features
        else:
            return (
                self.backbone.head[1].in_features
                if isinstance(self.backbone.head, nn.Sequential)
                else self.backbone.head.in_features
            )


class AdaptiveVideoTransformer(nn.Module):
    """
    Adaptive Video Transformer for variable-length surgical sequences.

    This is a custom transformer architecture that can handle variable-length
    video sequences commonly found in surgical procedures.

    Args:
        num_classes (int): Number of surgical phases to classify
        d_model (int): Transformer hidden dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
        max_seq_len (int): Maximum sequence length
    """

    def __init__(
        self,
        num_classes: int = 11,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        # Feature extraction backbone (could be any CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(
                3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # Spatial pooling, keep temporal
        )

        # Projection to transformer dimension
        self.feature_projection = nn.Linear(64, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )

        logger.info(f"Initialized AdaptiveVideoTransformer with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adaptive transformer.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        B, C, T, H, W = x.shape

        # Extract features
        features = self.feature_extractor(x)  # (B, 64, T, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 64, T)
        features = features.permute(0, 2, 1)  # (B, T, 64)

        # Project to transformer dimension
        features = self.feature_projection(features)  # (B, T, d_model)

        # Add positional encoding
        seq_len = min(T, self.max_seq_len)
        features = features[:, :seq_len] + self.positional_encoding[:seq_len].unsqueeze(
            0
        )

        # Apply transformer
        transformer_output = self.transformer(features)  # (B, T, d_model)

        # Global average pooling over time dimension
        pooled_features = transformer_output.mean(dim=1)  # (B, d_model)

        # Classification
        logits = self.classifier(pooled_features)  # (B, num_classes)

        return logits


def create_video_transformer(
    model_name: str, num_classes: int = 11, pretrained: bool = True, **kwargs
) -> nn.Module:
    """
    Factory function to create video transformer models.

    Args:
        model_name (str): Name of the model ('swin3d_t', 'mvit_v1_b', 'adaptive')
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained weights
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: The requested model

    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()

    if model_name == "swin3d_t":
        return Swin3DTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_name == "mvit_v1_b":
        return MViTTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_name == "adaptive":
        return AdaptiveVideoTransformer(
            num_classes=num_classes,
            d_model=kwargs.get("d_model", 512),
            nhead=kwargs.get("nhead", 8),
            num_layers=kwargs.get("num_layers", 6),
            dropout=kwargs.get("dropout", 0.1),
            max_seq_len=kwargs.get("max_seq_len", 100),
        )
    else:
        raise ValueError(f"Unsupported video transformer model: {model_name}")


# Model registry for easy access
VIDEO_TRANSFORMER_MODELS = {
    "swin3d_t": Swin3DTransformer,
    "mvit_v1_b": MViTTransformer,
    "adaptive": AdaptiveVideoTransformer,
}


if __name__ == "__main__":
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test Swin3D
    if TORCHVISION_AVAILABLE:
        model = create_video_transformer("swin3d_t", num_classes=11)
        model = model.to(device)

        # Test forward pass
        x = torch.randn(2, 3, 16, 224, 224).to(device)
        output = model(x)
        print(f"Swin3D output shape: {output.shape}")

        # Test MViT
        model = create_video_transformer("mvit_v1_b", num_classes=11)
        model = model.to(device)
        output = model(x)
        print(f"MViT output shape: {output.shape}")

    # Test Adaptive Transformer
    model = create_video_transformer("adaptive", num_classes=11, d_model=256)
    model = model.to(device)
    x = torch.randn(2, 3, 20, 224, 224).to(device)
    output = model(x)
    print(f"Adaptive Transformer output shape: {output.shape}")
