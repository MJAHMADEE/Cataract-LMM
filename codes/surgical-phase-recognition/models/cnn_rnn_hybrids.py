#!/usr/bin/env python3
"""
CNN-RNN Hybrid Models for Surgical Phase Recognition

This module implements hybrid architectures that combine CNN feature extraction
with RNN sequence modeling for surgical phase recognition. These models first
extract spatial features from individual frames, then model temporal dependencies
using recurrent networks.

Models included:
- ResNet50 + LSTM/GRU
- EfficientNetB5 + LSTM/GRU

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import torchvision.models as models

    TORCHVISION_AVAILABLE = True
except ImportError:
    logger.warning("torchvision not available. CNN backbones may not work.")
    TORCHVISION_AVAILABLE = False


class ResNet50LSTM(nn.Module):
    """
    ResNet50 + LSTM for surgical phase recognition.

    This model uses ResNet50 as a feature extractor for individual frames,
    followed by an LSTM to model temporal sequences for phase classification.

    Args:
        num_classes (int): Number of surgical phases to classify
        hidden_size (int): LSTM hidden state size
        dropout (float): Dropout rate for regularization
        mlp_hidden_size (int): Hidden size for the MLP classifier
        num_lstm_layers (int): Number of LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        pretrained_cnn (bool): Whether to use pretrained CNN weights
    """

    def __init__(
        self,
        num_classes: int = 11,
        hidden_size: int = 256,
        dropout: float = 0.5,
        mlp_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        bidirectional: bool = True,
        pretrained_cnn: bool = True,
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for ResNet50")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # CNN feature extractor (ResNet50)
        self.cnn = models.resnet50(weights="DEFAULT" if pretrained_cnn else None)
        # Remove the final classification layer
        self.cnn_features = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn_feature_dim = 2048  # ResNet50 output feature dimension

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Classifier MLP
        lstm_output_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, num_classes),
        )

        logger.info(f"Initialized ResNet50-LSTM with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet50-LSTM.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W) or (B, T, C, H, W)

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # Reshape for CNN processing: (B*T, C, H, W)
        x = x.reshape(B * T, C, H, W)

        # Extract CNN features
        with torch.set_grad_enabled(self.training):
            cnn_features = self.cnn_features(x)  # (B*T, 2048, 1, 1)
            cnn_features = cnn_features.squeeze(-1).squeeze(-1)  # (B*T, 2048)

        # Reshape back to sequence: (B, T, 2048)
        cnn_features = cnn_features.view(B, T, -1)

        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)  # (B, T, hidden_size * num_directions)

        # Use the last time step output for classification
        final_output = lstm_out[:, -1, :]  # (B, hidden_size * num_directions)

        # Classification
        logits = self.classifier(final_output)  # (B, num_classes)

        return logits


class ResNet50GRU(nn.Module):
    """
    ResNet50 + GRU for surgical phase recognition.

    Similar to ResNet50LSTM but uses GRU instead of LSTM, which can be
    more computationally efficient while maintaining good performance.

    Args:
        num_classes (int): Number of surgical phases to classify
        hidden_size (int): GRU hidden state size
        dropout (float): Dropout rate for regularization
        mlp_hidden_size (int): Hidden size for the MLP classifier
        num_gru_layers (int): Number of GRU layers
        bidirectional (bool): Whether to use bidirectional GRU
        pretrained_cnn (bool): Whether to use pretrained CNN weights
    """

    def __init__(
        self,
        num_classes: int = 11,
        hidden_size: int = 256,
        dropout: float = 0.5,
        mlp_hidden_size: int = 128,
        num_gru_layers: int = 2,
        bidirectional: bool = True,
        pretrained_cnn: bool = True,
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for ResNet50")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # CNN feature extractor (ResNet50)
        self.cnn = models.resnet50(weights="DEFAULT" if pretrained_cnn else None)
        self.cnn_features = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn_feature_dim = 2048

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=self.cnn_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Classifier MLP
        gru_output_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, num_classes),
        )

        logger.info(f"Initialized ResNet50-GRU with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet50-GRU."""
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # Reshape for CNN processing
        x = x.reshape(B * T, C, H, W)

        # Extract CNN features
        with torch.set_grad_enabled(self.training):
            cnn_features = self.cnn_features(x)
            cnn_features = cnn_features.squeeze(-1).squeeze(-1)

        # Reshape back to sequence
        cnn_features = cnn_features.view(B, T, -1)

        # GRU processing
        gru_out, _ = self.gru(cnn_features)

        # Use the last time step output
        final_output = gru_out[:, -1, :]

        # Classification
        logits = self.classifier(final_output)

        return logits


class EfficientNetB5LSTM(nn.Module):
    """
    EfficientNetB5 + LSTM for surgical phase recognition.

    Uses EfficientNetB5 as the CNN backbone, which is more parameter-efficient
    than ResNet50 while maintaining competitive performance.

    Args:
        num_classes (int): Number of surgical phases to classify
        hidden_size (int): LSTM hidden state size
        dropout (float): Dropout rate for regularization
        mlp_hidden_size (int): Hidden size for the MLP classifier
        num_lstm_layers (int): Number of LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        pretrained_cnn (bool): Whether to use pretrained CNN weights
    """

    def __init__(
        self,
        num_classes: int = 11,
        hidden_size: int = 256,
        dropout: float = 0.5,
        mlp_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        bidirectional: bool = True,
        pretrained_cnn: bool = True,
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for EfficientNet")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # CNN feature extractor (EfficientNetB5)
        self.cnn = models.efficientnet_b5(weights="DEFAULT" if pretrained_cnn else None)
        # Remove the final classification layer
        self.cnn_features = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn_feature_dim = 2048  # EfficientNetB5 output feature dimension

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Classifier MLP
        lstm_output_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, num_classes),
        )

        logger.info(f"Initialized EfficientNetB5-LSTM with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EfficientNetB5-LSTM."""
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # Reshape for CNN processing
        x = x.reshape(B * T, C, H, W)

        # Extract CNN features
        with torch.set_grad_enabled(self.training):
            cnn_features = self.cnn_features(x)
            cnn_features = cnn_features.squeeze(-1).squeeze(-1)

        # Reshape back to sequence
        cnn_features = cnn_features.view(B, T, -1)

        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)

        # Use the last time step output
        final_output = lstm_out[:, -1, :]

        # Classification
        logits = self.classifier(final_output)

        return logits


class EfficientNetB5GRU(nn.Module):
    """
    EfficientNetB5 + GRU for surgical phase recognition.

    Combines EfficientNetB5 with GRU for efficient temporal modeling.

    Args:
        num_classes (int): Number of surgical phases to classify
        hidden_size (int): GRU hidden state size
        dropout (float): Dropout rate for regularization
        mlp_hidden_size (int): Hidden size for the MLP classifier
        num_gru_layers (int): Number of GRU layers
        bidirectional (bool): Whether to use bidirectional GRU
        pretrained_cnn (bool): Whether to use pretrained CNN weights
    """

    def __init__(
        self,
        num_classes: int = 11,
        hidden_size: int = 256,
        dropout: float = 0.5,
        mlp_hidden_size: int = 128,
        num_gru_layers: int = 2,
        bidirectional: bool = True,
        pretrained_cnn: bool = True,
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for EfficientNet")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # CNN feature extractor (EfficientNetB5)
        self.cnn = models.efficientnet_b5(weights="DEFAULT" if pretrained_cnn else None)
        self.cnn_features = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn_feature_dim = 2048

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=self.cnn_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Classifier MLP
        gru_output_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, num_classes),
        )

        logger.info(f"Initialized EfficientNetB5-GRU with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EfficientNetB5-GRU."""
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # Reshape for CNN processing
        x = x.reshape(B * T, C, H, W)

        # Extract CNN features
        with torch.set_grad_enabled(self.training):
            cnn_features = self.cnn_features(x)
            cnn_features = cnn_features.squeeze(-1).squeeze(-1)

        # Reshape back to sequence
        cnn_features = cnn_features.view(B, T, -1)

        # GRU processing
        gru_out, _ = self.gru(cnn_features)

        # Use the last time step output
        final_output = gru_out[:, -1, :]

        # Classification
        logits = self.classifier(final_output)

        return logits


def create_cnn_rnn_model(
    model_name: str,
    num_classes: int = 11,
    hidden_size: int = 256,
    dropout: float = 0.5,
    mlp_hidden_size: int = 128,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create CNN-RNN hybrid models.

    Args:
        model_name (str): Name of the model
        num_classes (int): Number of classes
        hidden_size (int): RNN hidden size
        dropout (float): Dropout rate
        mlp_hidden_size (int): MLP hidden size
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: The requested model

    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()

    if model_name == "resnet50_lstm":
        return ResNet50LSTM(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            mlp_hidden_size=mlp_hidden_size,
            **kwargs,
        )
    elif model_name == "resnet50_gru":
        return ResNet50GRU(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            mlp_hidden_size=mlp_hidden_size,
            **kwargs,
        )
    elif model_name == "efficientnetb5_lstm":
        return EfficientNetB5LSTM(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            mlp_hidden_size=mlp_hidden_size,
            **kwargs,
        )
    elif model_name == "efficientnetb5_gru":
        return EfficientNetB5GRU(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            mlp_hidden_size=mlp_hidden_size,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported CNN-RNN model: {model_name}")


# Model registry for easy access
CNN_RNN_MODELS = {
    "resnet50_lstm": ResNet50LSTM,
    "resnet50_gru": ResNet50GRU,
    "efficientnetb5_lstm": EfficientNetB5LSTM,
    "efficientnetb5_gru": EfficientNetB5GRU,
}


if __name__ == "__main__":
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if TORCHVISION_AVAILABLE:
        # Test each model
        for model_name in CNN_RNN_MODELS.keys():
            try:
                model = create_cnn_rnn_model(
                    model_name,
                    num_classes=11,
                    hidden_size=128,
                    pretrained_cnn=False,  # For testing
                )
                model = model.to(device)

                # Test forward pass
                x = torch.randn(2, 3, 10, 224, 224).to(device)
                output = model(x)
                print(f"{model_name} output shape: {output.shape}")

            except Exception as e:
                print(f"Failed to test {model_name}: {e}")
    else:
        print("torchvision not available for testing")
