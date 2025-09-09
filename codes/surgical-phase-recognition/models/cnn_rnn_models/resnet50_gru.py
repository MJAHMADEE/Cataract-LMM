#!/usr/bin/env python3
"""
ResNet50 GRU model for surgical phase recognition.

This module provides a CNN-RNN hybrid model using ResNet50 as feature extractor
and GRU for temporal modeling of surgical phase sequences.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class Resnet50_GRU(nn.Module):
    """
    ResNet50 + GRU model for temporal sequence classification.

    This model uses ResNet50 as a feature extractor for individual frames,
    followed by a GRU to model temporal dependencies across the sequence.

    Args:
        num_classes (int): Number of output classes
        hidden_size (int): Hidden size of GRU layer
        dropout_rate (float): Dropout rate for regularization
        mlp_hidden_size (int): Hidden size of final MLP layer
        pretrained (bool): Whether to use pretrained ResNet50 weights
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        dropout_rate: float = 0.5,
        mlp_hidden_size: int = 128,
        pretrained: bool = True,
    ):
        super(Resnet50_GRU, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.mlp_hidden_size = mlp_hidden_size

        # ResNet50 feature extractor
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Feature dimension from ResNet50
        self.feature_dim = 2048

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate if hidden_size > 1 else 0,
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize GRU and classifier weights."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.size()

        # Reshape for feature extraction
        x = x.view(batch_size * seq_len, channels, height, width)

        # Extract features using ResNet50
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.feature_extractor(x)

        # Reshape features for GRU
        features = features.view(batch_size, seq_len, self.feature_dim)

        # GRU forward pass
        gru_out, hidden = self.gru(features)

        # Use the last output for classification
        last_output = gru_out[:, -1, :]

        # Final classification
        output = self.classifier(last_output)

        return output

    def get_features(self, x):
        """
        Extract features without classification for analysis.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (frame_features, gru_features)
        """
        batch_size, seq_len, channels, height, width = x.size()

        # Reshape for feature extraction
        x = x.view(batch_size * seq_len, channels, height, width)

        # Extract features using ResNet50
        frame_features = self.feature_extractor(x)
        frame_features = frame_features.view(batch_size, seq_len, self.feature_dim)

        # GRU features
        gru_features, _ = self.gru(frame_features)

        return frame_features, gru_features

    def freeze_backbone(self):
        """Freeze ResNet50 backbone for fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ResNet50 backbone."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


def create_resnet50_gru(
    num_classes: int,
    hidden_size: int = 256,
    dropout_rate: float = 0.5,
    mlp_hidden_size: int = 128,
    pretrained: bool = True,
) -> Resnet50_GRU:
    """
    Factory function to create ResNet50-GRU model.

    Args:
        num_classes (int): Number of output classes
        hidden_size (int): GRU hidden size
        dropout_rate (float): Dropout rate
        mlp_hidden_size (int): MLP hidden size
        pretrained (bool): Use pretrained weights

    Returns:
        Resnet50_GRU: Configured model instance
    """
    return Resnet50_GRU(
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        mlp_hidden_size=mlp_hidden_size,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    # Test the model
    model = create_resnet50_gru(num_classes=11)

    # Test input
    batch_size = 2
    seq_len = 10
    test_input = torch.randn(batch_size, seq_len, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")

        # Test feature extraction
        frame_features, gru_features = model.get_features(test_input)
        print(f"Frame features shape: {frame_features.shape}")
        print(f"GRU features shape: {gru_features.shape}")

    print("ResNet50-GRU model test completed!")
