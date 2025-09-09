#!/usr/bin/env python3
"""
TeCNO (Temporal Convolutional Networks for Operation) models for surgical phase recognition.

This module implements the TeCNO architecture for temporal modeling in surgical videos,
including both single-stage and multi-stage variants with dilated temporal convolutions.

Reference: TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    """
    Dilated residual layer for temporal modeling.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dilation (int): Dilation rate for temporal convolution
        dropout_rate (float): Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        dropout_rate: float = 0.3,
    ):
        super(DilatedResidualLayer, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout_rate)

        # Residual connection
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x

        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)

        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        out += residual
        out = F.relu(out)

        return out


class TemporalConvolutionalNetwork(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling.

    Args:
        input_size (int): Input feature dimension
        num_channels (List[int]): Number of channels in each layer
        kernel_size (int): Kernel size for temporal convolutions
        dropout_rate (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout_rate: float = 0.3,
    ):
        super(TemporalConvolutionalNetwork, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                DilatedResidualLayer(
                    in_channels,
                    out_channels,
                    dilation=dilation_size,
                    dropout_rate=dropout_rate,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through TCN."""
        return self.network(x)


class SingleStageModel(nn.Module):
    """
    Single-stage TeCNO model for surgical phase recognition.

    Args:
        input_size (int): Input feature dimension
        num_classes (int): Number of output classes
        num_channels (List[int]): TCN channel configuration
        dropout_rate (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_channels: List[int] = [64, 64, 64, 64],
        dropout_rate: float = 0.3,
    ):
        super(SingleStageModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Temporal Convolutional Network
        self.tcn = TemporalConvolutionalNetwork(
            input_size=input_size, num_channels=num_channels, dropout_rate=dropout_rate
        )

        # Classification head
        self.classifier = nn.Conv1d(num_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features, seq_len)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes, seq_len)
        """
        # TCN forward pass
        tcn_out = self.tcn(x)

        # Classification
        output = self.classifier(tcn_out)

        return output


class MultiStageModel(nn.Module):
    """
    Multi-stage TeCNO model for surgical phase recognition.

    This model implements a hierarchical approach with multiple temporal resolutions
    to capture both fine-grained and coarse-grained temporal patterns.

    Args:
        input_size (int): Input feature dimension
        num_classes (int): Number of output classes
        num_stages (int): Number of stages in the model
        stage_channels (List[List[int]]): Channel configuration for each stage
        dropout_rate (float): Dropout rate
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_stages: int = 4,
        stage_channels: Optional[List[List[int]]] = None,
        dropout_rate: float = 0.3,
    ):
        super(MultiStageModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.num_stages = num_stages

        if stage_channels is None:
            stage_channels = [
                [64, 64],  # Stage 1: fine-grained
                [128, 128],  # Stage 2: medium-grained
                [256, 256],  # Stage 3: coarse-grained
                [512, 512],  # Stage 4: very coarse-grained
            ]

        self.stages = nn.ModuleList()
        self.stage_classifiers = nn.ModuleList()

        current_input_size = input_size

        for i in range(num_stages):
            # Create TCN for this stage
            stage_tcn = TemporalConvolutionalNetwork(
                input_size=current_input_size,
                num_channels=stage_channels[i],
                dropout_rate=dropout_rate,
            )
            self.stages.append(stage_tcn)

            # Create classifier for this stage
            stage_classifier = nn.Conv1d(
                stage_channels[i][-1], num_classes, kernel_size=1
            )
            self.stage_classifiers.append(stage_classifier)

            # Update input size for next stage
            current_input_size = stage_channels[i][-1]

        # Final fusion layer
        self.fusion = nn.Conv1d(num_classes * num_stages, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through multi-stage model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features, seq_len)

        Returns:
            tuple: (final_output, stage_outputs)
                - final_output: Fused output of shape (batch_size, num_classes, seq_len)
                - stage_outputs: List of stage outputs
        """
        stage_outputs = []
        current_input = x

        # Forward through each stage
        for i, (stage, classifier) in enumerate(
            zip(self.stages, self.stage_classifiers)
        ):
            # TCN forward pass
            stage_features = stage(current_input)

            # Classification for this stage
            stage_output = classifier(stage_features)
            stage_outputs.append(stage_output)

            # Use features as input for next stage (residual connection)
            current_input = stage_features

        # Fuse all stage outputs
        fused_input = torch.cat(stage_outputs, dim=1)
        final_output = self.fusion(fused_input)

        return final_output, stage_outputs

    def get_stage_loss(
        self, stage_outputs: List[torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-stage loss for training.

        Args:
            stage_outputs (List[torch.Tensor]): Outputs from each stage
            targets (torch.Tensor): Ground truth targets

        Returns:
            torch.Tensor: Combined loss across all stages
        """
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        # Weight each stage differently (later stages get higher weight)
        stage_weights = (
            [0.1, 0.2, 0.3, 0.4]
            if len(stage_outputs) == 4
            else [1.0 / len(stage_outputs)] * len(stage_outputs)
        )

        for i, (output, weight) in enumerate(zip(stage_outputs, stage_weights)):
            # Reshape for loss computation
            output_reshaped = (
                output.transpose(1, 2).contiguous().view(-1, self.num_classes)
            )
            targets_reshaped = targets.view(-1)

            stage_loss = criterion(output_reshaped, targets_reshaped)
            total_loss += weight * stage_loss

        return total_loss


class TeCNOFeatureExtractor(nn.Module):
    """
    Feature extractor for TeCNO models using pre-trained CNN backbone.

    Args:
        backbone_name (str): Name of the backbone CNN
        feature_dim (int): Output feature dimension
        pretrained (bool): Whether to use pretrained weights
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        feature_dim: int = 2048,
        pretrained: bool = True,
    ):
        super(TeCNOFeatureExtractor, self).__init__()

        if backbone_name == "resnet50":
            import torchvision.models as models

            backbone = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Adaptive pooling to ensure consistent feature dimension
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Extract features from video frames.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)

        Returns:
            torch.Tensor: Features of shape (batch_size, feature_dim, seq_len)
        """
        batch_size, seq_len, channels, height, width = x.size()

        # Reshape for feature extraction
        x = x.view(batch_size * seq_len, channels, height, width)

        # Extract features
        features = self.feature_extractor(x)
        features = self.adaptive_pool(features)
        features = features.view(batch_size, seq_len, self.feature_dim)

        # Transpose for TCN input (batch_size, feature_dim, seq_len)
        features = features.transpose(1, 2)

        return features


def create_single_stage_tecno(
    input_size: int = 2048,
    num_classes: int = 11,
    num_channels: List[int] = [64, 64, 64, 64],
    dropout_rate: float = 0.3,
) -> SingleStageModel:
    """Factory function for single-stage TeCNO model."""
    return SingleStageModel(
        input_size=input_size,
        num_classes=num_classes,
        num_channels=num_channels,
        dropout_rate=dropout_rate,
    )


def create_multi_stage_tecno(
    input_size: int = 2048,
    num_classes: int = 11,
    num_stages: int = 4,
    stage_channels: Optional[List[List[int]]] = None,
    dropout_rate: float = 0.3,
) -> MultiStageModel:
    """Factory function for multi-stage TeCNO model."""
    return MultiStageModel(
        input_size=input_size,
        num_classes=num_classes,
        num_stages=num_stages,
        stage_channels=stage_channels,
        dropout_rate=dropout_rate,
    )


if __name__ == "__main__":
    # Test the models
    batch_size = 2
    seq_len = 100
    feature_dim = 2048
    num_classes = 11

    # Test input (features from CNN backbone)
    test_input = torch.randn(batch_size, feature_dim, seq_len)

    print("Testing Single-Stage TeCNO...")
    single_stage = create_single_stage_tecno(
        input_size=feature_dim, num_classes=num_classes
    )

    with torch.no_grad():
        single_output = single_stage(test_input)
        print(f"Single-stage input shape: {test_input.shape}")
        print(f"Single-stage output shape: {single_output.shape}")

    print("\nTesting Multi-Stage TeCNO...")
    multi_stage = create_multi_stage_tecno(
        input_size=feature_dim, num_classes=num_classes
    )

    with torch.no_grad():
        final_output, stage_outputs = multi_stage(test_input)
        print(f"Multi-stage input shape: {test_input.shape}")
        print(f"Multi-stage final output shape: {final_output.shape}")
        print(f"Number of stage outputs: {len(stage_outputs)}")
        for i, stage_out in enumerate(stage_outputs):
            print(f"Stage {i+1} output shape: {stage_out.shape}")

    print("\nTesting Feature Extractor...")
    feature_extractor = TeCNOFeatureExtractor()
    video_input = torch.randn(batch_size, seq_len, 3, 224, 224)

    with torch.no_grad():
        features = feature_extractor(video_input)
        print(f"Video input shape: {video_input.shape}")
        print(f"Extracted features shape: {features.shape}")

    print("TeCNO model testing completed!")
