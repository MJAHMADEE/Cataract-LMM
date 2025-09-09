#!/usr/bin/env python3
"""
Multi-Stage Models for Surgical Phase Recognition

This module implements advanced multi-stage architectures including the TeCNO model
and other sophisticated approaches for surgical phase recognition that process
video data in multiple stages or use hierarchical structures.

The TeCNO model is specifically designed for surgical workflow analysis and includes
temporal consistency objectives.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import logging
from typing import Dict, List, Optional, Tuple

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


class TeCNOModel(nn.Module):
    """
    TeCNO (Temporal Consistency Network for Operating) Model.

    TeCNO is a multi-stage model designed specifically for surgical workflow analysis.
    It combines spatial feature extraction with temporal consistency constraints
    to improve phase recognition accuracy.

    The model consists of:
    1. CNN backbone for spatial feature extraction
    2. Temporal feature aggregation
    3. Phase classification with temporal consistency
    4. Multi-scale temporal modeling

    Args:
        num_classes (int): Number of surgical phases
        backbone (str): CNN backbone ('resnet50', 'efficientnet_b5')
        hidden_size (int): Hidden dimension size
        temporal_window (int): Size of temporal window for consistency
        dropout (float): Dropout rate
        use_attention (bool): Whether to use attention mechanisms
        pretrained (bool): Whether to use pretrained backbone
    """

    def __init__(
        self,
        num_classes: int = 11,
        backbone: str = "resnet50",
        hidden_size: int = 512,
        temporal_window: int = 16,
        dropout: float = 0.3,
        use_attention: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for TeCNO model")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.temporal_window = temporal_window
        self.use_attention = use_attention

        # CNN Backbone
        self.backbone_name = backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
            self.feature_dim = 2048
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == "efficientnet_b5":
            self.backbone = models.efficientnet_b5(
                weights="DEFAULT" if pretrained else None
            )
            self.feature_dim = 2048
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Feature projection
        self.feature_projection = nn.Linear(self.feature_dim, hidden_size)

        # Temporal modeling components
        self.temporal_conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1
        )

        # Multi-scale temporal features
        self.temporal_conv_3 = nn.Conv1d(
            hidden_size, hidden_size // 2, kernel_size=3, padding=1
        )
        self.temporal_conv_5 = nn.Conv1d(
            hidden_size, hidden_size // 2, kernel_size=5, padding=2
        )
        self.temporal_conv_7 = nn.Conv1d(
            hidden_size, hidden_size // 2, kernel_size=7, padding=3
        )

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=8, dropout=dropout, batch_first=True
            )

        # Temporal consistency module
        self.consistency_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Final classifier
        classifier_input_size = (
            hidden_size + (hidden_size // 2) * 3
        )  # Multi-scale features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Temporal consistency loss weight
        self.consistency_weight = 0.1

        logger.info(f"Initialized TeCNO model with {backbone} backbone")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features using CNN backbone."""
        B, T, C, H, W = x.shape

        # Reshape for CNN processing
        x = x.reshape(B * T, C, H, W)

        # Extract features
        features = self.backbone(x)  # (B*T, feature_dim, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*T, feature_dim)

        # Project features
        features = self.feature_projection(features)  # (B*T, hidden_size)

        # Reshape back to sequence
        features = features.view(B, T, -1)  # (B, T, hidden_size)

        return features

    def temporal_modeling(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal modeling with multi-scale convolutions."""
        B, T, D = features.shape

        # Transpose for conv1d: (B, D, T)
        features_transposed = features.transpose(1, 2)

        # Base temporal convolution
        temporal_features = self.temporal_conv(features_transposed)
        temporal_features = F.relu(temporal_features)

        # Multi-scale temporal features
        temp_3 = F.relu(self.temporal_conv_3(features_transposed))
        temp_5 = F.relu(self.temporal_conv_5(features_transposed))
        temp_7 = F.relu(self.temporal_conv_7(features_transposed))

        # Concatenate multi-scale features
        multi_scale_features = torch.cat([temp_3, temp_5, temp_7], dim=1)

        # Transpose back: (B, T, D)
        temporal_features = temporal_features.transpose(1, 2)
        multi_scale_features = multi_scale_features.transpose(1, 2)

        return temporal_features, multi_scale_features

    def apply_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Apply self-attention mechanism."""
        if self.use_attention:
            attended_features, _ = self.attention(features, features, features)
            return attended_features
        return features

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TeCNO model.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W) or (B, C, T, H, W)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'logits': Main classification logits (B, num_classes)
                - 'consistency_logits': Temporal consistency logits (B, T, num_classes)
                - 'features': Extracted temporal features (B, T, hidden_size)
        """
        # Handle input format
        if x.dim() == 5:
            if x.shape[1] == 3:  # (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # Extract spatial features
        spatial_features = self.extract_features(x)  # (B, T, hidden_size)

        # Temporal modeling
        temporal_features, multi_scale_features = self.temporal_modeling(
            spatial_features
        )

        # Apply attention
        attended_features = self.apply_attention(temporal_features)

        # Temporal consistency prediction (for each frame)
        consistency_logits = self.consistency_fc(
            attended_features
        )  # (B, T, num_classes)

        # Global features for final classification
        # Use the last frame's attended features and multi-scale features
        final_features = torch.cat(
            [
                attended_features[:, -1, :],  # Last frame attended features
                multi_scale_features[:, -1, :],  # Last frame multi-scale features
            ],
            dim=1,
        )

        # Final classification
        main_logits = self.classifier(final_features)  # (B, num_classes)

        return {
            "logits": main_logits,
            "consistency_logits": consistency_logits,
            "features": attended_features,
        }

    def compute_temporal_consistency_loss(
        self, consistency_logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.

        Args:
            consistency_logits (torch.Tensor): Frame-level predictions (B, T, num_classes)
            targets (torch.Tensor): Target labels (B,)

        Returns:
            torch.Tensor: Temporal consistency loss
        """
        B, T, num_classes = consistency_logits.shape

        # Expand targets to match sequence length
        targets_expanded = targets.unsqueeze(1).expand(B, T)  # (B, T)

        # Compute cross-entropy loss for each frame
        consistency_loss = F.cross_entropy(
            consistency_logits.reshape(B * T, num_classes),
            targets_expanded.reshape(B * T),
            reduction="mean",
        )

        return consistency_loss


class HierarchicalPhaseModel(nn.Module):
    """
    Hierarchical model for surgical phase recognition.

    This model uses a hierarchical approach where it first classifies
    broad phase categories, then refines to specific phases.

    Args:
        num_classes (int): Number of fine-grained phases
        num_coarse_classes (int): Number of coarse phase categories
        backbone (str): CNN backbone architecture
        hidden_size (int): Hidden dimension size
        dropout (float): Dropout rate
        pretrained (bool): Whether to use pretrained backbone
    """

    def __init__(
        self,
        num_classes: int = 11,
        num_coarse_classes: int = 4,
        backbone: str = "resnet50",
        hidden_size: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for Hierarchical model")

        self.num_classes = num_classes
        self.num_coarse_classes = num_coarse_classes

        # CNN Backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
            self.feature_dim = 2048
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == "efficientnet_b5":
            self.backbone = models.efficientnet_b5(
                weights="DEFAULT" if pretrained else None
            )
            self.feature_dim = 2048
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Feature projection
        self.feature_projection = nn.Linear(self.feature_dim, hidden_size)

        # Temporal modeling
        self.temporal_conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1
        )

        # Coarse classifier (broad categories)
        self.coarse_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_coarse_classes),
        )

        # Fine classifier (specific phases)
        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_size + num_coarse_classes, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        logger.info(
            f"Initialized Hierarchical model with {num_coarse_classes} coarse and {num_classes} fine classes"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical model.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'logits': Fine-grained classification logits
                - 'coarse_logits': Coarse classification logits
                - 'features': Extracted features
        """
        # Handle input format
        if x.dim() == 5:
            if x.shape[1] == 3:  # (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # Reshape for CNN processing
        x = x.reshape(B * T, C, H, W)

        # Extract features
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)
        features = self.feature_projection(features)

        # Reshape back to sequence
        features = features.view(B, T, -1)

        # Temporal modeling
        temporal_features = features.transpose(1, 2)  # (B, D, T)
        temporal_features = F.relu(self.temporal_conv(temporal_features))
        temporal_features = temporal_features.transpose(1, 2)  # (B, T, D)

        # Use last frame features for classification
        final_features = temporal_features[:, -1, :]  # (B, D)

        # Coarse classification
        coarse_logits = self.coarse_classifier(final_features)
        coarse_probs = F.softmax(coarse_logits, dim=1)

        # Fine classification (conditioned on coarse prediction)
        combined_features = torch.cat([final_features, coarse_probs], dim=1)
        fine_logits = self.fine_classifier(combined_features)

        return {
            "logits": fine_logits,
            "coarse_logits": coarse_logits,
            "features": final_features,
        }


def create_multistage_model(
    model_name: str, num_classes: int = 11, **kwargs
) -> nn.Module:
    """
    Factory function to create multi-stage models.

    Args:
        model_name (str): Name of the model
        num_classes (int): Number of classes
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: The requested model

    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()

    if model_name == "tecno":
        return TeCNOModel(num_classes=num_classes, **kwargs)
    elif model_name == "hierarchical":
        return HierarchicalPhaseModel(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported multi-stage model: {model_name}")


# Model registry
MULTISTAGE_MODELS = {"tecno": TeCNOModel, "hierarchical": HierarchicalPhaseModel}


if __name__ == "__main__":
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if TORCHVISION_AVAILABLE:
        # Test TeCNO model
        try:
            model = create_multistage_model("tecno", num_classes=11, pretrained=False)
            model = model.to(device)

            x = torch.randn(2, 10, 3, 224, 224).to(device)
            output = model(x)

            print(f"TeCNO model output shapes:")
            for key, value in output.items():
                print(f"  {key}: {value.shape}")

        except Exception as e:
            print(f"Failed to test TeCNO model: {e}")

        # Test Hierarchical model
        try:
            model = create_multistage_model(
                "hierarchical", num_classes=11, pretrained=False
            )
            model = model.to(device)

            x = torch.randn(2, 10, 3, 224, 224).to(device)
            output = model(x)

            print(f"Hierarchical model output shapes:")
            for key, value in output.items():
                print(f"  {key}: {value.shape}")

        except Exception as e:
            print(f"Failed to test Hierarchical model: {e}")
    else:
        print("torchvision not available for testing")
