import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange


class CNNFeatureExtractor(nn.Module):
    """ResNet-based feature extractor for CNN-RNN models."""

    def __init__(self, pretrained=True):
        super().__init__()
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Process each frame through CNN
        x = rearrange(x, "b c t h w -> (b t) c h w")
        features = self.features(x)  # (B*T, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*T, 2048)
        features = rearrange(features, "(b t) d -> b t d", b=B, t=T)
        return features


class CNN_LSTM(nn.Module):
    """CNN backbone with Bidirectional LSTM for temporal modeling."""

    def __init__(self, num_classes, hidden_dim=512, num_layers=2, dropout=0.5):
        super().__init__()
        self.cnn = CNNFeatureExtractor(pretrained=True)
        self.lstm = nn.LSTM(
            input_size=self.cnn.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Bidirectional as required
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        # 2 * hidden_dim due to bidirectional
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        # Extract CNN features
        features = self.cnn(x)  # (B, T, D)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(features)  # lstm_out: (B, T, 2*hidden_dim)

        # Use the last hidden state from both directions
        # h_n shape: (num_layers * 2, B, hidden_dim)
        h_forward = h_n[-2]  # Last layer, forward direction
        h_backward = h_n[-1]  # Last layer, backward direction
        h_combined = torch.cat([h_forward, h_backward], dim=1)  # (B, 2*hidden_dim)

        # Classification
        h_combined = self.dropout(h_combined)
        output = self.classifier(h_combined)

        return output


class CNN_GRU(nn.Module):
    """CNN backbone with Bidirectional GRU for temporal modeling."""

    def __init__(self, num_classes, hidden_dim=512, num_layers=2, dropout=0.5):
        super().__init__()
        self.cnn = CNNFeatureExtractor(pretrained=True)
        self.gru = nn.GRU(
            input_size=self.cnn.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Bidirectional as required
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        # 2 * hidden_dim due to bidirectional
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        # Extract CNN features
        features = self.cnn(x)  # (B, T, D)

        # GRU processing
        gru_out, h_n = self.gru(features)  # gru_out: (B, T, 2*hidden_dim)

        # Use the last hidden state from both directions
        # h_n shape: (num_layers * 2, B, hidden_dim)
        h_forward = h_n[-2]  # Last layer, forward direction
        h_backward = h_n[-1]  # Last layer, backward direction
        h_combined = torch.cat([h_forward, h_backward], dim=1)  # (B, 2*hidden_dim)

        # Classification
        h_combined = self.dropout(h_combined)
        output = self.classifier(h_combined)

        return output
