# 🧠 CNN-RNN Hybrid Models

## Overview

This directory contains CNN-RNN hybrid model implementations specifically designed for surgical phase recognition in cataract surgery videos. These models combine convolutional neural networks for spatial feature extraction with recurrent neural networks for temporal sequence modeling.

## 📁 Model Architecture

### **Hybrid CNN-RNN Design**

The models in this directory implement a two-stage architecture:

1. **Spatial Feature Extraction (CNN)**: Extract visual features from individual video frames
2. **Temporal Sequence Modeling (RNN)**: Model temporal dependencies across frame sequences

```
Input Video Sequence
        ↓
[CNN Backbone] → Frame Features
        ↓
[RNN Layer] → Temporal Features  
        ↓
[Classifier] → Phase Predictions
```

## 🏗️ Available Models

### **Core Implementations**

| File | CNN Backbone | RNN Type | Description |
|------|--------------|----------|-------------|
| `resnet50_lstm.py` | ResNet-50 | LSTM | ResNet-50 features with LSTM temporal modeling |
| `resnet50_gru.py` | ResNet-50 | GRU | ResNet-50 features with GRU temporal modeling |
| `efficientnet_b5_lstm.py` | EfficientNet-B5 | LSTM | EfficientNet-B5 features with LSTM temporal modeling |
| `efficientnet_b5_gru.py` | EfficientNet-B5 | GRU | EfficientNet-B5 features with GRU temporal modeling |
| `advanced_tecno_models.py` | Various | TeCNO | Advanced TeCNO (Temporal Convolutional Network) implementations |

### **Model Specifications**

#### **ResNet-50 + LSTM/GRU**
```python
# Architecture Details
CNN_BACKBONE = "ResNet-50"
FEATURE_DIM = 2048
RNN_HIDDEN_SIZE = 512
RNN_LAYERS = 2
SEQUENCE_LENGTH = 16
NUM_CLASSES = 13  # Surgical phases
```

#### **EfficientNet-B5 + LSTM/GRU**
```python
# Architecture Details  
CNN_BACKBONE = "EfficientNet-B5"
FEATURE_DIM = 2048
RNN_HIDDEN_SIZE = 512
RNN_LAYERS = 2
SEQUENCE_LENGTH = 16
NUM_CLASSES = 13  # Surgical phases
```

## 🎯 Performance Metrics

### **Phase Recognition Results**

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|---------|
| **EfficientNet-B5 + GRU** | **82.1%** | **71.3%** | **76.0%** | **70.4%** |
| **EfficientNet-B5 + TeCNO** | **81.7%** | **71.2%** | **75.1%** | **71.2%** |
| **EfficientNet-B5 + LSTM** | **81.5%** | **70.0%** | **76.4%** | **69.4%** |
| **ResNet-50 + LSTM** | **79.2%** | **68.5%** | **73.8%** | **67.1%** |
| **ResNet-50 + GRU** | **78.9%** | **67.8%** | **73.2%** | **66.5%** |

### **Computational Efficiency**

| Model | Parameters | FLOPs | Inference Time |
|-------|------------|-------|----------------|
| ResNet-50 + LSTM | 45.2M | 8.1G | 85ms |
| ResNet-50 + GRU | 43.8M | 7.8G | 82ms |
| EfficientNet-B5 + LSTM | 52.3M | 6.2G | 92ms |
| EfficientNet-B5 + GRU | 50.9M | 5.9G | 89ms |

## 🔧 Model Components

### **CNN Backbone Features**

#### **ResNet-50**
- **Strengths**: Proven architecture, good gradient flow, robust feature extraction
- **Use Case**: Baseline models, comparison studies, resource-constrained environments
- **Features**: 2048-dimensional feature vectors per frame

#### **EfficientNet-B5**
- **Strengths**: Superior efficiency, advanced compound scaling, mobile-friendly
- **Use Case**: Production deployment, edge computing, high-accuracy requirements
- **Features**: 2048-dimensional feature vectors with improved representation

### **RNN Temporal Modeling**

#### **LSTM (Long Short-Term Memory)**
```python
class LSTMTemporalModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=2):
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
```

**Advantages:**
- Strong long-term memory capabilities
- Effective gradient flow through time
- Proven performance on sequential data

#### **GRU (Gated Recurrent Unit)**
```python
class GRUTemporalModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=2):
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
```

**Advantages:**
- Faster training and inference
- Fewer parameters than LSTM
- Often comparable performance

### **Advanced TeCNO Models**

The `advanced_tecno_models.py` implements Temporal Convolutional Networks with:

- **Causal Convolutions**: Ensure temporal causality
- **Dilated Convolutions**: Capture long-range dependencies
- **Residual Connections**: Improve gradient flow
- **Multi-scale Features**: Hierarchical temporal representation

## 🚀 Usage

### **Model Initialization**

```python
from models.cnn_rnn_models import EfficientNetB5GRU

# Initialize model
model = EfficientNetB5GRU(
    num_classes=13,
    sequence_length=16,
    hidden_size=512,
    num_layers=2,
    dropout=0.3
)

# Load pretrained weights
model.load_state_dict(torch.load('efficientnet_b5_gru_best.pth'))
```

### **Training Configuration**

```python
# Training hyperparameters
config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 100,
    'sequence_length': 16,
    'augmentation': True,
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR'
}
```

### **Inference Pipeline**

```python
# Video sequence inference
def predict_phases(model, video_frames):
    # Preprocess frames
    frames = preprocess_video_sequence(video_frames)
    
    # Extract CNN features
    features = model.extract_features(frames)
    
    # Temporal modeling
    phase_logits = model.temporal_classifier(features)
    
    # Get predictions
    predictions = torch.softmax(phase_logits, dim=-1)
    return predictions
```

## 📊 Training Details

### **Data Preparation**

- **Sequence Length**: 16 frames (optimal balance of context and efficiency)
- **Frame Sampling**: Uniform sampling from surgical video clips
- **Augmentation**: Random crops, color jittering, temporal shifts
- **Normalization**: ImageNet statistics for CNN backbones

### **Training Strategy**

1. **Two-Stage Training**: 
   - Stage 1: Freeze CNN, train RNN layers
   - Stage 2: End-to-end fine-tuning with reduced learning rate

2. **Loss Function**: Cross-entropy with class balancing for surgical phase distribution

3. **Regularization**:
   - Dropout in RNN layers (0.3)
   - Weight decay (1e-5)
   - Gradient clipping (max_norm=1.0)

### **Optimization**

- **Optimizer**: AdamW with decoupled weight decay
- **Learning Rate Schedule**: Cosine annealing with warm restarts
- **Early Stopping**: Patience of 10 epochs on validation F1-score

## 🔍 Model Analysis

### **Temporal Attention Visualization**

The models support attention weight visualization to understand temporal focus:

```python
# Extract attention weights
attention_weights = model.get_attention_weights(video_sequence)

# Visualize temporal attention
plot_temporal_attention(attention_weights, phase_labels)
```

### **Feature Analysis**

- **Spatial Features**: CNN backbone features capture surgical instruments and anatomical structures
- **Temporal Features**: RNN features model phase transitions and temporal patterns
- **Phase Transitions**: Models excel at detecting gradual phase changes

### **Error Analysis**

Common challenges and model behavior:

- **Phase Boundaries**: Difficulty with ambiguous transition periods
- **Similar Phases**: Confusion between visually similar phases (e.g., different irrigation phases)
- **Temporal Context**: Performance improves with longer sequence context

## 🎯 Best Practices

### **Model Selection**

- **High Accuracy**: Use EfficientNet-B5 + GRU for best performance
- **Fast Inference**: Choose ResNet-50 + GRU for speed-critical applications
- **Memory Constrained**: Consider TeCNO models for efficiency

### **Training Tips**

1. **Sequence Length**: Start with 16 frames, experiment with 8-32 range
2. **Batch Size**: Use gradient accumulation if GPU memory is limited
3. **Learning Rate**: Start with 1e-4, reduce by 10x if training plateaus
4. **Validation**: Use temporal validation splits (not random) for realistic evaluation

### **Deployment Considerations**

- **Model Size**: Consider quantization for mobile deployment
- **Inference Speed**: Profile models with realistic input sizes
- **Memory Usage**: Monitor GPU memory for batch processing
- **Preprocessing**: Optimize video decoding and frame extraction pipelines

---

*These CNN-RNN hybrid models provide state-of-the-art surgical phase recognition capabilities, combining the spatial understanding of CNNs with the temporal modeling power of RNNs.*
