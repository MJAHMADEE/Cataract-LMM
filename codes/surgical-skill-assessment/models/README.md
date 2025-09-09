# 🧠 Neural Network Architectures

The models module provides a comprehensive collection of state-of-the-art neural network architectures specifically designed for video-based surgical skill assessment. This module implements cutting-edge deep learning models optimized for spatiotemporal feature extraction and temporal pattern recognition.

## 🏗️ Architecture Overview

### 🎯 Model Categories

#### **🔄 Hybrid CNN-RNN Models** (`cnn_rnn.py`)
Advanced spatiotemporal architectures combining CNN spatial feature extraction with RNN temporal modeling:

#### **🚀 Transformer-Based Models** (`transformers.py`)
State-of-the-art attention-based architectures for video understanding:

#### **🏭 Model Factory** (`factory.py`)
Centralized model creation and configuration system:

## 🔬 Detailed Architecture Specifications

### 🎬 CNN-RNN Hybrid Models (`cnn_rnn.py`)

#### **`CNNFeatureExtractor`** - Spatial Feature Extraction
```python
# ResNet-based backbone with customizable depth
- Architecture: ResNet18/34/50/101/152
- Input: Individual video frames (3 × H × W)
- Output: High-dimensional feature vectors (512/2048-dim)
- Features:
  ✅ Transfer learning with ImageNet pretraining
  ✅ Configurable feature dimension output
  ✅ Batch normalization and dropout regularization
  ✅ Efficient GPU memory utilization
```

#### **`CNN_LSTM`** - Spatiotemporal LSTM Network
```python
# Bidirectional LSTM for temporal modeling
- Input: Sequential CNN features (T × Feature_Dim)
- Architecture: Bidirectional LSTM layers
- Hidden Dimensions: Configurable (256/512/1024)
- Output: Binary classification (lower/higher skill)
- Features:
  ✅ Bidirectional temporal context
  ✅ Gradient clipping for training stability
  ✅ Dropout layers for regularization
  ✅ Attention mechanism (optional)
```

#### **`CNN_GRU`** - Spatiotemporal GRU Network
```python
# Efficient GRU alternative to LSTM
- Architecture: Bidirectional GRU layers
- Computational Efficiency: ~25% faster than LSTM
- Memory Usage: ~20% less memory than LSTM
- Performance: Comparable accuracy with faster inference
- Features:
  ✅ Simplified gating mechanism
  ✅ Reduced parameter count
  ✅ Faster convergence
  ✅ Mobile-friendly deployment
```

### 🔮 Transformer Architectures (`transformers.py`)

#### **`MViT`** - Multiscale Vision Transformer
```python
# Hierarchical spatiotemporal attention
- Architecture: Multiscale attention blocks
- Spatial Scales: Multiple resolution processing
- Temporal Modeling: Factorized spatiotemporal attention
- Input Resolution: Adaptive (224×224 to 384×384)
- Features:
  ✅ Multi-scale feature learning
  ✅ Efficient attention computation
  ✅ State-of-the-art accuracy
  ✅ Scalable architecture
```

#### **`VideoMAE`** - Video Masked Autoencoder
```python
# Self-supervised pretraining + classification
- Pretraining: Masked autoencoder reconstruction
- Architecture: ViT backbone with specialized decoder
- Masking Strategy: High masking ratio (75-90%)
- Fine-tuning: Classification head adaptation
- Features:
  ✅ Strong representation learning
  ✅ Data-efficient training
  ✅ Robust to domain shift
  ✅ Excellent transfer learning
```

#### **`ViViT`** - Video Vision Transformer
```python
# Factorized spatiotemporal attention
- Attention Design: Separated spatial/temporal attention
- Computational Efficiency: O(HWT) instead of O((HWT)²)
- Patch Strategy: Tubelet tokenization
- Position Encoding: 3D learnable embeddings
- Features:
  ✅ Scalable to long sequences
  ✅ Memory-efficient attention
  ✅ Strong temporal modeling
  ✅ High-resolution processing
```

### 🏭 Model Factory (`factory.py`)

#### **`create_model`** - Universal Model Creator
```python
# Centralized model instantiation and configuration
Supported Architectures:
```

#### **📚 PyTorchVideo Models**
- **X3D Family**: `x3d_xs`, `x3d_s`, `x3d_m`, `x3d_l`
  - Efficient 3D CNNs with progressive expansion
  - Optimized for mobile and edge deployment
  - Channel-wise and depth-wise separable convolutions

- **SlowFast Networks**: `slowfast_r50`, `slowfast_r101`
  - Dual-pathway architecture for motion understanding
  - Slow pathway: High spatial resolution, low temporal resolution
  - Fast pathway: Low spatial resolution, high temporal resolution

- **R(2+1)D**: `r2plus1d_r18`, `r2plus1d_r34`, `r2plus1d_r50`
  - Factorized 3D convolutions for efficiency
  - Separated spatial and temporal convolution operations
  - Reduced computational complexity

#### **🎯 Custom Models**
- **CNN-LSTM/GRU**: Hybrid spatiotemporal architectures
- **TimeSformer**: Pure attention-based video model
- **Custom Transformers**: MViT, VideoMAE, ViViT implementations

## ⚙️ Configuration & Deployment

### 🔧 Model Configuration
```yaml
model:
  name: "cnn_lstm"  # Model architecture selection
  backbone: "resnet50"  # Feature extractor backbone
  num_classes: 2  # Binary skill classification
  hidden_dim: 512  # RNN hidden dimensions
  num_layers: 2  # Number of RNN/Transformer layers
  dropout: 0.3  # Regularization strength
  freeze_backbone: true  # Transfer learning strategy
  pretrained: true  # Use pretrained weights
```

### 🚀 Advanced Features

#### **Transfer Learning Support**
- **Backbone Freezing**: Selective layer freezing for efficient fine-tuning
- **Progressive Unfreezing**: Gradual unfreezing strategies
- **Layer-wise Learning Rates**: Differential learning rates for different layers
- **Domain Adaptation**: Specialized techniques for surgical video domain

#### **Model Optimization**
- **Quantization**: INT8 quantization for deployment efficiency
- **Pruning**: Structured and unstructured pruning techniques
- **Knowledge Distillation**: Teacher-student training for model compression
- **ONNX Export**: Cross-platform deployment support

#### **Memory Management**
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Mixed Precision**: FP16 training for speed and memory optimization
- **Dynamic Batching**: Adaptive batch sizing based on sequence length
- **Model Parallelism**: Multi-GPU model distribution

##  Integration Examples

### Model Creation
```python
from models.factory import create_model

# Create CNN-LSTM model
model = create_model(
    model_name="cnn_lstm",
    num_classes=2,
    config=config
)

# Create Transformer model
model = create_model(
    model_name="mvit_base",
    num_classes=2,
    config=config
)
```

### Custom Architecture
```python
from models.cnn_rnn import CNN_LSTM
from models.transformers import MViT

# Direct model instantiation
model = CNN_LSTM(
    num_classes=2,
    hidden_dim=512,
    backbone="resnet50"
)
```

This models module provides the architectural foundation for robust, efficient, and accurate surgical skill assessment through advanced deep learning techniques.
