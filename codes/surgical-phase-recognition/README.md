# 🏥 Surgical Phase Recognition Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

A comprehensive, production-ready framework for surgical phase recognition using deep learning, implementing the exact methodologies from the **"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"** academic paper.

## 🎯 Overview

This framework provides a complete solution for surgical phase recognition aligned with the Cataract-LMM benchmark specifications. It implements state-of-the-art architectures for automatic recognition of surgical phases in cataract surgery videos with exact paper compliance for reproducible research.

### ✨ Core Logic Reference

The **authoritative implementation** and core algorithms are located in the **`notebooks/`** directory, which contains the comprehensive research implementation from the Cataract-LMM paper. This production framework serves as:

- 🔧 **Modular utilities** for the core notebook research pipeline
- 📊 **Scalable data processing** components for large-scale evaluation  
- 🧪 **Testing infrastructure** to validate paper compliance
- 🚀 **Production deployment** tools for real-world applications

> **🔬 Paper Compliance**: Every component implements exact methodologies from Section 4 ("Experiments Methodology") of the Cataract-LMM academic paper for benchmark reproducibility.

### 📊 Academic Paper Alignment

This framework implements the **"Phase Recognition Dataset Description"** methodology with:

- ✅ **13-phase surgical taxonomy** exactly as defined in the paper
- ✅ **Multi-center evaluation** (Farabi vs Noor hospitals) 
- ✅ **Domain adaptation analysis** following Section 4.5
- ✅ **150 annotated videos** with 28.6 hours of surgical footage
- ✅ **Paper-validated metrics** (Macro F1-score, per-phase accuracy)

## 🏗️ Architecture

### 🎯 Surgical Phase Labels (Cataract-LMM Taxonomy)

**13-Phase Complete Taxonomy** (CATARACT_LMM_PHASES_13):
```python
CATARACT_LMM_PHASES_13 = {
    "Incision": 0,                     # Initial corneal incision
    "Viscoelastic": 1,                 # Viscoelastic agent injection  
    "Rhexis": 2,                       # Capsulorhexis (paper uses "Rhexis")
    "Hydrodissection": 3,              # Cortex-lens separation
    "Phacoemulsification": 4,          # Ultrasonic lens fragmentation
    "Aspiration": 5,                   # Cortex removal (I/A phase)
    "Capsule Polishing": 6,            # Posterior capsule cleaning
    "Lens Insertion": 7,               # Intraocular lens implantation
    "Viscoelastic Removal": 8,         # Viscoelastic material removal
    "Anterior Chamber Flushing": 9,    # Chamber irrigation
    "Tonifying-Antibiotics": 10,       # Final medication instillation
    "Idle": 11,                        # Surgical inactivity/instrument exchange
    "Wound Suturing": 12,              # Final wound closure
}
```

**11-Phase Evaluation Taxonomy** (CATARACT_LMM_PHASES_11):
Per paper methodology: *"visually similar and underrepresented phases, Viscoelastic and Anterior Chamber Flushing, were merged into a single class"*

### 🧠 Model Architectures (Paper Section 4.5)

**Following Cataract-LMM benchmark evaluation methodology:**

#### 📈 **Paper-Validated Performance** (Table 8 Results)
- **🏆 MViT-B**: 77.1% F1-score (best performing)
- **🎬 Swin-T**: 76.2% F1-score (video transformer)
- **🔗 CNN+GRU (EfficientNet-B5)**: 71.3% F1-score (hybrid)
- **🎥 Slow R50**: 69.8% F1-score (3D CNN)

#### **Two-Stage Framework** (CNN Backbone + Temporal)
- **CNN Backbones**: ResNet50, EfficientNet-B5 (ImageNet pre-trained)
- **Temporal Models**: LSTM, GRU, TeCNO Multi-scale Networks
- **Hybrid Approach**: Decoupled spatial-temporal learning

#### **End-to-End Video Models** (Kinetics-400 Pre-trained)
- **3D CNNs**: SlowFast, X3D, R(2+1)D, MC3, R3D  
- **Vision Transformers**: MViT, Video Swin Transformer
- **Direct Video Processing**: Full spatiotemporal modeling

#### **Multi-Stage Architectures** (TeCNO Framework)
- **TeCNO**: Multi-scale Temporal Convolutional Networks
- **Hierarchical Models**: Multi-stage temporal refinement  
- **Advanced Consistency**: Surgical workflow temporal modeling

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd surgical_phase_recognition

# Install dependencies
pip install torch torchvision
pip install pytorch-lightning torchmetrics  # Optional for full functionality
pip install opencv-python pillow pandas matplotlib seaborn  # Optional
```

### Basic Usage
```python
# Import core components (exact notebook compatibility)
from transform import transform_train, transform_test
from data.sequential_dataset import SequentialSurgicalPhaseDatasetAugOverlap
from models.individual import Resnet50_LSTM, EfficientNetB5_LSTM
from models.tecno import MultiStageModel
from validation.comprehensive_validator import validate_model

# Create dataset (matches notebook usage)
dataset = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=label_mapping,
    root_dir="/path/to/data", 
    lookback_window=10,
    max_sequences_per_phase=1500,
    overlap=0,
    frame_interval=1,
    test=True
)

# Initialize models (exact notebook parameters)
resnet_lstm = Resnet50_LSTM(num_classes=11, hidden_seq=256, dropout=0.5, hidden_MLP=128)

# TeCNO model (exact notebook usage)
import torchvision.models as models
resnet_backbone = models.resnet50()
tecno_model = MultiStageModel(resnet_backbone, 6, 1, 256, 2048, num_classes=11)

# Validate model (matches notebook validation)
results = validate_model("resnet50_lstm", checkpoint_path, resnet_lstm, test_loader, num_classes=11)
```

## 📁 Project Structure

```
surgical_phase_recognition/
├── 📋 README.md                    # This file - main documentation
├── 🔧 transform.py                 # Transform module (notebook compatible)
├── 📊 data/                        # Data handling and preprocessing
│   ├── sequential_dataset.py       # SequentialSurgicalPhaseDatasetAugOverlap
│   ├── datasets.py                 # Additional dataset classes
│   └── data_utils.py               # Data processing utilities
├── 🧠 models/                      # Model architectures
│   ├── individual/                 # Individual model implementations
│   ├── tecno.py                    # MultiStageModel (TeCNO) - notebook compatible
│   ├── multistage_models.py        # Advanced multi-stage architectures
│   ├── cnn_3d_models.py           # 3D CNN implementations
│   ├── video_transformers.py       # Video transformer architectures
│   └── cnn_rnn_hybrids.py         # CNN-RNN hybrid models
├── ✅ validation/                  # Model validation and evaluation
│   ├── comprehensive_validator.py  # ValidationModule & validate_model
│   ├── metrics.py                  # Evaluation metrics
│   └── training_framework.py       # Training utilities
├── ⚙️ preprocessing/               # Data preprocessing
├── 📈 analysis/                    # Analysis and visualization tools
├── ⚙️ configs/                     # Configuration files
├── 🛠️ utils/                       # Utility functions
├── 📓 notebooks/                   # Reference notebook documentation
└── 🧪 tests/                       # Unit tests
```

## 🎯 Key Features

### ✅ Production Ready
- **Modular architecture** with clean separation of concerns
- **Professional code structure** with comprehensive documentation
- **Error handling and logging** throughout the framework
- **Type hints and docstrings** for all major components

### ✅ Research Compatible
- **100% notebook compatibility** - all imports and usage patterns match the reference notebook
- **Exact parameter signatures** for seamless research-to-production transition
- **Comprehensive validation pipeline** matching published research standards

### ✅ Scalable Design
- **Plugin architecture** for easy model addition
- **Configurable training pipelines** for different experimental setups
- **Efficient data loading** with balanced sampling and augmentation
- **Multi-GPU support** through PyTorch Lightning integration

### ✅ Comprehensive Validation
- **Professional metrics** including accuracy, precision, recall, F1-score
- **Confusion matrix visualization** for detailed performance analysis
- **Per-class performance metrics** for surgical phase analysis
- **Model comparison utilities** for benchmarking

## 📊 Performance

The framework supports models evaluated on the Cataract-LMM dataset:

## 📊 Model Performance (In-Domain - Farabi Test Set)

| Model Architecture | Backbone | Accuracy | F1-Score | Precision | Recall |
|---------------------|----------|----------|----------|-----------|---------|
| **MViT-B** ⭐ | - | **85.7%** | **77.1%** | **77.1%** | **78.5%** |
| **Swin-T** | - | **85.5%** | **76.2%** | **77.5%** | **77.2%** |
| **CNN + GRU** | EfficientNet-B5 | **82.1%** | **71.3%** | **76.0%** | **70.4%** |
| **CNN + TeCNO** | EfficientNet-B5 | **81.7%** | **71.2%** | **75.1%** | **71.2%** |
| **CNN + LSTM** | EfficientNet-B5 | **81.5%** | **70.0%** | **76.4%** | **69.4%** |
| **CNN + GRU** | ResNet-50 | **79.8%** | **69.7%** | **70.1%** | **70.5%** |
| **Slow R50** | ResNet-50 | **79.6%** | **69.8%** | **70.7%** | **71.3%** |
| **CNN + LSTM** | ResNet-50 | **78.4%** | **67.0%** | **71.4%** | **66.0%** |
| **MC3-18** | ResNet-18 | **78.8%** | **67.0%** | **71.7%** | **69.6%** |
| **CNN + TeCNO** | ResNet-50 | **77.1%** | **66.9%** | **68.2%** | **69.2%** |
| **R3D-18** | ResNet-18 | **74.5%** | **64.0%** | **67.6%** | **66.6%** |
| **X3D-XS** | - | **73.3%** | **57.1%** | **62.3%** | **58.7%** |
| **R(2+1)D-18** | ResNet-18 | **64.2%** | **54.4%** | **66.6%** | **57.0%** |

## 🔬 Research Integration

This framework is designed to bridge the gap between research exploration and production deployment:

1. **📓 Research Phase**: Use the primary validation notebook for model exploration and validation
2. **🔧 Development Phase**: Leverage the modular framework for production implementation  
3. **🚀 Deployment Phase**: Utilize the professional structure for scalable deployment

### 📊 Primary Notebook Integration
The [`phase_validation_comprehensive.ipynb`](../Validation.ipynb) notebook demonstrates:
- Complete model validation pipeline
- Data loading with `SequentialSurgicalPhaseDatasetAugOverlap`
- Individual model usage (ResNet+LSTM, EfficientNet+GRU, etc.)
- TeCNO multi-stage model implementation
- 3D CNN and Video Transformer evaluation
- Professional visualization and metrics

---

> **💡 Pro Tip**: Start with the [primary validation notebook](../Validation.ipynb) to understand the complete workflow, then use this framework for production implementation!
