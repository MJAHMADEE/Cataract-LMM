# 🏥 Surgical Skill Assessment Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![Framework](https://img.shields.io/badge/Framework-Production%20Ready-green)](https://github.com)
[![Research](https://img.shields.io/badge/Research-Cataract%20LMM-purple)](https://github.com)

A comprehensive framework for automated surgical skill assessment using deep learning video analysis. This implementation follows the methodology described in the Cataract-LMM research paper for evaluating surgical competency through capsulorhexis video analysis.

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🔧 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📊 Dataset Format](#-dataset-format)
- [🏗️ Architecture](#-architecture)
- [⚙️ Configuration](#-configuration)
- [🔬 Model Architectures](#-model-architectures)
- [📈 Training](#-training)
- [🔍 Evaluation](#-evaluation)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)

## 🎯 Overview

This framework implements automated surgical skill assessment using video analysis of cataract surgery procedures, specifically focusing on the capsulorhexis phase. The system uses multiple deep learning architectures to classify surgical skill into binary categories (lower-skilled vs. higher-skilled) based on the methodology described in the Cataract-LMM research paper.

### Key Features

- **🎬 Multi-Architecture Support**: 11+ deep learning models including CNN, CNN-RNN, and Transformer architectures
- **📊 Binary Skill Classification**: K-means clustering approach for skill level categorization
- **🔄 Advanced Data Processing**: Comprehensive video preprocessing and augmentation pipelines
- **⚡ Mixed Precision Training**: GPU-optimized training with automatic mixed precision
- **📈 Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualization tools
- **🎛️ Dynamic Batch Sizing**: Automatic batch size optimization based on available GPU memory
- **🔧 Flexible Configuration**: YAML-based configuration management
- **📝 Extensive Logging**: Rich console output and comprehensive logging

### Research Context

This implementation is based on the Cataract-LMM dataset methodology with:
- **170 video clips** from capsulorhexis procedures
- **Binary skill classification** using K-means clustering
- **Lower-skilled group**: n=63, mean score = 3.12 ± 0.38
- **Higher-skilled group**: n=107, mean score = 4.24 ± 0.37
- **Cataract-LMM naming convention**: `SK_<ClipID>_S<Site>_P03.mp4`

## 🏆 Performance Benchmarks

Based on Cataract-LMM evaluation for binary skill classification:

| **Model** | **Accuracy (%)** | **Precision (%)** | **Recall (%)** | **F1-Score (%)** |
|-----------|------------------|-------------------|----------------|------------------|
| **TimeSformer** ⭐ | **82.50** | **86.00** | 82.00 | **83.90** |
| **R3D-18** | 81.67 | 82.35 | **84.85** | 83.58 |
| **Slow R50** | 80.00 | 81.82 | 81.82 | 81.82 |
| **X3D-M** | 80.00 | 83.87 | 78.79 | 81.25 |
| **R(2+1)D-18** | 72.92 | 79.31 | 76.67 | 77.97 |
| **CNN-LSTM** | 61.67 | 70.97 | 66.67 | 68.75 |
| **CNN-GRU** | 54.17 | 60.00 | 80.00 | 68.57 |

**🎯 Key Insights:**
- **TimeSformer** achieves the highest accuracy (82.5%) and F1-score (83.9%)
- **R3D-18** provides the best recall (84.85%) for identifying all skill levels
- **Transformer-based models** (TimeSformer) outperform traditional CNN architectures
- **3D CNN approaches** show strong performance for temporal skill analysis

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for full dataset processing
- 50GB+ storage for models and datasets

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/your-repo/cataract-lmm.git
cd cataract-lmm/codes/surgical-skill-assessment
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Basic Training

1. **Prepare your dataset** following the [Dataset Format](#-dataset-format) section

2. **Configure training parameters**:
```bash
cp configs/config.yaml configs/my_config.yaml
# Edit my_config.yaml with your specific settings
```

3. **Start training**:
```bash
python main.py --config configs/my_config.yaml
```

### Inference Only

```bash
python main.py --config configs/my_config.yaml --inference-only --resume checkpoints/best.pth
```

### Simple Training (Lightweight)

```bash
python main_simple.py --config configs/config.yaml
```

## 📊 Dataset Format

### Cataract-LMM Dataset Structure

```
data/
├── videos/
│   ├── SK_0001_S1_P03.mp4
│   ├── SK_0002_S1_P03.mp4
│   └── ...
├── annotations.csv
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Annotations Format

The `annotations.csv` file should contain:

| Column | Description | Example |
|--------|-------------|---------|
| video_filename | Video file name | SK_0001_S1_P03.mp4 |
| clip_id | Unique clip identifier | 0001 |
| site | Hospital site (1: Farabi, 2: Noor) | 1 |
| overall_score | Skill score (1-5 scale) | 4.2 |
| duration_seconds | Video duration | 45.3 |
| adverse_event | Adverse event indicator | 0 |

### Naming Convention

Videos must follow the Cataract-LMM naming convention:
```
SK_<ClipID>_S<Site>_P03.mp4
```
- `SK`: Skill assessment identifier
- `ClipID`: 4-digit clip identifier (0001-9999)
- `Site`: Hospital site number (1 or 2)
- `P03`: Phase identifier (capsulorhexis)

## 🏗️ Architecture

The framework consists of several modular components:

### Core Modules

```
📦 surgical-skill-assessment/
├── 🎬 data/           # Data processing and loading
├── 🧠 models/         # Neural network architectures
├── ⚙️ engine/         # Training, validation, and evaluation
├── 🛠️ utils/          # Helper utilities and tools
├── 📋 configs/        # Configuration files
├── 📓 notebooks/      # Jupyter notebooks for development
└── 🧪 tests/          # Unit and integration tests
```

### Data Flow

```mermaid
graph LR
    A[Video Files] --> B[Data Splitter]
    B --> C[Video Dataset]
    C --> D[Data Loaders]
    D --> E[Model Training]
    E --> F[Evaluation]
    F --> G[Results & Plots]
```

## ⚙️ Configuration

Configuration is managed through YAML files in the `configs/` directory:

### Main Configuration (`config.yaml`)

```yaml
# Data configuration
data:
  data_root: "data/videos"
  annotations_file: "data/annotations.csv"
  clip_len: 100
  image_size: [224, 224]
  fps: 10

# Model configuration  
model:
  model_name: "x3d_m"
  num_classes: 2
  pretrained: true

# Training configuration
train:
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 50
  optimizer: "adamw"
  scheduler: "cosine"
  mixed_precision: true
  
# Hardware configuration
hardware:
  gpus: 1
  num_workers: 4
```

### Available Configurations

- `config.yaml`: Default configuration
- `fast_training.yaml`: Reduced epochs for quick testing
- `high_memory.yaml`: Optimized for high-memory systems
- `cpu_only.yaml`: CPU-only training configuration

## 🔬 Model Architectures

The framework supports 11+ state-of-the-art video classification architectures:

### CNN-based Models
- **X3D-M**: Efficient 3D CNN for video understanding
- **SlowFast**: Dual-pathway architecture for temporal modeling
- **Slow R50**: Single-pathway slow architecture

### CNN-RNN Hybrid Models
- **ResNet + LSTM**: Spatial CNN features with temporal LSTM
- **ResNet + GRU**: Spatial CNN features with temporal GRU

### Transformer-based Models
- **TimeSformer**: Divided attention for video classification
- **MViT**: Multiscale Vision Transformer
- **ViViT**: Video Vision Transformer
- **VideoMAE**: Masked autoencoder for video

### Model Selection

```python
# Available models
AVAILABLE_MODELS = [
    'x3d_m', 'slowfast_r50', 'slow_r50',
    'resnet_lstm', 'resnet_gru',
    'timesformer', 'mvit_base_16x4', 'vivit_base_16x2',
    'videomae_vitb_16x4', 'videomae_vitl_16x4'
]
```

## 📈 Training

### Training Process

1. **Data Preparation**: Videos are processed into fixed-length clips
2. **Skill Classification**: K-means clustering for binary skill labels
3. **Model Training**: Multi-epoch training with validation
4. **Dynamic Optimization**: Automatic batch size and memory optimization
5. **Early Stopping**: Prevents overfitting with patience-based stopping
6. **Checkpointing**: Automatic model saving and resuming

### Training Outputs

```
outputs/
├── 📊 plots/
│   ├── training_progress.png
│   ├── confusion_matrix.png
│   └── skill_distribution.png
├── 💾 checkpoints/
│   ├── best.pth
│   ├── last.pth
│   └── epoch_*.pth
├── 📋 logs/
│   ├── training.log
│   ├── config.yaml
│   └── metrics.json
└── 📄 run_summary.json
```

### Monitoring Training

Training progress can be monitored through:
- **Rich Console Output**: Real-time progress bars and metrics
- **Log Files**: Detailed training logs in `outputs/logs/`
- **Plots**: Automatic generation of training curves and confusion matrices
- **Metrics JSON**: Structured metric data for analysis

## 🔍 Evaluation

### Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **AUC-ROC**: Area under receiver operating characteristic curve

### Evaluation Reports

```python
# Example evaluation output for TimeSformer (SOTA)
{
    "accuracy": 0.825,
    "precision": 0.850,
    "recall": 0.827,
    "f1_score": 0.839,
    "specificity": 0.822,
    "kappa": 0.782,
    "auc_roc": 0.882,
    "confusion_matrix": [[370, 80], [95, 455]]
}
```

### Custom Evaluation

```bash
# Evaluate specific model
python -m engine.evaluator --model checkpoints/best.pth --data data/test/

# Generate detailed analysis
python -m notebooks.analysis --results outputs/results.json
```

## 📁 Project Structure

```
📦 surgical-skill-assessment/
├── 📖 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package setup
├── 🐍 main.py                      # Main training script (comprehensive)
├── 🐍 main_simple.py               # Simplified training script
│
├── 📋 configs/                     # Configuration files
│   ├── 📄 README.md
│   ├── ⚙️ config.yaml              # Default configuration
│   ├── ⚙️ fast_training.yaml       # Quick training config
│   └── ⚙️ high_memory.yaml         # High-memory config
│
├── 🎬 data/                        # Data processing module
│   ├── 📄 README.md
│   ├── 🐍 __init__.py
│   ├── 🐍 dataset.py               # PyTorch datasets
│   ├── 🐍 loaders.py               # DataLoader utilities
│   ├── 🐍 splits.py                # Data splitting logic
│   ├── 🐍 transforms.py            # Video augmentations
│   └── 🐍 utils.py                 # Data utilities
│
├── 🧠 models/                      # Model architectures
│   ├── 📄 README.md
│   ├── 🐍 __init__.py
│   └── 🐍 factory.py               # Model factory
│
├── ⚙️ engine/                      # Training and evaluation
│   ├── 📄 README.md
│   ├── 🐍 __init__.py
│   ├── 🐍 trainer.py               # Training logic
│   ├── 🐍 evaluator.py             # Evaluation utilities
│   └── 🐍 predictor.py             # Inference utilities
│
├── 🛠️ utils/                       # Helper utilities
│   ├── 📄 README.md
│   ├── 🐍 __init__.py
│   └── 🐍 helpers.py               # Common utilities
│
├── 📓 notebooks/                   # Development notebooks
│   ├── 📄 README.md
│   └── 📔 video_classification_prototype.ipynb
│
└── 🧪 tests/                       # Test suite
    ├── 📄 README.md
    ├── 🐍 __init__.py
    ├── 🐍 test_data.py
    ├── 🐍 test_models.py
    └── 🐍 test_training.py
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_data.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Memory and speed benchmarks
- **Data Tests**: Dataset validation and integrity checks

## 📚 Documentation

### API Documentation

Generate comprehensive API documentation:

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Generate documentation
cd docs/
make html
```

### Jupyter Notebooks

Explore the framework through interactive notebooks:

- `notebooks/video_classification_prototype.ipynb`: Core implementation walkthrough
- `notebooks/data_analysis.ipynb`: Dataset analysis and visualization
- `notebooks/model_comparison.ipynb`: Architecture comparison studies

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black .
isort .
flake8 .
```

### Contribution Areas

- 🐛 Bug fixes and improvements
- 🚀 New model architectures
- 📊 Enhanced evaluation metrics
- 📖 Documentation improvements
- 🧪 Additional test coverage

## 🙏 Acknowledgments

- PyTorch and Torchvision teams
- Medical professionals who provided domain expertise
- Open-source community for foundational tools

---

<div align="center">

**Built with ❤️ for advancing surgical education through AI**

</div>
