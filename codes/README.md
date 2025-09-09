# 🏥 Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark

**Professional-grade implementation of the Cataract-LMM research framework for deep learning in surgical video analysis.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code Quality](https://img.shields.io/badge/code%20quality-professional-green.svg)]()

> **Research Paper**: "Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"

## 🎯 Overview

This repository contains the complete implementation of the Cataract-LMM framework, providing state-of-the-art deep learning tools for comprehensive surgical video analysis. The framework supports four core tasks aligned with the research paper methodology:

- **🎬 Surgical Video Processing**: Hospital-specific video preprocessing and quality control
- **🔍 Instrument Segmentation**: Instance segmentation of 10 surgical instruments + anatomical structures
- **⏱️ Phase Classification**: Automated recognition of 13 surgical phases
- **📊 Skill Assessment**: Objective surgical skill evaluation using 6-indicator rubric

## 📋 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/MJAHMADEE/Cataract-LMM.git
cd Cataract-LMM/codes

# Run comprehensive setup
python setup.py

# Or install dependencies manually
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration to match your data paths
nano .env
```

### 3. Quick Test

```bash
# Test video processing
cd surgical-video-processing
python main.py --help

# Test skill assessment
cd ../surgical-skill-assessment  
python main.py --help
```

## 🏗️ Project Structure

```
codes/
├── 📁 surgical-video-processing/    # Hospital-specific video preprocessing
├── 📁 surgical-instance-segmentation/  # YOLO/Mask R-CNN/SAM models
├── 📁 surgical-phase-recognition/     # Temporal workflow recognition
├── 📁 surgical-skill-assessment/         # Video-based skill evaluation
├── 📄 requirements.txt                   # Unified dependencies
├── 📄 .env.example                      # Environment template
├── 📄 .gitignore                        # Git ignore rules
└── 📄 setup.py                         # Master setup script
```

## 🎯 Core Modules

### 🎬 Surgical Video Processing
**Purpose**: Standardized preprocessing for multi-center video data

**Key Features**:
- Hospital-specific configurations (Farabi S1: 720×480@30fps, Noor S2: 1920×1080@60fps)
- Quality control and validation pipeline
- FFmpeg-based compression with GPU acceleration
- Metadata extraction and de-identification

**Usage**:
```bash
cd surgical-video-processing

# Process single video
python main.py --input video.mp4 --output processed.mp4 --hospital farabi

# Batch processing
python main.py --batch --input-dir ./videos --output-dir ./processed
```

### 🔍 Instrument Segmentation  
**Purpose**: Instance segmentation of surgical instruments and anatomical structures

**Supported Models**:
- **YOLOv8/YOLOv11**: Real-time segmentation
- **Mask R-CNN**: High-precision instance segmentation  
- **SAM/SAM2**: Zero-shot segmentation capabilities

**Classes**: 10 instruments + 2 anatomical structures (12 total)

**Usage**:
```bash
cd surgical-instance-segmentation

# Train YOLO model
python -m training.train_yolo --config configs/yolo_config.yaml

# Run inference
python -m inference.predict --model models/best.pt --input image.jpg
```

### ⏱️ Phase Classification
**Purpose**: Automated recognition of 13 surgical phases

**Supported Architectures**:
- **Hybrid Models**: ResNet/EfficientNet + LSTM/GRU/TeCNO
- **3D CNNs**: SlowFast, X3D, R(2+1)D, MC3, R3D  
- **Video Transformers**: MViT, Video Swin Transformer

**Domain Adaptation**: Cross-hospital evaluation (Farabi ↔ Noor)

**Usage**:
```bash
cd surgical-phase-recognition

# Train model
python -m validation.training_framework --config configs/default.yaml

# Evaluate with domain adaptation
python -m validation.evaluate --model checkpoints/best.pth --test-domain noor
```

### 📊 Skill Assessment
**Purpose**: Objective surgical skill evaluation using video analysis

**Assessment Framework**:
- **6-Indicator Rubric**: Instrument handling, motion economy, tissue handling, etc.
- **Video Classification**: Binary (higher/lower skill) and multi-class
- **Motion Analysis**: Instrument trajectory and kinematic features

**Supported Models**: TimeSformer, R3D, X3D, CNN-LSTM/GRU combinations

**Usage**:
```bash
cd surgical-skill-assessment

# Train skill classifier
python main.py --config configs/config.yaml

# Comprehensive evaluation
python main_comprehensive.py --config configs/config.yaml
```

## 📊 Dataset Information

The framework is designed to work with the **Cataract-LMM Dataset**:

| Component | Videos/Frames | Description |
|-----------|---------------|-------------|
| **Raw Videos** | 3,000 videos | Multi-center phacoemulsification procedures |
| **Phase Recognition** | 150 videos | 13-phase temporal annotations |
| **Instance Segmentation** | 6,094 frames | Pixel-wise masks for 12 classes |
| **Instrument Tracking** | 170 clips | Dense spatiotemporal annotations |
| **Skill Assessment** | 170 clips | 6-indicator skill ratings |

**Hospital Sources**:
- **Farabi Hospital (S1)**: 2,800 videos (720×480@30fps)
- **Noor Hospital (S2)**: 200 videos (1920×1080@60fps)

## 🔧 Technical Requirements

### System Requirements
- **Python**: 3.8+
- **CUDA**: 11.0+ (recommended for GPU acceleration)
- **Memory**: 16GB+ RAM, 8GB+ VRAM for training
- **Storage**: 50GB+ for models and processed data

### Key Dependencies
- **Deep Learning**: PyTorch 1.12+, TorchVision 0.13+
- **Computer Vision**: OpenCV, Ultralytics YOLO, Detectron2
- **Video Processing**: FFmpeg, PyTorchVideo, Decord
- **Scientific**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly

## 🚀 Advanced Usage

### Environment Configuration
```bash
# Set up environment variables
export PROJECT_ROOT=/path/to/Cataract_LMM/codes
export DATA_ROOT=/path/to/Cataract-LMM-Dataset  
export CUDA_VISIBLE_DEVICES=0
```

### Multi-GPU Training
```bash
# Phase classification with multiple GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    surgical-phase-recognition/validation/training_framework.py \
    --config configs/distributed.yaml
```

### Custom Dataset Integration
```bash
# Convert your dataset to Cataract-LMM format
python scripts/convert_dataset.py \
    --input /path/to/your/dataset \
    --output /path/to/converted \
    --format cataract_lmm
```

## 📈 Performance Benchmarks

Based on the research paper evaluation:

| Task | Best Model | Metric | Performance |
|------|------------|--------|-------------|
| **Phase Recognition** | MViT-B | Macro F1 | 77.1% (in-domain) |
| **Instance Segmentation** | YOLOv11-L | mask mAP | 73.9% |
| **Skill Assessment** | TimeSformer | F1-Score | 83.9% |

**Domain Adaptation**: ~22% average performance drop between hospitals, highlighting the need for robust domain adaptation techniques.

## 🧪 Testing & Validation

### Run Complete Test Suite
```bash
# Validate entire project
python setup.py --validate-only

# Test individual modules
cd surgical-skill-assessment
python -m pytest tests/ -v
```

### Benchmark Performance
```bash
# Video processing benchmarks
cd surgical-video-processing
python scripts/utilities.py benchmark --test-video sample.mp4
```

## 📚 Documentation

Each module contains comprehensive documentation:

- **📖 Module READMEs**: Detailed usage instructions
- **📋 Configuration Guides**: Parameter explanations
- **🔬 Research Alignment**: Implementation details matching the paper
- **📊 Evaluation Protocols**: Benchmarking methodologies

## 🤝 Contributing

1. **Code Quality**: All code follows PEP 8 standards with comprehensive docstrings
2. **Testing**: New features require corresponding test cases
3. **Documentation**: Update relevant README files for changes
4. **Research Alignment**: Ensure modifications align with paper methodology

## 📄 Citation

For citation information, please refer to the [main repository README](../README.md#-citation).

## 📜 License

This project is licensed under [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## 🙏 Acknowledgments

- **Farabi Eye Hospital & Noor Eye Hospital**: Data collection and clinical expertise
- **K.N. Toosi University of Technology**: Research infrastructure
- **Tehran University of Medical Sciences**: Medical oversight
- **University of Alberta**: Technical collaboration

---

**🏥 Built for advancing surgical AI through rigorous research and professional-grade implementation.**
