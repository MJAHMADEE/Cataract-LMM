# Cataract-LMM: Advanced Surgical Analysis Framework

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-enabled-green.svg)](https://docker.com)
[![Documentation](https://img.shields.io/badge/docs-sphinx-brightgreen.svg)](https://sphinx-doc.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Welcome to Cataract-LMM Documentation

**Cataract-LMM** is a comprehensive, production-grade framework for advanced surgical video analysis, leveraging state-of-the-art machine learning models and computer vision techniques for real-time surgical assistance and assessment.

```{toctree}
:maxdepth: 2
:caption: Contents:

getting_started
index.rst
api_reference
tutorials/index
deployment
contributing
```

## 🔬 Framework Overview

This framework encompasses four specialized modules designed for comprehensive surgical video analysis:

### 🎥 Surgical Video Processing
Advanced video preprocessing, compression, and quality control for surgical recordings.

- **Real-time Processing**: GPU-accelerated video stream analysis
- **Quality Assurance**: Automated quality metrics and validation
- **Compression**: Efficient video encoding with preservation of surgical details
- **Deidentification**: HIPAA-compliant patient data protection

### 🔪 Surgical Instance Segmentation
Precise identification and segmentation of surgical instruments using deep learning.

- **Multi-Model Support**: YOLOv8, YOLOv11, Mask R-CNN, SAM integration
- **Real-time Inference**: Optimized for live surgical video streams
- **High Precision**: Surgical-grade accuracy for instrument detection
- **Configurable Pipeline**: Flexible model selection and parameter tuning

### 📊 Surgical Phase Recognition
Automated recognition and classification of surgical procedure phases.

- **Temporal Analysis**: Sequential phase detection and transition modeling
- **Multi-Modal Input**: Video and audio feature integration
- **Clinical Workflow**: Integration with surgical protocol standards
- **Performance Monitoring**: Real-time phase progression tracking

### 🎯 Surgical Skill Assessment
Objective evaluation of surgical performance using motion analysis and computer vision.

- **Motion Tracking**: 3D instrument trajectory analysis
- **Skill Metrics**: Quantitative assessment of surgical dexterity
- **Training Support**: Feedback systems for surgical education
- **Comparative Analysis**: Benchmarking against expert performance

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker and Docker Compose
- 16GB+ RAM recommended

### Installation

#### Using Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd Cataract_LMM

# Build and run with Docker Compose
docker-compose up --build
```

#### Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from surgical_video_processing import VideoProcessor
from surgical_instance_segmentation import InstrumentSegmentor
from surgical_phase_recognition import PhaseClassifier
from surgical_skill_assessment import SkillAssessor

# Initialize components
video_processor = VideoProcessor()
segmentor = InstrumentSegmentor(model='yolov8n-seg')
classifier = PhaseClassifier()
assessor = SkillAssessor()

# Process surgical video
video_path = "path/to/surgical_video.mp4"
processed_video = video_processor.process(video_path)
instruments = segmentor.segment(processed_video)
phases = classifier.classify(processed_video)
skills = assessor.assess(processed_video, instruments)
```

## 📋 System Requirements

### Minimum Requirements
- **CPU**: 4-core processor (Intel i5 or AMD Ryzen 5 equivalent)
- **Memory**: 8GB RAM
- **Storage**: 10GB available space
- **GPU**: Optional but recommended for real-time processing

### Recommended Requirements
- **CPU**: 8-core processor (Intel i7 or AMD Ryzen 7 equivalent)
- **Memory**: 16GB RAM
- **Storage**: 50GB SSD storage
- **GPU**: NVIDIA RTX 3060 or better with 8GB+ VRAM

### Production Requirements
- **CPU**: 16-core processor (Intel Xeon or AMD EPYC)
- **Memory**: 32GB+ RAM
- **Storage**: 100GB+ NVMe SSD
- **GPU**: NVIDIA RTX 4080/A4000 or better with 12GB+ VRAM

- **Surgical Video Processing**: Hospital-specific preprocessing and quality control
- **Instrument Segmentation**: Instance segmentation of surgical instruments and anatomical structures  
- **Phase Classification**: Automated recognition of 13 surgical phases
- **Skill Assessment**: Objective surgical skill evaluation using video analysis

## Key Features

### 🏥 Multi-Center Dataset Support
- **Farabi Hospital (S1)**: 2,800 videos at 720×480@30fps
- **Noor Hospital (S2)**: 200 videos at 1920×1080@60fps
- Comprehensive domain adaptation capabilities

### 🤖 State-of-the-Art Models
- **YOLO/Ultralytics**: Real-time object detection and segmentation
- **Mask R-CNN**: High-precision instance segmentation
- **SAM/SAM2**: Zero-shot segmentation capabilities
- **Video Transformers**: MViT, Video Swin Transformer
- **CNN-RNN Hybrids**: ResNet/EfficientNet + LSTM/GRU combinations

### 📊 Comprehensive Evaluation
- COCO-style evaluation metrics
- Domain adaptation benchmarks
- Cross-hospital validation protocols
- Skill assessment rubrics aligned with ICO-OSCAR standards

### 🚀 Production-Ready Deployment
- Docker containerization with GPU support
- Multi-stage builds for development and production
- Comprehensive monitoring and health checks
- Professional documentation and API references

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/MJAHMADEE/Cataract-LMM.git
cd Cataract-LMM/codes

# Install dependencies
pip install -r requirements.txt

# Run setup and validation
python setup.py
```

### Docker Deployment
```bash
# Build production image
docker build --target production -t cataract-lmm:latest .

# Run with GPU support
docker run -it --gpus all \
  -v /path/to/data:/app/data \
  -v /path/to/outputs:/app/outputs \
  cataract-lmm:latest
```

### Basic Usage
```python
# Video processing
from surgical_video_processing import VideoProcessor
processor = VideoProcessor()
result = processor.process_video("input.mp4", "output.mp4")

# Instrument segmentation
from surgical_instance_segmentation import YOLOSegmenter
segmenter = YOLOSegmenter("yolov8n-seg.pt")
masks = segmenter.predict("image.jpg")

# Phase classification
from surgical_phase_recognition import PhaseClassifier
classifier = PhaseClassifier("mvit_model.pth")
phases = classifier.predict_video("surgery.mp4")

# Skill assessment
from surgical_skill_assessment import SkillAssessor
assessor = SkillAssessor("timesformer_model.pth")
score = assessor.evaluate_video("capsulorhexis.mp4")
```

## Research Citation

For citation information, please refer to the [main repository README](../../README.md#-citation).

## Indices and Tables

```{toctree}
:maxdepth: 1

genindex
modindex
search
```
