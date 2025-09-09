# 🔬 Surgical Instance Segmentation Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Framework](https://img.shields.io/badge/Framework-Production--Ready-brightgreen)](README.md)

A **comprehensive, production-ready framework** for surgical instance segmentation in cataract surgery videos using state-of-the-art deep learning architectures. This framework implements all models benchmarked in the **Cataract-LMM** academic paper with reproducible performance metrics.

> 🎯 **Academic Foundation**: Built upon the methodologies described in *"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"*

## 🚀 Project Overview

This framework transforms the research implementations from the **core reference notebooks** into a scalable, modular architecture designed for production deployment in surgical environments. The implementation provides complete support for the 3-task granularity system defined in the academic paper for surgical instrument instance segmentation.

### **🔬 Core Logic Reference**

The **primary implementation and algorithms** are located in the **`notebooks/`** directory, which contains the foundational research code:

- 📚 **`mask_rcnn_training.ipynb`**: ResNet-50 FPN Mask R-CNN implementation
- 🤖 **`sam_inference.ipynb`**: SAM foundation model with bbox prompts  
- ⚡ **`yolov8_segmentation_training.ipynb`**: YOLOv8-L segmentation pipeline
- 🚀 **`yolov11_segmentation_training.ipynb`**: YOLOv11-L segmentation (top performer)

This framework serves as the **production-ready implementation** of these core algorithms, providing enterprise-grade utilities, APIs, and deployment capabilities for surgical video analysis applications.

## 🏆 Performance Benchmarks

Achieving state-of-the-art results on the **6,094 annotated frames** from the Cataract-LMM dataset:

| **Model Architecture** | **mAP (Task 3: 12-class)** | **Top Performing Classes** |
|--------------------------|---------------------------|---------------------------|
| **YOLOv11** ⭐ | **73.9%** | Phaco. Handpiece (84.3%), Primary Knife (86.0%) |
| **YOLOv8** | **73.8%** | Lens Injector (84.2%), Primary Knife (89.1%) |
| **SAM** | **56.0%** | Primary Knife (86.7%) |
| **SAM2** | **55.2%** | Lens Injector (82.4%) |
| **Mask R-CNN** | **53.7%** | Cornea (94.7%), Pupil (91.2%) |

## 📁 Directory Structure

### 🧠 **[`models/`](./models/)** - Model Architectures
**State-of-the-art segmentation model implementations**
- **Purpose**: Production-ready implementations of all benchmarked architectures
- **Key Components**: Model factory, YOLO models, Mask R-CNN, SAM/SAM2 implementations
- **Performance**: Reproduces exact paper benchmarks (73.9% mAP top performance)

### 📊 **[`data/`](./data/)** - Data Management  
**Cataract-LMM dataset integration and preprocessing**
- **Purpose**: Robust data loading supporting 3-task granularity (3/9/12 classes)
- **Key Components**: SurgicalCocoDataset, multi-task class mappings, validation pipelines
- **Standards**: Full Cataract-LMM naming convention compliance (SE_*.png pattern)

### 🚀 **[`training/`](./training/)** - Training Pipelines
**Comprehensive training implementations for all architectures**  
- **Purpose**: Reproduce paper-exact training configurations and performance
- **Key Components**: YOLO trainer, Mask R-CNN trainer, SAM trainer, training manager
- **Features**: Multi-GPU support, experiment tracking, checkpoint management

### ⚡ **[`inference/`](./inference/)** - Inference Engine
**High-performance inference for real-time surgical video processing**
- **Purpose**: Production-ready inference with clinical deployment capabilities
- **Key Components**: Real-time engine, batch processing, SAM inference, predictor factory
- **Performance**: Real-time processing (>30 FPS) with maintained accuracy

### 📈 **[`evaluation/`](./evaluation/)** - Evaluation Suite
**Comprehensive evaluation and metrics calculation**
- **Purpose**: COCO evaluation protocol matching academic paper methodology
- **Key Components**: COCO evaluator, segmentation metrics, visualization tools
- **Standards**: Reproduces exact paper evaluation protocols and metrics

### 🧪 **[`tests/`](./tests/)** - Testing Suite
**Comprehensive testing for framework reliability**
- **Purpose**: Ensure framework functionality and performance validation
- **Key Components**: Unit tests, integration tests, performance benchmarks
- **Coverage**: All critical components and inference pipelines

### 🛠️ **[`utils/`](./utils/)** - Utility Functions
**Supporting utilities and configuration management**
- **Purpose**: Configuration management and helper functions
- **Key Components**: Config manager, logging utilities, helper functions
- **Integration**: Seamless integration across all framework components

### ⚙️ **[`configs/`](./configs/)** - Configuration Files
**Comprehensive configuration management aligned with academic paper**
- **Purpose**: Centralized configuration for all models, datasets, and training
- **Key Components**: Model configs, dataset configs, task definitions
- **Standards**: Paper-aligned hyperparameters and architectural specifications

## 🎯 Multi-Task Granularity System

Supporting the complete 3-task system defined in the Cataract-LMM paper:

### **Task 1: Binary Segmentation (3 Classes)**
- **Use Case**: High-level instrument detection for workflow recognition
- **Classes**: `cornea`, `pupil`, `instrument` (all 10 instruments merged)
- **YOLOv11 Performance**: 80.1% mAP

### **Task 2: Intermediate Segmentation (9 Classes)** 
- **Use Case**: Balanced granularity for robust instrument recognition  
- **Classes**: Merges only visually similar instruments (e.g., primary+secondary knife)
- **YOLOv11 Performance**: 75.2% mAP

### **Task 3: Fine-Grained Segmentation (12 Classes)**
- **Use Case**: Detailed analysis requiring specific instrument distinction
- **Classes**: All 12 distinct classes (2 anatomical + 10 instruments)  
- **YOLOv11 Performance**: 73.9% mAP

## 🔧 Quick Start

### **Installation**
```bash
# Clone the framework
git clone <repository-url>
cd surgical-instance-segmentation

# Install dependencies
pip install -r requirements.txt
```

### **Training Example (YOLOv11 - Top Performer)**
```python
from training.yolo_trainer import YOLOTrainer

# Initialize trainer with paper-exact configuration
trainer = YOLOTrainer(
    model_name='yolo11l-seg',
    data_yaml='./configs/cataract_lmm_data.yaml',
    epochs=80,
    imgsz=640,
    batch_size=20
)

# Train model
results = trainer.train()
print(f"Final mAP: {results['metrics/mAP50-95']:.1f}%")  # Expected: 73.9%
```

### **Real-Time Inference**
```python
from inference.real_time_engine import RealTimeEngine

# Initialize real-time segmentation engine  
engine = RealTimeEngine(
    model_type='yolov11',
    task_type='task_3'  # 12-class fine-grained segmentation
)

# Process surgical video stream
for frame in surgical_video:
    predictions = engine.predict_frame(frame)
    # Process predictions...
    annotated_frame = engine.visualize(frame, predictions)
```

### **Batch Evaluation**
```python
from evaluation.coco_evaluator import COCOEvaluator

# Evaluate model on Cataract-LMM test set
evaluator = COCOEvaluator(
    task_type='task_3'
)

# Run evaluation
results = evaluator.evaluate(
    test_dataset='./data/test_annotations.json',
    output_dir='./evaluation_results'
)

print(f"Model mAP: {results['mAP']:.1f}%")
```

## 📋 Dataset Integration

### **🔬 Cataract-LMM Dataset Support**
- **Total Frames**: 6,094 annotated surgical frames
- **Source Videos**: 150 cataract surgery procedures  
- **Clinical Centers**: 2 hospitals (Farabi S1, Noor S2)
- **Naming Convention**: `SE_<ClipID>_<RawVideoID>_S<Site>_<FrameIndex>.png`
- **Quality Standard**: Two-stage annotation with mIoU ≥ 0.95

### **🎯 Visual Challenge Handling**
- **Inter-Instrument Similarity**: Primary/secondary knives, forceps variants
- **Motion Blur & Occlusion**: Small instruments (cannula, second instrument)
- **Specular Reflections**: All metallic surgical instruments
- **Boundary Ambiguity**: Depth of field and motion artifacts

## 🏥 Clinical Deployment Features

### **🚀 Production-Ready Capabilities**
- **Real-Time Processing**: >30 FPS surgical video segmentation
- **Multi-GPU Support**: Distributed training and inference
- **Clinical Integration**: HIPAA-compliant processing pipelines
- **Fault Tolerance**: Robust error handling and recovery

### **🔌 API Integration**
```python
from inference.api import SurgicalSegmentationAPI

# Clinical-grade REST API
api = SurgicalSegmentationAPI(
    model_checkpoint='./models/production_yolov11.pt',
    enable_logging=True,
    require_authentication=True
)

# Deployment ready
api.deploy(port=8080, ssl_enabled=True)
```

## 🔗 Framework Integration

This surgical instance segmentation framework integrates seamlessly with:

- **Surgical Phase Recognition**: Temporal workflow analysis  
- **Skill Assessment**: Instrument-based performance evaluation
- **Tracking Systems**: Spatiotemporal instrument analysis
- **Clinical Workflows**: Real-time surgical assistance systems

## 📚 Academic Alignment

**100% compatibility** with reference notebook implementations:
- All models reproduce exact paper performance benchmarks
- Training configurations match academic specifications  
- Evaluation protocols follow paper methodology
- Dataset handling preserves research reproducibility

---

> 💡 **Note**: This framework serves as the production-ready implementation of the core research algorithms found in the `notebooks/` directory, providing enterprise-grade capabilities for surgical video analysis deployment.
