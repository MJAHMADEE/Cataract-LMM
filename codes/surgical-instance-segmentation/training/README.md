# 🚀 Training Pipeline - Surgical Instance Segmentation

This directory contains comprehensive training implementations for all segmentation architectures benchmarked in the Cataract-LMM paper, ensuring reproducible performance across the reported metrics.

## 🎯 Objective

Provide production-ready training pipelines that achieve the exact performance benchmarks reported in the academic paper:
- **YOLOv11-L**: 73.9% mAP (80 epochs, batch=20, imgsz=640)
- **YOLOv8-L**: 73.8% mAP (80 epochs, batch=20, imgsz=640)  
- **Mask R-CNN**: 53.7% mAP (100 epochs, batch=4, lr=0.005)
- **SAM**: Pre-trained foundation model (inference-focused)

## 📂 Contents

### 🏭 **Training Manager** (`training_manager.py`)
**Unified training coordination and pipeline management**
- **Purpose**: Central orchestration of all training workflows
- **Features**: Multi-model support, experiment tracking, checkpoint management
- **Integration**: Seamless integration with all model architectures

```python
from training.training_manager import TrainingManager

# Initialize unified training manager
manager = TrainingManager(
    model_type='yolov11',
    task_type='task_3',  # 12-class fine-grained segmentation
    data_config='./configs/dataset_config.yaml'
)

# Execute training with paper-exact configuration
results = manager.train(
    epochs=80,
    batch_size=20,
    learning_rate='auto',  # Model-specific optimal rates
    save_checkpoints=True
)
```

### ⚡ **YOLO Trainer** (`yolo_trainer.py`) 
**High-performance training for YOLOv8 and YOLOv11 models**

#### **Core Features**
- **YOLOv11 Training**: 73.9% mAP top performance configuration
- **YOLOv8 Training**: 73.8% mAP proven baseline configuration  
- **Ultralytics Integration**: Native compatibility with YOLO ecosystem
- **Real-time Monitoring**: Live training metrics and visualization

#### **Paper-Exact Configuration**
```python
from training.yolo_trainer import YOLOTrainer

# YOLOv11 - Top performing configuration
yolo11_trainer = YOLOTrainer(
    model_name='yolo11l-seg',
    data_yaml='./data.yaml',
    
    # Exact paper training parameters
    epochs=80,
    imgsz=640,
    batch_size=20,
    device=0,  # GPU acceleration
    plots=True,
    resume=True
)

# Execute training
results = yolo11_trainer.train()
print(f"Final mAP: {results['metrics/mAP50-95']:.1f}%")  # Expected: 73.9%
```

#### **Advanced Features**
- **Multi-GPU Support**: Distributed training for large datasets
- **Automatic Mixed Precision**: Memory and speed optimization
- **Hyperparameter Optimization**: Built-in hyperparameter tuning
- **Custom Callbacks**: Surgical-specific monitoring and validation

### 🎭 **Mask R-CNN Trainer** (`mask_rcnn_trainer.py`)
**Precise instance segmentation training with ResNet-50 FPN**

#### **Architecture-Specific Training**
- **Backbone**: ResNet-50 Feature Pyramid Network
- **Performance**: 92.9% mAP on anatomical structures
- **Optimization**: AdamW optimizer with StepLR scheduling
- **Configuration**: Exact match with `mask_rcnn_training.ipynb`

```python
from training.mask_rcnn_trainer import MaskRCNNTrainer
from data.dataset_utils import SurgicalCocoDataset

# Dataset setup matching reference notebook
train_dataset = SurgicalCocoDataset(
    root='./data/images',
    annotation='./data/annotations.json',
    task_type='task_3'
)

# Paper-exact training configuration
trainer = MaskRCNNTrainer(
    num_classes=12,  # 12 surgical classes
    
    # Training hyperparameters from paper
    learning_rate=0.005,
    batch_size=4,
    num_epochs=100,
    
    # Optimizer configuration
    optimizer='AdamW',
    scheduler='StepLR',
    step_size=30,
    gamma=0.1
)

# Execute training with surgical dataset
model = trainer.train(train_dataset, val_dataset)
```

#### **Specialized Features**
- **COCO Evaluation**: Built-in COCO metrics during training
- **Instance Validation**: Real-time mask quality assessment
- **Surgical Metrics**: Custom metrics for instrument detection
- **Memory Optimization**: Efficient handling of large surgical images

### 🔮 **SAM Trainer** (`sam_trainer.py`)
**Foundation model adaptation and inference pipeline training**

#### **SAM-Specific Training**
- **Pre-trained Models**: SAM ViT-H, SAM2 foundation models
- **Prompt Engineering**: Optimized bbox prompt strategies
- **Zero-shot Evaluation**: 56.0% mAP without fine-tuning
- **Adaptation Protocols**: Optional surgical domain adaptation

```python
from training.sam_trainer import SAMTrainer

# SAM foundation model setup
sam_trainer = SAMTrainer(
    model_type='vit_h',  # Vision Transformer Huge
    checkpoint_path='./checkpoints/sam_vit_h.pth',
    
    # Evaluation configuration matching paper
    prompt_type='bbox',  # Bounding box prompts
    confidence_threshold=0.8,
    stability_score=0.95
)

# Zero-shot evaluation on surgical data
results = sam_trainer.evaluate_zero_shot(test_dataset)
print(f"SAM Zero-shot mAP: {results['mAP']:.1f}%")  # Expected: 56.0%
```

## 🏆 Training Performance Targets

Achieve paper-reported benchmarks across all architectures:

| **Model** | **Target mAP** | **Training Config** | **Key Parameters** |
|-----------|---------------|--------------------|--------------------|
| **YOLOv11-L** ⭐ | **73.9%** | 80 epochs, batch=20 | imgsz=640, device=GPU |
| **YOLOv8-L** | **73.8%** | 80 epochs, batch=20 | imgsz=640, device=GPU |
| **Mask R-CNN** | **53.7%** | 100 epochs, batch=4 | lr=0.005, AdamW+StepLR |
| **SAM ViT-H** | **56.0%** | Zero-shot inference | bbox prompts, conf=0.8 |

## 🔧 Advanced Training Features

### **Experiment Tracking**
- **MLflow Integration**: Comprehensive experiment logging
- **Weights & Biases**: Real-time metrics visualization  
- **TensorBoard**: Training progress monitoring
- **Custom Metrics**: Surgical-specific performance indicators

### **Data Augmentation Pipeline**
```python
# Surgical-optimized augmentation matching paper methodology
augmentation_config = {
    'random_horizontal_flip': 0.5,
    'gaussian_blur': {'kernel_size': 5, 'sigma': [0.1, 2.0]},
    'brightness_adjustment': {'factor': [-0.2, 0.2]},
    'hsv_jitter': {'hue': 0.015, 'saturation': 0.7, 'value': 0.4}
}
```

### **Multi-Task Support**
- **Task 1 Training**: 3-class granularity (instrument detection)
- **Task 2 Training**: 9-class granularity (balanced approach)
- **Task 3 Training**: 12-class granularity (fine-grained analysis)

### **Validation Protocols**
- **COCO Evaluation**: Standard instance segmentation metrics
- **Surgical Metrics**: Domain-specific performance indicators
- **Visual Quality**: Real-time mask quality assessment
- **Cross-Center Validation**: S1 (Farabi) vs S2 (Noor) performance

## 📊 Training Monitoring

### **Real-Time Metrics**
- **Loss Tracking**: Segmentation, classification, and total loss
- **mAP Progression**: Continuous validation performance
- **Learning Rate**: Automatic scheduling and optimization
- **GPU Utilization**: Resource usage optimization

### **Checkpoint Management**
- **Best Model Saving**: Automatic best mAP checkpoint preservation
- **Resume Capability**: Seamless training continuation
- **Model Versioning**: Comprehensive checkpoint metadata
- **Export Optimization**: Production-ready model formats

## 🎯 Multi-GPU & Distributed Training

```python
# Distributed training for large-scale experiments
from training.distributed import DistributedTrainer

# Multi-GPU YOLOv11 training
trainer = DistributedTrainer(
    model_type='yolov11',
    num_gpus=4,
    batch_size_per_gpu=5,  # Total batch_size = 20
    epochs=80
)

results = trainer.train_distributed()
```

## 📚 Reference Notebook Alignment

Each trainer maintains 100% compatibility with reference implementations:

- **`yolov11_segmentation_training.ipynb`** → `yolo_trainer.py` (YOLOv11)
- **`yolov8_segmentation_training.ipynb`** → `yolo_trainer.py` (YOLOv8)
- **`mask_rcnn_training.ipynb`** → `mask_rcnn_trainer.py`
- **`sam_inference.ipynb`** → `sam_trainer.py`

All training configurations exactly match the notebook implementations to ensure reproducible benchmark performance.
