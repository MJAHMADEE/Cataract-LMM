# 🎭 Mask R-CNN Implementation

> **Advanced Mask R-CNN architecture for surgical instrument instance segmentation with ResNet50-FPN backbone**

## 📋 Overview

This module provides a complete **Mask R-CNN** implementation specifically optimized for surgical instrument detection and segmentation in cataract surgery videos. Based on the research methodologies from the Cataract-LMM dataset, achieving **53.7% mAP** performance on surgical instrument segmentation tasks.

### 🔬 Model Architecture

- **🧠 Backbone**: ResNet50 with Feature Pyramid Network (FPN)
- **📊 Output**: Simultaneous object detection and instance segmentation
- **🎯 Classes**: 13 surgical instrument categories from Cataract-LMM dataset
- **📏 Performance**: 53.7% mAP on Task 3 (12-class granularity)
- **� Performance**: 53.7% mAP on Task 3 (12-class granularity)

## 📁 Contents

| File / Component | Description |
|:----------------|:------------|
| `model.py` | 🎭 **Mask R-CNN Model Architecture** - Complete ResNet50-FPN implementation with custom predictors for surgical instruments |
| `predictor.py` | 🔍 **Inference Engine** - Production-ready predictor with preprocessing, post-processing, and visualization capabilities |
| `__init__.py` | 📦 **Module Exports** - Public API exports for model instantiation and usage |

## 🚀 Quick Start

### Basic Usage

```python
from models.mask_rcnn import SurgicalMaskRCNN

# Initialize model
model = SurgicalMaskRCNN(
    num_classes=13,  # Cataract-LMM 13-class taxonomy
    pretrained=True
)

# Load pre-trained weights
model.load_state_dict(torch.load('surgical_maskrcnn.pth'))

# Inference on surgical image
predictions = model(image_tensor)
masks = predictions['masks']
boxes = predictions['boxes']
scores = predictions['scores']
```

### Advanced Configuration

```python
# Custom backbone configuration
model = SurgicalMaskRCNN(
    backbone_name='resnet50',
    pretrained_backbone=True,
    num_classes=13,
    min_size=800,
    max_size=1333,
    box_score_thresh=0.5,
    box_nms_thresh=0.5,
    mask_thresh=0.5
)

# Training mode setup
model.train()
for images, targets in dataloader:
    losses = model(images, targets)
    total_loss = sum(loss for loss in losses.values())
```

## 🎯 Key Features

### 🔧 Model Capabilities
- **Instance Segmentation**: Pixel-precise masks for each detected instrument
- **Multi-Scale Detection**: Effective on instruments of varying sizes
- **COCO Format**: Compatible with standard detection datasets
- **Transfer Learning**: Leverages COCO pre-training for medical domain

### ⚙️ Technical Specifications
- **Input Resolution**: 800×1333 (adaptable)
- **Anchor Generation**: Multi-scale anchors for various instrument sizes
- **Loss Functions**: Combined classification, regression, and mask losses
- **Optimization**: Support for various optimizers and learning rate schedules

### 📊 Performance Metrics
Based on Cataract-LMM evaluation (Task 3 - 12 classes):
- **Overall mAP**: 53.7% (mAP@0.5:0.95)
- **🏆 Exceptional Tissue Performance**:
  - Cornea: 94.7% (best across all models)
  - Pupil: 91.2% (best across all models)
  - All Tissue Classes: 92.9% (best across all models)
- **🔧 Instrument Performance**:
  - Primary Knife: 79.2%
  - Secondary Knife: 60.2%
  - Phaco. Handpiece: 58.9%
  - I/A Handpiece: 57.9%
  - All Instrument Classes: 45.9%

## 🔬 Implementation Details

### Architecture Components

1. **🖼️ ResNet50-FPN Backbone**
   - Pre-trained on ImageNet
   - Feature pyramid for multi-scale detection
   - Optimized for medical imaging characteristics

2. **🎯 RPN (Region Proposal Network)**
   - Generates object proposals
   - Optimized anchor ratios for surgical instruments
   - Multi-scale proposal generation

3. **🎭 Mask Head**
   - FCN-based mask prediction
   - 28×28 mask resolution
   - Binary segmentation for each detected instance

### Training Configuration
```yaml
# Example training configuration
training:
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 100
  optimizer: 'SGD'
  momentum: 0.9
  weight_decay: 0.0001
  lr_scheduler: 'StepLR'
  step_size: 30
  gamma: 0.1
```

## 🧪 Evaluation Metrics

The model supports comprehensive evaluation metrics:

- **Detection Metrics**: mAP@0.5, mAP@0.75, mAP@0.5:0.95
- **Segmentation Metrics**: Mask mAP, IoU scores
- **Class-Specific Performance**: Per-instrument evaluation
- **Speed Metrics**: FPS, inference time per image

## 🔧 Usage Examples

### Training Example
```python
from models.mask_rcnn import train_mask_rcnn

# Train model
model = train_mask_rcnn(
    train_dataset=train_data,
    val_dataset=val_data,
    num_epochs=100,
    batch_size=4,
    learning_rate=0.001
)
```

### Inference Example
```python
from models.mask_rcnn import SurgicalMaskRCNNPredictor

# Initialize predictor
predictor = SurgicalMaskRCNNPredictor('model_weights.pth')

# Process surgical image
results = predictor.predict(image_path)
predictor.visualize_results(image_path, results, save_path='output.jpg')
```

## 📚 References

- **Research Paper**: *"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"*
- **Original Paper**: He, K. et al. "Mask R-CNN" ICCV 2017
- **Implementation**: Based on torchvision.models.detection.maskrcnn_resnet50_fpn

---

**💡 Note**: This implementation is optimized for surgical instrument segmentation and follows the exact specifications from the Cataract-LMM research methodology for reproducible results.
