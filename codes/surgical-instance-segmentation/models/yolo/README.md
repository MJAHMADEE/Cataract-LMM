# ⚡ YOLO Segmentation Models

> **High-performance real-time YOLO architectures for surgical instrument detection and segmentation**

## 📋 Overview

This module provides state-of-the-art **YOLO segmentation models** (YOLOv8, YOLOv11) specifically optimized for real-time surgical instrument detection and segmentation. Achieving industry-leading performance with **73.9% mAP** (YOLOv11) and **73.8% mAP** (YOLOv8) on the Cataract-LMM dataset.

### 🔬 Model Architecture

- **🧠 Architectures**: YOLOv8-L, YOLOv11-L (Large variants for best accuracy)
- **📊 Output**: Unified detection + instance segmentation
- **🎯 Classes**: 13 surgical instrument categories from Cataract-LMM
- **📏 Performance**: 73.9% mAP (YOLOv11), 73.8% mAP (YOLOv8)
- **� Performance**: 73.9% mAP (YOLOv11), 73.8% mAP (YOLOv8)

## 📁 Contents

| File / Component | Description |
|:----------------|:------------|
| `yolo_models.py` | ⚡ **YOLO Model Implementations** - Complete YOLOv8/v11 segmentation models with surgical-specific optimizations |
| `predictor.py` | 🔍 **Real-time Inference Engine** - High-performance predictor with GPU acceleration and batch processing |
| `__init__.py` | 📦 **Module Exports** - Public API exports for YOLO model instantiation and real-time inference |

## 🚀 Quick Start

### Basic Usage

```python
from models.yolo import SurgicalYOLO

# Initialize YOLOv11 (best performance)
model = SurgicalYOLO(
    model_version='yolov11l-seg',
    num_classes=13,
    pretrained=True
)

# Load surgical-trained weights
model.load_weights('surgical_yolov11_best.pt')

# Real-time inference
results = model.predict(image_path, conf=0.25, iou=0.5)

# Extract predictions
boxes = results.boxes.xyxy    # Bounding boxes
masks = results.masks.data    # Segmentation masks
scores = results.boxes.conf   # Confidence scores
classes = results.boxes.cls   # Class predictions
```

### Batch Processing

```python
from models.yolo import YOLOBatchProcessor

# Initialize batch processor
processor = YOLOBatchProcessor(
    model_path='surgical_yolov11_best.pt',
    batch_size=8,
    device='cuda'
)

# Process video frames or image batch
results = processor.process_batch(
    image_paths=['frame1.jpg', 'frame2.jpg', ...],
    output_dir='./results'
)
```

## 🏆 Model Performance Comparison

| Model | mAP@0.5:0.95 | Best Classes |
|-------|-------------|--------------|
| **YOLOv11** ⭐ | **73.9%** | Phaco. Handpiece (84.3%), Primary Knife (86.0%) |
| **YOLOv8** | **73.8%** | Lens Injector (84.2%), Primary Knife (89.1%) |

## 🎯 Key Features

### 🔧 Model Capabilities
- **🎭 Instance Segmentation**: Pixel-precise masks with detection
- **🎯 Multi-Scale Detection**: Effective on various instrument sizes
- **🔄 End-to-End Training**: Unified loss for detection and segmentation
- **📱 Deployment Ready**: ONNX, TensorRT, mobile optimization support

### ⚙️ Technical Specifications
- **Input Resolution**: 640×640 (default), 1280×1280 (high-res mode)
- **Architecture**: CSPDarknet backbone with PANet/FPN neck
- **Anchor-Free**: Modern anchor-free detection paradigm
- **Loss Functions**: Combined box, class, and mask losses with DFL

### 📊 Surgical Performance Metrics
Based on Cataract-LMM evaluation (Task 3 - 12 classes):
- **🏆 Best Overall**: YOLOv11 (73.9% mAP@0.5:0.95)
- **🥈 Runner-up**: YOLOv8 (73.8% mAP@0.5:0.95)
- **🎯 Top Class Performances**:
  - Primary Knife: 89.1% (YOLOv8), 86.0% (YOLOv11)
  - Phaco. Handpiece: 84.3% (YOLOv11), 82.7% (YOLOv8)
  - Lens Injector: 84.2% (YOLOv8), 82.3% (YOLOv11)
  - I/A Handpiece: 74.8% (YOLOv11), 73.9% (YOLOv8)
- **📊 Category Performance**:
  - All Tissue Classes: 83.4% (both models)
  - All Instrument Classes: 72.0% (YOLOv11), 71.9% (YOLOv8)

## 🔬 Architecture Details

### YOLOv11 Improvements
```python
# Key architectural enhancements in YOLOv11
yolov11_features = {
    'backbone': 'Enhanced CSPDarknet with C2PSA blocks',
    'neck': 'Improved PANet with spatial attention',
    'head': 'Decoupled detection and segmentation heads',
    'loss': 'VFL + DFL + CIoU for better localization',
    'optimization': 'Improved NMS and post-processing'
}
```

### Model Configuration
```yaml
# Example YOLOv11 configuration
model:
  type: 'yolov11l-seg'
  num_classes: 13
  input_size: [640, 640]
  
training:
  batch_size: 16
  epochs: 300
  learning_rate: 0.01
  warmup_epochs: 3
  mosaic: 1.0
  mixup: 0.1
  copy_paste: 0.3
  
inference:
  conf_threshold: 0.25
  iou_threshold: 0.5
  max_detections: 300
```

## 🧪 Training and Fine-tuning

### Custom Training Pipeline
```python
from models.yolo import train_yolo_segmentation

# Train on surgical dataset
model = train_yolo_segmentation(
    model='yolov11l-seg.pt',
    data='surgical_dataset.yaml',
    epochs=300,
    batch_size=16,
    device='cuda',
    project='surgical_segmentation',
    name='yolov11_experiment'
)
```

### Data Augmentation for Surgery
```python
# Surgical-specific augmentations
surgical_augmentations = {
    'hsv_h': 0.015,      # Slight hue variation for lighting
    'hsv_s': 0.7,        # Saturation for surgical lighting
    'hsv_v': 0.4,        # Brightness variation
    'degrees': 10.0,     # Limited rotation for surgical context
    'translate': 0.1,    # Small translation
    'scale': 0.9,        # Scale variation for zoom levels
    'shear': 0.0,        # No shear (preserve surgical geometry)
    'perspective': 0.0,  # No perspective (maintain surgical view)
    'flipud': 0.0,       # No vertical flip (surgical orientation)
    'fliplr': 0.5,       # Horizontal flip OK for instruments
    'mosaic': 1.0,       # Mosaic augmentation
    'mixup': 0.1,        # Mix-up for better generalization
    'copy_paste': 0.3    # Copy-paste for instance variation
}
```

## 🔧 Advanced Usage

### Real-Time Video Processing
```python
from models.yolo import YOLOVideoProcessor

# Real-time surgical video analysis
processor = YOLOVideoProcessor(
    model_path='surgical_yolov11_best.pt',
    input_source='surgical_video.mp4',  # or webcam: 0
    output_path='annotated_video.mp4',
    conf_threshold=0.3,
    tracker='bytetrack'  # Multi-object tracking
)

# Process with real-time visualization
processor.run(
    show_preview=True,
    save_results=True,
    track_instruments=True
)
```

### Model Export and Optimization
```python
# Export for deployment
from models.yolo import export_yolo_model

# Export to various formats
export_yolo_model(
    model_path='surgical_yolov11_best.pt',
    formats=['onnx', 'tensorrt', 'torchscript'],
    optimize=True,
    half=True  # FP16 precision
)

# TensorRT optimization for NVIDIA GPUs
export_yolo_model(
    model_path='surgical_yolov11_best.pt',
    format='engine',
    workspace=4,  # GB
    precision='fp16'
)
```

### Custom Post-Processing
```python
def surgical_post_processing(results, confidence_threshold=0.3):
    """Custom post-processing for surgical instruments"""
    filtered_results = []
    
    for result in results:
        # Filter by confidence
        high_conf_mask = result.boxes.conf > confidence_threshold
        
        # Apply surgical-specific filters
        boxes = result.boxes.xyxy[high_conf_mask]
        masks = result.masks.data[high_conf_mask]
        classes = result.boxes.cls[high_conf_mask]
        
        # Size-based filtering (remove very small detections)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        size_mask = areas > 100  # Minimum 100 pixels area
        
        filtered_results.append({
            'boxes': boxes[size_mask],
            'masks': masks[size_mask],
            'classes': classes[size_mask],
            'scores': result.boxes.conf[high_conf_mask][size_mask]
        })
    
    return filtered_results
```

## 📊 Evaluation and Benchmarking

### Performance Evaluation
```python
from models.yolo import evaluate_yolo_model

# Comprehensive evaluation
metrics = evaluate_yolo_model(
    model_path='surgical_yolov11_best.pt',
    data_yaml='surgical_test.yaml',
    iou_thresholds=[0.5, 0.75, 0.9],
    conf_threshold=0.001,
    save_json=True
)

print(f"mAP@0.5: {metrics['mAP_50']:.3f}")
print(f"mAP@0.5:0.95: {metrics['mAP']:.3f}")
```

## 🚀 Deployment Options

### Edge Deployment
```python
# Lightweight model for edge devices
edge_model = SurgicalYOLO(
    model_version='yolov11n-seg',  # Nano version
    optimization_level='edge',
    precision='fp16'
)

# Mobile-optimized inference
mobile_results = edge_model.predict(
    image, 
    optimize_for_mobile=True,
    max_detections=50
)
```

### Cloud API Integration
```python
# RESTful API wrapper
from models.yolo import YOLOAPIHandler

api = YOLOAPIHandler('surgical_yolov11_best.pt')

@api.route('/predict', methods=['POST'])
def predict_surgical_instruments():
    image = request.files['image']
    results = api.predict(image)
    return jsonify(results)
```

## 📚 References

- **YOLOv8 Paper**: "YOLOv8: A New Era of Object Detection" Ultralytics 2023
- **YOLOv11**: Enhanced architecture with improved accuracy and speed
- **Research Application**: *"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"*
- **Implementation**: Based on Ultralytics YOLO framework

---

**💡 Note**: YOLO models represent the best balance of accuracy and speed for real-time surgical video analysis, making them ideal for live surgical guidance systems and video processing pipelines.
