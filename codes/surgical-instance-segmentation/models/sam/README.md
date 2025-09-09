# 🎯 SAM (Segment Anything Model) Implementation

> **Advanced vision transformer-based foundation model for prompt-guided surgical instrument segmentation**

## 📋 Overview

This module provides a complete **SAM (Segment Anything Model)** implementation specifically adapted for surgical instrument segmentation in cataract surgery videos. Using bounding box prompts, it achieves **56.0% mAP** on the Cataract-LMM dataset with remarkable zero-shot capabilities for unseen surgical instruments.

### 🔬 Model Architecture

- **🧠 Backbone**: Vision Transformer (ViT-H, ViT-L, ViT-B variants)
- **📊 Output**: High-quality instance segmentation masks
- **🎯 Prompting**: Bounding box, point, and text prompt support
- **📏 Performance**: 56.0% mAP with zero-shot capabilities
- **� Performance**: 56.0% mAP with zero-shot capabilities

## 📁 Contents

| File / Component | Description |
|:----------------|:------------|
| `model.py` | 🎯 **SAM Model Architecture** - Complete Vision Transformer implementation with prompt encoder and mask decoder |
| `predictor.py` | 🔍 **Inference Engine** - Production-ready predictor with prompt processing and interactive segmentation |
| `__init__.py` | 📦 **Module Exports** - Public API exports for model instantiation and prompt-based inference |

## 🚀 Quick Start

### Basic Usage

```python
from models.sam import SurgicalSAM

# Initialize SAM model
model = SurgicalSAM(
    model_type='vit_h',  # ViT-Huge for best performance
    checkpoint_path='sam_vit_h_4b8939.pth'
)

# Bbox-prompted segmentation
bbox_prompt = [100, 150, 200, 300]  # [x1, y1, x2, y2]
mask = model.segment_with_bbox(image, bbox_prompt)
```

### Interactive Segmentation

```python
from models.sam import SurgicalSAMPredictor

# Initialize predictor
predictor = SurgicalSAMPredictor('sam_vit_h_4b8939.pth')

# Set image
predictor.set_image(surgical_image)

# Interactive prompting
point_prompt = [(150, 200)]  # Click on instrument
point_labels = [1]  # Foreground point

masks, scores, logits = predictor.predict(
    point_coords=np.array(point_prompt),
    point_labels=np.array(point_labels),
    multimask_output=True
)
```

## 🎯 Key Features

### 🔧 Model Capabilities
- **🎭 Zero-Shot Segmentation**: No training required for new instrument types
- **🎯 Prompt Flexibility**: Supports bbox, point, and text prompts
- **🔄 Interactive Refinement**: Iterative mask refinement with additional prompts
- **📏 High Resolution**: Maintains fine-grained segmentation quality
- **🧠 Foundation Model**: Pre-trained on 1B+ masks for robust generalization

### ⚙️ Technical Specifications
- **Input Resolution**: 1024×1024 (automatically resized)
- **Prompt Types**: Points, bounding boxes, masks
- **Output**: Multiple mask candidates with confidence scores

### 📊 Performance Metrics
Based on Cataract-LMM evaluation (Task 3 - 12 classes):
- **Overall mAP**: 56.0% (SAM ViT-H)
- **🎯 Top Class Performances (SAM)**:
  - Primary Knife: 86.7%
  - Pupil: 73.5%
  - Phaco. Handpiece: 52.4%
  - Forceps: 48.2%
- **📊 Category Performance**:
  - All Tissue Classes: 63.1%
  - All Instrument Classes: 54.5%
- **Zero-Shot Performance**: Effective on unseen instrument types

## 🔬 Implementation Details

### Architecture Components

1. **🖼️ Image Encoder (ViT)**
   - Vision Transformer backbone
   - Processes 1024×1024 images
   - Generates 256×64×64 feature embeddings

2. **🎯 Prompt Encoder**
   - Handles multiple prompt modalities
   - Sparse prompts (points, boxes)
   - Dense prompts (masks)

3. **🎭 Mask Decoder**
   - Lightweight decoder architecture
   - Produces high-quality segmentation masks
   - Multiple mask candidates per prompt

### Prompt Engineering for Surgery
```python
# Optimized prompting strategies for surgical instruments
def create_surgical_prompts(bbox_annotations):
    """Convert surgical instrument bboxes to SAM prompts"""
    prompts = []
    for bbox in bbox_annotations:
        # Use bbox corners and center for robust prompting
        x1, y1, x2, y2 = bbox
        prompts.append({
            'bbox': [x1, y1, x2, y2],
            'center_point': [(x1 + x2) / 2, (y1 + y2) / 2],
            'confidence': 1.0
        })
    return prompts
```

## 🧪 Model Performance

| Model | mAP@0.5:0.95 | Best Class Performance |
|-------|-------------|----------------------|
| **SAM** | **56.0%** | Primary Knife (86.7%) |

## 🔧 Usage Examples

### Batch Processing with Bounding Boxes
```python
from models.sam import batch_segment_with_sam

# Process multiple surgical images
results = batch_segment_with_sam(
    images=image_list,
    bbox_prompts=bbox_list,
    model_type='vit_h',
    output_dir='./segmentation_results'
)
```

### Integration with Detection Models
```python
# Combine YOLO detection + SAM segmentation
from models.yolo import YOLOPredictor
from models.sam import SurgicalSAMPredictor

# Detect instruments with YOLO
yolo = YOLOPredictor('yolov11_surgical.pt')
detections = yolo.predict(image)

# Refine with SAM
sam = SurgicalSAMPredictor('sam_vit_h.pth')
sam.set_image(image)

refined_masks = []
for bbox in detections['boxes']:
    mask = sam.predict_from_bbox(bbox)
    refined_masks.append(mask)
```

### Performance Optimization
```python
# Optimized inference configuration
sam_config = {
    'model_type': 'vit_b',  # Faster variant
    'points_per_side': 32,  # Grid sampling density
    'pred_iou_thresh': 0.88,  # Quality threshold
    'stability_score_thresh': 0.95,
    'crop_n_layers': 1,  # Multi-crop processing
    'crop_n_points_downscale_factor': 2
}
```

## 🔄 Integration with Other Models

### SAM + YOLO Pipeline
```python
def sam_yolo_pipeline(image_path):
    """Combined detection and refinement pipeline"""
    # Step 1: Detect with YOLO
    detections = yolo_model(image_path)
    
    # Step 2: Refine with SAM
    refined_results = []
    for det in detections:
        mask = sam_model.segment_with_bbox(image_path, det.bbox)
        refined_results.append({
            'class': det.class_name,
            'confidence': det.confidence,
            'refined_mask': mask,
            'original_bbox': det.bbox
        })
    
    return refined_results
```

## 📊 Evaluation and Metrics

### Zero-Shot Evaluation
```python
# Evaluate zero-shot performance on new instrument types
zero_shot_results = evaluate_zero_shot_segmentation(
    model=sam_model,
    test_dataset=unseen_instruments,
    prompt_strategy='bbox_center_point'
)

print(f"Zero-shot mAP: {zero_shot_results['mAP']:.3f}")
print(f"IoU Score: {zero_shot_results['mean_iou']:.3f}")
```

## 📚 References

- **Foundation Paper**: Kirillov, A. et al. "Segment Anything" ICCV 2023
- **Research Application**: *"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"*
- **Model Weights**: Available from Meta AI Segment Anything project

---

**💡 Note**: SAM's zero-shot capabilities make it particularly valuable for surgical applications where annotated data is limited. The model excels at generalizing to unseen surgical instruments and scenarios.
