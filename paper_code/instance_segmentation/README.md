# 🎯 Instance Segmentation

## Notebooks Overview

This directory contains the core notebooks used for training and evaluating instance segmentation models on the Cataract-LMM dataset.

### **📓 Notebooks**

| Notebook | Model | Description |
|----------|-------|-------------|
| `yolov8_segmentation_training.ipynb` | YOLOv8 | Training pipeline for YOLOv8-large segmentation model |
| `yolov11_segmentation_training.ipynb` | YOLOv11 | Training pipeline for YOLOv11-large segmentation model |
| `mask_rcnn_training.ipynb` | Mask R-CNN | Complete training pipeline for Mask R-CNN with ResNet50-FPN backbone |
| `sam_inference.ipynb` | SAM | Segment Anything Model inference pipeline |

### **🎯 Training Configuration**

**Common Parameters:**
- **Image Size**: 640x640
- **Batch Size**: 8-20 (depending on model)
- **Epochs**: 80
- **Classes**: 12 surgical instruments + background
- **Data Format**: COCO annotations

**YOLOv8/v11 Configuration:**
```yaml
epochs: 80
imgsz: 640
batch: 20
device: 0 (GPU)
plots: True
resume: True
```

**Mask R-CNN Configuration:**
- **Optimizer**: AdamW (lr=0.0005)
- **Scheduler**: StepLR (step_size=3, gamma=0.1)
- **Batch Size**: 8
- **Data Split**: 70% train, 20% validation, 10% test

### **📊 Expected Results**

**Performance Metrics:**
- **YOLOv8**: mAP@0.5 for multi-class segmentation
- **YOLOv11**: Enhanced performance with architectural improvements
- **Mask R-CNN**: Precise instance-level segmentation masks
- **SAM**: Zero-shot and fine-tuned segmentation capabilities

### **🔧 Setup Requirements**

**Dependencies:**
```bash
pip install ultralytics torch torchvision pycocotools pillow numpy tqdm
```

**Data Structure:**
```
dataset/
├── train/
│   ├── images/
│   └── _annotations.coco.json
└── data.yaml  # For YOLO models
```

### **💡 Usage Tips**

1. **Update dataset paths** in each notebook to match your local setup
2. **Adjust batch size** based on GPU memory
3. **Monitor training progress** using built-in plotting functions
4. **Save checkpoints** regularly for long training runs
5. **Validate models** on separate test set before final evaluation

---

## 👨‍💻 Contributors

[![MJAHMADEE](https://img.shields.io/badge/Lead%20Developer-@MJAHMADEE-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MJAHMADEE)
[![shahedmomenzadeh](https://img.shields.io/badge/Instance%20Segmentation%20Expert-@shahedmomenzadeh-orange?style=for-the-badge&logo=github&logoColor=white)](https://github.com/shahedmomenzadeh)
