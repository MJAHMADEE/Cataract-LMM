# 📄 Paper Code Repository

<div align="center">

[![Paper Code](https://img.shields.io/badge/Paper_Code-Nature_Scientific_Data-blue?style=for-the-badge)]()
[![Notebooks](https://img.shields.io/badge/Notebooks-Jupyter-orange?style=for-the-badge&logo=jupyter)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)]()

**Main code and notebooks used to generate results for the Cataract-LMM paper**

</div>

---

## 🎯 Overview

This directory contains the **core code and notebooks** used to generate the experimental results presented in the Cataract-LMM paper submitted to **Nature Scientific Data**. All notebooks have been cleaned of outputs for reproducibility and clarity.

---

## 📁 Code Organization

### **🎯 Instance Segmentation**
| Notebook | Description | Task |
|----------|-------------|------|
| [`yolov8_segmentation_training.ipynb`](instance_segmentation/yolov8_segmentation_training.ipynb) | YOLOv8 training pipeline for surgical instrument segmentation | Instance Segmentation |
| [`yolov11_segmentation_training.ipynb`](instance_segmentation/yolov11_segmentation_training.ipynb) | YOLOv11 advanced segmentation training | Instance Segmentation |
| [`mask_rcnn_training.ipynb`](instance_segmentation/mask_rcnn_training.ipynb) | Mask R-CNN training for precise instance segmentation | Instance Segmentation |
| [`sam_inference.ipynb`](instance_segmentation/sam_inference.ipynb) | Segment Anything Model (SAM) inference pipeline | Instance Segmentation |

### **⏱️ Phase Recognition**
| Notebook | Description | Task |
|----------|-------------|------|
| [`phase_validation_comprehensive.ipynb`](phase_recognition/phase_validation_comprehensive.ipynb) | Comprehensive phase recognition validation and evaluation | Phase Recognition |

### **🏆 Skill Assessment** 
| Notebook | Description | Task |
|----------|-------------|------|
| [`video_classification_prototype.ipynb`](skill_assessment/video_classification_prototype.ipynb) | Video-based skill assessment classification prototype | Skill Assessment |

### **🎬 Video Processing**
| Script | Description | Task |
|--------|-------------|------|
| [`compress_video.ipynb`](video_processing/compress_video.ipynb) | Video compression and preprocessing pipeline | Video Processing |
| [`process_video.sh`](video_processing/process_video.sh) | Shell script for batch video processing | Video Processing |
| [`process_videos.bat`](video_processing/process_videos.bat) | Batch script for Windows video processing | Video Processing |

---

## 🚀 Quick Start

### **Prerequisites**
```bash
# Install required dependencies
pip install torch torchvision ultralytics opencv-python pandas numpy matplotlib seaborn
pip install transformers timm albumentations segmentation-models-pytorch
```

### **Running the Notebooks**
1. **Navigate to the specific task directory**
2. **Open the Jupyter notebook**
3. **Follow the step-by-step execution**
4. **Modify parameters as needed for your experiments**

---

## 📊 Expected Results

### **Instance Segmentation Performance**
- **YOLOv8**: mAP@0.5 results for 12-class surgical instrument segmentation
- **YOLOv11**: Enhanced performance with latest architecture improvements  
- **Mask R-CNN**: Precise pixel-level segmentation masks
- **SAM**: Zero-shot and fine-tuned segmentation capabilities

### **Phase Recognition Accuracy**
- **Comprehensive validation**: Frame-level and clip-level phase recognition metrics
- **Temporal consistency**: Phase transition analysis and smoothing

### **Skill Assessment Metrics**
- **Video classification**: Skill level prediction accuracy
- **Feature analysis**: Important temporal and spatial features for skill assessment

### **Video Processing Benchmarks**
- **Compression ratios**: Quality vs. file size optimization
- **Processing speed**: Throughput metrics for different video formats

---

## 🔧 Configuration

### **Dataset Paths**
Update the dataset paths in each notebook to match your local setup:
```python
# Example configuration
DATA_ROOT = "/path/to/your/cataract-lmm/data"
TRAIN_IMAGES = f"{DATA_ROOT}/3_Instance_Segmentation/images"
ANNOTATIONS = f"{DATA_ROOT}/3_Instance_Segmentation/annotations_coco"
```

### **Model Checkpoints**
Specify your model checkpoint directories:
```python
MODEL_DIR = "/path/to/your/models"
CHECKPOINT_PATH = f"{MODEL_DIR}/best_model.pth"
```

---

## 📈 Reproducibility

### **Environment Setup**
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (for GPU training)

### **Seed Configuration**
All notebooks include deterministic training setup:
```python
import torch
import numpy as np
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

### **Hardware Requirements**
- **GPU Memory**: 8GB+ recommended for training
- **RAM**: 16GB+ recommended
- **Storage**: 100GB+ for full dataset processing

---

## 📖 Usage Guidelines

### **Research Use**
1. **Follow ethical guidelines** for medical data
2. **Cite the main paper** when using these codes
3. **Respect data licensing** requirements
4. **Share improvements** with the community

### **Commercial Use**
- **Review licensing terms** in the main repository
- **Contact author** for commercial applications
- **Ensure compliance** with medical data regulations

---

## 🔗 Related Resources

- **Main Repository**: [Cataract-LMM](../README.md)
- **Dataset Documentation**: [Data README](../data/README.md)
- **Full Codebase**: [Codes Directory](../codes/README.md)
- **Paper Citation**: [Citation Information](../README.md#-citation)

---

## 🐛 Issues & Support

For issues with these notebooks:

1. **Check dataset paths** and ensure data is properly extracted
2. **Verify dependencies** are correctly installed
3. **Review hardware requirements** for your specific setup
4. **Open an issue** on the main repository for bug reports

---

## 👨‍💻 Author

**Mohammad Javad Ahmadi**  
For complete contact information and links, see the [main repository README](../README.md#-author).

---

<div align="center">

**🧬 Advancing Surgical AI Through Reproducible Research 🧬**

*These notebooks represent the core experimental work underlying the Cataract-LMM dataset and framework.*

</div>
