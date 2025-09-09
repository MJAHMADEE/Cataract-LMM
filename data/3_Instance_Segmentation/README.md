# 🎯 Instance Segmentation Dataset

[![Dataset Type](https://img.shields.io/badge/Task-Instance%20Segmentation-red)](.)
[![Clips](https://img.shields.io/badge/Source%20Clips-150-green)](.)
[![Frames](https://img.shields.io/badge/Annotated%20Frames-6,094-blue)](.)
[![Classes](https://img.shields.io/badge/Classes-12-orange)](.)

## 🎯 Overview

This comprehensive dataset provides **instrument and tissue instance segmentation** resources for computer vision applications in cataract surgery. The annotations are available in both industry-standard formats to support various machine learning frameworks and workflows.

### 📊 Dataset Composition
- **🎬 Source Video Clips:** 150 curated surgical segments
- **🖼️ Annotated Frames:** 6,094 high-quality annotated frames
- **📝 Annotation Formats:** COCO JSON and YOLO TXT formats
- **🏷️ Object Classes:** 12 distinct surgical instruments and anatomical structures

---

## 📁 Directory Structure

```
3_Instance_Segmentation/
├── videos/                          # 🎥 Source video clips (150 files)
├── annotations/
│   ├── coco_json/                   # 📊 COCO-style annotations
│   │   ├── images/                  # 🖼️ Annotated frame images
│   │   └── annotations.json         # 📋 COCO format annotation file
│   └── yolo_txt/                    # 🎯 YOLO-style annotations
│       ├── images/                  # 🖼️ Annotated frame images
│       └── labels/                  # 📝 Individual TXT label files
```

---

## 🏷️ Object Classes (12 Classes)

The dataset includes annotations for **2 anatomical structures** and **10 surgical instruments**:

### 👁️ Anatomical Structures
| Class ID | 🏷️ Name | 📝 Description |
|----------|----------|----------------|
| `01` | 👁️ **Pupil** | Central opening of the iris |
| `02` | 🔵 **Cornea** | Transparent front layer of the eye |

### 🔧 Surgical Instruments
| Class ID | 🏷️ Name | 📝 Description |
|----------|----------|----------------|
| `03` | 🔪 **Primary Knife** | Main incision instrument |
| `04` | 🗡️ **Secondary Knife** | Secondary cutting tool |
| `05` | 🪝 **Capsulorhexis Cystotome** | Capsule cutting instrument |
| `06` | 🔧 **Capsulorhexis Forceps** | Capsule gripping tool |
| `07` | ⚡ **Phaco Handpiece** | Phacoemulsification device |
| `08` | 🌊 **I/A Handpiece** | Irrigation/Aspiration tool |
| `09` | 🔨 **Second Instrument** | Supporting instrument |
| `10` | 🤏 **Forceps** | General grasping tool |
| `11` | 💉 **Cannula** | Injection/aspiration tube |
| `12` | 👁️ **Lens Injector** | IOL implantation device |

---

## 📊 Annotation Formats

### 🎯 COCO Format (`annotations/coco_json/`)
- **Format:** JSON-based annotation following COCO standards
- **Structure:** Single `annotations.json` file with all annotations
- **Features:** 
  - Bounding boxes and segmentation masks
  - Category information and metadata
  - Image references and annotation IDs
- **Use Case:** Ideal for frameworks like Detectron2, MMDetection

### 🎯 YOLO Format (`annotations/yolo_txt/`)
- **Format:** Individual TXT files per image
- **Structure:** One `.txt` file per annotated image in `labels/`
- **Features:**
  - Normalized bounding box coordinates
  - Class IDs and confidence scores
  - Polygon segmentation coordinates
- **Use Case:** Perfect for YOLO-based training pipelines

---

## 🔍 Data Statistics

| Metric | Value |
|--------|-------|
| 🎬 **Source Videos** | 150 clips |
| 🖼️ **Total Annotated Frames** | 6,094 frames |
| 🏷️ **Object Classes** | 12 classes |
| 🎯 **Annotation Quality** | Professional medical annotation |

---

## 💡 Usage Examples

### 🐍 Loading COCO Annotations (Python)
```python
import json
from pycocotools.coco import COCO

# Load COCO annotations
coco = COCO('annotations/coco_json/annotations.json')

# Get all category information
categories = coco.loadCats(coco.getCatIds())
print(f"Found {len(categories)} categories")
```

### 🎯 Loading YOLO Annotations (Python)
```python
import os
import glob

# Load YOLO label files
label_files = glob.glob('annotations/yolo_txt/labels/*.txt')
print(f"Found {len(label_files)} annotated images")

# Read a sample label file
with open(label_files[0], 'r') as f:
    annotations = f.readlines()
    print(f"Sample has {len(annotations)} objects")
```

---

## ⚠️ Data Access Notice

> **📁 Repository Contents:** This package contains **documentation only** — no media files or annotation data are included here.
> 
> **🔑 Data Access:** For information on accessing the actual dataset files, please refer to [`../README.md`](../README.md) for detailed access instructions.

---

## 🔗 Related Datasets

This instance segmentation dataset works well with:
- 🎥 [**Raw Videos**](../1_Raw_Videos/) - Original source material
- 🔄 [**Phase Recognition**](../2_Phase_Recognition/) - Temporal workflow context
- 🔍 [**Instrument Tracking**](../4_Instrument_Tracking/) - Temporal instrument analysis
- ⭐ [**Skill Assessment**](../5_Skill_Assessment/) - Performance evaluation context
