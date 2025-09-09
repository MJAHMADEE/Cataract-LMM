# 🔍 Instrument Tracking Dataset (Capsulorhexis)

[![Dataset Type](https://img.shields.io/badge/Task-Instrument%20Tracking-cyan)](.)
[![Clips](https://img.shields.io/badge/Clips-170-green)](.)
[![Phase](https://img.shields.io/badge/Phase-P03%20Capsulorhexis-purple)](.)
[![Format](https://img.shields.io/badge/Annotations-JSON-blue)](.)

## 🎯 Overview

This specialized dataset focuses on **instrument tracking during the Capsulorhexis phase** of cataract surgery. It provides frame-by-frame tracking annotations for surgical instruments, enabling the development of computer vision models for real-time instrument tracking and surgical guidance systems.

### 📊 Dataset Composition
- **🎬 Tracking Clips:** 170 video segments from the Capsulorhexis phase
- **🏷️ Target Phase:** P03 (Capsulorhexis) - circular anterior capsulotomy
- **📋 Annotations:** Frame-level JSON tracking data
- **🖼️ Frame Extraction:** Individual frame images for each tracking sequence

---

## 📁 Directory Structure

```
4_Instrument_Tracking/
├── videos/                          # 🎥 Capsulorhexis video clips (170 files)
└── annotations/                     # 📊 Tracking annotation folders
    ├── TR_S1_<ClipID>_P03/         # 🏥 Farabi Hospital annotations
    │   ├── TR_<ClipID>_S1_P03_<FrameIndex>.jpg  # 🖼️ Individual frames
    │   └── TR_<ClipID>_S1_P03.json             # 📋 Tracking data
    └── TR_S2_<ClipID>_P03/         # 🏥 Noor Hospital annotations
        ├── TR_<ClipID>_S2_P03_<FrameIndex>.jpg  # 🖼️ Individual frames
        └── TR_<ClipID>_S2_P03.json             # 📋 Tracking data
```

---

## 🏷️ File Naming Conventions

### 🎬 Video Files (`videos/`)
```
TR_<ClipID>_S<Site>_P03.mp4
```

### 📁 Annotation Directories
```
TR_S<Site>_<ClipID>_P03/
```
*Note: Site code appears before ClipID in folder names*

### 🖼️ Individual Frame Images
```
TR_<ClipID>_S<Site>_P03_<FrameIndex>.jpg
```

### 📋 Tracking JSON Files
```
TR_<ClipID>_S<Site>_P03.json
```

---

## 🔍 Naming Components Explained

| Component | Description | Format | Example |
|-----------|-------------|---------|----------|
| `TR` | Tracking dataset prefix | Fixed | `TR` |
| `ClipID` | Unique clip identifier | 4-digit zero-padded | `0001`, `0042` |
| `S<Site>` | Hospital site code | S + digit | `S1`, `S2` |
| `P03` | Phase identifier (Capsulorhexis) | Fixed | `P03` |
| `FrameIndex` | Frame number within clip | 7-digit zero-padded | `0000001`, `0000142` |

### 🎯 Example File Structure
```
# Video clip
TR_0042_S1_P03.mp4

# Corresponding annotation folder
TR_S1_0042_P03/
├── TR_0042_S1_P03_0000001.jpg    # First frame
├── TR_0042_S1_P03_0000002.jpg    # Second frame
├── ...
├── TR_0042_S1_P03_0000142.jpg    # Last frame
└── TR_0042_S1_P03.json           # Tracking annotations
```

---

## 📊 Tracking Annotations Format

The JSON files contain detailed tracking information for each frame:

### 📋 JSON Structure Example
```json
{
  "clip_info": {
    "clip_id": "0042",
    "site": "S1",
    "phase": "P03",
    "total_frames": 142,
    "fps": 30
  },
  "tracking_data": [
    {
      "frame_index": 1,
      "timestamp": 0.033,
      "instruments": [
        {
          "instrument_id": "capsulorhexis_forceps",
          "bbox": [x, y, width, height],
          "center_point": [x, y],
          "confidence": 0.95,
          "visible": true
        }
      ]
    }
  ]
}
```

---

## 🏥 Capsulorhexis Phase Context

### ⭕ What is Capsulorhexis?
Capsulorhexis (Phase P03) is a critical step in cataract surgery involving:
- **🎯 Purpose:** Creating a circular opening in the anterior lens capsule
- **🔧 Instruments:** Primarily capsulorhexis forceps and cystotome
- **⏱️ Duration:** Typically 30-120 seconds per procedure
- **🎯 Precision:** Requires high manual dexterity and spatial awareness

### 🎯 Tracking Challenges
- **🔄 Continuous Motion:** Instruments move in circular patterns
- **👁️ Occlusion:** Instruments may temporarily disappear behind tissue
- **💧 Visual Interference:** Viscoelastic substances affect visibility
- **⚡ Speed Variation:** Motion varies from slow positioning to rapid cutting

---

## 🔢 Dataset Statistics

| Metric | Value |
|--------|-------|
| 🎬 **Total Clips** | 170 sequences |
| 📊 **Annotation Rate** | 100% frame coverage |

---

## ⚠️ Data Access Notice

> **📁 Repository Contents:** This package contains **documentation only** — no video files, images, or JSON annotations are included here.
> 
> **🔑 Data Access:** For information on accessing the actual dataset files, please refer to [`../README.md`](../README.md) for detailed access instructions.

---

## 🔗 Related Datasets

This instrument tracking dataset complements:
- 🎥 [**Raw Videos**](../1_Raw_Videos/) - Original source material
- 🔄 [**Phase Recognition**](../2_Phase_Recognition/) - P03 phase identification context
- 🎯 [**Instance Segmentation**](../3_Instance_Segmentation/) - Instrument segmentation masks
- ⭐ [**Skill Assessment**](../5_Skill_Assessment/) - P03 skill evaluation using same clips
