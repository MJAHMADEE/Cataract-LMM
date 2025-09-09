# 🔄 Phase Recognition Dataset

[![Dataset Type](https://img.shields.io/badge/Task-Surgical%20Workflow%20Analysis-purple)](.)
[![Clips](https://img.shields.io/badge/Clips-150-green)](.)
[![Phases](https://img.shields.io/badge/Phases-13-blue)](.)

## 🎯 Overview

This specialized subset enables **surgical workflow analysis** through comprehensive phase recognition capabilities. The dataset provides curated video clips with detailed temporal annotations for understanding the sequential flow of cataract surgery procedures.

### 📊 Dataset Composition
- **🎬 Video Clips:** 150 carefully curated surgical segments
- **📋 Full-Clip Annotations:** 150 CSV files with frame-level phase labels
- **🔍 Phase-Specific Sub-clips:** Organized temporal segments for detailed analysis

---

## 📁 Directory Structure

```
2_Phase_Recognition/
├── videos/                     # 🎥 Source video clips (150 files)
├── annotations_full_video/     # 📊 Frame-level CSV annotations (150 files)
└── annotations_sub_clips/      # 🎞️ Phase-specific temporal segments
```

---

## 🏷️ File Naming Conventions

### 🎬 Video Clips (`videos/`)
```
PH_<ClipID>_<RawVideoID>_S<Site>.mp4
```

### 📊 Full-Clip Annotations (`annotations_full_video/`)
```
PH_<ClipID>_<RawVideoID>_S<Site>.csv
```
*Note: CSV filename matches corresponding video file*

### 🎞️ Phase Sub-Clips (`annotations_sub_clips/`)
```
PH_<ClipID>_<RawVideoID>_S<Site>_<SubclipOrder>_P<PhaseID>_<PhaseOccurrence>.mp4
```

---

## 🔍 Naming Components Explained

| Component | Description | Format | Example |
|-----------|-------------|---------|----------|
| `PH` | Phase Recognition prefix | Fixed | `PH` |
| `ClipID` | Unique clip identifier | 4-digit zero-padded | `0001` |
| `RawVideoID` | Source raw video ID | 4-digit zero-padded | `0042` |
| `S<Site>` | Hospital site code | S + digit | `S1`, `S2` |
| `SubclipOrder` | **Temporal order** in original raw video | 4-digit zero-padded | `0001`, `0015` |
| `PhaseID` | Phase identifier | P + 2-digit | `P01`, `P03` |
| `PhaseOccurrence` | **n-th occurrence** of phase in raw video | 2-digit zero-padded | `01`, `02` |

### 🎯 Example Files
```
PH_0001_0042_S1.mp4                    ← Video clip
PH_0001_0042_S1.csv                    ← Corresponding annotations
PH_0001_0042_S1_0001_P03_01.mp4       ← First Capsulorhexis sub-clip
PH_0001_0042_S1_0015_P07_02.mp4       ← Second I/A sub-clip (15th in order)
```

---

## 🏥 Surgical Phases Classification

The dataset covers **13 distinct surgical phases** in cataract surgery:

| Phase ID | 🏷️ Phase Name | 📝 Description |
|----------|---------------|----------------|
| `P01` | 🔪 **Incision** | Initial corneal incision |
| `P02` | 💧 **Viscoelastic** | Viscoelastic substance injection |
| `P03` | ⭕ **Capsulorhexis** | Circular anterior capsulotomy |
| `P04` | 💦 **Hydrodissection** | Lens cortex separation |
| `P05` | ⚡ **Phacoemulsification** | Lens nucleus fragmentation |
| `P06` | 🌊 **Irrigation-Aspiration** | Lens cortex removal |
| `P07` | ✨ **Capsule Polishing** | Capsule cleaning |
| `P08` | 👁️ **Lens Implantation** | IOL insertion |
| `P09` | 📐 **Lens Positioning** | IOL final positioning |
| `P10` | 🧹 **Viscoelastic Suction** | Viscoelastic removal |
| `P11` | 🚿 **Anterior Chamber Flushing** | Chamber irrigation |
| `P12` | 💊 **Tonifying-Antibiotics** | Medication administration |
| `P13` | ⏸️ **Idle** | Non-active surgical time |

> **💡 Key Note:** Phase `P03` (Capsulorhexis) is particularly important as it's featured in both [Instrument Tracking](../4_Instrument_Tracking/) and [Skill Assessment](../5_Skill_Assessment/) datasets.

---

## ⚠️ Data Access Notice

> **📁 Repository Contents:** This package includes **documentation only** — no videos or CSV files are included here.
> 
> **🔑 Data Access:** For information on accessing the actual dataset files, please refer to [`../data/README.md`](../README.md) for detailed access instructions.

---

## 🔗 Related Datasets

This phase recognition dataset complements:
- 🎥 [**Raw Videos**](../1_Raw_Videos/) - Source material
- 🎯 [**Instance Segmentation**](../3_Instance_Segmentation/) - Visual segmentation of the same clips  
- 🔍 [**Instrument Tracking**](../4_Instrument_Tracking/) - P03 (Capsulorhexis) tracking
- ⭐ [**Skill Assessment**](../5_Skill_Assessment/) - P03 (Capsulorhexis) skill evaluation
