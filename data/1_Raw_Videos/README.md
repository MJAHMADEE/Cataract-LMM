# 🎥 Raw Videos Dataset

[![Dataset Status](https://img.shields.io/badge/Dataset-Documentation%20Only-blue)](.) 
[![Total Videos](https://img.shields.io/badge/Videos-3,000-green)](.)
[![Sites](https://img.shields.io/badge/Sites-2-orange)](.)

## 📊 Dataset Overview

This collection contains the **raw, unprocessed source videos** from which all specialized subsets in the Cataract-LMM dataset were derived. These videos represent the foundational material for surgical workflow analysis, instance segmentation, instrument tracking, and skill assessment tasks.

### 🏥 Distribution Summary
- **📹 Total Procedures:** 3,000 complete cataract surgeries
- **🏥 Farabi Hospital (S1):** 2,930 procedures (97.7%)
- **🏥 Noor Hospital (S2):** 70 procedures (2.3%)

---

## 📝 File Naming Convention

### 🏷️ Pattern Structure
```
RV_<RawVideoID>_S<Site>.mp4
```

### 📋 Components Breakdown
| Component | Description | Format | Example |
|-----------|-------------|---------|----------|
| `RV` | Raw Video prefix | Fixed | `RV` |
| `RawVideoID` | Unique video identifier | 4-digit zero-padded | `0001`, `0002` |
| `S<Site>` | Hospital site code | S + digit | `S1`, `S2` |

### 🎯 Example Files
```
RV_0001_S1.mp4  ← First video from Farabi Hospital
RV_0002_S2.mp4  ← Second video from Noor Hospital
RV_2930_S1.mp4  ← Last video from Farabi Hospital
```

### 🏥 Site Code Reference
| Code | Hospital | Location |
|------|----------|----------|
| `S1` | 🏥 Farabi Hospital | Primary site (2,930 videos) |
| `S2` | 🏥 Noor Hospital | Secondary site (70 videos) |

---

## ⚠️ Data Access Notice

> **📁 Repository Contents:** This package contains **documentation only** — no actual video files are included here.
> 
> **🔑 Data Access:** For information on accessing the actual video files, please refer to the main [`data/README.md`](../README.md) file for detailed access instructions.

---

## 🔗 Related Datasets

This raw video collection serves as the source for:
- 📊 [**Phase Recognition**](../2_Phase_Recognition/) - Surgical workflow analysis
- 🎯 [**Instance Segmentation**](../3_Instance_Segmentation/) - Instrument and tissue segmentation  
- 🔍 [**Instrument Tracking**](../4_Instrument_Tracking/) - Capsulorhexis instrument tracking
- ⭐ [**Skill Assessment**](../5_Skill_Assessment/) - Surgical skill evaluation
