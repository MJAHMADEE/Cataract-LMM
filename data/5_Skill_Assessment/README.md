# ⭐ Skill Assessment Dataset (Capsulorhexis)

[![Dataset Type](https://img.shields.io/badge/Task-Skill%20Assessment-gold)](.)
[![Clips](https://img.shields.io/badge/Clips-170-green)](.)
[![Phase](https://img.shields.io/badge/Phase-P03%20Capsulorhexis-purple)](.)
[![Metrics](https://img.shields.io/badge/Assessment%20Metrics-6-orange)](.)

## 🎯 Overview

This dataset provides **surgical skill assessment** for the Capsulorhexis phase of cataract surgery. It includes expert evaluations across multiple performance dimensions, enabling the development of automated skill assessment systems and surgical training tools.

### 📊 Dataset Composition
- **🎬 Assessment Clips:** 170 Capsulorhexis (P03) video segments
- **📋 Skill Annotations:** Multi-dimensional scoring across 6 key metrics
- **🏥 Expert Evaluation:** Professional surgical skill assessments
- **🎯 Objective Metrics:** Standardized scoring rubric for consistent evaluation

---

## 📁 Directory Structure

```
5_Skill_Assessment/
├── videos/                          # 🎥 Capsulorhexis video clips (170 files)
└── annotations/
    └── skill_scores.csv            # 📊 Comprehensive skill assessment scores
```

---

## 🏷️ File Naming Convention

### 🎬 Video Files (`videos/`)
```
SK_<ClipID>_S<Site>_P03.mp4
```

### 🔍 Naming Components
| Component | Description | Format | Example |
|-----------|-------------|---------|----------|
| `SK` | Skill Assessment prefix | Fixed | `SK` |
| `ClipID` | Unique clip identifier | 4-digit zero-padded | `0001`, `0042` |
| `S<Site>` | Hospital site code | S + digit | `S1`, `S2` |
| `P03` | Phase identifier (Capsulorhexis) | Fixed | `P03` |

### 🎯 Example Files
```
SK_0001_S1_P03.mp4    # First skill assessment clip from Farabi
SK_0042_S2_P03.mp4    # Clip from Noor Hospital
SK_0170_S1_P03.mp4    # Last assessment clip
```

---

## 📊 Skill Assessment Metrics

The dataset evaluates surgical performance across **6 comprehensive dimensions**:

| Metric ID | 🏷️ Assessment Dimension | 📝 Focus Area |
|-----------|-------------------------|---------------|
| **1** | 🔧 **Instrument Handling** | Tool grip, control, and manipulation efficiency |
| **2** | 🌊 **Motion** | Movement smoothness, precision, and economy |
| **3** | 🧬 **Tissue Handling** | Gentleness, respect for anatomical structures |
| **4** | 🔬 **Microscope Use** | Optimal visualization and focus management |
| **5** | 🎯 **Commencement of Flap** | Initial capsule engagement technique |
| **6** | ⭕ **Circular Completion** | Quality and continuity of circular tear |

### 📈 Scoring System
- **Scale:** Standardized scoring scale (specific details intentionally omitted for proprietary reasons)
- **Evaluators:** Expert ophthalmic surgeons with extensive cataract surgery experience
- **Consistency:** Inter-rater reliability ensured through calibration sessions

---

## 📋 Annotations Format

### 📊 skill_scores.csv Structure
```csv
clip_key,instrument_handling,motion,tissue_handling,microscope_use,commencement_of_flap,circular_completion
SK_0001_S1_P03,4.2,3.8,4.5,4.1,3.9,4.3
SK_0002_S1_P03,3.5,3.2,3.8,3.6,3.4,3.7
...
```

### 🔑 Key Features
- **🔑 Primary Key:** `clip_key` matches video filename stems (SK/TR prefix)
- **📊 Score Columns:** Six assessment dimensions as separate columns
- **🎯 Compatibility:** Keys work with both SK (Skill) and TR (Tracking) datasets
- **📈 Granular Data:** Enables detailed performance analysis

---

## 🏥 Capsulorhexis Skill Context

### ⭕ Why Capsulorhexis Assessment Matters
Capsulorhexis is considered one of the most technically demanding steps in cataract surgery:

- **🎯 Precision Required:** Creates foundation for entire procedure success
- **⚠️ Risk Factors:** Improper technique can lead to complications
- **👨‍⚕️ Learning Curve:** Significant skill development needed for mastery
- **🎓 Training Value:** Ideal for automated skill assessment systems

### 📊 Assessment Applications
- **🏥 Surgical Training:** Objective feedback for residents and fellows
- **🔬 Research:** Understanding skill acquisition patterns
- **🤖 AI Development:** Training automated assessment algorithms
- **📈 Quality Improvement:** Standardizing surgical competency evaluation

---

## 🔢 Dataset Statistics

| Metric | Value |
|--------|-------|
| 🎬 **Total Clips** | 170 procedures |
| ⏱️ **Assessment Duration** | Complete Capsulorhexis phase |
| 📊 **Evaluation Dimensions** | 6 skill metrics |
| 👨‍⚕️ **Expert Assessors** | Board-certified ophthalmologists |

---

## 💡 Usage Examples

### 📊 Loading Skill Scores (Python)
```python
import pandas as pd

# Load skill assessment data
scores_df = pd.read_csv('annotations/skill_scores.csv')

# Display basic statistics
print(f"Total assessments: {len(scores_df)}")
print(f"Average overall score: {scores_df.iloc[:, 1:].mean().mean():.2f}")

# Analyze specific metrics
print("\nMetric averages:")
for col in scores_df.columns[1:]:
    avg_score = scores_df[col].mean()
    print(f"{col}: {avg_score:.2f}")
```

### 🎯 Correlation Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create correlation matrix of skill metrics
corr_matrix = scores_df.iloc[:, 1:].corr()

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Skill Assessment Metrics Correlation')
plt.show()
```

---

## ⚠️ Data Access Notice

> **📁 Repository Contents:** This package contains **documentation only** — no video files or CSV data are included here.
> 
> **🔑 Data Access:** For information on accessing the actual dataset files, please refer to [`../README.md`](../README.md) for detailed access instructions.

---

## 🔗 Related Datasets

This skill assessment dataset works in conjunction with:
- 🎥 [**Raw Videos**](../1_Raw_Videos/) - Original source material
- 🔄 [**Phase Recognition**](../2_Phase_Recognition/) - P03 phase context and timing
- 🎯 [**Instance Segmentation**](../3_Instance_Segmentation/) - Visual analysis of instruments
- 🔍 [**Instrument Tracking**](../4_Instrument_Tracking/) - Same P03 clips with tracking data

> **🔗 Cross-Dataset Compatibility:** The skill scores CSV uses keys compatible with both `SK_` (skill assessment) and `TR_` (instrument tracking) filename prefixes, enabling multi-modal analysis of the same surgical procedures.
