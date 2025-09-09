# ⏱️ Phase Recognition

## Notebook Overview

This directory contains the comprehensive validation notebook for surgical phase recognition on the Cataract-LMM dataset.

### **📓 Notebook**

| Notebook | Description | Purpose |
|----------|-------------|---------|
| `phase_validation_comprehensive.ipynb` | Comprehensive phase recognition validation and evaluation | Complete pipeline for model validation on surgical phase data |

### **🎯 Validation Framework**

**Phase Recognition Task:**
- **Input**: Video clips from cataract surgery procedures
- **Output**: Frame-level phase predictions for 11 surgical phase classes
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and confusion matrix analysis

**Dataset Configuration:**
- **Training Data**: Phase-annotated surgical videos
- **Preprocessing Pipeline**: Videos are processed by extracting frames at 4 FPS and resizing them to 224x224 pixels. Annotations are mapped from the original video framerate to the extracted frames.
- **Validation Split**: The notebook implements a random split of video folders into training and testing sets.
- **Annotation Format**: CSV files with frame-level phase labels for each video.

### **📊 Evaluation Metrics**

**Performance Measures:**
- **Frame-level Accuracy**: Percentage of correctly classified frames
- **Phase-level Metrics**: Precision, Recall, F1-score (macro-averaged and per-class)
- **Confusion Matrix**: Detailed analysis of inter-phase misclassifications

**Clinical Phases (12 categories mapped to 11 classes):**
The model is trained to recognize the following surgical phases. Note that `Viscoelastic` and `AnteriorChamberFlushing` are mapped to the same class label during training.

1. Incision
2. Viscoelastic / AnteriorChamberFlushing
3. Capsulorhexis
4. Hydrodissection
5. Phacoemulsification
6. IrrigationAspiration
7. CapsulePolishing
8. LensImplantation
9. LensPositioning
10. ViscoelasticSuction
11. TonifyingAntibiotics

### **🔧 Model Architecture**

**Validated Models:**
The notebook includes validation code for a wide range of architectures:

- **Video Transformers**:
    - `mvit_v1_b`
    - `swin3d_t`
- **3D CNNs**:
    - `slow_r50`
    - `r3d_18`
    - `mc3_18`
    - `r2plus1d_18`
    - `x3d_xs`
- **Two-Stage & Hybrid Models (CNN + Sequential)**:
    - **CNN+RNN**: `Resnet50_LSTM`, `EfficientNetB5_LSTM`, `Resnet50_GRU`, `EfficientNetB5_GRU`
    - **CNN+TCN**: `Resnet50_tecno` (MultiStageModel), `EfficientNetB5_tecno` (MultiStageModel)

### **💡 Validation Strategy**

**Cross-Validation:**
- **Random split**: The primary strategy shown in the notebook is a random split of surgical videos into training and validation sets.
- **Site-based validation**: The final analysis cells demonstrate evaluation across different datasets (e.g., "noor" and "farabi"), effectively performing cross-site or cross-hospital validation.

**Quality Metrics:**
- **Clinical correlation**: Agreement with expert annotations is measured via standard classification metrics.
- **Robustness testing**: The framework supports evaluating performance across different video conditions.
- **Generalization**: Cross-site validation results are generated to assess model generalization.

### **🚀 Usage Instructions**

1.  **Prepare Video Data**: Place surgical videos in a designated folder.
2.  **Run Preprocessing**: Use the notebook cells to extract, resize, and crop frames from videos (e.g., at 4 FPS, 224x224 resolution). The notebook will also map frame numbers to ground-truth phase labels from annotation CSVs.
3.  **Split Data**: Organize the processed frame folders into `train` and `test` directories.
4.  **Configure Validation**: In the notebook, select a model architecture, provide the path to its pre-trained checkpoint (`.ckpt` file), and point the data loader to the validation set.
5.  **Run Validation Pipeline**: Execute the validation cells to run the model on the test data and compute performance metrics.
6.  **Analyze and Report**: The notebook will generate CSV files with detailed metrics and a confusion matrix image for each model. The final cells can be used to merge results and create summary tables.

### **📋 Expected Outputs**

**Validation Results:**
- CSV file per model with comprehensive performance metrics (accuracy, precision, recall, F1).
- Per-class breakdown of all metrics.
- Confusion matrix visualization saved as a PNG image.
- Final summary tables comparing the performance of all validated models across different datasets.

---

## 👨‍💻 Contributors

[![MJAHMADEE](https://img.shields.io/badge/Lead%20Developer-@MJAHMADEE-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MJAHMADEE)
[![amirtaslimi](https://img.shields.io/badge/Phase%20Recognition%20Expert-@amirtaslimi-green?style=for-the-badge&logo=github&logoColor=white)](https://github.com/amirtaslimi)
