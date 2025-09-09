# 🧪 Testing Suite - Surgical Instance Segmentation

This directory contains comprehensive testing frameworks to ensure the reliability, accuracy, and performance of the surgical instance segmentation framework.

## 🎯 Objective

Provide comprehensive testing infrastructure for:
- **🔍 Framework Validation**: Complete system integration testing
- **🧪 Unit Testing**: Individual component validation and reliability
- **📊 Performance Testing**: Speed and accuracy benchmarking
- **🏥 Medical Compliance**: Clinical deployment readiness validation

## 📂 Contents

### 📄 **Framework Validation** (`test_framework_validation.py`)
**Comprehensive integration testing suite**
- **Purpose**: Validate complete framework functionality
- **Coverage**: All modules, imports, configurations, and integrations
- **Validation**: Performance metrics consistency, naming conventions
- **Execution**: `python test_framework_validation.py`

### 📄 **Model Testing** (`test_models.py`)
**Model-specific functionality testing**
- **Purpose**: Validate model implementations and performance
- **Coverage**: YOLOv11, YOLOv8, Mask R-CNN, SAM model testing
- **Benchmarks**: Verify paper-reported performance metrics

### 📄 **Data Pipeline Testing** (`test_data_pipeline.py`)
**Dataset and preprocessing validation**
- **Purpose**: Ensure data integrity and processing accuracy
- **Coverage**: COCO loading, task mappings, augmentation pipelines
- **Standards**: Cataract-LMM naming convention compliance

## 🚀 Quick Testing

Run the comprehensive validation:
```bash
cd surgical-instance-segmentation
python tests/test_framework_validation.py
```

Expected output: All tests passing with framework validation success.
