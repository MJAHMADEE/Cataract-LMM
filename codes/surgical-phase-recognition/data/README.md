# 📊 Data Pipeline for Cataract-LMM Phase Recognition

This directory contains the data handling and dataset implementations for the Cataract-LMM surgical phase recognition task. The data pipeline implements the exact specifications described in the academic paper's "Phase Recognition Dataset Description" section.

## 🎯 Objective

Provide a comprehensive data pipeline for the Cataract-LMM phase recognition task:
- **150 annotated videos** (129 from Farabi, 21 from Noor hospitals)  
- **28.6 hours** total duration with frame-level phase labels
- **13-phase taxonomy** covering complete phacoemulsification procedure
- **Multi-center domain adaptation** support for generalization evaluation

## 📚 Academic Paper Alignment

All data components implement the dataset specifications from:
*"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"*

The pipeline supports the paper's experimental protocol:
- Training exclusively on **Farabi hospital** videos (80 videos)
- Validation on **Farabi hospital** videos (26 videos)  
- Testing on both **in-domain (Farabi)** and **out-of-domain (Noor)** videos

## 📁 Contents

### � Core Dataset Classes

- **`sequential_dataset.py`** - 📊 Primary `SequentialSurgicalPhaseDatasetAugOverlap` class (paper-compatible)
- **`datasets.py`** - 📋 Extended dataset classes for various experimental setups
- **`data_utils.py`** - 🔧 Data processing utilities and helper functions
- **`__init__.py`** - 📦 Data module interface and exports

## �🏗️ Core Components

### 📊 SequentialSurgicalPhaseDatasetAugOverlap (`sequential_dataset.py`)
**Primary dataset class** implementing the exact data loading protocol used in the paper's experiments:

```python
from data.sequential_dataset import SequentialSurgicalPhaseDatasetAugOverlap
from transform import get_phase_mapping

# Paper-compliant dataset setup
label_mapping = get_phase_mapping("11_phase")  # Current implementation

# Training dataset (Farabi hospital only)
train_dataset = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=label_mapping,
    root_dir="/path/to/farabi/videos",  # Training on Farabi only
    lookback_window=10,  # Paper's temporal window
    max_sequences_per_phase=1500,  # Balanced sampling  
    overlap=0,  # No overlap for training
    frame_interval=1,  # Paper's frame interval
    test=False  # Training mode with augmentation
)

# Test dataset (for domain evaluation)
test_dataset_farabi = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=label_mapping,
    root_dir="/path/to/farabi/test",  # In-domain test
    lookback_window=10,
    max_sequences_per_phase=1500,
    overlap=0,
    frame_interval=1,
    test=True  # Test mode without augmentation
)

test_dataset_noor = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=label_mapping,
    root_dir="/path/to/noor/videos",  # Out-of-domain test
    lookback_window=10,
    max_sequences_per_phase=1500,
    overlap=0,
    frame_interval=1,
    test=True  # Test mode for domain adaptation evaluation
)
```

**Key Features (Paper-Validated):**
- ✅ **Multi-center support**: Farabi and Noor hospital data handling
- ✅ **Temporal modeling**: 10-frame lookback window (paper specification)
- ✅ **Balanced sampling**: Equal representation across surgical phases
- ✅ **Domain evaluation**: Separate datasets for in-domain vs out-of-domain testing
- ✅ **Frame-level labels**: Precise phase annotations for each video frame

# Temporal dataset
temporal_dataset = TemporalSurgicalDataset(data_dir="/path/to/data", sequence_length=16)
```

### 🛠️ Data Utilities (`data_utils.py`)
**Comprehensive utilities** for data processing and management:

### 🔧 Data Utilities (`data_utils.py`)
**Supporting utilities** for data management and validation:

```python
from data.data_utils import (
    CataractLMMDataManager,
    PhaseAnnotationProcessor, 
    DomainAdaptationSplitter,
    DatasetValidator
)

# Cataract-LMM dataset management
manager = CataractLMMDataManager()
dataset_info = manager.load_phase_recognition_subset()

# Domain adaptation splits (paper protocol)
splitter = DomainAdaptationSplitter()
splits = splitter.create_paper_splits(
    farabi_videos=farabi_video_list,
    noor_videos=noor_video_list,
    train_ratio=0.8  # 80% Farabi for training
)

# Validate dataset integrity
validator = DatasetValidator()
validation_results = validator.validate_cataract_lmm_format(dataset_path)
```

## 📊 Cataract-LMM Phase Mapping

### Standard 13-Phase Taxonomy (Paper Definition)
```python
CATARACT_LMM_PHASES_13 = {
    "Incision": 0,                   # Initial corneal incision
    "Viscoelastic": 1,               # Viscoelastic agent injection
    "Capsulorhexis": 2,              # Opening of the anterior capsule
    "Hydrodissection": 3,            # Separation of lens nucleus from cortex
    "Phacoemulsification": 4,        # Ultrasonic lens fragmentation and removal
    "Irrigation Aspiration": 5,      # Cortex removal using irrigation/aspiration
    "Capsule Polishing": 6,          # Posterior capsule cleaning
    "Lens Implantation": 7,          # Intraocular lens implantation
    "Lens Positioning": 8,           # Adjustment of lens position in capsule
    "Viscoelastic Suction": 9,       # Removal of viscoelastic material
    "Anterior Chamber Flushing": 10, # Final irrigation of anterior chamber
    "Tonifying Antibiotics": 11,     # Instillation of antibiotics/medication
    "Idle": 12                       # Surgical inactivity or instrument exchange
}
```

### Reduced 11-Phase Mapping (Current Implementation)
```python
# 11-phase mapping used in current notebooks for compatibility
CATARACT_LMM_PHASES_11 = {
    "Incision": 0,
    "Viscoelastic": 1,
    "Capsulorhexis": 2,
    "Hydrodissection": 3,
    "Phacoemulsification": 4,
    "IrrigationAspiration": 5,
    "CapsulePolishing": 6,
    "LensImplantation": 7,
    "LensPositioning": 8,
    "ViscoelasticSuction": 9,
    "TonifyingAntibiotics": 10,
}
```

## 🎯 Dataset Features (Paper-Validated)

### ⚖️ Multi-Center Balanced Sampling
The dataset provides sophisticated balancing following paper methodology:

- **Phase balancing**: Equal representation across all surgical phases
- **Hospital balancing**: Appropriate splits between Farabi and Noor data
- **Temporal balancing**: Maintains sequence integrity while balancing classes
- **Domain-aware sampling**: Supports in-domain and out-of-domain evaluation

### 🕒 Temporal Modeling (Paper Specification)
Advanced temporal sequence handling matching paper's approach:

- **Lookback windows**: 10-frame temporal context (paper standard)
- **Frame intervals**: 1-frame interval with 4 fps effective sampling
- **Sequence extraction**: Non-overlapping sequences for training
- **Boundary handling**: Smart handling of video start/end boundaries

Seamless integration with the transform module following paper preprocessing:

```python
from transform import transform_train, transform_test, get_phase_mapping

# Automatic transform application based on dataset mode
phase_mapping = get_phase_mapping("11_phase")

train_dataset = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=phase_mapping,
    root_dir="/path/to/farabi/training",
    test=False  # Uses transform_train with augmentation
)

test_dataset = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=phase_mapping,
    root_dir="/path/to/test",
    test=True   # Uses transform_test without augmentation
)
```

## 📊 Dataset Statistics (Paper Reference)

### 📈 Phase Recognition Subset Overview
Based on the academic paper's "Phase Recognition Dataset Description":

| **Metric** | **Value** | **Details** |
|------------|-----------|-------------|
| **Total Videos** | 150 | Subset from 3,000 total Cataract-LMM videos |
| **Total Duration** | 28.6 hours | Complete annotated content |
| **Farabi Hospital** | 129 videos | Source hospital for training/validation |
| **Noor Hospital** | 21 videos | Target hospital for domain adaptation |
| **Phase Count** | 13 phases | Complete phacoemulsification taxonomy |
| **Frame-level Labels** | Yes | Precise temporal annotations |

### 🏥 Multi-Center Distribution
```python
HOSPITAL_DISTRIBUTION = {
    "Farabi": {
        "videos": 129,
        "percentage": 86.0,
        "role": "Training/Validation (In-domain)"
    },
    "Noor": {
        "videos": 21, 
        "percentage": 14.0,
        "role": "Testing (Out-of-domain)"
    }
}
```

### ⚖️ Class Imbalance Characteristics
The dataset exhibits natural class imbalance reflecting real surgical procedures:
- **Dominant phases**: Phacoemulsification (longest duration)
- **Brief phases**: Capsule Polishing, Incision (shorter duration)
- **Balancing strategy**: Up to 1500 sequences per phase maximum

## 🚀 Quick Start Guide

### 1. Basic Dataset Creation (Paper Protocol)
```python
from data.sequential_dataset import SequentialSurgicalPhaseDatasetAugOverlap
from transform import get_phase_mapping

# Paper-standard phase mapping
label_to_idx = get_phase_mapping("11_phase")

# Training dataset (Farabi only - paper protocol)
train_dataset = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=label_to_idx,
    root_dir="/path/to/farabi/training",
    lookback_window=10,  # Paper's temporal window
    max_sequences_per_phase=1500,  # Paper's balancing limit
    overlap=0,  # No overlap for training
    frame_interval=1,  # Paper's frame interval
    test=False  # Training mode with augmentation
)

# In-domain test dataset (Farabi)
test_farabi = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=label_to_idx,
    root_dir="/path/to/farabi/test",
    lookback_window=10,
    max_sequences_per_phase=1500,
    overlap=0,
    frame_interval=1,
    test=True  # Test mode without augmentation
)

# Out-of-domain test dataset (Noor) - for domain adaptation evaluation
test_noor = SequentialSurgicalPhaseDatasetAugOverlap(
    label_to_idx=label_to_idx,
    root_dir="/path/to/noor/test",
    lookback_window=10,
    max_sequences_per_phase=1500,
    overlap=0,
    frame_interval=1,
    test=True
```

### 2. DataLoader Creation (Paper Protocol)
```python
from torch.utils.data import DataLoader

# Training DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,  # Paper's batch size
    shuffle=True,  # Shuffle for training
    num_workers=4,
    pin_memory=True,
    drop_last=True  # Consistent batch sizes
)

# In-domain test DataLoader
test_farabi_loader = DataLoader(
    test_farabi,
    batch_size=32,
    shuffle=False,  # No shuffle for evaluation
    num_workers=4,
    pin_memory=True
)

# Out-of-domain test DataLoader (domain adaptation)
test_noor_loader = DataLoader(
    test_noor,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

### 3. Domain Adaptation Evaluation (Paper Method)
```python
# Evaluate domain generalization as described in paper
def evaluate_domain_adaptation(model, farabi_loader, noor_loader):
    """
    Evaluate domain adaptation following paper's protocol.
    Expected: ~22% average F1-score drop from Farabi to Noor.
    """
    # In-domain evaluation (Farabi)
    farabi_results = evaluate_model(model, farabi_loader)
    
    # Out-of-domain evaluation (Noor)  
    noor_results = evaluate_model(model, noor_loader)
    
    # Calculate domain shift metrics
    f1_drop = farabi_results['macro_f1'] - noor_results['macro_f1']
    performance_degradation = (f1_drop / farabi_results['macro_f1']) * 100
    
    print(f"In-domain (Farabi) F1: {farabi_results['macro_f1']:.1f}%")
    print(f"Out-of-domain (Noor) F1: {noor_results['macro_f1']:.1f}%") 
    print(f"Domain shift F1 drop: {f1_drop:.1f}% ({performance_degradation:.1f}%)")
    
    return {
        'farabi_f1': farabi_results['macro_f1'],
        'noor_f1': noor_results['macro_f1'], 
        'domain_shift': f1_drop,
        'degradation_percent': performance_degradation
    }
```

## 📁 Expected Data Structure

The dataset expects the following directory structure following Cataract-LMM format:

```
phase_recognition_data/
├── farabi_hospital/
│   ├── training_videos/
│   │   ├── video_001/
│   │   │   ├── frame_phase_mapping.txt  # Frame-to-phase mapping
│   │   │   ├── frame_000001.jpg
│   │   │   ├── frame_000002.jpg
│   │   │   └── ...
│   │   └── video_002/
│   │       └── ...
│   └── test_videos/
│       └── ...
└── noor_hospital/
    └── test_videos/
        ├── video_201/
        │   ├── frame_phase_mapping.txt
        │   ├── frame_000001.jpg
        │   └── ...
        └── ...
```

### Frame-Phase Mapping Format
```txt
frame_000001.jpg Incision
frame_000002.jpg Incision  
frame_000003.jpg Viscoelastic
frame_000004.jpg Viscoelastic
...
```

## 🔗 Integration Points

- **📓 Core Logic**: Dataset classes integrate with [`../notebooks/`](../notebooks/) for research workflows
- **🧠 Models**: Compatible with all model architectures in [`../models/`](../models/)
- **🔧 Transforms**: Works with preprocessing pipeline in [`../transform.py`](../transform.py)
- **✅ Validation**: Used by [`../validation/`](../validation/) for systematic evaluation

---

> **💡 Pro Tip**: Use the paper's exact dataset splits and parameters to reproduce published results and ensure fair comparisons with the benchmarks!
```

### 3. Data Inspection
```python
# Print dataset information
print(f"Dataset length: {len(test_dataset)}")

# Get sample batch
for frames, labels, frame_paths in test_loader:
    print(f"Batch shape: {frames.shape}")  # [batch_size, sequence_length, channels, height, width]
    print(f"Labels shape: {labels.shape}")  # [batch_size]
    break
```

## 🔧 Data Processing Pipeline

### 1. Video Frame Extraction
```python
from data.data_utils import VideoProcessor

processor = VideoProcessor()
frames = processor.extract_frames("/path/to/video.mp4", output_dir="/path/to/frames")
```

### 2. Annotation Processing
```python
from data.data_utils import PhaseAnnotationProcessor

processor = PhaseAnnotationProcessor()
annotations = processor.load_annotations("/path/to/annotations.json")
processed = processor.convert_to_frame_labels(annotations, fps=30)
```

### 3. Data Validation
```python
from data.data_utils import DataValidator

validator = DataValidator()
report = validator.validate_dataset("/path/to/dataset", label_mapping)
```

## 📈 Performance Optimization

### Efficient Loading
- **Multi-processing**: Configurable `num_workers` for parallel data loading
- **Pin memory**: GPU memory optimization with `pin_memory=True`
- **Caching**: Intelligent frame caching for repeated access
- **Lazy loading**: Memory-efficient frame loading on demand

### Memory Management
- **Batch processing**: Configurable batch sizes for memory constraints
- **Frame compression**: Optional frame compression for storage efficiency
- **Garbage collection**: Automatic cleanup of unused frames

## 🔬 Research Integration

### Notebook Compatibility
The data pipeline exactly matches the notebook's data loading patterns:

1. **Same class names**: `SequentialSurgicalPhaseDatasetAugOverlap`
2. **Same parameters**: All constructor parameters match notebook usage
3. **Same behavior**: Identical balancing and sampling logic
4. **Same output format**: Compatible tensor shapes and data types

### Validation Integration
Perfect compatibility with the validation pipeline:

```python
from validation.comprehensive_validator import validate_model

# Use dataset directly with validation function
results = validate_model("model_name", checkpoint_path, model, test_loader, num_classes=11)
```

## 🛡️ Error Handling

Robust error handling throughout the data pipeline:

- **File validation**: Automatic checking of video file integrity
- **Path verification**: Validation of data directory structures
- **Format checking**: Ensuring compatible video and annotation formats
- **Graceful degradation**: Fallback options for missing dependencies

---

> **💡 Pro Tip**: Always start with the exact dataset configuration shown in the primary validation notebook to ensure compatibility with trained models!
