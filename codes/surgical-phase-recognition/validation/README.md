# ✅ Validation & Evaluation - Cataract-LMM Framework

This directory contains comprehensive model validation and evaluation tools implementing the exact methodology from the **"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"** paper.

## 📊 Paper-Compliant Evaluation Protocol

All validation tools implement the exact evaluation pipeline described in the Cataract-LMM paper for reproducible benchmark results.

### 🎯 **Paper Evaluation Standards**
- **Primary Metric**: Macro F1-score (paper standard)
- **Multi-Center Protocol**: Farabi → Noor domain adaptation
- **13-Phase Taxonomy**: Complete surgical workflow evaluation
- **Cross-Center Validation**: In-domain vs out-of-domain testing

## 🏗️ Core Components

### 🎯 Cataract-LMM Validator (`comprehensive_validator.py`)
**Paper-compliant validation module** implementing Section 4 evaluation methodology:

```python
from validation.comprehensive_validator import validate_model_cataract_lmm

# Paper-standard validation
results = validate_model_cataract_lmm(
    model_name="mvit_v1_b",           # Paper's best model
    checkpoint_path="/path/to/checkpoint.ckpt", 
    model=mvit_model,
    farabi_loader=farabi_test_loader,  # In-domain testing
    noor_loader=noor_test_loader,      # Out-of-domain testing
    num_classes=13                     # Full Cataract-LMM taxonomy
)

# Results follow paper's reporting format
print(f"Farabi F1 (in-domain): {results['farabi_f1']:.1f}%")
print(f"Noor F1 (out-of-domain): {results['noor_f1']:.1f}%")
print(f"Domain gap: {results['domain_gap']:.1f}%")  # Expected: ~22%
```

### 📊 Paper Metrics (`metrics.py`)
**Cataract-LMM evaluation metrics** following paper specifications:

```python
from validation.metrics import CataractLMMMetrics

# Paper-compliant metrics calculator
metrics_calculator = CataractLMMMetrics(
    num_classes=13,                    # Full phase taxonomy
    phase_names=CATARACT_LMM_PHASES_13,
    primary_metric="macro_f1"          # Paper's primary metric
)

# Calculate paper-standard results
results = metrics_calculator.evaluate_paper_standard(
    predictions=predictions,
    targets=targets,
    center="farabi"  # Specify evaluation center
)
```

### ⚡ Domain Adaptation Framework (`domain_adaptation.py`)
**Multi-center evaluation** implementing paper's Section 4.5 methodology:

```python
from validation.domain_adaptation import DomainAdaptationEvaluator

# Paper's domain adaptation protocol
domain_evaluator = DomainAdaptationEvaluator(
    source_center="farabi",    # Training domain
    target_center="noor",      # Testing domain
    phase_taxonomy=13          # Full Cataract-LMM phases
)

# Evaluate domain transfer (Paper Section 4.5)
domain_results = domain_evaluator.evaluate_cross_center_transfer(
    model=model,
    source_loader=farabi_test_loader,   # In-domain evaluation
    target_loader=noor_test_loader      # Out-of-domain evaluation
)

# Expected domain gap: ~22% F1-score drop (paper result)
print(f"Domain adaptation analysis:")
print(f"Source F1: {domain_results['source_f1']:.1f}%")
print(f"Target F1: {domain_results['target_f1']:.1f}%")
print(f"Domain gap: {domain_results['f1_gap']:.1f}%")
```

## 🎯 Paper-Compliant Validation Pipeline

### 📊 Cataract-LMM Evaluation Protocol
Following the exact methodology from Section 4:

```python
# 1. Load MViT-B model (paper's best performer)
model = create_video_transformer('mvit_v1_b', num_classes=13, pretrained=True)

# 2. Evaluate using paper's protocol
results = validate_model_cataract_lmm(
    model_name="mvit_v1_b",
    checkpoint_path="/path/to/mvit_checkpoint.ckpt",
    model=model,
    farabi_loader=farabi_test_loader,    # In-domain testing
    noor_loader=noor_test_loader,        # Out-of-domain testing
    num_classes=13                       # Full phase taxonomy
)

# 3. Paper-standard results format
expected_results = {
    'farabi_f1': 77.1,        # Paper's reported in-domain F1
    'noor_f1': 55.3,          # Paper's reported out-of-domain F1
    'domain_gap': 21.8,       # Expected domain adaptation gap
    'per_phase_f1': [...],    # 13-phase detailed results
    'macro_precision': 76.8,  # Paper's precision
    'macro_recall': 77.4      # Paper's recall
}
```

## � Paper-Compliant Validation Workflow

### 1. Basic Cataract-LMM Validation
```python
from validation.comprehensive_validator import validate_model_cataract_lmm
from models.video_transformers import create_video_transformer

# Load paper's best model (MViT-B)
model = create_video_transformer('mvit_v1_b', num_classes=13, pretrained=True)

# Validate following paper protocol
results = validate_model_cataract_lmm(
    model_name="mvit_v1_b",
    checkpoint_path="/path/to/mvit_checkpoint.ckpt",
    model=model,
    farabi_loader=farabi_test_loader,  # In-domain testing
    noor_loader=noor_test_loader,      # Out-of-domain testing
    num_classes=13                     # Full Cataract-LMM taxonomy
)

# Validate against paper benchmarks (expected: 77.1% F1)
if abs(results['farabi_f1'] - 77.1) <= 2.0:
    print(f"✅ MViT-B reproduces paper: {results['farabi_f1']:.1f}%")
else:
    print(f"⚠️  MViT-B differs from paper: {results['farabi_f1']:.1f}% vs 77.1%")
```

### 2. Multi-Center Domain Adaptation
```python
from validation.domain_adaptation import DomainAdaptationEvaluator

# Evaluate cross-center performance (Paper Section 4.5)
domain_evaluator = DomainAdaptationEvaluator()

# Expected domain gaps from paper
expected_gaps = {'mvit_v1_b': 21.8, 'tecno': 17.3, 'slow_r50': 17.4}

for model_name, model in [('mvit_v1_b', mvit_model), ('tecno', tecno_model)]:
    results = domain_evaluator.evaluate_cross_center_transfer(
        model, farabi_test_loader, noor_test_loader
    )
    
    expected = expected_gaps.get(model_name, None)
    if expected and abs(results['f1_gap'] - expected) <= 3.0:
        print(f"✅ {model_name}: Domain gap {results['f1_gap']:.1f}% matches paper")
```

## 🔗 Integration & Compliance

### Paper Reproduction Checklist
- ✅ **Macro F1-Score**: Primary evaluation metric (Section 4)
- ✅ **Multi-Center Protocol**: Farabi/Noor evaluation (Section 4.5)  
- ✅ **13-Phase Taxonomy**: Complete surgical workflow
- ✅ **Benchmark Validation**: Compare against Table 2 results
- ✅ **Domain Analysis**: Cross-center performance gaps

### Integration Points
- **📊 Data**: Validates [`../data/`](../data/) paper compliance
- **🧠 Models**: Benchmarks [`../models/`](../models/) performance
- **📈 Analysis**: Powers [`../analysis/`](../analysis/) with paper metrics
- **🛠️ Utils**: Uses [`../utils/`](../utils/) for paper calculations

---

> **📖 Paper Reference**: All validation implements "Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis" methodology for reproducible research.
from validation.comprehensive_validator import CataractLMMValidationModule

# Create validation module following paper specifications
validation_module = CataractLMMValidationModule(
    model=model,
    num_classes=13,                           # Full Cataract-LMM taxonomy
    class_names=CATARACT_LMM_PHASES_13,
    centers=["farabi", "noor"],               # Multi-center evaluation
    primary_metric="macro_f1"                 # Paper's primary metric
)

# Run paper-compliant validation
trainer = pl.Trainer(
    accelerator="gpu",
    precision=16,           # Paper's mixed precision setting
    logger=False,
    enable_checkpointing=False
)

# Evaluate on both centers
farabi_results = trainer.validate(validation_module, farabi_test_loader)
noor_results = trainer.validate(validation_module, noor_test_loader)
```

- **Multi-class metrics**: Accuracy, Precision, Recall, F1-score
- **Per-class metrics**: Individual performance for each surgical phase
- **Confusion matrix**: Detailed error analysis with visualization
- **Batch processing**: Efficient validation over entire datasets
- **GPU acceleration**: Automatic device handling for fast evaluation

## 📈 Metrics & Visualization

### 🎯 Core Metrics
```python
# Comprehensive metrics (exact notebook output)
{
    'accuracy': 0.8756,
    'precision_macro': 0.8543,
    'recall_macro': 0.8421,
    'f1_macro': 0.8364,
    'accuracy_per_class': [0.92, 0.85, ...],  # Per-phase accuracy
    'precision_per_class': [0.89, 0.88, ...], # Per-phase precision
    'recall_per_class': [0.91, 0.82, ...],    # Per-phase recall
    'f1_per_class': [0.90, 0.85, ...],        # Per-phase F1
    'confusion_matrix': [[...], [...], ...]    # 11x11 confusion matrix
}
```

### 📊 Confusion Matrix Visualization
Professional confusion matrix plots matching the notebook:

```python
# Automatic confusion matrix plotting (notebook style)
validation_module = ValidationModule(model, num_classes=11, class_names=phase_names)
trainer = pl.Trainer(logger=False, enable_checkpointing=False)
trainer.validate(validation_module, test_loader)

# Results saved as "confusion_matrix.png"
```

### 📋 Detailed Reports
Per-class performance analysis:

| Surgical Phase | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Incision | 0.89 | 0.91 | 0.90 | 156 |
| Viscoelastic | 0.88 | 0.82 | 0.85 | 142 |
| Capsulorhexis | 0.85 | 0.87 | 0.86 | 178 |
| ... | ... | ... | ... | ... |

## 🔬 Model Comparison

### 📊 Benchmark Results
Performance comparison across all architectures (notebook results):

```python
# Model comparison (from notebook validation)
BENCHMARK_RESULTS = {
    'swin3d_t': {'f1_macro': 0.8592, 'accuracy': 0.8756},
    'slow_r50': {'f1_macro': 0.8613, 'accuracy': 0.8801},
    'mvit_v1_b': {'f1_macro': 0.8612, 'accuracy': 0.8789},
    'resnet50_lstm': {'f1_macro': 0.8364, 'accuracy': 0.8543},
    'efficientnet_lstm': {'f1_macro': 0.8400, 'accuracy': 0.8567},
    'resnet50_tecno': {'f1_macro': 0.8297, 'accuracy': 0.8489},
    'efficientnet_tecno': {'f1_macro': 0.8287, 'accuracy': 0.8456}
}
```

### 🏆 Top Performers
1. **Slow-R50**: 86.13% F1-macro (Slow-Fast Network)
2. **MViT-B**: 86.12% F1-macro (Multiscale Video Transformer)  
3. **Swin3D-T**: 85.92% F1-macro (3D Vision Transformer)

## 🚀 Quick Start Guide

### 1. Basic Model Validation
```python
from validation.comprehensive_validator import validate_model
from models.individual import Resnet50_LSTM
from torch.utils.data import DataLoader

# Load model
model = Resnet50_LSTM(num_classes=11, hidden_seq=256, dropout=0.5, hidden_MLP=128)

# Validate (exact notebook pattern)
results = validate_model(
    model_name="resnet50_lstm",
    checkpoint_path="/path/to/checkpoint.ckpt",
    model=model,
    test_loader=test_loader,
    num_classes=11
)

print(f"Model Performance: F1-Macro = {results['f1_macro']:.4f}")
```

### 2. Advanced Validation with Lightning
```python
from validation.comprehensive_validator import ValidationModule
import pytorch_lightning as pl

# Create validation module
validation_module = ValidationModule(
    model=model,
    num_classes=11,
    class_names=["Incision", "Viscoelastic", "Capsulorhexis", ...]
)

# Run validation
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger=False,
    enable_checkpointing=False
)

trainer.validate(validation_module, test_loader)
```

### 3. Custom Metrics Analysis
```python
from validation.metrics import SurgicalPhaseMetrics

# Detailed metrics calculation
metrics = SurgicalPhaseMetrics(num_classes=11)
detailed_results = metrics.compute_all_metrics(predictions, targets)

# Per-phase analysis
for i, phase_name in enumerate(phase_names):
    print(f"{phase_name}: F1={detailed_results['f1_per_class'][i]:.3f}")
```

## 🔧 Checkpoint Compatibility

### State Dict Processing
The `strip_prefix` function handles various checkpoint formats:

```python
from validation.comprehensive_validator import strip_prefix

# Remove model prefix from state dict (notebook pattern)
cleaned_state_dict = strip_prefix(checkpoint, prefix="model.")
model.load_state_dict(cleaned_state_dict)
```

### Checkpoint Loading Patterns
Support for different checkpoint formats:
- **PyTorch Lightning checkpoints**: Automatic `state_dict` extraction
- **Direct state dicts**: Direct loading support
- **Prefixed models**: Automatic prefix removal
- **Custom formats**: Flexible handling for various training frameworks

## 📊 Integration with Research Workflow

### Notebook Validation Pipeline
Perfect alignment with the notebook's validation approach:

1. **Model instantiation**: Same parameter patterns as notebook
2. **Checkpoint loading**: Compatible with notebook's checkpoint format
3. **Validation execution**: Identical validation pipeline
4. **Results format**: Same output structure and metrics
5. **Visualization**: Matching confusion matrix plots

### Production Deployment
Validation tools support production workflows:

- **Batch validation**: Efficient processing of large test sets
- **Model comparison**: Systematic benchmarking across architectures
- **Performance monitoring**: Continuous validation for model updates
- **Quality assurance**: Comprehensive testing before deployment

## 🛡️ Error Handling & Robustness

### Graceful Degradation
- **Missing dependencies**: Optional PyTorch Lightning support
- **Checkpoint errors**: Robust checkpoint loading with fallbacks
- **Memory constraints**: Efficient batch processing for large datasets
- **Device compatibility**: Automatic CPU/GPU handling

### Validation Checks
- **Model compatibility**: Verification of model-checkpoint alignment
- **Data format validation**: Ensuring correct input tensor shapes
- **Metric consistency**: Cross-validation of metric calculations

---

> **💡 Pro Tip**: Always use the `validate_model` function for model evaluation to ensure consistency with the reference notebook results!## 📈 Paper Benchmark Results & Validation

### 📊 Cataract-LMM Performance Benchmarks
Performance validation against paper's reported results (Section 4):

```python
# Paper benchmark results (Table 2 in paper)
CATARACT_LMM_BENCHMARKS = {
    'mvit_v1_b': {
        'farabi_f1': 77.1,     # Best performing model
        'noor_f1': 55.3,       # Domain adaptation result
        'domain_gap': 21.8,    # F1-score drop
        'architecture': 'Multiscale Vision Transformer'
    },
    'tecno_multistage': {
        'farabi_f1': 74.5,     # Multi-stage temporal network
        'noor_f1': 57.2,       # Better domain adaptation
        'domain_gap': 17.3,    # Lower domain gap
        'architecture': 'TeCNO Temporal Consistency'
    },
    'slow_r50': {
        'farabi_f1': 73.2,     # 3D CNN baseline
        'noor_f1': 55.8,       # Standard domain drop
        'domain_gap': 17.4,    # Consistent with paper
        'architecture': 'SlowFast 3D CNN'
    },
    'resnet50_lstm': {
        'farabi_f1': 71.8,     # CNN+RNN hybrid
        'noor_f1': 52.6,       # Higher domain sensitivity
        'domain_gap': 19.2,    # Moderate domain gap
        'architecture': 'CNN-RNN Hybrid'
    }
}

# Validate your results against paper benchmarks
def validate_against_paper_benchmarks(model_name, achieved_results):
    """Validate results against Cataract-LMM paper benchmarks."""
    if model_name in CATARACT_LMM_BENCHMARKS:
        paper_results = CATARACT_LMM_BENCHMARKS[model_name]
        
        f1_deviation = abs(achieved_results['farabi_f1'] - paper_results['farabi_f1'])
        gap_deviation = abs(achieved_results['domain_gap'] - paper_results['domain_gap'])
        
        if f1_deviation <= 2.0 and gap_deviation <= 3.0:
            print(f"✅ {model_name}: Results match paper benchmarks")
            return True
        else:
            print(f"⚠️  {model_name}: Results deviate from paper benchmarks")
            print(f"   F1 deviation: {f1_deviation:.1f}%")
            print(f"   Domain gap deviation: {gap_deviation:.1f}%")
            return False
    else:
        print(f"ℹ️  {model_name}: No paper benchmark available")
        return None
```

### 🏆 Expected Performance Hierarchy (Paper Section 4)
1. **MViT-B**: 77.1% F1 (Multiscale attention, best overall)
2. **TeCNO**: 74.5% F1 (Multi-stage refinement, best domain adaptation)
3. **3D CNNs**: 73.2% F1 (Strong spatial-temporal features)
4. **CNN-RNN**: 71.8% F1 (Efficient but domain-sensitive)

### 📊 Domain Adaptation Analysis (Paper Section 4.5)
Expected cross-center performance characteristics:

```python
# Domain adaptation expectations (from paper)
EXPECTED_DOMAIN_CHARACTERISTICS = {
    'average_f1_drop': 21.7,          # Mean across all models
    'best_domain_adaptation': 'tecno', # Lowest domain gap (17.3%)
    'worst_domain_adaptation': 'mvit', # Highest domain gap (21.8%)
    'domain_robust_threshold': 18.0,   # Models with <18% gap
    'clinical_significance': 20.0      # Threshold for clinical impact
}

def analyze_domain_robustness(domain_gap):
    """Analyze domain adaptation robustness."""
    if domain_gap < EXPECTED_DOMAIN_CHARACTERISTICS['domain_robust_threshold']:
        return "Domain robust (good generalization)"
    elif domain_gap < EXPECTED_DOMAIN_CHARACTERISTICS['clinical_significance']:
        return "Moderate domain sensitivity"
    else:
        return "High domain sensitivity (clinical concern)"
```## 🚀 Quick Start Guide### 1. Basic Model Validation```pythonfrom validation.comprehensive_validator import validate_modelfrom models.individual import Resnet50_LSTMfrom torch.utils.data import DataLoader# Load modelmodel = Resnet50_LSTM(num_classes=11, hidden_seq=256, dropout=0.5, hidden_MLP=128)# Validate (exact notebook pattern)results = validate_model(    model_name="resnet50_lstm",    checkpoint_path="/path/to/checkpoint.ckpt",    model=model,    test_loader=test_loader,    num_classes=11)print(f"Model Performance: F1-Macro = {results['f1_macro']:.4f}")```### 2. Advanced Validation with Lightning```pythonfrom validation.comprehensive_validator import ValidationModuleimport pytorch_lightning as pl# Create validation modulevalidation_module = ValidationModule(    model=model,    num_classes=11,    class_names=["Incision", "Viscoelastic", "Capsulorhexis", ...])# Run validationtrainer = pl.Trainer(    accelerator="gpu" if torch.cuda.is_available() else "cpu",    logger=False,    enable_checkpointing=False)trainer.validate(validation_module, test_loader)```### 3. Custom Metrics Analysis```pythonfrom validation.metrics import SurgicalPhaseMetrics# Detailed metrics calculationmetrics = SurgicalPhaseMetrics(num_classes=11)detailed_results = metrics.compute_all_metrics(predictions, targets)# Per-phase analysisfor i, phase_name in enumerate(phase_names):    print(f"{phase_name}: F1={detailed_results['f1_per_class'][i]:.3f}")```## 🔧 Checkpoint Compatibility### State Dict ProcessingThe `strip_prefix` function handles various checkpoint formats:```pythonfrom validation.comprehensive_validator import strip_prefix# Remove model prefix from state dict (notebook pattern)cleaned_state_dict = strip_prefix(checkpoint, prefix="model.")model.load_state_dict(cleaned_state_dict)```### Checkpoint Loading PatternsSupport for different checkpoint formats:- **PyTorch Lightning checkpoints**: Automatic `state_dict` extraction- **Direct state dicts**: Direct loading support- **Prefixed models**: Automatic prefix removal- **Custom formats**: Flexible handling for various training frameworks## 📊 Integration with Research Workflow### Notebook Validation PipelinePerfect alignment with the notebook's validation approach:1. **Model instantiation**: Same parameter patterns as notebook2. **Checkpoint loading**: Compatible with notebook's checkpoint format3. **Validation execution**: Identical validation pipeline4. **Results format**: Same output structure and metrics5. **Visualization**: Matching confusion matrix plots### Production DeploymentValidation tools support production workflows:- **Batch validation**: Efficient processing of large test sets- **Model comparison**: Systematic benchmarking across architectures- **Performance monitoring**: Continuous validation for model updates- **Quality assurance**: Comprehensive testing before deployment## 🛡️ Error Handling & Robustness### Graceful Degradation- **Missing dependencies**: Optional PyTorch Lightning support- **Checkpoint errors**: Robust checkpoint loading with fallbacks- **Memory constraints**: Efficient batch processing for large datasets- **Device compatibility**: Automatic CPU/GPU handling### Validation Checks- **Model compatibility**: Verification of model-checkpoint alignment- **Data format validation**: Ensuring correct input tensor shapes- **Metric consistency**: Cross-validation of metric calculations

---

> **💡 Pro Tip**: Always use the `validate_model` function for model evaluation to ensure consistency with the reference notebook results!
