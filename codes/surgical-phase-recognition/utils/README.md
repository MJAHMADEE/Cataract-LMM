# 🛠️ Utilities - Cataract-LMM Framework

This directory contains essential utility functions and helper modules for the Cataract-LMM surgical phase recognition framework, providing paper-compliant tools and support utilities for reproducible research.

## � Paper-Aligned Utilities

All utility components implement functionalities specifically designed to support the **"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"** paper methodology.

### 🎯 **Research Focus**
- **Multi-Center Support**: Farabi and Noor hospital data handling
- **13-Phase Taxonomy**: Complete surgical workflow utilities
- **Domain Adaptation**: Cross-center evaluation tools
- **Benchmark Metrics**: Paper-standard evaluation functions

## 📁 Contents

### Core Utility Files

- **`helpers.py`** - Cataract-LMM specific utility functions and helper classes
- **`__init__.py`** - Utilities module interface and exports

## 🛠️ Cataract-LMM Utility Categories

### Multi-Center Data Management
- **`organize_multicenter_data()`** - Farabi/Noor dataset organization
- **`validate_hospital_splits()`** - Verify 80/20 center distribution
- **`cross_center_metrics()`** - Domain adaptation evaluation tools
- **`phase_distribution_analysis()`** - 13-phase occurrence statistics

### Paper-Compliant Metrics
- **`calculate_macro_f1()`** - Primary evaluation metric (paper standard)
- **`domain_gap_analysis()`** - Cross-center performance measurement
- **`phase_wise_accuracy()`** - Per-phase performance breakdown
- **`temporal_consistency_score()`** - Sequence-level evaluation

## 🔧 Core Paper-Compliant Utilities

### Multi-Center Evaluation System
```python
from utils import CataractLMMEvaluator

# Paper's domain adaptation evaluation
evaluator = CataractLMMEvaluator(
    phase_taxonomy=13,  # Full Cataract-LMM phases
    primary_metric="macro_f1",  # Paper's primary metric
    centers=["farabi", "noor"]  # Multi-center setup
)

# Evaluate domain adaptation (Paper Section 4.5)
results = evaluator.evaluate_domain_adaptation(
    model=model,
    farabi_loader=farabi_test_loader,  # In-domain
    noor_loader=noor_test_loader      # Out-of-domain
)

print(f"Farabi F1: {results['farabi_f1']:.1f}%")
print(f"Noor F1: {results['noor_f1']:.1f}%") 
print(f"Domain gap: {results['domain_gap']:.1f}%")  # Expected: ~22%
```

### Paper Metrics Calculator
```python
from utils import calculate_cataract_lmm_metrics

# Calculate all paper-standard metrics
def evaluate_model_paper_standard(model, test_loader, center="farabi"):
    """
    Evaluate model using Cataract-LMM paper metrics.
    
    Primary Metric: Macro F1-score
    Secondary: Per-phase accuracy, precision, recall
    """
    predictions, targets = get_model_predictions(model, test_loader)
    
    # Paper's primary evaluation metric
    metrics = calculate_cataract_lmm_metrics(
        predictions=predictions,
        targets=targets,
        phase_names=CATARACT_LMM_PHASES_13,
        center=center
    )
    
    return {
        'macro_f1': metrics['macro_f1'],           # Primary paper metric
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'per_phase_f1': metrics['per_phase_f1'],   # 13-phase breakdown
        'accuracy': metrics['accuracy'],
        'center': center
    }
```

### Domain Adaptation Analysis
```python
from utils import DomainAdaptationAnalyzer

# Analyze cross-center performance following paper methodology
analyzer = DomainAdaptationAnalyzer(
    source_center="farabi",    # Training domain
    target_center="noor",      # Testing domain  
    phase_count=13             # Full taxonomy
)

# Generate domain shift analysis (Paper Section 4.5)
domain_analysis = analyzer.analyze_domain_shift(
    source_results=farabi_results,
    target_results=noor_results
)

# Expected results based on paper
expected_gaps = {
    'mvit_b': 21.8,      # MViT-B domain gap (paper result)
    'tecno': 17.3,       # TeCNO domain gap
    '3d_cnn': 17.4,      # 3D CNN average gap
    'cnn_rnn': 19.2      # CNN+RNN average gap
}

# Validate against paper benchmarks
for model_name, gap in domain_analysis['per_model_gaps'].items():
    expected = expected_gaps.get(model_name, None)
    if expected and abs(gap - expected) < 3.0:
        print(f"✅ {model_name}: {gap:.1f}% gap matches paper ({expected:.1f}%)")
    else:
        print(f"⚠️  {model_name}: {gap:.1f}% gap differs from paper")
```
    }, step=epoch)

# Get best metrics
best_acc = tracker.get_best_metric('val_accuracy')
print(f"Best validation accuracy: {best_acc['value']:.4f} at epoch {best_acc['step']}")

# Plot training curves
tracker.plot_metrics(['train_accuracy', 'val_accuracy'], save_path='metrics.png')

# Save metrics history
tracker.save('training_metrics.json')
```

### Configuration Validation
```python
from utils import ConfigValidator

# Validate individual configuration sections
try:
    ConfigValidator.validate_model_config(config['model'])
    ConfigValidator.validate_data_config(config['data'])
    ConfigValidator.validate_training_config(config['training'])
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## 📊 Data Analysis Utilities

### Video Dataset Statistics
```python
from utils import create_video_summary_stats

# Analyze video dataset
stats = create_video_summary_stats("./video_dataset")

print(f"Total videos: {stats['total_videos']}")
print(f"Total size: {stats['total_size_formatted']}")
print(f"Average size: {stats['average_size_formatted']}")
print(f"Formats: {stats['video_formats']}")
```

### Annotation Validation
```python
from utils import validate_phase_annotations

# Validate surgical phase annotations
validation_results = validate_phase_annotations(annotations_list)

print(f"Valid annotations: {validation_results['valid_annotations']}")
print(f"Errors found: {len(validation_results['errors'])}")
print(f"Warnings: {len(validation_results['warnings'])}")
print(f"Success rate: {validation_results['validation_summary']['success_rate']:.2%}")

# Check phase distribution
for phase, count in validation_results['phase_distribution'].items():
    print(f"{phase}: {count} annotations")
```

## 💾 File Operations

### Safe File Handling
```python
from utils import save_json, load_json, save_pickle, load_pickle

# JSON operations with error handling
data = {"model": "swin3d", "accuracy": 0.95}
save_json(data, "results.json")
loaded_data = load_json("results.json")

# Pickle operations for complex objects
model_state = {"weights": model.state_dict(), "config": config}
save_pickle(model_state, "model_checkpoint.pkl")
restored_state = load_pickle("model_checkpoint.pkl")

# File integrity verification
file_hash = calculate_file_hash("model_checkpoint.pkl", algorithm="sha256")
print(f"File hash: {file_hash}")
```

### Directory Management
```python
from utils import ensure_dir

# Create directory structure safely
output_dir = ensure_dir("./experiments/run_001/checkpoints")
results_dir = ensure_dir("./results/analysis")
logs_dir = ensure_dir("./logs/training")

# Directories are created with parents if they don't exist
print(f"Output directory ready: {output_dir}")
```

## 🔍 System Information

### GPU and Hardware Utilities
```python
from utils import get_gpu_memory_info, format_bytes

## 📈 Paper Integration Examples

### Complete Cataract-LMM Evaluation Pipeline
```python
from utils import (
    CataractLMMEvaluator, DomainAdaptationAnalyzer,
    CataractLMMPhaseManager, ReproducibilityManager
)

def run_cataract_lmm_evaluation(model, config):
    """
    Complete evaluation following Cataract-LMM paper methodology.
    """
    
    # Initialize paper-compliant components
    evaluator = CataractLMMEvaluator()
    phase_manager = CataractLMMPhaseManager()
    reproducer = ReproducibilityManager()
    
    # Set up reproducible environment
    reproducer.setup_reproducible_environment()
    
    # Evaluate on Farabi (in-domain)
    farabi_results = evaluator.evaluate_center(
        model=model,
        test_loader=farabi_test_loader,
        center="farabi",
        phase_taxonomy=phase_manager.get_13_phase_taxonomy()
    )
    
    # Evaluate on Noor (out-of-domain) 
    noor_results = evaluator.evaluate_center(
        model=model,
        test_loader=noor_test_loader,
        center="noor",
        phase_taxonomy=phase_manager.get_13_phase_taxonomy()
    )
    
    # Calculate domain adaptation metrics
    domain_analysis = DomainAdaptationAnalyzer.analyze_cross_center(
        farabi_results, noor_results
    )
    
    # Generate paper-compliant report
    report = {
        'model_name': config['model_name'],
        'farabi_macro_f1': farabi_results['macro_f1'],
        'noor_macro_f1': noor_results['macro_f1'],
        'domain_gap': domain_analysis['f1_gap'],
        'per_phase_performance': farabi_results['per_phase_f1'],
        'meets_paper_benchmark': validate_against_paper_benchmarks(config['model_name'], farabi_results['macro_f1'])
    }
    
    return report

# Execute full evaluation
evaluation_results = run_cataract_lmm_evaluation(model, model_config)
print(f"Paper-compliant evaluation completed:")
print(f"Farabi F1: {evaluation_results['farabi_macro_f1']:.1f}%")
print(f"Noor F1: {evaluation_results['noor_macro_f1']:.1f}%")
print(f"Domain gap: {evaluation_results['domain_gap']:.1f}%")
```

### Paper Results Validation
```python
# Validate all results against paper benchmarks
def validate_complete_paper_compliance():
    """
    Comprehensive validation of framework compliance with paper.
    """
    
    validation_checks = {
        'dataset_statistics': validate_dataset_statistics(dataset_path),
        'phase_taxonomy': validate_13_phase_taxonomy(),
        'model_benchmarks': validate_model_benchmarks(all_model_results),
        'domain_adaptation': validate_domain_gaps(domain_results),
        'evaluation_metrics': validate_evaluation_protocol()
    }
    
    all_compliant = all(check['compliant'] for check in validation_checks.values())
    
    if all_compliant:
        print("✅ Complete framework compliance with Cataract-LMM paper")
    else:
        print("⚠️  Some components deviate from paper specifications")
        for check_name, result in validation_checks.items():
            if not result['compliant']:
                print(f"  - {check_name}: {result['message']}")
    
    return validation_checks
```

## 🔗 Integration Points

- **📊 Data Pipeline**: Validates [`../data/`](../data/) dataset compliance with paper
- **🧠 Models**: Benchmarks [`../models/`](../models/) against paper performance
- **📈 Analysis**: Powers [`../analysis/`](../analysis/) with paper metrics
- **✅ Validation**: Supports [`../validation/`](../validation/) evaluation protocol
- **� Notebooks**: Used throughout [`../notebooks/`](../notebooks/) research workflows

## 📖 Research Compliance

### Paper Methodology Checklist
- ✅ **Multi-Center Protocol**: Farabi/Noor evaluation setup
- ✅ **13-Phase Taxonomy**: Complete surgical workflow support
- ✅ **Macro F1-Score**: Primary evaluation metric implementation
- ✅ **Domain Adaptation**: Cross-center performance analysis
- ✅ **Benchmark Validation**: Compare against paper results

### Expected Performance Validation
Use these utilities to ensure your results align with the Cataract-LMM paper:

| Component | Paper Benchmark | Validation Function |
|-----------|----------------|-------------------|
| Dataset Stats | 150 videos, 28.6h | `validate_dataset_statistics()` |
| MViT-B F1 | 77.1% | `benchmark_against_paper()` |
| Domain Gap | ~22% average | `validate_domain_gaps()` |
| Phase Count | 13 phases | `validate_13_phase_taxonomy()` |

---

> **📖 Paper Reference**: All utilities implement the exact specifications and evaluation protocols from "Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis" for reproducible research.
