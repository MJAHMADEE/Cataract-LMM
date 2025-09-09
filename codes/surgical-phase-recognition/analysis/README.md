# 📊 Analysis & Performance Evaluation

This directory contains comprehensive analysis tools for surgical phase recognition models used in the Cataract-LMM dataset research. These tools provide detailed performance evaluation, error analysis, and visualization capabilities following the methodologies described in the academic paper.

## 🎯 Objective

Provide advanced analytical capabilities for evaluating model performance across the 13 surgical phases defined in the Cataract-LMM taxonomy, with specific focus on:

- **Phase-wise performance analysis** across different surgical centers
- **Domain adaptation evaluation** between Farabi and Noor hospitals  
- **Temporal consistency assessment** for surgical workflow modeling
- **Error pattern analysis** for model improvement insights

## 📚 Academic Paper Alignment

All analysis tools implement evaluation methodologies consistent with the technical validation described in:
*"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"*

The analysis framework supports the paper's benchmarking experiments for phase recognition baselines and domain generalization assessment.

## 📁 Contents

### 🎯 Core Analysis Files

- **`model_analyzer.py`** - 📈 Comprehensive model performance analysis with metrics computation
- **`error_analyzer.py`** - 🔍 Detailed error pattern analysis and domain generalization assessment  
- **`__init__.py`** - 🔧 Analysis module interface and exports

## 📊 Analysis Tools & Metrics

### 📈 ModelAnalyzer
Implements the evaluation metrics and performance analysis methodology described in the academic paper's technical validation section.

**Key Metrics Supported:**
- **Accuracy**: Proportion of correctly classified frames
- **Macro-averaged Precision, Recall, F1-score**: Balanced assessment across all 13 phases
- **Per-phase performance**: Individual phase classification accuracy
- **Domain shift analysis**: In-domain vs out-of-domain performance comparison

```python
from analysis import ModelAnalyzer
from transform import get_phase_mapping

# Initialize with Cataract-LMM phase mapping
phase_mapping = get_phase_mapping("13_phase")  # or "11_phase" 
analyzer = ModelAnalyzer(model=trained_model, phase_mapping=phase_mapping)

# Complete performance analysis (matches paper methodology)
results = analyzer.analyze_model_performance(
    test_dataloader,
    compute_macro_metrics=True,  # Paper's evaluation approach
    save_confusion_matrix=True,
    output_dir="./analysis_results"
)

# Domain adaptation analysis (Farabi vs Noor evaluation)
domain_results = analyzer.evaluate_domain_generalization(
    farabi_dataloader,  # In-domain test set
    noor_dataloader,    # Out-of-domain test set
)

# Per-phase analysis (as shown in paper's Figure 9)
phase_results = analyzer.analyze_per_phase_performance(y_true, y_pred)
```

### 🔍 ErrorAnalyzer  
Specialized error analysis for identifying failure patterns and challenging phases mentioned in the paper's technical validation.

**Analysis Focus:**
- **Phase confusion patterns**: Which phases are frequently misclassified (e.g., "Capsule Polishing" difficulties)
- **Multi-center consistency**: Error patterns across Farabi vs Noor hospitals
- **Temporal boundary errors**: Misclassifications at phase transitions
- **Visual similarity challenges**: Confusion between visually similar phases

```python
from analysis import ErrorAnalyzer
from transform import get_phase_mapping

# Initialize with Cataract-LMM phases
phase_mapping = get_phase_mapping("11_phase")
error_analyzer = ErrorAnalyzer(phase_mapping=phase_mapping)

# Misclassification analysis (identifies challenging phases)
error_results = error_analyzer.analyze_misclassifications(
    y_true, y_pred, y_prob, 
    video_ids=video_ids,  # For per-video analysis
    hospital_ids=hospital_ids  # For domain analysis
)

# Phase difficulty assessment (matches paper's observations)
difficulty_analysis = error_analyzer.analyze_phase_difficulty(
    y_true, y_pred,
    focus_phases=["CapsulePolishing", "Phacoemulsification"]  # Paper highlights
)

# Domain adaptation error patterns
domain_errors = error_analyzer.analyze_domain_errors(
    farabi_pred, noor_pred, farabi_true, noor_true
)
```

## 📈 Analysis Capabilities

### 🎯 Paper-Validated Metrics
- **Macro F1-Score**: Primary evaluation metric used in paper (enables fair comparison across phases)
- **Accuracy**: Overall frame-level classification performance  
- **Per-Phase F1-Scores**: Individual phase performance analysis (as shown in paper's Figure 9)
- **Domain Shift Assessment**: Performance degradation between hospitals (e.g., 77.1% → 57.6% for MViT-B)

### 🔍 Advanced Error Analysis
- **Phase Confusion Mapping**: Identifies most commonly confused phase pairs
- **Temporal Consistency**: Analyzes prediction smoothness over video sequences
- **Multi-Center Generalization**: Quantifies domain adaptation challenges
- **Visual Similarity Analysis**: Groups phases by classification difficulty

### 📊 Visualization & Reporting
- **Performance Hierarchy Charts**: Model ranking by macro F1-score
- **Phase Difficulty Rankings**: Based on per-phase F1-scores (Phacoemulsification → highest, Capsule Polishing → lowest)
- **Domain Adaptation Plots**: In-domain vs out-of-domain performance comparison
- **Confusion Matrix Heatmaps**: Detailed misclassification pattern visualization

## 🔍 Key Analysis Insights

The analysis tools help identify patterns consistent with the paper's findings:

### 📊 Model Performance Hierarchy
Based on paper's Table 8 results:
1. **Video Transformers**: MViT-B (77.1% macro F1) → Best overall performance
2. **3D CNNs**: Slow R50 (69.8% macro F1) → Strong temporal modeling  
3. **Hybrid Models**: CNN+GRU (71.3% macro F1) → Balanced performance
4. **End-to-End Models**: Various 3D architectures → Architecture-dependent results

### 🏥 Domain Adaptation Challenges
Consistent ~22% average F1-score drop when models trained on Farabi are tested on Noor hospital data, highlighting the value of multi-center training data.

### ⚡ Phase-Specific Insights
- **Easiest Phase**: Phacoemulsification (distinctive visual characteristics)
- **Most Challenging**: Capsule Polishing (visual similarity to other phases)
- **Best Separable**: Phases with unique instruments or anatomical changes
- **Pattern Analysis**: Common misclassification types
- **Recommendations**: Suggested improvements based on analysis

### Visualizations
- **Confusion Matrices**: Interactive and static heatmaps
- **ROC Curves**: Per-class receiver operating characteristic
- **Feature Maps**: Learned representation visualization
- **Temporal Analysis**: Sequence-level error patterns

## 📊 Example Analysis Workflow

```python
# Complete analysis pipeline
from analysis import ModelAnalyzer, ErrorAnalyzer

# Initialize analyzers
model_analyzer = ModelAnalyzer(model=trained_model)
error_analyzer = ErrorAnalyzer()

# Run model predictions
## 🚀 Quick Start Example

```python
from analysis import ModelAnalyzer, ErrorAnalyzer  
from transform import get_phase_mapping
import torch

# Initialize analysis tools with Cataract-LMM phase mapping
phase_mapping = get_phase_mapping("11_phase")  # Use 11-phase for compatibility
model_analyzer = ModelAnalyzer(model=your_trained_model, phase_mapping=phase_mapping)
error_analyzer = ErrorAnalyzer(phase_mapping=phase_mapping)

# Run comprehensive analysis matching paper's evaluation protocol
y_true, y_pred, y_prob = model_analyzer.predict(test_dataloader)

# Calculate paper-standard metrics
results = model_analyzer.calculate_macro_metrics(y_true, y_pred)
print(f"Macro F1-Score: {results['macro_f1']:.3f}")  # Primary paper metric
print(f"Accuracy: {results['accuracy']:.3f}")

# Analyze phase-specific performance (Figure 9 recreation)
phase_performance = model_analyzer.analyze_per_phase_performance(y_true, y_pred)
for phase, f1_score in phase_performance.items():
    print(f"{phase}: {f1_score:.3f}")

# Domain adaptation analysis
if domain_info_available:
    domain_results = model_analyzer.evaluate_domain_shift(
        farabi_results=(farabi_true, farabi_pred),
        noor_results=(noor_true, noor_pred)
    )
    print(f"Domain shift F1 drop: {domain_results['f1_drop']:.1f}%")
```

## � Expected Output Format

Analysis results follow the paper's reporting standards:

```bash
=== Cataract-LMM Phase Recognition Analysis ===
Model: MViT-B (example)
Dataset: 11-phase classification

Overall Metrics:
  Accuracy: 85.7%
  Macro F1-Score: 77.1%
  Macro Precision: 77.1%  
  Macro Recall: 78.5%

Per-Phase Performance (F1-Scores):
  Phacoemulsification: 0.881 ⭐ (Best performing)
  Lens Implantation: 0.824
  Incision: 0.802
  ...
  Capsule Polishing: 0.632 ⚠️ (Most challenging)

Domain Adaptation:
  In-domain (Farabi): 77.1% F1
  Out-of-domain (Noor): 57.6% F1  
  Performance drop: 22.0% ⚠️
```

## � Integration Points

- **📓 Core Logic**: Analysis tools integrate with [`../notebooks/`](../notebooks/) for research workflows
- **🧠 Models**: Compatible with all model architectures in [`../models/`](../models/)
- **📊 Data**: Works with datasets from [`../data/`](../data/) module
- **✅ Validation**: Used by [`../validation/`](../validation/) for comprehensive model testing

---

> **💡 Pro Tip**: Use the analysis tools to replicate the paper's Table 8 results and validate your own model implementations against published benchmarks!
