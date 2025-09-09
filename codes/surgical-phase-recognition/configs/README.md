# ⚙️ Configuration Management System

This directory contains the configuration management system for surgical phase recognition experiments following the Cataract-LMM dataset methodology. The configurations implement the exact hyperparameter settings and training protocols described in the academic paper.

## 🎯 Objective  

Provide standardized, reproducible configuration management for:
- **Model architectures** benchmarked in the paper (MViT, Slow-Fast, CNNs, etc.)
- **Training hyperparameters** following paper's experimental setup  
- **Dataset parameters** matching Cataract-LMM specifications
- **Evaluation protocols** for multi-center domain adaptation

## 📚 Academic Paper Alignment

All configurations implement the experimental design described in:
*"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"*

The configuration system supports the paper's methodology for training on Farabi hospital data and evaluating domain generalization to Noor hospital.

## 📁 Contents

### 🎯 Configuration Files

- **`config_manager.py`** - 🔧 Central configuration management with validation
- **`default.yaml`** - 📋 Standard configuration based on paper's methodology  
- **`fast_training.yaml`** - ⚡ Quick experimentation setup (reduced epochs)
- **`high_accuracy.yaml`** - 🎯 Maximum performance configuration (paper's best settings)
- **`tecno.yaml`** - 🧠 Specialized configuration for TeCNO multi-stage models
- **`__init__.py`** - 📦 Configuration module interface

## ⚙️ Paper-Based Configuration System

### 🔧 ConfigManager
Configuration management implementing paper's experimental protocols.

```python
from configs import ConfigManager
from transform import get_phase_mapping

config_manager = ConfigManager()

# Load paper-validated configurations
paper_config = config_manager.load_config('high_accuracy.yaml')  # Best paper settings
tecno_config = config_manager.load_config('tecno.yaml')         # TeCNO architecture

# Create domain adaptation configuration 
domain_config = config_manager.create_domain_adaptation_config(
    train_hospital='farabi',  # Paper's training protocol
    test_hospitals=['farabi', 'noor'],  # Domain generalization setup
    phase_mapping=get_phase_mapping('11_phase')
)

# Validate against paper standards
is_valid = config_manager.validate_paper_compliance(paper_config)
```

### 📊 Paper-Validated Model Configurations

#### ModelConfig (Paper Architectures)
```python
@dataclass
class ModelConfig:
    type: str = 'mvit_b'  # Paper's best performer (77.1% macro F1)
    num_classes: int = 11  # 11-phase classification
    pretrained: bool = True  # Pre-trained on Kinetics-400
    input_size: tuple = (224, 224)  # Paper's input resolution
    temporal_window: int = 10  # Sequence length from paper
    dropout_rate: float = 0.5  # Paper's dropout setting
```

#### DataConfig (Paper Dataset Settings)
```python
@dataclass  
class DataConfig:
    batch_size: int = 32  # Paper's batch size
    num_workers: int = 4
    sequence_length: int = 10  # Lookback window from paper
    frame_interval: int = 1  # Temporal downsampling (4 fps effective)
    target_fps: int = 4  # Paper: "downsampled to 4 frames per second"
    image_size: Tuple[int, int] = (224, 224)  # Paper's input resolution
    data_augmentation: bool = True  # Training augmentation enabled
    max_sequences_per_phase: int = 1500  # Paper's balancing parameter

#### TrainingConfig (Paper Hyperparameters)  
```python
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4  # Paper's learning rate
    optimizer: str = 'Adam'  # Paper's optimizer choice
    weight_decay: float = 1e-3  # Paper's weight decay
    max_epochs: int = 100  # Training duration
    dropout_rate: float = 0.5  # Paper's dropout setting
    mlp_hidden_dim: int = 128  # Paper's MLP hidden layer
    temporal_model_hidden_dim: int = 256  # Paper's temporal hidden dim

#### DomainAdaptationConfig (Multi-Center Setup)
```python
@dataclass
class DomainAdaptationConfig:
```

## 📄 Paper-Based Configuration Presets

### 🎯 High Accuracy Configuration (`high_accuracy.yaml`)
Implements the best-performing settings from the paper:
```yaml
model:
  type: "mvit_b"  # Paper's top performer
  num_classes: 11
  pretrained: true
  input_size: [224, 224]

data:
  batch_size: 32
  sequence_length: 10  # Paper's lookback window
  frame_interval: 1
  target_fps: 4  # Paper's effective fps
  max_sequences_per_phase: 1500

training:
  learning_rate: 1.0e-4  # Paper's learning rate
  optimizer: "Adam"
  weight_decay: 1.0e-3
  dropout_rate: 0.5
  
domain_adaptation:
  train_hospitals: ["farabi"]
  test_hospitals: ["farabi", "noor"]  # Domain generalization protocol
```

### ⚡ Fast Training Configuration (`fast_training.yaml`) 
Quick experimentation with reduced complexity:
```yaml
model:
  type: "resnet50_lstm"  # Faster than video transformers
  num_classes: 11

data:
  batch_size: 16  # Smaller batch for speed
  sequence_length: 5   # Shorter sequences
  
training:
  max_epochs: 20  # Quick training
  learning_rate: 2.0e-4  # Slightly higher for faster convergence
```

### 🧠 TeCNO Configuration (`tecno.yaml`)
Specialized settings for TeCNO multi-stage models:
```yaml
model:
  type: "tecno_multi_stage"
  num_classes: 11
  num_stages: 6  # Paper's TeCNO configuration
  num_layers: 1
  num_f_maps: 256
  backbone_dim: 2048

training:
  learning_rate: 1.0e-4
```

## 🚀 Configuration Usage Examples

### 📊 Reproducing Paper Results
```python
from configs import ConfigManager
from transform import get_phase_mapping

config_manager = ConfigManager()

# Load paper's best configuration
paper_config = config_manager.load_config('high_accuracy.yaml')

# Override for specific experiments
mvit_config = config_manager.load_config(
    'high_accuracy.yaml',
    overrides={
        'model.type': 'mvit_b',
        'training.learning_rate': 1e-4,
        'data.sequence_length': 10,
        'experiment.name': 'mvit_b_replication'
    }
)

# Domain adaptation experiment setup
domain_config = config_manager.create_domain_experiment_config(
    base_config='high_accuracy.yaml',
    train_hospital='farabi',
    test_hospitals=['farabi', 'noor'],
    phase_mapping=get_phase_mapping('11_phase')
)
```

### 🔧 Custom Configuration Creation
```python
# Create paper-compliant configuration
paper_compliant_config = config_manager.create_paper_config(
    model_architecture='slow_r50',  # Paper architecture
    hospital_setup='farabi_to_noor',  # Domain adaptation
    phase_count=11,  # 11-phase classification
    experiment_name='slow_r50_domain_adaptation'
)

# Save for reproducibility
config_manager.save_config(paper_compliant_config, 'slow_r50_paper.yaml')
```

### ✅ Paper Compliance Validation
```python
from configs import PaperValidator

validator = PaperValidator()

# Validate against paper standards
compliance_check = validator.validate_paper_compliance(config)
print(f"Paper compliance: {compliance_check['is_compliant']}")

if not compliance_check['is_compliant']:
    print("Issues found:")
    for issue in compliance_check['issues']:
        print(f"  - {issue}")

# Check specific paper requirements
phase_check = validator.validate_phase_mapping(config)
domain_check = validator.validate_domain_setup(config)
hyperparams_check = validator.validate_hyperparameters(config)
```

## � Available Model Configurations

Based on paper's Table 8 architectures:

| Configuration Key | Architecture | Paper F1-Score | Best Use Case |
|------------------|--------------|----------------|---------------|
| `mvit_b` | MViT-B | 77.1% | 🏆 Best overall performance |
| `swin_transformer` | Swin-T | 76.2% | Attention-based modeling |
| `slow_r50` | Slow R50 | 69.8% | 3D CNN baseline |
| `resnet50_lstm` | CNN+LSTM | - | Hybrid approach |
| `efficientnet_gru` | CNN+GRU | 71.3% | Efficient processing |
| `tecno_multi` | TeCNO | 82.97% | Multi-stage temporal |

## 🏥 Domain Adaptation Presets

```yaml
# Farabi → Noor (Paper's domain adaptation setup)
domain_adaptation:
  name: "farabi_to_noor_adaptation"
  source_hospital: "farabi"
  target_hospital: "noor" 
  expected_performance_drop: 22.0  # Average from paper
  
  evaluation_metrics:
    - "macro_f1_score"  # Primary paper metric
    - "accuracy"
    - "per_phase_f1"
    - "domain_shift_gap"
```

## 🔗 Integration Points

- **📓 Core Logic**: Configurations integrate with [`../notebooks/`](../notebooks/) for research workflows
- **🧠 Models**: Model configs work with all architectures in [`../models/`](../models/)
- **📊 Data**: Data configs compatible with [`../data/`](../data/) pipeline
- **✅ Validation**: Used by [`../validation/`](../validation/) for systematic evaluation

---

> **💡 Pro Tip**: Use `high_accuracy.yaml` as the baseline for replicating paper results, then customize for your specific experimental needs!
ConfigValidator.validate_training_config(config['training'])

# Validate complete configuration
is_valid = config_manager.validate_config(config)
```

## 📊 Configuration Comparison

| Preset | Model | Batch Size | Sequence Length | Epochs | Use Case |
|--------|-------|------------|-----------------|--------|----------|
| Default | Swin3D | 8 | 32 | 100 | General purpose |
| Fast | R3D-18 | 16 | 16 | 30 | Quick experiments |
| High Accuracy | Swin3D | 4 | 64 | 200 | Maximum performance |
| TeCNO | TeCNO | 12 | 48 | 150 | Multi-stage analysis |

## 🛠️ Advanced Configuration

### Environment-Specific Configs
```python
# Development configuration
dev_config = config_manager.load_config('default.yaml')
dev_config['training']['max_epochs'] = 5
dev_config['data']['num_workers'] = 2

# Production configuration
prod_config = config_manager.load_config('high_accuracy.yaml')
prod_config['experiment']['log_level'] = 'INFO'
```

### Dynamic Configuration
```python
# Adjust based on available resources
import torch

if torch.cuda.device_count() > 1:
    config['data']['batch_size'] *= torch.cuda.device_count()
    config['training']['learning_rate'] *= torch.cuda.device_count()
```

## 📖 Usage Examples

See the comprehensive reference notebook: [`../notebooks/phase_validation_comprehensive.ipynb`](../notebooks/phase_validation_comprehensive.ipynb)
