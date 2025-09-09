# ⚙️ Configuration Management

This directory contains YAML configuration files that control all aspects of the surgical skill assessment pipeline. The configuration system provides comprehensive control over training, evaluation, and inference without requiring code modifications.

## 📁 Configuration Files

### 🔧 `config.yaml` - Main Configuration

The primary configuration file that orchestrates the entire pipeline. Written in YAML format with organized sections for different aspects of the system:

#### 📍 **Paths Configuration**
Configure input and output directories for streamlined project management:
- `data_root`: Root directory containing your Cataract-LMM video dataset
- `output_root`: Destination directory for all pipeline outputs (logs, checkpoints, visualizations)

#### 🖥️ **Hardware Configuration**
Optimize performance and resource utilization:
- `gpus`: Number of GPUs to utilize for parallel processing
- `mixed_precision`: Enable Automatic Mixed Precision (AMP) for accelerated training

#### 🎬 **Data Processing Configuration**
Control video loading and preprocessing parameters:
- `frame_rate`: Target sampling rate for video frames
- `clip_len`: Number of frames per video clip for temporal analysis
- `split_mode`: Data splitting strategy (`stratified` or `manual`)

#### 🧠 **Model Architecture Configuration**
Select and customize neural network architectures:
- `model_name`: Architecture selection (e.g., `x3d_m`, `timesformer`, `slowfast_r50`)
- `freeze_backbone`: Control pretrained weight freezing for transfer learning

#### 🚀 **Training Hyperparameters**
Fine-tune the learning process:
- `epochs`: Maximum number of training iterations
- `batch_size`: Batch size for training optimization
- `lr`: Learning rate for gradient descent
- `scheduler`: Learning rate scheduling strategy

#### 📊 **Metrics Configuration**
Define evaluation metrics for comprehensive assessment:
- Configurable list of metrics to track during training and validation

#### 📝 **Logging Configuration**
Control output verbosity and console formatting:
- Customize logging levels and console output styling

#### 🎯 **Pipeline Modes**
Toggle different execution stages:
- Training, evaluation, and inference mode controls

#### 🔄 **Override Settings**
Manual configuration overrides:
- Override automatic class detection when needed

## 🚀 Usage Examples

### Basic Training Configuration
```yaml
data:
  data_root: "data/videos"
  clip_len: 100
  
model:
  model_name: "x3d_m"
  
train:
  epochs: 50
  batch_size: 4
  lr: 0.001
```

### High-Performance Setup
```yaml
hardware:
  gpus: 2
  mixed_precision: true
  
train:
  batch_size: 8
  lr: 0.002
```

## 🎛️ Configuration Best Practices

- **🔍 Start Small**: Begin with smaller batch sizes and fewer epochs for initial testing
- **📈 Scale Gradually**: Increase complexity after validating basic functionality
- **💾 Save Configs**: Maintain separate configuration files for different experiments
- **🔄 Version Control**: Track configuration changes alongside code modifications
