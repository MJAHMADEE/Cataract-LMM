# 🛠️ Utility Functions & Helper Tools

The utils module provides essential supporting infrastructure for the surgical skill assessment framework. This collection of utility functions, helper tools, and configuration management systems ensures robust, reproducible, and efficient operation across all framework components.

## 🎯 Module Overview

### 🔧 Core Utilities (`helpers.py`)

Essential functions providing foundational support for reproducibility, hardware management, output organization, and development workflow optimization.

#### **🎲 `seed_everything`** - Reproducibility Management
```python
def seed_everything(seed: int = 42) -> None:
    """Comprehensive random seed configuration for reproducible experiments."""
    
# Supported Libraries & Systems:
✅ Python built-in random module
✅ NumPy random number generation
✅ PyTorch CPU and GPU random states
✅ CUDA random number generators
✅ Deterministic PyTorch operations
✅ Reproducible data loading order
```

**Advanced Reproducibility Features:**
- **Cross-Platform Consistency**: Ensures identical results across different operating systems
- **Multi-GPU Determinism**: Synchronized random states across distributed training
- **Library Compatibility**: Comprehensive coverage of all major ML libraries
- **Debugging Support**: Detailed logging of seed application status

#### **🖥️ `check_gpu_memory`** - Hardware Resource Management
```python
def check_gpu_memory() -> Dict[str, Union[bool, int, str]]:
    """Comprehensive GPU availability and memory analysis."""
    
# Hardware Detection Capabilities:
✅ GPU availability verification
✅ Total GPU memory capacity (GB)
✅ Available GPU memory monitoring
✅ GPU model identification and specifications
✅ CUDA version compatibility checking
✅ Multi-GPU configuration detection
```

**Resource Management Features:**
- **Memory Optimization**: Real-time memory usage tracking for optimal batch sizing
- **Hardware Adaptation**: Automatic configuration adjustment based on available resources
- **Performance Monitoring**: GPU utilization tracking and optimization recommendations
- **Error Prevention**: Early detection of memory constraints before training failures

#### **📋 `print_section`** - Professional Console Output
```python
def print_section(title: str, symbol: str = "=", width: int = 80) -> None:
    """Beautiful, professional console output formatting for enhanced readability."""
    
# Output Formatting Features:
✅ Customizable section dividers and symbols
✅ Adaptive width for different terminal sizes
✅ Hierarchical formatting for nested sections
✅ Color-coded output for different message types
✅ Timestamp integration for detailed logging
✅ Professional presentation for production environments
```

**Console Enhancement Benefits:**
- **Development Experience**: Clear visual separation of training phases and operations
- **Debug Efficiency**: Easy identification of different processing stages
- **Professional Presentation**: Clean, organized output suitable for demonstrations and reports
- **Log Analysis**: Structured output facilitating automated log parsing and analysis

#### **📁 `setup_output_dirs`** - Intelligent Directory Management
```python
def setup_output_dirs(base_path: str = "outputs") -> Dict[str, str]:
    """Comprehensive directory structure creation with intelligent organization."""
    
# Directory Structure Creation:
outputs/
├── 📅 {timestamp}_surgical_skill_assessment/    # Timestamped experiment directory
│   ├── 🔖 checkpoints/                          # Model checkpoints and saved states
│   ├── 📊 logs/                                 # Training logs and metrics
│   ├── 📈 plots/                                # Visualization and analysis plots
│   ├── 🔮 predictions/                          # Model predictions and results
│   ├── ⚙️ configs/                              # Saved configuration files
│   └── 📋 reports/                              # Generated reports and summaries
```

**Advanced Directory Management:**
- **Timestamp Organization**: Automatic experiment timestamping for version control
- **Hierarchical Structure**: Logical organization of different output types
- **Collision Prevention**: Automatic handling of duplicate experiment names
- **Cleanup Support**: Optional automatic cleanup of old experiments
- **Integration Ready**: Direct compatibility with logging and monitoring systems

## 🚀 Advanced Utility Features

### 🔄 Workflow Integration
```python
# Seamless integration with main training pipeline
from utils.helpers import seed_everything, setup_output_dirs, check_gpu_memory

# Initialize reproducible experiment environment
seed_everything(config['seed'])
gpu_info = check_gpu_memory()
output_dirs = setup_output_dirs(config['output_path'])

# Professional progress tracking
print_section("Training Phase 1: Data Loading", "🔄")
print_section("Training Phase 2: Model Initialization", "🧠")
print_section("Training Phase 3: Optimization", "⚡")
```

### 📊 Performance Monitoring Integration
```python
# Real-time resource monitoring
def monitor_training_resources():
    """Continuous monitoring of system resources during training."""
    gpu_status = check_gpu_memory()
    
    if gpu_status['available_memory_gb'] < 2.0:
        logger.warning("Low GPU memory detected - consider reducing batch size")
    
    return gpu_status
```

### 🎛️ Configuration-Driven Operation
```python
# YAML configuration integration
utils_config:
  reproducibility:
    seed: 42
    deterministic: true
  
  output:
    base_path: "experiments"
    timestamp_format: "%Y%m%d_%H%M%S"
    auto_cleanup: false
  
  hardware:
    gpu_monitoring: true
    memory_threshold: 2.0  # GB
```

## 🔧 Development Support Features

### 🎯 Debug & Development Utilities
```python
# Enhanced debugging support
def debug_tensor_info(tensor: torch.Tensor, name: str = "tensor"):
    """Comprehensive tensor analysis for debugging."""
    print_section(f"Tensor Analysis: {name}", "🔍")
    print(f"Shape: {tensor.shape}")
    print(f"Device: {tensor.device}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Memory: {tensor.element_size() * tensor.nelement() / 1024**2:.2f} MB")
    print(f"Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")

# Performance benchmarking
def benchmark_operation(func, *args, iterations=100, **kwargs):
    """Accurate performance benchmarking for optimization."""
    import time
    
    # Warm-up
    for _ in range(10):
        func(*args, **kwargs)
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    
    avg_time = (time.time() - start_time) / iterations
    print_section(f"Performance: {func.__name__}", "⚡")
    print(f"Average time: {avg_time*1000:.3f} ms")
    
    return result
```

### 📈 Experiment Tracking Integration
```python
# Integration with experiment tracking systems
def setup_experiment_tracking(config, output_dirs):
    """Initialize comprehensive experiment tracking."""
    
    # TensorBoard integration
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_dir = os.path.join(output_dirs['logs'], 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    
    # MLflow integration (optional)
    try:
        import mlflow
        mlflow.start_run(experiment_id=config.get('experiment_id'))
        mlflow.log_params(config)
    except ImportError:
        pass
    
    return {'tensorboard': writer, 'mlflow': mlflow}
```

## 🎯 Production-Ready Features

### 🛡️ Error Handling & Validation
```python
# Robust error handling throughout utilities
def safe_gpu_operation(func):
    """Decorator for safe GPU operations with fallback handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print_section("GPU Memory Error - Implementing Fallback", "⚠️")
                torch.cuda.empty_cache()
                # Implement fallback strategy
                return func(*args, **kwargs)
            raise
    return wrapper
```

### 🔄 Scalability & Optimization
```python
# Automatic resource optimization
def optimize_batch_size(model, dataloader, device, max_memory_gb=8.0):
    """Automatic batch size optimization based on available memory."""
    gpu_info = check_gpu_memory()
    available_memory = gpu_info['available_memory_gb']
    
    # Calculate optimal batch size
    memory_ratio = min(available_memory / max_memory_gb, 1.0)
    optimal_batch_size = int(dataloader.batch_size * memory_ratio)
    
    print_section(f"Optimized Batch Size: {optimal_batch_size}", "⚡")
    return optimal_batch_size
```

## 💡 Usage Examples

### 🚀 Quick Start Integration
```python
# Complete setup in main training script
from utils.helpers import *

def main():
    # Initialize environment
    print_section("Cataract-LMM: Surgical Skill Assessment", "🎯")
    
    # Setup reproducibility
    seed_everything(42)
    
    # Check hardware resources
    gpu_info = check_gpu_memory()
    print(f"GPU Available: {gpu_info['available']}")
    print(f"GPU Memory: {gpu_info['total_memory_gb']} GB")
    
    # Create output structure
    output_dirs = setup_output_dirs("experiments")
    print(f"Experiment directory: {output_dirs['base']}")
    
    # Begin training
    print_section("Starting Training Pipeline", "🚀")
```

### 🔧 Advanced Configuration
```python
# Production deployment configuration
production_config = {
    'reproducibility': {
        'seed': 42,
        'deterministic_ops': True,
        'benchmark_mode': False
    },
    'resource_management': {
        'gpu_memory_fraction': 0.8,
        'automatic_batch_optimization': True,
        'memory_monitoring': True
    },
    'output_management': {
        'experiment_tracking': True,
        'automatic_cleanup': True,
        'checkpoint_retention': 5
    }
}
```

## 🔗 Integration Points

### 🎯 Framework Compatibility
- **Main Pipeline**: Direct integration with `main.py` execution flow
- **Training Engine**: Resource monitoring and progress tracking in training loops
- **Model Factory**: Hardware-aware model instantiation and optimization
- **Data Pipeline**: Reproducible data loading and preprocessing
- **Evaluation Framework**: Structured output organization and result management

### 📊 External Tool Integration
- **TensorBoard**: Automatic logging directory setup and configuration
- **MLflow**: Experiment tracking and parameter logging
- **Weights & Biases**: Advanced experiment monitoring and visualization
- **Docker**: Container-friendly configuration and resource management

This utils module provides the essential foundation ensuring robust, reproducible, and professional operation of the surgical skill assessment framework across development, testing, and production environments.
