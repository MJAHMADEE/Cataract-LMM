# 🛠️ Utility Functions - Surgical Instance Segmentation

This directory provides essential utility functions and configuration management for the surgical instance segmentation framework, supporting all components with shared functionality.

## 🎯 Objective

Deliver robust utility functions for:
- **⚙️ Configuration Management**: Centralized settings and parameter management
- **📝 Logging Systems**: Comprehensive logging for debugging and monitoring
- **🛠️ Helper Functions**: Common operations used across the framework
- **📊 Performance Monitoring**: System performance tracking and optimization

## 📂 Contents

### 📄 **Configuration Manager** (`config_manager.py`)
**Centralized configuration management system**
- **Purpose**: Unified configuration handling for all framework components
- **Features**: YAML/JSON loading, validation, defaults management
- **Integration**: Model configs, training configs, dataset configs
- **Standards**: Paper-aligned hyperparameters and specifications

```python
from utils.config_manager import ConfigManager

# Load model configuration
config = ConfigManager.load_config('model_configs.yaml')
yolov11_config = config['yolov11']

# Access training parameters
epochs = yolov11_config['training']['epochs']  # 80
batch_size = yolov11_config['training']['batch_size']  # 20
```

### 📄 **Logging Configuration** (`logging_config.py`)
**Structured logging for framework operations**
- **Purpose**: Comprehensive logging system for debugging and monitoring
- **Features**: Multi-level logging, file/console output, performance tracking
- **Integration**: Training logs, inference logs, error tracking

### 📄 **Helper Functions** (`helpers.py`)
**Common utility functions used across components**
- **Purpose**: Shared functionality to avoid code duplication
- **Features**: File operations, image processing, format conversions
- **Optimization**: Performance-optimized implementations

### 📄 **Performance Monitor** (`performance_monitor.py`)
**System performance tracking and optimization**
- **Purpose**: Monitor framework performance and resource usage
- **Features**: GPU memory tracking, FPS monitoring, latency measurement
- **Integration**: Real-time inference monitoring, training performance

## 🔧 Configuration Architecture

### **Hierarchical Configuration System**
```
configs/
├── task_definitions.yaml     # 3-task granularity system
├── model_configs.yaml       # All model specifications
└── dataset_config.yaml      # Dataset paths and settings

utils/config_manager.py       # Configuration loading and validation
```

### **Model Configuration Examples**
```python
# YOLOv11 configuration (top performer)
yolov11_config = {
    'training': {
        'epochs': 80,
        'batch_size': 20,
        'imgsz': 640
    },
    'performance': {
        'task_3_map': 73.9,  # Paper benchmark
        'real_time_fps': 45
    }
}
```

## 📊 Performance Monitoring

### **Real-Time Metrics**
```python
from utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

# Track inference performance
with monitor.track_inference():
    predictions = model.predict(frame)
    
print(f"FPS: {monitor.get_fps():.1f}")
print(f"GPU Memory: {monitor.get_gpu_memory():.1f}GB")
```

### **Training Monitoring**
- **Loss Tracking**: Real-time loss visualization
- **Memory Usage**: GPU memory optimization
- **Speed Metrics**: Training throughput monitoring
- **Checkpoint Management**: Automatic best model saving

## 🔗 Framework Integration

All utility functions integrate seamlessly with:
- **Models**: Configuration loading for all architectures
- **Training**: Performance monitoring and logging
- **Inference**: Real-time performance tracking
- **Evaluation**: Results logging and visualization
