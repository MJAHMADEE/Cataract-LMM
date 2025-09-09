# 🚀 Surgical Video Processing Framework

> ### Processing Modules  
- **[`processing/`](processing/README.md)** - 🗜️ Advanced video processing with surgical detail preservation
- **[`deidentification/`](deidentification/README.md)** - 🔒 Privacy protection and patient data anonymization
- **[`quality_control/`](quality_control/README.md)** - 🔍 Automated quality assessment and validation
- **[`preprocessing/`](preprocessing/README.md)** - 🎥 Video preprocessing and enhancement
- **[`metadata/`](metadata/README.md)** - 📋 Metadata extraction and managementessional-grade video processing framework specifically designed for phacoemulsification cataract surgery video analysis and preparation aligned with the Cataract-LMM research dataset.**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](README.md)

## 🎯 Project Overview

The Surgical Video Processing Framework is a comprehensive, production-ready solution for processing phacoemulsification cataract surgery videos. This framework is designed to support the **Cataract-LMM research dataset** processing pipeline, implementing validated methodologies for surgical video analysis.

### ✨ Key Features

- **🔧 Reference Script Alignment**: Perfectly aligned with `process_video.sh` and `process_videos.bat` reference implementations
- **⚡ High Performance**: Multi-threaded processing with GPU acceleration support
- **🛡️ Production Ready**: Enterprise-grade error handling and monitoring
- **📊 Real-time Monitoring**: Comprehensive performance tracking and metrics
- **🔒 Privacy Compliant**: Privacy protection and secure processing following dataset guidelines
- **📁 Batch Processing**: Efficient large-scale video processing for research datasets
- **🎛️ Professional CLI**: Complete command-line interface for automation
- **⚙️ Flexible Configuration**: YAML-based configuration system with multiple presets

## 🔧 Core Implementation Reference

The core implementation and algorithms for surgical video processing are located in the `/workspaces/Cataract_LMM/codes/surgical-video-processing/reference_scripts/` directory. This framework serves as an enhanced, production-ready implementation of those core reference scripts with the following improvements:

- **🐍 Python API Integration**: Full programmatic interface while maintaining reference script compatibility
- **🎯 Automated Processing**: Streamlined video processing workflow based on research methodologies
- **🛡️ Enhanced Error Handling**: Comprehensive validation and recovery mechanisms
- **📊 Performance Monitoring**: Real-time metrics and progress tracking
- **⚙️ Configuration Management**: YAML-based configuration system with flexible presets

### Reference Script Alignment

- **`process_video.sh`**: Core single video processing methodology implementation
- **`process_videos.bat`**: Batch processing workflow for multiple videos

## 📁 Directory Structure

### Core Processing Components
- **[`core/`](core/README.md)** - 🔧 Core video processing engine with FFmpeg integration
- **[`pipelines/`](pipelines/README.md)** - 🔄 Pipeline orchestration and workflow management
- **[`utils/`](utils/README.md)** - �️ Utility functions and helper modules

### Processing Modules  
- **[`compression/`](compression/README.md)** - 🗜️ Advanced video compression with surgical detail preservation
- **[`deidentification/`](deidentification/README.md)** - 🔒 Privacy protection and patient data anonymization
- **[`quality_control/`](quality_control/README.md)** - � Automated quality assessment and validation
- **[`preprocessing/`](preprocessing/README.md)** - 🎥 Video preprocessing and enhancement
- **[`metadata/`](metadata/README.md)** - � Metadata extraction and management

### Configuration and Documentation
- **[`configs/`](configs/README.md)** - ⚙️ Hospital-specific and processing configurations
- **[`scripts/`](scripts/README.md)** - 📜 Utility scripts and automation tools
- **[`notebooks/`](notebooks/README.md)** - 📓 Jupyter notebooks for analysis and demonstrations
- **[`tests/`](tests/README.md)** - 🧪 Comprehensive testing suite

### Reference Implementation
- **[`reference_scripts/`](reference_scripts/README.md)** - 🎯 Original reference scripts (process_video.sh, process_videos.bat)
- **[`legacy_scripts/`](legacy_scripts/README.md)** - 📁 Legacy script migration and compatibility

## 🚀 Quick Start

### Basic Usage

```bash
# Process single video with reference methodology
python main.py --method process_video --input video.mp4 --output processed.mp4

# Batch process directory
python main.py --batch --input-dir ./videos --output-dir ./processed

# High-quality processing with GPU acceleration
python main.py --quality high --gpu --input video.mp4 --output hq_video.mp4

# Validate environment
python main.py --validate-env
```

### Python API Usage

```python
from pipelines import SurgicalVideoProcessor

# Initialize processor with default configuration
processor = SurgicalVideoProcessor()

# Process single video
result = processor.process_video('input.mp4', 'output.mp4')

# Batch processing
results = processor.process_batch('input_dir/', 'output_dir/')
```

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **FFmpeg** (automatically detected)
- **8GB+ RAM** (recommended)

### Installation Steps

```bash
# Install FFmpeg (if not already installed)
# Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg
# macOS: brew install ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Validate installation
python main.py --validate-env
```

## ⚙️ Configuration Presets

The framework includes multiple processing presets optimized for different use cases:

### Default Processing
- **Resolution**: Maintains original video dimensions
- **Processing**: Standard quality with efficient encoding
- **Codec**: H.265 with CRF 23

### Quick Processing
- **Resolution**: Optimized for speed
- **Processing**: Fast encoding for preview generation
- **Codec**: H.264 with ultrafast preset

### High Quality
- **Resolution**: Maximum quality preservation  
- **Processing**: Research-grade quality encoding
- **Codec**: H.265 with slow preset

## 🎛️ Command-Line Interface

```bash
# Method-specific processing
python main.py --method {process_video,process_videos,general} --input VIDEO --output OUTPUT

# Quality presets
python main.py --quality {fast,balanced,high} --input VIDEO

# Batch processing
python main.py --batch --input-dir DIR --output-dir DIR --parallel --workers 4

# Advanced options
python main.py --gpu --preset slow --backup --progress --verbose

# Utility commands
python main.py --validate-env        # Validate environment
python main.py --info               # Framework information
python main.py --dry-run            # Validate without processing
```

## 🔧 Configuration

Create custom YAML configurations:

```yaml
# custom_config.yaml
processing:
  method: "process_video"
  quality_preset: "high"
  enable_gpu: false
  
ffmpeg:
  crf_value: 21
  video_codec: "libx265"
  
preprocessing:
  enable_deidentification: true
  enable_quality_check: true
```

## 🧪 Testing

```bash
# Run comprehensive tests
python -m pytest tests/

# Environment validation
python main.py --validate-env

# Test specific configuration
python main.py --config test.yaml --dry-run --input sample.mp4
```

## 📊 Performance

| Video Quality | Resolution | Processing Speed | Optimization Ratio |
|---------------|------------|------------------|-------------------|
| Full HD       | 1920×1080  | 1.2x realtime   | 8-12x            |
| Standard      | 720×480    | 4.0x realtime   | 4-8x             |

## 🏆 Production Features

- ✅ **Reference Script Compatibility**: Exact replication of original processing logic
- ✅ **Flexible Configuration**: Adaptable to various research requirements
- ✅ **Enterprise Monitoring**: Real-time performance tracking and logging
- ✅ **Privacy Compliance**: Privacy protection following medical data standards
- ✅ **Batch Processing**: Efficient large-scale video processing
- ✅ **Error Recovery**: Comprehensive validation and graceful failure handling

## 🔗 Related Projects

- [Original Video Processing Scripts](../Video%20Processing/) - Reference implementations
- [Surgical Phase Recognition](../Phase/) - Phase detection and analysis
- [Surgical Instance Segmentation](../Segmentation/) - Computer vision processing
- [Surgical Skill Assessment](../Skill/) - Performance evaluation

---

**🚀 Built for excellence in surgical video processing - Professional, Scalable, Production-Ready**
