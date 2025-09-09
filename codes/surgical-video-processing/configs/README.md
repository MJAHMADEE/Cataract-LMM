# ⚙️ Configuration System

> **Workflow-optimized configuration presets for Cataract-LMM surgical video processing**

## 🎯 Objective

This directory contains YAML configuration files that define processing parameters for different quality presets and workflow scenarios. Each configuration ensures exact alignment with the Cataract-LMM dataset specifications and reference script methodologies.

## 📂 Configuration Inventory

### Processing Configuration Files

| Configuration | Purpose | Quality Level | Processing Speed |
|---------------|---------|---------------|------------------|
| **`default.yaml`** | 🎯 Balanced processing | Medium | Standard |
| **`high_quality.yaml`** | 🔍 Research analysis | Maximum | Slow |
| **`quick_processing.yaml`** | ⚡ Preview generation | Low-Medium | Fast |
| **`farabi_config.yaml`** | 📋 Example configuration preset | Custom | Variable |
| **`noor_config.yaml`** | 📋 Example configuration preset | Custom | Variable |

---

## 🎯 Configuration Details

### Default Configuration (`default.yaml`)

```yaml
# Default Processing Configuration
# Balanced quality and performance for general use

processing:
  method: "process_video"
  quality_preset: "balanced"
  enable_gpu: false
  
video:
  input_format: "mp4"
  output_format: "mp4"
  
ffmpeg:
  # Standard encoding settings
  video_codec: "libx265"
  crf: 23
  audio_codec: "copy"
  
  # Optimization
  movflags: "+faststart"

output:
  prefix: "processed_"
  suffix: ""
  container: "mp4"
  
quality_control:
  enable_validation: true
  check_encoding_quality: true
  validate_audio_sync: true
```

### High Quality Configuration (`high_quality.yaml`)

```yaml
# High Quality Processing Configuration  
# Research-grade quality for detailed analysis

processing:
  method: "process_video"
  quality_preset: "high"
  enable_gpu: false
  
ffmpeg:
  # High quality encoding settings
  video_codec: "libx265"
  crf: 18
  preset: "slow"
  audio_codec: "copy"
  
  # Advanced optimization
  movflags: "+faststart"
  tune: "grain"
  
  # Direct compression without filtering
  filter_complex: null
  
  # Video encoding settings
  video_codec: "libx265"
  crf: 23
  audio_codec: "copy"
  
  # Web optimization
  movflags: "+faststart"

output:
  prefix: "compressed_"
  suffix: ""
  container: "mp4"
  
quality_control:
  verify_compression_efficiency: true
  check_web_compatibility: true
  validate_hd_quality: true
```

---

## 🎚️ Quality Preset Configurations

### Default Configuration (`default.yaml`)

```yaml
# Balanced processing for general use
processing:
  video_codec: "libx265"
  crf: 28
  preset: "medium"
  audio_codec: "aac"
  audio_bitrate: "128k"
  
performance:
  threads: "auto"
  memory_limit: "2GB"
  batch_size: 5
  
monitoring:
  progress_reporting: true
  performance_metrics: true
  error_logging: true
```

### High Quality Configuration (`high_quality.yaml`)

```yaml
# Maximum quality for research analysis
processing:
  video_codec: "libx265"
  crf: 18
  preset: "veryslow"
  audio_codec: "flac"
  
  # Advanced encoding options
  x265_params: "bframes=8:psy-rd=1:aq-mode=3"
  
performance:
  threads: "auto"
  memory_limit: "4GB"
  batch_size: 2
  
quality_control:
  verify_lossless_audio: true
  check_video_integrity: true
  validate_research_standards: true
```

### Quick Processing Configuration (`quick_processing.yaml`)

```yaml
# Fast processing for previews
processing:
  video_codec: "libx264"
  crf: 28
  preset: "ultrafast"
  audio_codec: "aac"
  audio_bitrate: "96k"
  
performance:
  threads: "auto"
  memory_limit: "1GB"
  batch_size: 10
  
optimization:
  skip_audio_processing: false
  reduce_frame_analysis: true
  fast_seek: true
```

---

## 🔄 Configuration Usage

### Framework Integration

```python
from utils.config_manager import ConfigManager

# Load hospital-specific configuration
config = ConfigManager.load_config("farabi_config.yaml")
processor = VideoProcessor(config)

# Process with Farabi hospital standards
result = processor.process_video("surgery.mp4", "compressed.mp4")
```

### CLI Usage

```bash
# Use specific hospital configuration
python main.py --config configs/farabi_config.yaml input.mp4 output.mp4

# Use quality preset
python main.py --config configs/high_quality.yaml input.mp4 output.mp4

# Batch processing with configuration
python main.py batch --config configs/noor_config.yaml input_dir/ output_dir/
```

### Configuration Override

```bash
# Override specific parameters
python main.py --config configs/default.yaml --crf 25 --preset fast input.mp4 output.mp4
```

---

## 🏗️ Configuration Structure

### Standard Configuration Schema

```yaml
# Hospital identification
hospital:
  name: string
  equipment: string
  location: string

# Input video specifications
video:
  input_resolution: string  # "WIDTHxHEIGHT"
  input_framerate: number
  expected_format: string

# Processing parameters
processing:
  privacy_protection: boolean
  deidentification_method: string
  filter_complex: string|null
  video_codec: string
  crf: number
  preset: string
  audio_codec: string
  movflags: string

# Output settings
output:
  prefix: string
  suffix: string
  container: string

# Quality control
quality_control:
  verify_privacy_removal: boolean
  check_compression_ratio: boolean
  validate_audio_sync: boolean
```

### Advanced Options

```yaml
# Performance tuning
performance:
  threads: number|"auto"
  memory_limit: string
  batch_size: number
  gpu_acceleration: boolean

# Monitoring and logging
monitoring:
  progress_reporting: boolean
  performance_metrics: boolean
  error_logging: boolean
  debug_mode: boolean

# Custom FFmpeg parameters
ffmpeg:
  input_options: list
  output_options: list
  global_options: list
```

---

## 🔧 Configuration Management

### Loading Configurations

```python
# Load specific configuration
config = ConfigManager.load_config("farabi_config.yaml")

# Load with environment overrides
config = ConfigManager.load_config("default.yaml", env_override=True)

# Merge multiple configurations
config = ConfigManager.merge_configs([
    "default.yaml",
    "farabi_config.yaml",
    "custom_overrides.yaml"
])
```

### Runtime Configuration

```python
# Modify configuration at runtime
config.set("processing.crf", 20)
config.set("output.prefix", "enhanced_")

# Validate configuration
is_valid = ConfigManager.validate_config(config)

# Get effective configuration
effective_config = ConfigManager.get_effective_config(config)
```

### Environment Variables

```bash
# Override configuration with environment variables
export SURGICAL_VIDEO_CRF=25
export SURGICAL_VIDEO_PRESET=slow
export SURGICAL_VIDEO_OUTPUT_DIR=/path/to/output

python main.py --config configs/default.yaml input.mp4
```

---

## 🧪 Configuration Testing

### Validation Tests

```python
def test_farabi_config_compliance():
    """Test Farabi configuration matches compress_video.sh"""
    config = ConfigManager.load_config("farabi_config.yaml")
    
    assert config.processing.filter_complex == "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]"
    assert config.processing.video_codec == "libx265"
    assert config.processing.crf == 23
    assert config.processing.audio_codec == "copy"

def test_noor_config_compliance():
    """Test Noor configuration matches compress_videos.bat"""
    config = ConfigManager.load_config("noor_config.yaml")
    
    assert config.processing.filter_complex is None
    assert config.processing.video_codec == "libx265"
    assert config.processing.crf == 23
    assert config.processing.movflags == "+faststart"
```

### Benchmarking

```bash
# Test configuration performance
python -m pytest tests/test_config_performance.py

# Benchmark against reference scripts
python scripts/benchmark_configs.py --config configs/farabi_config.yaml
```

---

## 📊 Configuration Comparison

| Feature | Farabi Config | Noor Config | Default Config | High Quality |
|---------|---------------|-------------|----------------|--------------|
| **Privacy Protection** | ✅ Crop+Blur+Overlay | ❌ None | ⚙️ Configurable | ⚙️ Configurable |
| **Video Codec** | libx265 | libx265 | libx265 | libx265 |
| **CRF Setting** | 23 | 23 | 28 | 18 |
| **Preset** | Default | Default | Medium | Very Slow |
| **Audio Handling** | Copy | Copy | AAC 128k | FLAC |
| **Web Optimization** | ✅ FastStart | ✅ FastStart | ✅ FastStart | ✅ FastStart |
| **Processing Speed** | Fast | Fast | Medium | Slow |
| **Quality Level** | High | High | Balanced | Maximum |

---

## 🚀 Best Practices

### Configuration Selection Guidelines

#### **Use Farabi Config when:**
- ✅ Processing videos from Haag-Streit HS Hi-R NEO 900
- ✅ Patient privacy protection required
- ✅ 720×480 resolution input videos
- ✅ Need exact compress_video.sh replication

#### **Use Noor Config when:**
- ✅ Processing videos from ZEISS ARTEVO 800
- ✅ High-definition 1080p source material
- ✅ Web streaming optimization needed
- ✅ Need exact compress_videos.bat replication

#### **Use Quality Presets when:**
- ✅ **High Quality**: Research analysis, archival storage
- ✅ **Default**: General purpose, balanced workflow
- ✅ **Quick Processing**: Preview generation, testing

### Configuration Customization

```yaml
# Custom configuration template
custom_config:
  extends: "default.yaml"
  
  overrides:
    processing:
      crf: 25
      preset: "slow"
    
    output:
      prefix: "custom_"
    
    monitoring:
      debug_mode: true
```

---

**⚙️ These configurations ensure consistent, hospital-specific processing aligned with Cataract-LMM dataset standards and reference script methodologies.**
