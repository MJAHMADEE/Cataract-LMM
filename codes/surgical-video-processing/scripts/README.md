# Utility Scripts

Collection of utility scripts for system administration, data migration, and maintenance tasks.

## Overview

This directory contains utility scripts that help with setup, maintenance, and administrative tasks for the surgical video processing framework.

## Available Scripts

### System Setup (`setup`)
Verify system dependencies and prepare the environment:

```bash
python scripts/utilities.py setup
```

**Features:**
- ✅ **FFmpeg verification**: Checks FFmpeg installation and accessibility
- ✅ **Python dependencies**: Verifies all required packages are installed
- ✅ **Directory creation**: Creates necessary working directories
- ✅ **System validation**: Ensures environment is ready for processing

**Output:**
```
Setting up surgical video processing environment...
✓ FFmpeg is installed and accessible
✓ Python dependencies are installed
✓ Created directory: ./logs
✓ Created directory: ./metadata
✓ Created directory: ./temp
✓ Created directory: ./output
✓ Created directory: ./backup
✓ System setup complete!
```

### Legacy Migration (`migrate`)
Migrate from original Video Processing scripts:

```bash
python scripts/utilities.py migrate
```

**Features:**
- 📂 **Script backup**: Preserves original compress_video.sh and compress_videos.bat
- 🔄 **Parameter mapping**: Maps legacy FFmpeg parameters to new configurations
- 📝 **Migration documentation**: Creates migration_info.json with equivalents
- ⚙️ **Configuration guidance**: Recommends equivalent modern configurations

**Migration Mapping:**
```json
{
  "legacy_parameters": {
    "crop_filter": "crop=268:58:6:422,avgblur=10",
    "codec": "libx265",
    "crf": 23,
    "copy_audio": true
  },
  "equivalent_config": "farabi_config.yaml"
}
```

### Metadata Extraction (`extract-metadata`)
Batch extract metadata from video collections:

```bash
python scripts/utilities.py extract-metadata --input ./videos --output ./metadata --format json
```

**Options:**
- `--input`, `-i`: Input directory containing videos
- `--output`, `-o`: Output directory for metadata files (default: ./metadata)
- `--format`: Output format - json or yaml (default: json)

**Features:**
- 🎥 **Comprehensive extraction**: Technical specs, quality metrics, hospital detection
- 📊 **Batch reporting**: Generates aggregate statistics
- 🏥 **Hospital identification**: Automatic hospital and equipment detection
- 📁 **Organized output**: Individual metadata files plus summary report

**Output Structure:**
```
metadata/
├── video_001_metadata.json
├── video_002_metadata.json
├── ...
└── batch_report.json
```

### Configuration Validation (`validate-config`)
Validate all YAML configuration files:

```bash
python scripts/utilities.py validate-config
```

**Features:**
- ✅ **Syntax validation**: YAML syntax and structure checking
- 🔧 **Schema validation**: Ensures required fields are present
- 🏥 **Hospital configs**: Validates hospital-specific configurations
- 📋 **Comprehensive reporting**: Shows validation status for all configs

**Sample Output:**
```
Validating configuration files...
Validating default.yaml...
✓ default.yaml is valid
Validating farabi_config.yaml...
✓ farabi_config.yaml is valid
Validating noor_config.yaml...
✓ noor_config.yaml is valid
Configuration validation complete!
```

### Dataset Analysis (`analyze-dataset`)
Comprehensive dataset analysis and reporting:

```bash
python scripts/utilities.py analyze-dataset --input ./videos --output dataset_report.json
```

**Options:**
- `--input`, `-i`: Directory containing video dataset
- `--output`, `-o`: Output report file (default: ./dataset_analysis.json)

**Analysis Features:**
- 🏥 **Hospital distribution**: Video counts per hospital
- 📐 **Resolution analysis**: Resolution distribution across dataset
- 🎞️ **Frame rate analysis**: FPS distribution and patterns
- ⏱️ **Duration statistics**: Total, mean, min, max duration
- 💾 **Storage analysis**: File sizes and total storage requirements
- 🎬 **Codec distribution**: Video codec usage patterns

**Sample Report:**
```json
{
  "total_videos": 3000,
  "hospitals": {
    "farabi": 1500,
    "noor": 1500,
    "unknown": 0
  },
  "resolutions": {
    "720x480": 1500,
    "1920x1080": 1500
  },
  "duration_stats": {
    "total": 1250.5,
    "mean": 15.2,
    "min": 8.3,
    "max": 28.7
  },
  "file_sizes": {
    "total_gb": 245.8,
    "mean_mb": 84.2
  }
}
```

### System Cleanup (`cleanup`)
Clean temporary files and reset working directories:

```bash
python scripts/utilities.py cleanup
```

**Features:**
- 🧹 **Temporary file removal**: Cleans temp processing files
- 📝 **Log rotation**: Removes old log files
- 🗂️ **Cache cleanup**: Removes Python cache files
- 🔄 **Directory reset**: Recreates necessary directories

**Cleaned Directories:**
- `./temp/` - Temporary processing files
- `./logs/` - Old log files (recreated empty)
- `./__pycache__/` - Python bytecode cache
- `./surgical_video_processing/__pycache__/` - Module cache

### Performance Benchmarking (`benchmark`)
Test processing performance across different configurations:

```bash
python scripts/utilities.py benchmark --test-video sample.mp4 --output benchmark_results.json
```

**Options:**
- `--test-video`: Video file to use for benchmarking
- `--output`: Results file (default: ./benchmark_results.json)

**Benchmark Configurations:**
- **Quick Processing**: Speed-optimized settings
- **Default Processing**: Balanced quality/speed
- **High Quality**: Maximum quality settings
- **Farabi Hospital**: Hospital-specific optimization
- **Noor Hospital**: Equipment-specific optimization

**Benchmark Metrics:**
- ⏱️ **Processing time**: Time to complete processing
- ✅ **Success rate**: Whether processing completed successfully
- 📊 **Output size**: Size of processed video file
- 🎯 **Quality score**: Resulting quality assessment score

**Sample Results:**
```json
{
  "test_video": "sample.mp4",
  "benchmarks": [
    {
      "config_name": "Quick Processing",
      "processing_time": 12.5,
      "success": true,
      "output_size": 15728640,
      "quality_score": 78.2
    },
    {
      "config_name": "High Quality",
      "processing_time": 45.8,
      "success": true,
      "output_size": 52428800,
      "quality_score": 94.1
    }
  ]
}
```

## Usage Examples

### Complete System Setup
```bash
# Initial system setup
python scripts/utilities.py setup

# Migrate from legacy scripts (if applicable)
python scripts/utilities.py migrate

# Validate all configurations
python scripts/utilities.py validate-config
```

### Dataset Preparation
```bash
# Analyze existing dataset
python scripts/utilities.py analyze-dataset --input /path/to/videos

# Extract metadata for all videos
python scripts/utilities.py extract-metadata --input /path/to/videos --output ./metadata

# Validate processing readiness
python scripts/utilities.py validate-config
```

### Performance Testing
```bash
# Test with sample video
python scripts/utilities.py benchmark --test-video sample_surgery.mp4

# Compare different hospital configurations
python scripts/utilities.py benchmark --test-video farabi_sample.mp4
python scripts/utilities.py benchmark --test-video noor_sample.mp4
```

### Maintenance Tasks
```bash
# Regular cleanup
python scripts/utilities.py cleanup

# Re-validate configurations after changes
python scripts/utilities.py validate-config

# Re-analyze dataset after additions
python scripts/utilities.py analyze-dataset --input ./videos
```

## Integration with Main Framework

### Command Line Integration
```bash
# Process videos after setup
python scripts/utilities.py setup
python main.py --hospital farabi --input ./videos --output ./processed
```

### Automated Workflows
```bash
#!/bin/bash
# Complete processing workflow

# Setup and validation
python scripts/utilities.py setup
python scripts/utilities.py validate-config

# Dataset analysis
python scripts/utilities.py analyze-dataset --input ./raw_videos

# Batch processing
python main.py --hospital farabi --input ./raw_videos/farabi --output ./processed/farabi
python main.py --hospital noor --input ./raw_videos/noor --output ./processed/noor

# Cleanup
python scripts/utilities.py cleanup
```

### Configuration Management
```python
# Use utilities in Python scripts
from surgical_video_processing.scripts import (
    setup_system, validate_configuration, analyze_dataset
)

# Programmatic setup
if setup_system():
    print("System ready for processing")

# Validate before processing
validate_configuration()

# Analyze results
analyze_dataset("./processed_videos", "./analysis_report.json")
```

## Error Handling

### Common Issues and Solutions

**FFmpeg Not Found:**
```bash
# Install FFmpeg on Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Install FFmpeg on macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

**Missing Python Dependencies:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Install specific missing packages
pip install opencv-python numpy scikit-image PyYAML
```

**Permission Errors:**
```bash
# Fix directory permissions
chmod -R 755 ./surgical_video_processing
sudo chown -R $USER:$USER ./surgical_video_processing
```

**Configuration Errors:**
```bash
# Validate and fix configurations
python scripts/utilities.py validate-config

# Reset to default configurations
cp configs/default.yaml configs/default.yaml.backup
git checkout configs/default.yaml
```

## Advanced Usage

### Custom Analysis Scripts
```python
# Extend dataset analysis
def custom_analysis(video_dir):
    from surgical_video_processing.scripts import analyze_dataset
    
    # Standard analysis
    analyze_dataset(video_dir, "standard_report.json")
    
    # Custom quality analysis
    # ... additional analysis code
```

### Automated Monitoring
```bash
# Cron job for regular cleanup (daily at 2 AM)
0 2 * * * cd /path/to/framework && python scripts/utilities.py cleanup

# Weekly dataset analysis
0 3 * * 0 cd /path/to/framework && python scripts/utilities.py analyze-dataset --input ./data
```

### Integration Testing
```python
# Test complete workflow
def test_workflow():
    import subprocess
    
    # Setup
    result = subprocess.run(['python', 'scripts/utilities.py', 'setup'])
    assert result.returncode == 0
    
    # Validate
    result = subprocess.run(['python', 'scripts/utilities.py', 'validate-config'])
    assert result.returncode == 0
    
    # Process
    result = subprocess.run(['python', 'main.py', '--quick', '--input', 'test.mp4', '--output', './test_output'])
    assert result.returncode == 0
```

## Best Practices

1. **Regular Maintenance**: Run cleanup and validation scripts regularly
2. **Performance Monitoring**: Benchmark performance after system changes
3. **Dataset Tracking**: Analyze datasets before and after processing
4. **Configuration Management**: Validate configurations before deployment
5. **Backup Management**: Use migration scripts to preserve legacy workflows
