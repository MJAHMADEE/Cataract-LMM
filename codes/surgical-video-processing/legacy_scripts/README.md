# Legacy Scripts

This directory contains the original video processing scripts that the surgical video processing framework replaces, preserved for reference and migration purposes.

## Overview

These are the original scripts from `/workspaces/Cataract_LMM/codes/Video Processing/` that were used for basic video compression before the comprehensive framework was developed. They are maintained here for:

- **Historical reference** and migration documentation
- **Backward compatibility** for legacy workflows
- **Parameter mapping** to new framework configurations
- **Validation** of equivalent functionality in new system

## Files

### `compress_video.sh` (Linux/macOS Script)
Original bash script for video compression on Unix-like systems.

**Original Purpose:**
- Batch compression of MP4 files in current directory
- Apply cropping and blur filters to remove patient information
- Use H.265 encoding for file size reduction

**Key Parameters:**
```bash
# FFmpeg command from original script
ffmpeg -i "$input_file" \
    -filter_complex "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]" \
    -map "[v]" -map 0:a \
    -c:v libx265 -crf 23 -c:a copy -movflags +faststart "$output_file"
```

**Filter Breakdown:**
- `crop=268:58:6:422`: Crop 268×58 region starting at position (6,422)
- `avgblur=10`: Apply average blur with radius 10
- `overlay=6:422`: Overlay blurred region back to original position
- `libx265 -crf 23`: H.265 encoding with CRF 23 (good quality)
- `copy` audio: Preserve original audio track
- `+faststart`: Optimize for web streaming

### `compress_videos.bat` (Windows Script)
Original Windows batch script for video compression.

**Original Purpose:**
- Batch processing of MP4 files on Windows systems
- Simplified compression without complex filtering
- Basic H.265 encoding for storage optimization

**Key Parameters:**
```batch
ffmpeg -i "!input_file!" -vcodec libx265 -crf 23 -movflags +faststart "!output_file!"
```

**Differences from Unix version:**
- No cropping or blurring filters
- Simpler compression-only approach
- Windows batch syntax with delayed expansion

## Migration to New Framework

### Equivalent Modern Configuration
The legacy scripts' functionality is replicated and enhanced in the new framework through hospital-specific configurations.

**For `compress_video.sh` (Farabi Hospital equivalent):**
```yaml
# farabi_config.yaml equivalent settings
deidentification:
  blur_patient_info: true
  timestamp_regions:
    - [6, 422, 268, 58]  # Equivalent to crop region
  blur_strength: 30     # Enhanced blur (was avgblur=10)

compression:
  video_codec: "libx265"
  crf_value: 23
  movflags: ["+faststart"]
  preserve_audio: true  # Equivalent to -c:a copy
```

**For `compress_videos.bat` (Simple compression):**
```yaml
# quick_processing.yaml equivalent
compression:
  video_codec: "libx265" 
  crf_value: 23
  movflags: ["+faststart"]
  compression_speed: "fast"
```

### Enhanced Capabilities in New Framework
The modern framework provides significant improvements over legacy scripts:

**Privacy Protection:**
- **Legacy**: Fixed crop region with blur
- **Modern**: Intelligent text detection, adaptive blurring, comprehensive metadata removal

**Quality Control:**
- **Legacy**: No quality assessment
- **Modern**: Automated focus, glare, exposure, and motion analysis

**Hospital Support:**
- **Legacy**: One-size-fits-all approach
- **Modern**: Hospital-specific optimizations for different equipment

**Processing Pipeline:**
- **Legacy**: Single-step compression
- **Modern**: Multi-stage pipeline with preprocessing, quality control, de-identification, and compression

**Error Handling:**
- **Legacy**: Basic error reporting
- **Modern**: Comprehensive error handling, logging, and recovery

**Configuration:**
- **Legacy**: Hard-coded parameters
- **Modern**: Flexible YAML configurations with environment overrides

## Usage Comparison

### Legacy Workflow
```bash
# Original bash script usage
cd /path/to/videos
./compress_video.sh
# Output: compressed_*.mp4 files in same directory
```

```batch
# Original Windows batch usage
cd C:\path\to\videos
compress_videos.bat
# Output: compressed_*.mp4 files in same directory
```

### Modern Framework Equivalent
```bash
# Modern framework usage
python main.py --hospital farabi --input /path/to/videos --output /path/to/processed

# With same compression settings as legacy
python main.py --config legacy_equivalent.yaml --input /path/to/videos --output /path/to/processed
```

### Direct Parameter Mapping
```python
# Convert legacy parameters to modern configuration
legacy_params = {
    "crop_region": "268:58:6:422",
    "blur_radius": 10,
    "codec": "libx265", 
    "crf": 23,
    "audio_copy": True,
    "faststart": True
}

modern_config = convert_legacy_parameters(legacy_params)
# Generates equivalent YAML configuration
```

## Performance Comparison

### Legacy Scripts Performance
- **Processing Speed**: ~1.0x real-time (no optimization)
- **Quality Control**: None (no quality assessment)
- **Error Handling**: Basic (stops on first error)
- **Parallel Processing**: None (sequential only)
- **Memory Usage**: Uncontrolled (depends on FFmpeg)

### Modern Framework Performance
- **Processing Speed**: 0.1x to 3.0x real-time (configurable)
- **Quality Control**: Comprehensive automated assessment
- **Error Handling**: Robust with detailed logging
- **Parallel Processing**: Multi-worker batch processing
- **Memory Usage**: Controlled and optimized

## Validation Scripts

### Legacy Output Validation
```python
# Validate legacy script output
def validate_legacy_compression(input_file, output_file):
    """Validate that legacy compression worked correctly."""
    
    # Check file exists and is playable
    assert Path(output_file).exists()
    assert is_valid_video(output_file)
    
    # Check compression achieved size reduction
    original_size = Path(input_file).stat().st_size
    compressed_size = Path(output_file).stat().st_size
    compression_ratio = compressed_size / original_size
    
    assert compression_ratio < 0.8  # At least 20% reduction
    
    # Check codec is H.265
    metadata = extract_video_metadata(output_file)
    assert metadata['codec'] == 'hevc'  # H.265
    
    return True
```

### Framework Equivalency Test
```python
# Test framework produces equivalent results
def test_framework_equivalency():
    """Test that framework produces equivalent results to legacy scripts."""
    
    # Process with legacy-equivalent configuration
    config = ConfigManager.load_config("legacy_equivalent.yaml")
    processor = SurgicalVideoProcessor(config)
    
    result = processor.process_video("test_input.mp4", "./output/")
    
    # Compare with legacy output
    legacy_output = process_with_legacy_script("test_input.mp4")
    framework_output = result.output_path
    
    # Validate similar compression and quality
    assert compare_video_quality(legacy_output, framework_output) > 0.95
    assert compare_file_sizes(legacy_output, framework_output) < 0.1  # Within 10%
```

## Migration Guide

### Step 1: Backup Legacy Scripts
```bash
# Backup original scripts
mkdir -p ./legacy_backup
cp "/workspaces/Cataract_LMM/codes/Video Processing/"* ./legacy_backup/
```

### Step 2: Test Framework Equivalency
```bash
# Test with sample video
python main.py --config legacy_equivalent.yaml --input sample.mp4 --output ./test_output

# Compare results with legacy output
./legacy_backup/compress_video.sh  # Process with legacy
python scripts/utilities.py compare-outputs ./test_output ./legacy_output
```

### Step 3: Create Custom Configuration
```yaml
# custom_legacy_equivalent.yaml
# Replicate exact legacy behavior

deidentification:
  # Replicate crop=268:58:6:422,avgblur=10
  roi_blur_regions:
    - [6, 422, 268, 58]
  blur_strength: 10  # Match original avgblur=10

compression:
  video_codec: "libx265"
  crf_value: 23
  audio_codec: "copy"  # Preserve original audio
  movflags: ["+faststart"]
  
# Disable other processing for exact equivalency
preprocessing:
  enhance_contrast: false
  reduce_noise: false
  
quality_control:
  enable_focus_check: false
  enable_glare_check: false
```

### Step 4: Batch Migration
```bash
# Process entire dataset with new framework
python main.py --config custom_legacy_equivalent.yaml \
               --input /path/to/original/videos \
               --output /path/to/migrated/videos \
               --backup
```

### Step 5: Validation and Cleanup
```bash
# Validate migration results
python scripts/utilities.py validate-migration \
       --original /path/to/original/videos \
       --migrated /path/to/migrated/videos

# Clean up if validation passes
python scripts/utilities.py cleanup
```

## Deprecation Notice

**⚠️ Legacy Script Status: DEPRECATED**

These legacy scripts are maintained for historical reference only. For new processing tasks, use the modern framework:

**Instead of:**
```bash
./compress_video.sh
```

**Use:**
```bash
python main.py --hospital farabi --input ./videos --output ./processed
```

**Benefits of Migration:**
- ✅ **Enhanced privacy protection** with intelligent de-identification
- ✅ **Quality control** with automated assessment and filtering  
- ✅ **Hospital optimization** with equipment-specific configurations
- ✅ **Robust error handling** with detailed logging and recovery
- ✅ **Batch processing** with parallel execution and progress tracking
- ✅ **Flexible configuration** with YAML-based settings management

## Support and Questions

For questions about legacy script migration or framework equivalency:

1. **Review migration documentation** in this README
2. **Test with sample videos** using legacy_equivalent.yaml configuration  
3. **Compare outputs** using validation scripts
4. **Check migration logs** for detailed processing information

The framework is designed to provide a smooth migration path while offering significantly enhanced capabilities for surgical video processing workflows.
