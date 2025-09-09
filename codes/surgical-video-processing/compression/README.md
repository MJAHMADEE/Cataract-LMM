# 🗜️ Video Compression Module

> **Advanced video compression optimized for surgical video preservation with medical-grade quality retention**

## 📋 Overview

This module provides **intelligent video compression** specifically designed for **phacoemulsification cataract surgery videos**. It implements advanced algorithms that balance file size reduction with preservation of critical surgical details essential for medical analysis, training, and AI model development.

### 🎯 Medical-Grade Compression Philosophy

Following **surgical video quality standards** with:
- **🔍 Detail Preservation**: Maintains surgical instrument visibility and texture details
- **⚖️ Optimal Balance**: Maximum compression while preserving diagnostic quality
- **📊 Content-Aware**: Adaptive compression based on surgical scene analysis
- **🚀 Performance Optimized**: GPU-accelerated encoding for high-throughput processing
- **🏥 Clinical Standards**: Meets medical video archival and analysis requirements

## 🧠 Compression Intelligence

### 📊 Content-Aware Analysis

The compression system analyzes surgical video content to optimize encoding parameters:

- **🔬 Instrument Detection**: Identifies surgical tools and preserves their detail
- **💧 Fluid Recognition**: Optimizes compression for irrigation and aspiration phases
- **🎯 ROI Prioritization**: Higher quality encoding for critical surgical regions
- **⚡ Motion Analysis**: Adapts to surgical movement patterns and camera stability

### 🎛️ Adaptive Compression Strategies

| Surgical Phase | Compression Strategy | Quality Priority |
|:--------------|:--------------------|:-----------------|
| **Incision** | 🔍 High Detail Retention | Instrument edges, tissue detail |
| **Phacoemulsification** | ⚡ Motion Optimization | Ultrasonic tip visibility, debris tracking |
| **Aspiration** | 💧 Fluid-Aware Encoding | Clear fluid flow visualization |
| **Lens Insertion** | 🎯 Precision Preservation | IOL positioning, haptic detail |

## 🏗️ Architecture Components

### 🔧 Core Classes

#### **AdaptiveCompressor**
*Intelligent compression based on real-time content analysis*

```python
from compression import AdaptiveCompressor

# Initialize with surgical-optimized settings
compressor = AdaptiveCompressor(
    quality_target='surgical_archive',  # or 'ai_training', 'streaming'
    preserve_instruments=True,
    adaptive_bitrate=True,
    gpu_acceleration=True
)

# Process surgical video with content analysis
result = compressor.compress_video(
    input_path='cataract_surgery.mp4',
    output_path='compressed_surgery.mp4',
    analysis_mode='full'  # Analyzes entire video for optimal settings
)
```

#### **QualityPreservingCompressor**
*Maximum detail preservation for diagnostic and training purposes*

```python
from compression import QualityPreservingCompressor

# Medical-grade quality preservation
preserving_compressor = QualityPreservingCompressor(
    min_quality_threshold=0.95,  # Minimum SSIM score
    preserve_surgical_detail=True,
    instrument_enhancement=True,
    texture_preservation='high'
)

# Compress while maintaining diagnostic quality
result = preserving_compressor.compress(
    'high_quality_surgery.mov',
    target_size_reduction=0.7,  # 70% size reduction
    quality_validation=True
)
```

#### **BatchCompressor**
*High-throughput processing for surgical video archives*

```python
from compression import BatchCompressor

# Setup batch processing for surgical archives
batch_compressor = BatchCompressor(
    workers=8,
    gpu_workers=2,
    progress_callback=True,
    quality_monitoring=True
)

# Process entire surgical video library
results = batch_compressor.compress_directory(
    input_dir='/surgical_videos/raw/',
    output_dir='/surgical_videos/compressed/',
    recursive=True,
    filter_extensions=['.mp4', '.mov', '.avi']
)
```

## 📊 Compression Algorithms

### 🎥 Medical-Optimized Encoding

#### H.264 Surgical Profile
```python
h264_surgical_config = {
    'codec': 'libx264',
    'profile': 'high',
    'preset': 'slow',      # Quality over speed for medical use
    'crf': 18,             # High quality (lower = better)
    'tune': 'grain',       # Preserve surgical texture
    'x264opts': {
        'bframes': 3,
        'ref': 5,
        'deblock': '1:1',
        'subme': 8,
        'trellis': 2
    }
}
```

#### H.265/HEVC Medical Profile
```python
hevc_medical_config = {
    'codec': 'libx265',
    'preset': 'slow',
    'crf': 20,             # Balanced quality/size for H.265
    'tune': 'grain',
    'x265params': {
        'bframes': 4,
        'ref': 4,
        'rd': 4,
        'subme': 3,
        'range': 'limited',
        'colorprim': 'bt709'
    }
}
```

### 🧠 Content-Aware Optimization

#### Surgical Scene Analysis
```python
def analyze_surgical_content(video_path):
    """
    Analyze surgical video content to optimize compression.
    
    Returns:
        Dict with surgical scene characteristics and recommended settings
    """
    analysis = {
        'instrument_density': calculate_instrument_presence(video_path),
        'motion_complexity': analyze_surgical_motion(video_path),
        'lighting_stability': measure_illumination_variance(video_path),
        'fluid_presence': detect_irrigation_phases(video_path),
        'critical_regions': identify_surgical_roi(video_path)
    }
    
    # Generate optimized encoding parameters
    return optimize_for_surgical_content(analysis)
```

#### ROI-Based Quality Allocation
```python
def apply_roi_compression(video_path, surgical_phases):
    """
    Apply region-of-interest based quality allocation.
    
    Higher quality for:
    - Surgical instrument tips
    - Incision sites
    - Critical anatomical structures
    """
    roi_map = generate_surgical_roi_map(video_path, surgical_phases)
    
    # Create quality mask
    quality_params = {
        'instrument_regions': {'crf': 16, 'priority': 'high'},
        'surgical_field': {'crf': 18, 'priority': 'medium'},
        'background': {'crf': 24, 'priority': 'low'}
    }
    
    return apply_variable_quality_encoding(video_path, roi_map, quality_params)
```

## 🚀 Usage Examples

### 🔍 Single Video Compression

```python
from compression import AdaptiveCompressor

# Initialize compressor for surgical videos
compressor = AdaptiveCompressor(
    target_quality='medical_grade',
    gpu_acceleration=True,
    content_analysis=True
)

# Compress surgical video with quality preservation
result = compressor.compress_video(
    input_path='phaco_surgery_4k.mp4',
    output_path='phaco_surgery_compressed.mp4',
    preserve_metadata=True,
    quality_validation=True
)

# Review compression results
print(f"Original size: {result['original_size_mb']:.1f} MB")
print(f"Compressed size: {result['compressed_size_mb']:.1f} MB")
print(f"Size reduction: {result['compression_ratio']:.1f}x")
print(f"Quality score (SSIM): {result['quality_metrics']['ssim']:.3f}")
print(f"Surgical detail preservation: {result['surgical_quality_score']:.3f}")
```

### 📊 Batch Processing with Analysis

```python
from compression import BatchCompressor, CompressionAnalyzer

# Setup batch processing
batch_processor = BatchCompressor(
    workers=6,
    analysis_enabled=True,
    quality_monitoring=True,
    progress_bar=True
)

# Process surgical video library
results = batch_processor.process_directory(
    input_dir='/medical_videos/cataract_surgeries/',
    output_dir='/medical_videos/compressed/',
    settings={
        'target_bitrate': 'adaptive',
        'preserve_surgical_detail': True,
        'gpu_acceleration': True,
        'quality_threshold': 0.92
    }
)

# Analyze compression performance
analyzer = CompressionAnalyzer()
report = analyzer.generate_report(results)

print(f"Total videos processed: {report['total_videos']}")
print(f"Average compression ratio: {report['avg_compression_ratio']:.1f}x")
print(f"Average quality retention: {report['avg_quality_score']:.3f}")
print(f"Processing speed: {report['avg_fps']:.1f} FPS")
```

### 🏥 Hospital-Specific Optimization

```python
# Farabi Hospital optimization (720p@30fps)
farabi_compressor = AdaptiveCompressor.create_hospital_profile(
    hospital='farabi',
    input_specs={'resolution': '720p', 'fps': 30},
    target_specs={'resolution': '720p', 'fps': 30},
    quality_priority='balanced'
)

# Noor Hospital optimization (1080p@60fps)  
noor_compressor = AdaptiveCompressor.create_hospital_profile(
    hospital='noor',
    input_specs={'resolution': '1080p', 'fps': 60},
    target_specs={'resolution': '1080p', 'fps': 30},  # Downsample FPS
    quality_priority='maximum'
)
```

## 📈 Performance Optimization

### 🚀 GPU Acceleration

```python
# NVENC hardware encoding for NVIDIA GPUs
gpu_config = {
    'encoder': 'h264_nvenc',    # or 'hevc_nvenc'
    'preset': 'slow',           # Quality over speed
    'profile': 'high',
    'rc': 'vbr',               # Variable bitrate
    'cq': 20,                  # Constant quality
    'gpu': 0,                  # GPU device ID
    'surfaces': 32             # Encoding surfaces
}

# Enable GPU acceleration
compressor = AdaptiveCompressor(
    gpu_acceleration=True,
    gpu_config=gpu_config,
    fallback_to_cpu=True
)
```

### ⚡ Performance Monitoring

```python
def monitor_compression_performance(compressor):
    """Monitor compression performance metrics."""
    metrics = compressor.get_performance_metrics()
    
    return {
        'encoding_fps': metrics['frames_per_second'],
        'gpu_utilization': metrics['gpu_usage_percent'],
        'memory_usage': metrics['memory_mb'],
        'cpu_usage': metrics['cpu_percent'],
        'throughput_mbps': metrics['throughput']
    }
```

## 🧪 Quality Assessment

### 📊 Medical Quality Metrics

The compression module includes specialized quality assessment for surgical videos:

```python
from compression import CompressionQualityAssessment

# Comprehensive quality analysis
quality_assessor = CompressionQualityAssessment()

quality_report = quality_assessor.analyze_compressed_video(
    original_path='original_surgery.mp4',
    compressed_path='compressed_surgery.mp4',
    assessment_type='surgical_comprehensive'
)

# Quality metrics specific to surgical videos
surgical_metrics = {
    'instrument_detail_preservation': quality_report['instrument_ssim'],
    'tissue_texture_retention': quality_report['tissue_quality'],
    'fluid_flow_clarity': quality_report['fluid_visibility'],
    'overall_diagnostic_quality': quality_report['diagnostic_score']
}
```

### 🎯 Validation Thresholds

```python
# Medical-grade quality thresholds
SURGICAL_QUALITY_THRESHOLDS = {
    'diagnostic_use': {'min_ssim': 0.95, 'min_psnr': 40},
    'training_videos': {'min_ssim': 0.92, 'min_psnr': 38},
    'archival_storage': {'min_ssim': 0.90, 'min_psnr': 36},
    'streaming_preview': {'min_ssim': 0.85, 'min_psnr': 32}
}
```

## ⚙️ Configuration Management

### 📋 Compression Profiles

```yaml
# compression_profiles.yaml
profiles:
  diagnostic_archive:
    codec: libx265
    crf: 18
    preset: slow
    tune: grain
    quality_threshold: 0.95
    preserve_metadata: true
    
  ai_training:
    codec: libx264  
    crf: 20
    preset: medium
    batch_optimization: true
    augmentation_safe: true
    
  streaming_delivery:
    codec: libx264
    crf: 23
    preset: fast
    adaptive_bitrate: true
    low_latency: true
```

## 📚 Integration Examples

### 🔄 Pipeline Integration

```python
# Integration with video processing pipeline
from surgical_video_processing import SurgicalVideoProcessor
from compression import AdaptiveCompressor

pipeline = SurgicalVideoProcessor()
compressor = AdaptiveCompressor(quality_target='ai_training')

# Combined processing and compression
def process_and_compress_surgical_video(input_path, output_path):
    # Step 1: Apply surgical video processing
    processed_path = pipeline.process_video(
        input_path, 
        hospital_config='auto_detect',
        deidentification=True
    )
    
    # Step 2: Intelligent compression
    compressed_result = compressor.compress_video(
        processed_path,
        output_path,
        preserve_processing_metadata=True
    )
    
    return compressed_result
```

## 📚 References

- **FFmpeg Medical Encoding**: Advanced encoding techniques for medical video
- **H.264/H.265 Standards**: Video compression standards optimized for medical use
- **Surgical Video Quality**: Medical imaging quality assessment standards
- **GPU Acceleration**: NVENC/QuickSync hardware encoding optimization

---

**💡 Note**: This compression module is specifically designed for surgical video content and maintains the visual quality necessary for medical diagnosis, training, and AI model development while achieving significant file size reductions.
