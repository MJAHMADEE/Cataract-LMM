# Video Preprocessing Module

## Overview

The preprocessing module handles video standardization, enhancement, and format conversion for phacoemulsification cataract surgery videos. It ensures consistent video parameters across different recording equipment and applies quality improvements optimized for surgical microscopy.

## Key Components

### VideoStandardizer
Standardizes video format, resolution, and frame rate across different source hospitals:
- **Resolution normalization**: Converts videos to consistent resolution
- **Frame rate standardization**: Normalizes frame rates for uniform processing
- **Format conversion**: Ensures consistent video format (MP4, H.264)
- **Cropping**: Removes unwanted margins and artifacts

### VideoEnhancer
Applies enhancement techniques specifically for surgical video quality:
- **Contrast enhancement**: Improves visibility of surgical details
- **Noise reduction**: Reduces camera noise and compression artifacts
- **Brightness/gamma correction**: Optimizes lighting conditions
- **Sharpening**: Enhances edge definition for better clarity
- **Color adjustment**: Normalizes color balance and saturation

### FrameExtractor
Extracts representative frames for analysis and quality assessment:
- **Temporal sampling**: Extracts frames at specified intervals
- **Quality assessment**: Generates samples for automated quality checks
- **Thumbnail generation**: Creates preview images
- **Analysis preparation**: Provides frames for computer vision tasks

### HospitalSpecificPreprocessor
Applies hospital-specific adjustments for different recording equipment:
- **Farabi Hospital (Haag-Streit HS Hi-R NEO 900)**:
  - 720×480 resolution, 30 fps
  - Slight brightness boost for better visibility
  - Noise reduction to compensate for lower resolution
  - Margin cropping to remove equipment artifacts
  
- **Noor Hospital (ZEISS ARTEVO 800)**:
  - 1920×1080 resolution, 60 fps
  - Minimal processing due to higher quality source
  - Preservation of original color characteristics
  - No cropping required

## Configuration

### PreprocessingConfig
Comprehensive configuration for all preprocessing operations:

```python
@dataclass
class PreprocessingConfig:
    # Standardization settings
    standardize_resolution: bool = True
    target_width: int = 1280
    target_height: int = 720
    standardize_fps: bool = True
    target_fps: float = 30.0
    
    # Enhancement settings
    enhance_contrast: bool = True
    reduce_noise: bool = True
    brightness_adjustment: int = 0  # -100 to 100
    contrast_adjustment: float = 1.0  # 0.5 to 2.0
    gamma_correction: float = 1.0  # 0.1 to 3.0
    
    # Advanced settings
    crop_margins: Tuple[int, int, int, int] = (0, 0, 0, 0)
    noise_reduction_strength: int = 3  # 1-10
    sharpen_kernel_size: int = 3
```

## Usage Examples

### Basic Video Standardization

```python
from surgical_video_processing.preprocessing import VideoStandardizer, PreprocessingConfig
from surgical_video_processing.core import ProcessingConfig

# Create configurations
processing_config = ProcessingConfig()
preprocessing_config = PreprocessingConfig(
    target_width=1280,
    target_height=720,
    target_fps=30.0
)

# Initialize standardizer
standardizer = VideoStandardizer(processing_config, preprocessing_config)

# Process video
result = standardizer.process(
    input_path="raw_surgery.mp4",
    output_path="standardized_surgery.mp4"
)
```

### Video Enhancement

```python
from surgical_video_processing.preprocessing import VideoEnhancer

# Create enhancement configuration
enhancement_config = PreprocessingConfig(
    enhance_contrast=True,
    reduce_noise=True,
    brightness_adjustment=10,
    contrast_adjustment=1.2,
    gamma_correction=1.1
)

# Initialize enhancer
enhancer = VideoEnhancer(processing_config, enhancement_config)

# Enhance video
result = enhancer.process(
    input_path="standardized_surgery.mp4",
    output_path="enhanced_surgery.mp4"
)
```

### Frame Extraction

```python
from surgical_video_processing.preprocessing import FrameExtractor

# Initialize frame extractor
extractor = FrameExtractor(
    config=processing_config,
    extraction_interval=5.0,  # Every 5 seconds
    max_frames=100,
    output_format='jpg'
)

# Extract frames
result = extractor.process(
    input_path="surgery_video.mp4",
    output_path="frames_output_dir/"
)

# Access extracted frame paths
frame_paths = result.metrics['extracted_files']
```

### Hospital-Specific Processing

```python
from surgical_video_processing.preprocessing import HospitalSpecificPreprocessor
from surgical_video_processing.core import VideoMetadata, HospitalSource

# Create metadata with hospital information
metadata = VideoMetadata(
    file_path=Path("surgery.mp4"),
    original_filename="surgery.mp4",
    hospital_source=HospitalSource.FARABI
)

# Initialize hospital-specific processor
processor = HospitalSpecificPreprocessor(processing_config)

# Process with hospital-specific settings
result = processor.process(
    input_path="surgery.mp4",
    output_path="processed_surgery.mp4",
    metadata=metadata
)
```

### Complete Preprocessing Pipeline

```python
from surgical_video_processing.core import ProcessingPipeline
from surgical_video_processing.preprocessing import (
    HospitalSpecificPreprocessor, VideoStandardizer, VideoEnhancer
)

# Create pipeline
pipeline = ProcessingPipeline(processing_config)

# Add processors in order
pipeline.add_processor(HospitalSpecificPreprocessor(processing_config))
pipeline.add_processor(VideoStandardizer(processing_config, preprocessing_config))
pipeline.add_processor(VideoEnhancer(processing_config, preprocessing_config))

# Execute complete preprocessing
results = pipeline.process(
    input_path="raw_surgery.mp4",
    output_dir="preprocessed_output/"
)
```

## Quality Considerations

### Resolution Standardization
- **Target Resolution**: 1280×720 provides optimal balance between quality and file size
- **Aspect Ratio**: Maintains original aspect ratio while standardizing dimensions
- **Upscaling**: Uses high-quality algorithms for upscaling lower resolution sources
- **Downscaling**: Preserves detail when downscaling high-resolution sources

### Frame Rate Normalization
- **Target FPS**: 30 fps standard for consistent processing
- **Temporal Interpolation**: Smart frame interpolation for upsampling
- **Frame Dropping**: Intelligent frame selection for downsampling
- **Motion Preservation**: Maintains smooth motion during rate conversion

### Enhancement Parameters
- **Contrast Enhancement**: Optimized for surgical lighting conditions
- **Noise Reduction**: Balanced to preserve detail while reducing noise
- **Color Correction**: Maintains natural tissue colors
- **Sharpening**: Enhances instrument and anatomical detail visibility

## Performance Optimization

### Hardware Acceleration
- **GPU Processing**: Utilizes CUDA/OpenCL when available
- **Multi-threading**: Parallel processing for batch operations
- **Memory Management**: Efficient memory usage for large videos
- **Disk I/O**: Optimized reading/writing for fast processing

### Batch Processing
```python
# Process multiple videos efficiently
results = []
for video_path in video_list:
    result = processor.process(video_path, output_dir)
    results.append(result)
```

## Error Handling

The preprocessing module provides comprehensive error handling:
- **Input validation**: Checks file existence and format compatibility
- **Processing errors**: Captures FFmpeg errors and provides detailed messages
- **Recovery mechanisms**: Fallback options for failed operations
- **Progress tracking**: Real-time processing status updates

## Integration

### With Quality Control
```python
# Preprocessing followed by quality assessment
preprocessed = preprocessor.process(input_video, temp_output)
if preprocessed.status == ProcessingStatus.COMPLETED:
    quality_result = quality_checker.process(temp_output, final_output)
```

### With Compression
```python
# Preprocessing followed by compression
enhanced = enhancer.process(input_video, temp_enhanced)
compressed = compressor.process(temp_enhanced, final_output)
```

## Monitoring and Metrics

Each processor provides detailed metrics:
- **Processing time**: Execution duration for performance monitoring
- **Quality metrics**: Before/after quality measurements
- **File size changes**: Compression ratios and size variations
- **Parameter tracking**: Applied enhancement parameters
- **Error statistics**: Failure rates and common issues
