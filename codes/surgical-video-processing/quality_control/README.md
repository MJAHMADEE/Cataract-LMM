# Quality Control Module

## Overview

The quality control module provides comprehensive automated quality assessment for phacoemulsification cataract surgery videos. It implements technical quality screening to exclude recordings based on pre-defined criteria such as incomplete procedures, poor focus, excessive glare, and other quality issues that could compromise surgical analysis.

## Key Components

### QualityAssessment
Comprehensive video quality analysis framework:
- **Multi-metric evaluation**: Combines multiple quality indicators
- **Standardized scoring**: 0-100 scale for all quality metrics
- **Threshold-based filtering**: Configurable quality thresholds
- **Detailed reporting**: Comprehensive quality analysis reports

### FocusQualityChecker
Analyzes video focus and sharpness using multiple algorithms:
- **Laplacian Variance**: Measures edge sharpness
- **Sobel Gradient**: Evaluates gradient magnitude
- **Brenner Gradient**: Modified Brenner focus measure
- **ROI Analysis**: Focus assessment in surgical field regions
- **Temporal Consistency**: Focus stability across frames

### GlareDetector
Detects excessive glare and overexposure:
- **Brightness Analysis**: Pixel-level brightness evaluation
- **Region Detection**: Identifies overexposed areas
- **Surgical Impact**: Assesses impact on surgical field visibility
- **Temporal Tracking**: Glare patterns over time
- **Adaptive Thresholds**: Hospital-specific glare thresholds

### ExposureAnalyzer
Evaluates lighting and exposure quality:
- **Histogram Analysis**: Brightness distribution evaluation
- **Dynamic Range**: Contrast and detail preservation
- **Lighting Uniformity**: Even illumination assessment
- **Shadow/Highlight**: Clipping detection and analysis

### MotionAnalyzer
Analyzes camera stability and motion artifacts:
- **Optical Flow**: Motion vector analysis
- **Stability Metrics**: Camera shake and movement detection
- **Surgical Motion**: Distinguishes surgical from camera motion
- **Stabilization Assessment**: Need for post-processing stabilization

### CompletenessChecker
Verifies procedure completeness:
- **Phase Detection**: Identifies surgical phases
- **Duration Analysis**: Procedure length validation
- **Interruption Detection**: Identifies recording gaps
- **Content Verification**: Ensures complete procedure capture

## Configuration

### QualityControlConfig
Comprehensive configuration for all quality assessments:

```python
@dataclass
class QualityControlConfig:
    # Quality thresholds
    min_overall_score: float = 60.0
    min_focus_score: float = 50.0
    max_glare_percentage: float = 15.0
    min_exposure_score: float = 40.0
    max_motion_threshold: float = 20.0
    min_completeness_score: float = 70.0
    
    # Analysis settings
    sample_frame_count: int = 50
    sample_interval: float = 2.0
    surgical_field_roi: Optional[Tuple[int, int, int, int]] = None
    
    # Feature toggles
    enable_focus_check: bool = True
    enable_glare_check: bool = True
    enable_exposure_check: bool = True
    enable_motion_check: bool = True
    enable_completeness_check: bool = True
    
    # Output settings
    generate_report: bool = True
    save_analysis_frames: bool = False
```

## Quality Metrics

### QualityMetrics
Standardized quality metrics for comprehensive assessment:

```python
@dataclass
class QualityMetrics:
    overall_score: float = 0.0
    focus_score: float = 0.0
    glare_score: float = 0.0
    exposure_score: float = 0.0
    motion_score: float = 0.0
    completeness_score: float = 0.0
    surgical_field_visibility: float = 0.0
    instrument_clarity: float = 0.0
    tissue_contrast: float = 0.0
    lighting_uniformity: float = 0.0
```

## Usage Examples

### Basic Quality Assessment

```python
from surgical_video_processing.quality_control import QualityControlPipeline, QualityControlConfig
from surgical_video_processing.core import ProcessingConfig

# Create configurations
processing_config = ProcessingConfig()
quality_config = QualityControlConfig(
    min_overall_score=60.0,
    min_focus_score=50.0,
    max_glare_percentage=15.0
)

# Initialize quality control pipeline
quality_controller = QualityControlPipeline(processing_config, quality_config)

# Analyze video quality
result = quality_controller.process(
    input_path="surgery_video.mp4",
    output_path="quality_analysis/"
)

# Check results
print(f"Quality acceptable: {result.metrics['quality_acceptable']}")
print(f"Overall score: {result.metrics['quality_metrics']['overall_score']:.1f}")
print(f"Focus score: {result.metrics['quality_metrics']['focus_score']:.1f}")
print(f"Glare score: {result.metrics['quality_metrics']['glare_score']:.1f}")
```

### Focus Quality Assessment

```python
from surgical_video_processing.quality_control import FocusQualityChecker

# Configure focus analysis
focus_config = QualityControlConfig(
    min_focus_score=60.0,
    sample_frame_count=30,
    sample_interval=3.0,
    surgical_field_roi=(100, 100, 500, 400)  # Focus on surgical field
)

# Initialize focus checker
focus_checker = FocusQualityChecker(processing_config, focus_config)

# Analyze focus quality
result = focus_checker.process(
    input_path="surgery_video.mp4",
    output_path="focus_analysis/"
)

# Review focus analysis
focus_metrics = result.metrics
print(f"Average focus score: {focus_metrics['average_focus_score']:.1f}")
print(f"Focus variance: {focus_metrics['focus_variance']:.2f}")
print(f"Focus distribution: {focus_metrics['focus_distribution']}")
```

### Glare Detection

```python
from surgical_video_processing.quality_control import GlareDetector

# Configure glare detection
glare_config = QualityControlConfig(
    glare_threshold=240,  # Brightness threshold
    max_glare_percentage=10.0,
    save_analysis_frames=True
)

# Initialize glare detector
glare_detector = GlareDetector(processing_config, glare_config)

# Detect glare
result = glare_detector.process(
    input_path="surgery_video.mp4",
    output_path="glare_analysis/"
)

# Review glare analysis
glare_metrics = result.metrics
print(f"Average glare: {glare_metrics['average_glare_percentage']:.1f}%")
print(f"Frames with excessive glare: {glare_metrics['frames_with_excessive_glare']}")
print(f"Glare severity distribution: {glare_metrics['glare_severity_distribution']}")
```

### Custom ROI Analysis

```python
# Define surgical field region of interest
surgical_roi = (150, 100, 600, 450)  # x, y, width, height

# Configure for ROI-specific analysis
roi_config = QualityControlConfig(
    surgical_field_roi=surgical_roi,
    sample_frame_count=40,
    sample_interval=2.5
)

quality_controller = QualityControlPipeline(processing_config, roi_config)
result = quality_controller.process("surgery.mp4", "roi_analysis/")
```

### Batch Quality Assessment

```python
from pathlib import Path

def batch_quality_assessment(input_dir: str, output_dir: str, min_score: float = 60.0):
    """Assess quality of all videos in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Configure quality control
    quality_config = QualityControlConfig(min_overall_score=min_score)
    quality_controller = QualityControlPipeline(processing_config, quality_config)
    
    results = []
    passed_videos = []
    failed_videos = []
    
    for video_file in input_path.glob("*.mp4"):
        analysis_dir = output_path / video_file.stem
        
        result = quality_controller.process(
            input_path=video_file,
            output_path=analysis_dir
        )
        
        quality_data = {
            'video': str(video_file),
            'overall_score': result.metrics['quality_metrics']['overall_score'],
            'focus_score': result.metrics['quality_metrics']['focus_score'],
            'glare_score': result.metrics['quality_metrics']['glare_score'],
            'acceptable': result.metrics['quality_acceptable'],
            'processing_time': result.processing_time
        }
        
        results.append(quality_data)
        
        if result.metrics['quality_acceptable']:
            passed_videos.append(str(video_file))
        else:
            failed_videos.append(str(video_file))
    
    # Generate batch report
    with open(output_path / 'batch_quality_report.json', 'w') as f:
        import json
        json.dump({
            'summary': {
                'total_videos': len(results),
                'passed': len(passed_videos),
                'failed': len(failed_videos),
                'pass_rate': len(passed_videos) / len(results) * 100
            },
            'passed_videos': passed_videos,
            'failed_videos': failed_videos,
            'detailed_results': results
        }, f, indent=2)
    
    return results

# Process entire directory
results = batch_quality_assessment("input_videos/", "quality_reports/", min_score=65.0)
```

## Quality Control Pipeline

### Sequential Processing
The quality control pipeline processes videos through multiple stages:

1. **Frame Sampling**: Extracts representative frames for analysis
2. **Focus Analysis**: Evaluates sharpness and clarity
3. **Glare Detection**: Identifies overexposed regions
4. **Exposure Assessment**: Analyzes lighting quality
5. **Motion Analysis**: Evaluates camera stability
6. **Completeness Check**: Verifies procedure coverage
7. **Overall Scoring**: Combines individual metrics
8. **Report Generation**: Creates comprehensive analysis reports

### Decision Making
```python
def quality_decision_logic(metrics: QualityMetrics, config: QualityControlConfig) -> bool:
    """Determine if video passes quality control"""
    
    # Individual metric checks
    focus_pass = metrics.focus_score >= config.min_focus_score
    glare_pass = (100 - metrics.glare_score) <= config.max_glare_percentage
    exposure_pass = metrics.exposure_score >= config.min_exposure_score
    motion_pass = metrics.motion_score <= config.max_motion_threshold
    completeness_pass = metrics.completeness_score >= config.min_completeness_score
    
    # Overall score check
    overall_pass = metrics.overall_score >= config.min_overall_score
    
    # Combined decision (all checks must pass)
    return all([focus_pass, glare_pass, exposure_pass, motion_pass, completeness_pass, overall_pass])
```

## Reporting and Visualization

### Comprehensive Reports
The module generates detailed quality control reports including:

- **Quality Dashboard**: Visual overview of all metrics
- **Individual Analysis**: Detailed results for each quality aspect
- **Trend Analysis**: Quality patterns over time
- **Recommendation Engine**: Actionable improvement suggestions
- **Comparative Analysis**: Benchmarking against quality standards

### Visual Outputs
- **Quality Score Gauges**: Circular progress indicators
- **Metric Bar Charts**: Individual score comparisons
- **Time Series Plots**: Quality trends across frames
- **Heatmaps**: Regional quality analysis
- **Annotated Frames**: Visual indication of quality issues

### Report Formats
- **JSON**: Machine-readable detailed results
- **HTML**: Interactive web-based reports
- **PDF**: Publication-ready quality certificates
- **CSV**: Spreadsheet-compatible summaries

## Integration with Processing Pipeline

### Pre-processing Quality Gate
```python
# Quality check before processing
quality_result = quality_controller.process(input_video, temp_analysis)
if quality_result.metrics['quality_acceptable']:
    # Proceed with processing pipeline
    processed = processor.process(input_video, output_video)
else:
    # Reject video or apply quality improvement
    logger.warning(f"Video rejected: {quality_result.warnings}")
```

### Post-processing Validation
```python
# Quality validation after processing
original_quality = quality_controller.process(original_video, temp_analysis)
processed_quality = quality_controller.process(processed_video, temp_analysis)

# Ensure processing didn't degrade quality
quality_preserved = (
    processed_quality.metrics['quality_metrics']['overall_score'] >= 
    original_quality.metrics['quality_metrics']['overall_score'] - 5.0  # Allow 5% degradation
)
```

## Performance Optimization

### Efficient Sampling
- **Adaptive Sampling**: Adjusts frame sampling based on video length
- **ROI Processing**: Focuses analysis on surgical field regions
- **Multi-threading**: Parallel analysis of different quality aspects
- **Caching**: Reuses intermediate calculations when possible

### GPU Acceleration
```python
# Enable GPU acceleration for quality analysis
quality_config = QualityControlConfig(
    use_gpu_acceleration=True,
    gpu_device_id=0
)
```

### Memory Management
- **Streaming Analysis**: Processes videos without loading entirely into memory
- **Batch Processing**: Efficient handling of multiple videos
- **Resource Cleanup**: Proper memory and resource management

## Error Handling and Edge Cases

### Robust Analysis
- **Corrupted Frames**: Handles damaged or missing frames gracefully
- **Variable Quality**: Adapts to videos with varying quality sections
- **Short Videos**: Adjusts analysis for videos shorter than expected
- **Format Variations**: Handles different video formats and codecs

### Fallback Mechanisms
```python
try:
    # Primary analysis method
    result = advanced_quality_analysis(video)
except Exception as e:
    logger.warning(f"Advanced analysis failed: {e}")
    # Fallback to basic analysis
    result = basic_quality_analysis(video)
```
