# 🔄 PThe pipelines module provides comprehensive workflow orchestration and batch processing management for the surgical video processing framework. It coordinates all processing steps, manages complex workflows, and provides the main programmatic interface for both single video and batch processing operations.

## 📂 Contents

### Primary Componentsne Orchestration

> **Workflow coordination and batch processing management for surgical video processing**

## 🎯 Objective

The pipelines module provides comprehensive workflow orchestration and batch processing management for the surgical video processing framework. It coordinates all processing steps, manages complex workflows, and provides the main programmatic interface for both single video and batch processing operations.

## � Contents

### Primary Components

| File | Purpose | Description |
|------|---------|-------------|
| **`__init__.py`** | 🎛️ Main Pipeline Interface | Primary interface classes and configuration |
| **`orchestrator.py`** | 🎼 Workflow Coordination | Complete workflow management and orchestration |

### Key Classes

#### **SurgicalVideoProcessor** (Main Interface)
The primary interface for all video processing operations:
- **Single Video Processing**: Process individual videos with full workflow
- **Batch Processing**: Handle multiple videos with parallel processing
- **Configuration Management**: Workflow-specific and custom configurations
- **Progress Tracking**: Real-time progress monitoring and callbacks

#### **PipelineOrchestrator** (Workflow Engine)
Manages the complete processing workflow:
- **Component Coordination**: Orchestrates quality control, de-identification, processing
- **Error Recovery**: Comprehensive error handling and recovery mechanisms
- **Performance Monitoring**: Real-time metrics and processing statistics
- **Workflow Customization**: Configurable processing steps and options

## 💡 Usage Examples

### Single Video Processing

```python
from pipelines import SurgicalVideoProcessor

# Initialize with standard configuration
processor = SurgicalVideoProcessor(config={'workflow_type': 'standard'})

# Process single video
result = processor.process_video(
    input_path="RV_0001_S1.mp4",
    output_path="processed_RV_0001_S1.mp4"
)

# Check results
if result['success']:
    print(f"Processing completed: {result['output_path']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Processing ratio: {result['processing_ratio']:.1f}x")
else:
    print(f"Processing failed: {result['error']}")
```

### Batch Processing

```python
from pipelines import SurgicalVideoProcessor
from pathlib import Path

# Initialize processor with advanced configuration
config = {
    'workflow_type': 'standard',
    'quality_preset': 'high',
    'enable_gpu': True,
    'batch_mode': True,
    'resume_processing': True
}

processor = SurgicalVideoProcessor(config)

# Define progress callback
def progress_callback(current, total, filename, status):
    percentage = (current / total) * 100
    print(f"[{percentage:5.1f}%] {status}: {filename}")

# Process batch with monitoring
results = processor.process_batch(
    input_directory="./cataract_videos",
    output_directory="./processed_videos",
    progress_callback=progress_callback
)

# Analyze batch results
print(f"Batch processing completed:")
print(f"  Total videos: {results['total_files']}")
print(f"  Successful: {results['successful_files']}")
print(f"  Failed: {results['failed_files']}")
print(f"  Total time: {results['total_processing_time']:.1f}s")
```

## 🔧 Configuration Options

### Standard Processing Configurations

#### Standard Workflow Configuration
```python
config = {
    'workflow_type': 'standard',
    'processing_method': 'standard_processing',
    'video_codec': 'libx265',
    'crf_value': 23,
    'audio_handling': 'copy',
    'optimization': 'faststart'
}
```

#### High Quality Configuration
```python
config = {
    'workflow_type': 'high_quality',
    'processing_method': 'quality_processing',
    'video_codec': 'libx265',
    'crf_value': 18,
    'optimization': 'faststart',
    'quality_preset': 'high',
    'resolution': [1920, 1080],
    'fps': 30
}
```

## 🔄 Workflow Architecture

### Complete Processing Pipeline

```
Input Video
    ↓
1. Input Validation
   - File existence and format check
   - Video metadata extraction
   - Hospital source detection
    ↓
2. Quality Control (Optional)
   - Focus analysis
   - Exposure assessment
   - Motion detection
   - Surgical field validation
    ↓
3. De-identification (Optional)
   - Metadata stripping
   - Visual anonymization
   - Audio processing
   - Timestamp removal
    ↓
4. Hospital-Specific Processing
   - Farabi: Crop+blur+overlay compression
   - Noor: High-quality direct compression
   - Generic: Configurable compression
    ↓
5. Output Generation
   - File optimization
   - Metadata preservation
   - Quality validation
    ↓
Processed Video + Metrics
```

## 🎯 Cataract-LMM Dataset Integration

### Dataset Naming Convention Support

The pipeline automatically handles Cataract-LMM naming conventions:

```python
# Input file examples
input_files = [
    "RV_0001_S1.mp4",           # Raw video from Farabi (S1)
    "PH_0001_0002_S2.mp4",      # Phase recognition from Noor (S2)
    "SE_0001_0003_S1.mp4",      # Instance segmentation from Farabi
    "TR_0001_S2_P03.mp4",       # Tracking video from Noor
    "SK_0001_S1_P03.mp4"        # Skill assessment from Farabi
]

# Automatic hospital detection and appropriate processing
for input_file in input_files:
    result = processor.process_video(input_file, f"compressed_{input_file}")
```

## 📊 Performance Metrics

| Hospital | Average Speed | Compression Ratio | Quality Score |
|----------|---------------|-------------------|---------------|
| **Farabi** | 4.0x realtime | 6-8x reduction | 8.5/10 |
| **Noor** | 1.2x realtime | 8-12x reduction | 9.2/10 |
- **Quality Control**: Integrated quality validation pipelines

### Primary Classes

#### SurgicalVideoProcessor
**The main interface for all video processing operations**

```python
from pipelines import SurgicalVideoProcessor

# Initialize with hospital configuration
processor = SurgicalVideoProcessor(config={'hospital': 'farabi'})

# Process single video
result = processor.process_video(input_path, output_path)

# Batch processing
results = processor.process_batch(input_dir, output_dir)
```

#### PipelineConfig
**Configuration management for processing pipelines**

```python
@dataclass
class PipelineConfig:
    hospital: str = "general"
    quality: str = "balanced"
    parallel_processing: bool = True
    max_workers: int = 4
    enable_quality_control: bool = True
    enable_deidentification: bool = True
    backup_originals: bool = False
```

### Pipeline Types

#### QualityControlPipeline
```python
class QualityControlPipeline(SurgicalVideoProcessor):
    """Pipeline focused on quality validation and control"""
    
    def process_video(self, input_path, output_path, **kwargs):
        # Enhanced quality validation
        # Multiple quality checkpoints
        # Automated quality scoring
```

#### AnalysisPreparationPipeline
```python
class AnalysisPreparationPipeline(SurgicalVideoProcessor):
    """Pipeline for preparing videos for analysis"""
    
    def process_video(self, input_path, output_path, **kwargs):
        # Analysis-optimized processing
        # Metadata preservation
        # Format standardization
```

## 🎼 Workflow Orchestration (`orchestrator.py`)

### Features
- **Advanced Workflow Management**: Complex processing workflow coordination
- **Performance Monitoring**: Real-time metrics collection and reporting
- **Resource Management**: Intelligent resource allocation and optimization
- **Parallel Processing**: Multi-threaded batch processing capabilities
- **Resume Functionality**: Interrupted processing recovery

### Core Classes

#### PipelineOrchestrator
**Advanced workflow coordination and management**

```python
from pipelines.orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator(
    max_workers=8,
    enable_monitoring=True,
    performance_tracking=True
)

# Process single video with full orchestration
result = orchestrator.process_single_video(
    input_path=input_path,
    output_path=output_path,
    hospital="farabi",
    quality_control=True,
    deidentification=True
)

# Advanced batch processing
results = orchestrator.process_batch(
    input_files=file_list,
    output_dir=output_dir,
    parallel=True,
    resume_on_failure=True
)
```

#### ProcessingMetrics
**Comprehensive performance metrics and monitoring**

```python
@dataclass
class ProcessingMetrics:
    total_videos: int = 0
    processed_videos: int = 0
    failed_videos: int = 0
    total_processing_time: float = 0.0
    average_processing_speed: float = 0.0
    total_size_reduction: float = 0.0
    average_quality_score: float = 0.0
    system_resource_usage: Dict[str, float] = field(default_factory=dict)
```

## 🚀 Usage Examples

### Basic Pipeline Usage

```python
from pipelines import SurgicalVideoProcessor

# Simple processing
processor = SurgicalVideoProcessor()
result = processor.process_video("input.mp4", "output.mp4")

# Hospital-specific processing
farabi_processor = SurgicalVideoProcessor(config={'hospital': 'farabi'})
result = farabi_processor.process_video("surgery.mp4", "compressed.mp4")
```

### Advanced Orchestration

```python
from pipelines.orchestrator import PipelineOrchestrator
from pathlib import Path

# Initialize with monitoring
orchestrator = PipelineOrchestrator(
    max_workers=4,
    enable_monitoring=True,
    performance_tracking=True
)

# Process with full workflow
result = orchestrator.process_single_video(
    input_path=Path("surgery.mp4"),
    output_path=Path("processed.mp4"),
    hospital="noor",
    quality_control=True,
    deidentification=True,
    backup_original=True
)

# Get processing metrics
metrics = orchestrator.get_processing_metrics()
print(f"Processing speed: {metrics.average_processing_speed:.2f}x")
print(f"Quality score: {metrics.average_quality_score:.1f}/10")
```

### Batch Processing

```python
from pipelines import SurgicalVideoProcessor
from pathlib import Path

# Batch processing with configuration
processor = SurgicalVideoProcessor(config={
    'hospital': 'farabi',
    'quality': 'high',
    'parallel_processing': True,
    'max_workers': 8
})

# Process entire directory
input_dir = Path("./raw_videos")
output_dir = Path("./processed_videos")

results = processor.process_batch(
    input_dir=input_dir,
    output_dir=output_dir,
    file_pattern="*.mp4",
    resume_on_failure=True
)

# Summary report
successful = sum(1 for r in results if r.status == "completed")
print(f"Successfully processed {successful}/{len(results)} videos")
```

## 🔧 Configuration Options

### Pipeline Configuration

```python
# Default configuration
{
    "hospital": "general",
    "quality": "balanced",
    "parallel_processing": True,
    "max_workers": 4,
    "enable_quality_control": True,
    "enable_deidentification": True,
    "backup_originals": False,
    "resume_on_failure": True
}

# High-performance configuration
{
    "hospital": "noor",
    "quality": "high",
    "parallel_processing": True,
    "max_workers": 16,
    "enable_quality_control": True,
    "enable_deidentification": True,
    "backup_originals": True,
    "resume_on_failure": True,
    "performance_monitoring": True
}
```

### Orchestrator Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_workers` | 4 | Maximum concurrent processing threads |
| `enable_monitoring` | True | Real-time performance monitoring |
| `performance_tracking` | True | Detailed performance metrics collection |
| `resume_on_failure` | True | Resume interrupted batch processing |
| `quality_control` | True | Enable quality validation pipeline |
| `deidentification` | True | Enable privacy protection pipeline |

## 📊 Performance Features

### Monitoring Capabilities
- **Real-time Metrics**: Live processing statistics
- **Resource Tracking**: CPU, memory, disk usage monitoring
- **Performance Analytics**: Speed, quality, efficiency metrics
- **Error Tracking**: Comprehensive error logging and analysis

### Optimization Features
- **Intelligent Scheduling**: Optimal task distribution
- **Resource Management**: Dynamic resource allocation
- **Load Balancing**: Automatic workload balancing
- **Memory Optimization**: Efficient memory usage patterns

## 🛡️ Error Handling

### Robust Recovery Mechanisms

```python
try:
    result = orchestrator.process_single_video(input_path, output_path)
    if result.status == "failed":
        # Automatic retry with alternative settings
        result = orchestrator.retry_processing(input_path, output_path)
        
except ProcessingTimeoutError:
    # Handle timeout with extended limits
    result = orchestrator.process_with_extended_timeout(input_path, output_path)
    
except InsufficientResourcesError:
    # Reduce processing requirements
    result = orchestrator.process_with_reduced_requirements(input_path, output_path)
```

### Error Recovery Strategies
- **Automatic Retry**: Failed processing automatic retry with different settings
- **Graceful Degradation**: Fallback to lower quality settings on resource constraints
- **Resume Capability**: Continue interrupted batch processing from last successful point
- **Error Isolation**: Prevent single failure from affecting entire batch

## 🧪 Testing

### Pipeline Testing

```bash
# Run pipeline tests
python -m pytest tests/test_pipelines.py

# Test orchestration functionality
python -m pytest tests/test_orchestrator.py

# Integration tests
python -m pytest tests/integration/test_pipeline_integration.py
```

### Performance Testing

```bash
# Performance benchmarks
python -m pytest tests/performance/test_pipeline_performance.py

# Load testing
python -m pytest tests/load/test_batch_processing.py
```

## 📚 API Reference

### Main Classes

```python
class SurgicalVideoProcessor:
    """Main pipeline interface for surgical video processing"""
    
    def __init__(self, config: Dict = None)
    def process_video(self, input_path, output_path, **kwargs) -> ProcessingResult
    def process_batch(self, input_dir, output_dir, **kwargs) -> List[ProcessingResult]

class PipelineOrchestrator:
    """Advanced workflow orchestration and management"""
    
    def __init__(self, max_workers: int = 4, **kwargs)
    def process_single_video(self, input_path, output_path, **kwargs) -> ProcessingResult
    def process_batch(self, input_files, output_dir, **kwargs) -> List[ProcessingResult]
    def get_processing_metrics(self) -> ProcessingMetrics
```

---

**🔄 Professional pipeline orchestration for surgical video processing workflows**
