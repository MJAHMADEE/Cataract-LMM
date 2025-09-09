# Metadata Management

This module provides comprehensive metadata extraction, management, and anonymization capabilities for surgical videos.

## Overview

The metadata module handles:
- **Video metadata extraction** using OpenCV and FFmpeg
- **Technical specifications** (resolution, frame rate, codec, etc.)
- **Quality metrics** integration from quality control module
- **Processing history** tracking throughout the pipeline
- **Privacy-compliant anonymization** for research and archival
- **Batch reporting** for dataset analysis

## Components

### VideoMetadata Class
A comprehensive data structure that stores all video-related information:

```python
from surgical_video_processing.metadata import VideoMetadata

metadata = VideoMetadata(
    file_path="surgery_video.mp4",
    width=1920, height=1080, fps=60.0,
    hospital_source="noor",
    equipment_model="ZEISS ARTEVO 800"
)
```

**Key Fields:**
- **File Information**: path, size, hash, timestamps
- **Technical Specs**: resolution, fps, codec, bitrate
- **Hospital Data**: source, equipment, anonymized identifiers
- **Quality Metrics**: focus, glare, exposure scores
- **Processing History**: complete audit trail

### MetadataExtractor Class
Extracts metadata from video files using multiple methods:

```python
from surgical_video_processing.metadata import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract_metadata("video.mp4")

print(f"Resolution: {metadata.width}x{metadata.height}")
print(f"Hospital: {metadata.hospital_source}")
print(f"Equipment: {metadata.equipment_model}")
```

**Features:**
- **Multi-source extraction**: OpenCV + FFmpeg for comprehensive data
- **Automatic hospital detection**: Based on resolution patterns
- **Equipment identification**: Matches hospital-specific setups
- **Error handling**: Graceful fallbacks for corrupted files

### MetadataManager Class
Centralized management of metadata throughout processing:

```python
from surgical_video_processing.metadata import MetadataManager

manager = MetadataManager("./metadata")
metadata = manager.process_video_metadata("video.mp4")

# Update processing history
manager.update_processing_history(
    metadata, "quality_control", 
    {"threshold": 70.0}, 
    {"passed": True, "score": 85.2}
)
```

**Capabilities:**
- **Automated processing**: Extract and save metadata
- **History tracking**: Record all processing steps
- **Batch reporting**: Generate dataset summaries
- **Integration ready**: Works with all pipeline modules

### MetadataAnonymizer Class
HIPAA-compliant anonymization for research use:

```python
from surgical_video_processing.metadata import MetadataAnonymizer

anonymizer = MetadataAnonymizer()
anonymized_metadata = anonymizer.anonymize_metadata(original_metadata)

# Original surgeon ID is hashed, dates are generalized
print(anonymized_metadata.surgeon_id)  # "a1b2c3d4e5f6g7h8"
```

**Privacy Features:**
- **Identifier hashing**: Secure SHA-256 with salt
- **Date generalization**: Preserve temporal patterns
- **Path anonymization**: Remove identifying file paths
- **Configurable salt**: Institution-specific anonymization

## Hospital Detection

The system automatically detects hospital source and equipment based on video characteristics:

### Farabi Hospital Detection
- **Resolution**: 720×480 pixels
- **Frame Rate**: ~30 fps
- **Equipment**: Haag-Streit HS Hi-R NEO 900

```python
# Automatically detected for 720x480 videos
if metadata.width == 720 and metadata.height == 480:
    assert metadata.hospital_source == "farabi"
    assert "Haag-Streit" in metadata.equipment_model
```

### Noor Hospital Detection
- **Resolution**: 1920×1080 pixels
- **Frame Rate**: ~60 fps  
- **Equipment**: ZEISS ARTEVO 800

```python
# Automatically detected for 1080p videos
if metadata.width == 1920 and metadata.height == 1080:
    assert metadata.hospital_source == "noor"
    assert "ZEISS" in metadata.equipment_model
```

## Usage Examples

### Basic Metadata Extraction
```python
from surgical_video_processing.metadata import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract_metadata("cataract_surgery.mp4")

print(f"Video: {metadata.file_name}")
print(f"Duration: {metadata.duration_seconds:.1f} seconds")
print(f"Quality: {metadata.quality_score or 'Not assessed'}")
print(f"Hospital: {metadata.hospital_source}")
```

### Batch Processing with Metadata
```python
from surgical_video_processing.metadata import MetadataManager
from pathlib import Path

manager = MetadataManager("./video_metadata")
video_files = Path("./videos").glob("*.mp4")

metadata_list = []
for video_file in video_files:
    metadata = manager.process_video_metadata(str(video_file))
    metadata_list.append(metadata)

# Generate batch report
manager.generate_batch_report(metadata_list, "batch_report.json")
```

### Privacy-Compliant Anonymization
```python
from surgical_video_processing.metadata import MetadataAnonymizer

anonymizer = MetadataAnonymizer(salt="research_project_2024")

# Anonymize for research dataset
research_metadata = []
for original_metadata in all_metadata:
    anonymized = anonymizer.anonymize_metadata(original_metadata)
    research_metadata.append(anonymized)
    
    # Save anonymized version
    anonymized.save_to_file(f"research_metadata/{anonymized.file_name}.json")
```

### Integration with Quality Control
```python
from surgical_video_processing.metadata import MetadataManager
from surgical_video_processing.quality_control import QualityControlPipeline

manager = MetadataManager()
quality_pipeline = QualityControlPipeline(config)

# Extract metadata and run quality control
metadata = manager.process_video_metadata("video.mp4")
quality_result = quality_pipeline.analyze_video("video.mp4")

# Update metadata with quality scores
metadata.quality_score = quality_result.overall_score
metadata.focus_score = quality_result.focus_score
metadata.glare_percentage = quality_result.glare_percentage

# Save updated metadata
metadata.save_to_file("video_metadata.json")
```

## Metadata Schema

### Complete VideoMetadata Fields
```yaml
# File Information
file_path: str                    # Original file path
file_name: str                    # File name
file_size_bytes: int              # File size in bytes
file_hash: str                    # SHA-256 hash
creation_date: datetime           # File creation timestamp
modification_date: datetime       # Last modification timestamp

# Technical Specifications  
width: int                        # Video width in pixels
height: int                       # Video height in pixels
fps: float                        # Frames per second
duration_seconds: float           # Video duration
frame_count: int                  # Total number of frames
codec: str                        # Video codec (e.g., h264)
bitrate_kbps: int                 # Bitrate in kilobits/second
color_space: str                  # Color space (e.g., yuv420p)
pixel_format: str                 # Pixel format

# Hospital and Equipment
hospital_source: str              # "farabi", "noor", or "unknown"
equipment_model: str              # Specific equipment model
surgeon_id: str                   # Anonymized surgeon identifier
procedure_date: datetime          # Anonymized procedure date
case_id: str                      # Anonymized case identifier

# Quality Metrics
quality_score: float              # Overall quality score (0-100)
focus_score: float                # Focus quality score
glare_percentage: float           # Percentage of glare in video
exposure_score: float             # Exposure quality score
motion_score: float               # Motion stability score
completeness_score: float        # Procedure completeness score

# Processing History
processing_history: list          # List of processing steps
original_metadata: dict           # Raw metadata from FFmpeg
```

## Output Files

### Individual Metadata Files
Each processed video generates a JSON metadata file:
```json
{
  "file_name": "cataract_surgery_001.mp4",
  "width": 1920,
  "height": 1080,
  "fps": 60.0,
  "hospital_source": "noor",
  "equipment_model": "ZEISS ARTEVO 800",
  "quality_score": 87.5,
  "processing_history": [
    {
      "timestamp": "2024-08-30T10:30:00",
      "process_name": "quality_control",
      "result": {"passed": true, "score": 87.5}
    }
  ]
}
```

### Batch Reports
Aggregate statistics for entire datasets:
```json
{
  "generated_at": "2024-08-30T15:45:00",
  "total_videos": 3000,
  "hospitals": {
    "farabi": 1500,
    "noor": 1500
  },
  "quality_summary": {
    "mean_quality": 78.3,
    "videos_passed": 2847
  },
  "technical_summary": {
    "resolutions": {
      "720x480": 1500,
      "1920x1080": 1500
    }
  }
}
```

## Error Handling

The metadata module includes comprehensive error handling:

```python
try:
    metadata = extractor.extract_metadata("corrupted_video.mp4")
except FileNotFoundError:
    print("Video file not found")
except ValueError as e:
    print(f"Unsupported format: {e}")
except Exception as e:
    print(f"Metadata extraction failed: {e}")
```

**Common Issues:**
- **Corrupted files**: Graceful fallback to partial metadata
- **Unsupported formats**: Clear error messages
- **Missing FFmpeg**: Automatic detection and warnings
- **Permission errors**: Helpful error reporting

## Performance Considerations

- **Efficient extraction**: Optimized FFmpeg probe commands
- **Minimal memory usage**: Streaming file processing
- **Batch optimization**: Parallel metadata extraction
- **Caching support**: Avoid re-processing unchanged files

## Integration Points

The metadata module integrates with all other framework components:

- **Quality Control**: Stores quality assessment results
- **De-identification**: Tracks anonymization history  
- **Compression**: Records compression parameters
- **Pipelines**: Maintains complete processing audit trail
- **Configuration**: Respects hospital-specific settings
