# 🔒 De-identification Module

## 🎯 Overview

The de-identification module provides comprehensive privacy protection for phacoemulsification cataract surgery videos, ensuring compliance with medical data protection regulations. It removes or anonymizes all potentially identifying information from both video content and metadata.

## 🔧 Key Components

### MetadataStripper
Removes all metadata from video files including:
- **📊 EXIF data**: Camera settings, timestamps, device information
- **📅 Creation metadata**: File creation dates and modification times
- **📱 Device information**: Camera model, manufacturer details
- **🌍 Geographic data**: Location information if present
- **💬 User comments**: Any embedded text or annotations

### VisualDeidentifier
Removes visual identifiers from video content:
- **⏰ Timestamp overlays**: Date/time stamps burned into video
- **🏥 Institutional watermarks**: Logos and institutional branding
- **👤 Patient information**: Any visible patient identifiers
- **🎯 Region of interest blurring**: Selective area anonymization
- **📝 Text overlay removal**: Equipment readings and annotations

### TimestampRemover
Specialized removal of timestamp overlays:
- **🔍 Automatic detection**: AI-powered timestamp region identification
- **📋 Multiple formats**: Supports various timestamp formats
- **🎨 Inpainting**: Intelligent filling of removed timestamp areas
- **🛡️ Preservation**: Maintains surgical field integrity

### ComprehensiveDeidentifier
Orchestrates all de-identification processes:
- **🔄 Multi-stage processing**: Sequential application of all methods
- **💎 Quality preservation**: Maintains video quality during anonymization
- **📋 Audit trail**: Complete record of de-identification steps
- **🆔 Anonymized identifiers**: Generation of privacy-safe file identifiers

## ⚙️ Configuration

### DeidentificationConfig
Comprehensive configuration for all de-identification operations:

```python
@dataclass
class DeidentificationConfig:
    # Core de-identification settings
    remove_metadata: bool = True
    remove_timestamps: bool = True
    remove_watermarks: bool = True
    remove_audio: bool = True
    blur_patient_info: bool = True
    
    # Advanced settings
    roi_blur_regions: List[Tuple[int, int, int, int]] = None
    timestamp_regions: List[Tuple[int, int, int, int]] = None
    watermark_regions: List[Tuple[int, int, int, int]] = None
    blur_strength: int = 25
    replacement_color: Tuple[int, int, int] = (0, 0, 0)
    
    # Identifier handling
    hash_identifiers: bool = True
    anonymize_audio: bool = False
```

## Usage Examples

### Basic De-identification

```python
from surgical_video_processing.deidentification import ComprehensiveDeidentifier, DeidentificationConfig
from surgical_video_processing.core import ProcessingConfig

# Create configurations
processing_config = ProcessingConfig()
deident_config = DeidentificationConfig(
    remove_metadata=True,
    remove_timestamps=True,
    remove_watermarks=True,
    remove_audio=True
)

# Initialize de-identifier
deidentifier = ComprehensiveDeidentifier(processing_config, deident_config)

# Process video
result = deidentifier.process(
    input_path="surgery_with_identifiers.mp4",
    output_path="anonymous_surgery.mp4"
)

print(f"De-identification completed: {result.status}")
print(f"Processing steps: {result.metrics['processing_steps']}")
print(f"Anonymized ID: {result.metrics['anonymized_id']}")
```

### Metadata Stripping Only

```python
from surgical_video_processing.deidentification import MetadataStripper

# Initialize metadata stripper
stripper = MetadataStripper(processing_config, deident_config)

# Strip metadata while preserving video content
result = stripper.process(
    input_path="surgery_with_metadata.mp4",
    output_path="surgery_no_metadata.mp4"
)
```

### Custom Region De-identification

```python
# Define custom regions to blur or remove
custom_config = DeidentificationConfig(
    # Blur specific patient information areas
    roi_blur_regions=[
        (50, 50, 200, 100),    # Patient ID area
        (600, 400, 150, 50),   # Doctor notes area
    ],
    
    # Remove timestamp from specific locations
    timestamp_regions=[
        (0, 0, 300, 50),       # Top-left timestamp
        (720, 480, 200, 30),   # Bottom-right timestamp
    ],
    
    # Remove hospital watermarks
    watermark_regions=[
        (600, 0, 120, 80),     # Hospital logo
    ],
    
    blur_strength=30,
    replacement_color=(128, 128, 128)  # Gray replacement
)

deidentifier = ComprehensiveDeidentifier(processing_config, custom_config)
result = deidentifier.process("input.mp4", "output.mp4")
```

### Visual De-identification Only

```python
from surgical_video_processing.deidentification import VisualDeidentifier

# Configure for visual de-identification only
visual_config = DeidentificationConfig(
    remove_metadata=False,  # Keep metadata
    remove_timestamps=True,
    remove_watermarks=True,
    blur_patient_info=True,
    remove_audio=False      # Keep audio
)

visual_deidentifier = VisualDeidentifier(processing_config, visual_config)
result = visual_deidentifier.process("input.mp4", "visually_deidentified.mp4")
```

### Timestamp Detection and Removal

```python
from surgical_video_processing.deidentification import TimestampRemover

# Automatic timestamp detection and removal
timestamp_config = DeidentificationConfig(
    timestamp_regions=[],  # Empty list triggers auto-detection
    remove_timestamps=True
)

timestamp_remover = TimestampRemover(processing_config, timestamp_config)
result = timestamp_remover.process("timestamped_surgery.mp4", "clean_surgery.mp4")

# Check detection results
print(f"Timestamp regions removed: {result.metrics['timestamp_regions_removed']}")
print(f"Auto-detection used: {result.metrics['auto_detection_used']}")
```

### Batch De-identification

```python
import os
from pathlib import Path

def batch_deidentify(input_dir: str, output_dir: str):
    """De-identify all videos in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    deidentifier = ComprehensiveDeidentifier(processing_config, deident_config)
    
    results = []
    for video_file in input_path.glob("*.mp4"):
        output_file = output_path / video_file.name
        
        result = deidentifier.process(
            input_path=video_file,
            output_path=output_file
        )
        
        results.append({
            'input': str(video_file),
            'output': str(output_file),
            'status': result.status,
            'anonymized_id': result.metrics.get('anonymized_id'),
            'processing_time': result.processing_time
        })
    
    return results

# Process entire directory
results = batch_deidentify("raw_surgeries/", "deidentified_surgeries/")
```

## Privacy and Compliance

### HIPAA Compliance
The de-identification module addresses HIPAA requirements by:
- **Safe Harbor Method**: Removes all 18 types of identifiers listed in HIPAA
- **Expert Determination**: Provides statistical disclosure control
- **Audit Trails**: Maintains records of de-identification processes
- **Quality Assurance**: Validates removal of identifying information

### GDPR Compliance
For European data protection:
- **Pseudonymization**: Converts identifiers to pseudonyms
- **Data Minimization**: Removes unnecessary identifying data
- **Technical Measures**: Implements appropriate technical safeguards
- **Documentation**: Provides comprehensive processing records

### Supported Identifier Types
The module removes or anonymizes:
1. **Names and initials**
2. **Medical record numbers**
3. **Dates** (except year)
4. **Geographic identifiers**
5. **Telephone/fax numbers**
6. **Email addresses**
7. **Account numbers**
8. **Device identifiers**
9. **IP addresses**
10. **Biometric identifiers**
11. **Photographic images**
12. **Other unique identifiers**

## Technical Implementation

### Video Processing Pipeline
```
Input Video
    ↓
1. Visual De-identification
   - Timestamp removal
   - Watermark removal
   - Region blurring
    ↓
2. Audio Processing
   - Removal or anonymization
    ↓
3. Metadata Stripping
   - Complete metadata removal
    ↓
4. Identifier Generation
   - Anonymous file naming
    ↓
De-identified Video
```

### Quality Preservation
- **Minimal quality loss**: Optimized processing to preserve surgical detail
- **Selective processing**: Only processes areas requiring de-identification
- **High-quality inpainting**: Advanced algorithms for content removal
- **Format consistency**: Maintains video format and compatibility

### Performance Optimization
- **GPU acceleration**: Utilizes hardware acceleration when available
- **Parallel processing**: Concurrent processing of multiple videos
- **Memory efficiency**: Optimized memory usage for large files
- **Progress tracking**: Real-time processing status updates

## Validation and Quality Assurance

### Automated Validation
```python
from surgical_video_processing.deidentification import validate_deidentification

# Validate de-identification completeness
validation_result = validate_deidentification(
    original_video="original.mp4",
    deidentified_video="anonymous.mp4",
    config=deident_config
)

print(f"Validation passed: {validation_result.is_valid}")
print(f"Issues found: {validation_result.issues}")
print(f"Confidence score: {validation_result.confidence_score}")
```

### Manual Review Support
- **Side-by-side comparison**: Tools for manual verification
- **Highlighted changes**: Visual indication of modified regions
- **Metadata comparison**: Before/after metadata analysis
- **Quality metrics**: Quantitative assessment of changes

## Error Handling and Recovery

### Robust Processing
- **Input validation**: Comprehensive file and format checking
- **Graceful degradation**: Fallback options for processing failures
- **Partial processing**: Continue with successful steps if others fail
- **Recovery mechanisms**: Automatic retry with alternative methods

### Logging and Audit
```python
# Access detailed processing logs
print(f"Processing history: {result.metadata.processing_history}")
print(f"De-identification steps: {result.metrics['processing_steps']}")
print(f"Quality preservation: {result.metrics['quality_score']}")
```

## Integration Examples

### With Quality Control
```python
# De-identification followed by quality assessment
deidentified = deidentifier.process(input_video, temp_output)
if deidentified.status == ProcessingStatus.COMPLETED:
    quality_result = quality_checker.process(temp_output, final_output)
```

### With Compression
```python
# De-identification followed by compression
anonymous = deidentifier.process(input_video, temp_anonymous)
compressed = compressor.process(temp_anonymous, final_output)
```

## Security Considerations

### Data Security
- **Temporary file handling**: Secure creation and deletion of temporary files
- **Memory management**: Secure memory allocation and cleanup
- **Access controls**: Appropriate file permissions and access restrictions
- **Audit logging**: Complete audit trail of all operations

### Cryptographic Functions
- **Hash generation**: Secure hash functions for identifier generation
- **Random number generation**: Cryptographically secure randomness
- **Key management**: Secure handling of any cryptographic keys
