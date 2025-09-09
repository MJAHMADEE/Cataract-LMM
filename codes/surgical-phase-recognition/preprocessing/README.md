# 🎬 Video Preprocessing - Cataract-LMM Protocol

This directory contains video preprocessing and data preparation tools following the **"Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis"** paper specifications.

## � Paper-Compliant Processing

All preprocessing components implement the exact specifications from the Cataract-LMM paper for reproducible benchmark results.

### 🎥 **Paper Specifications**
- **Input Resolution**: Variable (original surgical video quality)
- **Output Resolution**: 224×224 pixels (paper standard)
- **Frame Rate**: 25 FPS (paper's extraction rate)
- **Total Dataset**: 150 videos, 28.6 hours of surgical footage
- **Multi-Center**: Farabi Hospital (80%) + Noor Hospital (20%)

## 📁 Contents

### Core Preprocessing Files

- **`video_preprocessing.py`** - Cataract-LMM compliant video processing
- **`__init__.py`** - Preprocessing module interface and exports

## 🎬 Video Processing Capabilities

### VideoPreprocessor (Paper Protocol)
Surgical video processing following Cataract-LMM specifications:

```python
from preprocessing import VideoPreprocessor

# Paper-compliant configuration
preprocessor = VideoPreprocessor(
    input_dir="./cataract_raw_videos",
    output_dir="./cataract_processed",
    target_fps=25,              # Paper's frame rate
    target_resolution=(224, 224), # Paper's input size
    center_specific=True        # Separate Farabi/Noor processing
)

# Process with paper protocol
preprocessor.process_video(
    "farabi_surgery_001.mp4",
    hospital_center="farabi",   # Multi-center labeling
    phase_annotations="phase_labels.txt"
)
```

## 🔧 Cataract-LMM Processing Features

### Paper-Compliant Frame Extraction
Following the exact protocol from Section 3.1:

```python
# Extract frames following paper methodology
def extract_frames_cataract_lmm(video_path, output_dir):
    """
    Extract frames at 25 FPS following Cataract-LMM protocol.
    Paper: "frames are extracted at 25 FPS and resized to 224×224"
    """
    frame_extractor = FrameExtractor(
        fps=25,                    # Paper specification
        resolution=(224, 224),     # Paper input size
        quality='high',            # Maintain surgical detail
        format='jpg'               # Standard image format
    )
    
    return frame_extractor.extract(video_path, output_dir)
```

### Multi-Center Data Organization
Organizing data according to paper's multi-center protocol:

```python
# Paper's multi-center structure
def organize_multicenter_data(input_videos, phase_annotations):
    """
    Organize videos by hospital center following paper protocol.
    
    Paper Distribution:
    - Farabi Hospital: 80% (120 videos)
    - Noor Hospital: 20% (30 videos)
    """
    farabi_videos = []  # Training + in-domain testing
    noor_videos = []    # Out-of-domain testing only
    
    for video, annotation in zip(input_videos, phase_annotations):
        center = detect_hospital_center(video)
        if center == "farabi":
            farabi_videos.append((video, annotation))
        elif center == "noor":
            noor_videos.append((video, annotation))
    
    return farabi_videos, noor_videos
```

### Quality Control (Paper Standards)
Ensuring surgical video quality meets paper requirements:

```python
# Paper-compliant quality validation
def validate_surgical_video(video_path):
    """
    Validate video quality following Cataract-LMM standards.
    Paper requirements: clear surgical view, stable camera, adequate lighting.
    """
    metrics = {
        'resolution_adequate': check_min_resolution(video_path, min_res=480),
        'duration_sufficient': check_duration(video_path, min_duration=300),  # 5+ minutes
        'lighting_quality': assess_lighting_consistency(video_path),
        'camera_stability': measure_camera_shake(video_path),
        'surgical_visibility': detect_surgical_instruments(video_path)
    }
    
    return all(metrics.values())
```

### Video Quality Enhancement
- **Noise Reduction**: Gaussian and bilateral filtering
- **Contrast Enhancement**: Histogram equalization
- **Stabilization**: Motion compensation algorithms
- **Color Correction**: White balance and saturation adjustment

## 📊 Processing Parameters

### Standard Configuration
```python
processing_config = {
    'target_fps': 30,
    'target_resolution': (224, 224),
    'quality': 'high',
    'codec': 'h264',
    'pixel_format': 'yuv420p'
}
```

### Advanced Options
```python
advanced_config = {
    'denoise': True,
    'stabilize': False,
    'enhance_contrast': True,
    'preserve_aspect_ratio': True,
    'generate_thumbnails': True,
    'extract_metadata': True
}
```

## 🎯 Surgical Video Specific Features

### Clinical Video Optimization
- **Endoscopic Enhancement**: Specialized filters for surgical cameras
- **Light Adjustment**: Compensation for surgical lighting variations
- **Instrument Detection**: Optional surgical tool highlighting
- **Privacy Protection**: Automatic patient information removal

### Quality Assessment
- **Blur Detection**: Automatic quality scoring
- **Motion Analysis**: Camera movement quantification
- **Illumination Metrics**: Lighting quality assessment
- **Frame Consistency**: Temporal stability validation

## 📈 Performance Optimization

### Batch Processing
- **Parallel Processing**: Multi-threaded video conversion
- **Memory Management**: Efficient buffer handling
- **Progress Tracking**: Real-time processing status
- **Error Recovery**: Automatic retry mechanisms

### GPU Acceleration
- **CUDA Support**: GPU-accelerated video decoding
- **Hardware Encoding**: Leveraging video encoding hardware
- **Memory Optimization**: GPU memory management
- **Fallback Options**: CPU processing when GPU unavailable

## 🔍 Quality Control

### Validation Checks
```python
# Automatic video validation
validation_results = preprocessor.validate_video("input_video.mp4")

checks = {
    'format_supported': validation_results['is_valid_format'],
    'duration_adequate': validation_results['duration'] > 30,
    'resolution_sufficient': validation_results['resolution'][0] >= 480,
    'frame_rate_stable': validation_results['fps_variance'] < 0.1
}
```

### Error Handling
- **Corruption Detection**: Automatic damaged file identification
- **Format Compatibility**: Comprehensive codec support
- **Metadata Validation**: Frame count and duration verification
- **Recovery Procedures**: Automatic error correction attempts

## 💾 Output Management

### Directory Structure
```
processed_videos/
├── videos/              # Processed video files
│   ├── surgery_001.mp4
│   ├── surgery_002.mp4
│   └── ...
├── frames/              # Extracted frames (optional)
│   ├── surgery_001/
│   │   ├── frame_000001.jpg
│   │   ├── frame_000002.jpg
│   │   └── ...
├── metadata/            # Video metadata
│   ├── surgery_001.json
│   ├── surgery_002.json
│   └── ...
└── thumbnails/          # Video thumbnails
    ├── surgery_001.jpg
    ├── surgery_002.jpg
    └── ...
```

### Metadata Information
```json
{
  "original_file": "surgery_001.avi",
  "processed_file": "surgery_001.mp4",
  "duration": 1820.5,
  "fps": 30.0,
  "resolution": [224, 224],
  "frame_count": 54615,
  "file_size": "2.3 GB",
  "processing_time": "00:03:45",
  "quality_score": 0.89
}
```

## 🛠️ Paper-Compliant Usage Examples

### Basic Cataract-LMM Processing
```python
from preprocessing import VideoPreprocessor

# Initialize with paper specifications
preprocessor = VideoPreprocessor(
    input_dir="./cataract_raw_videos",
    output_dir="./cataract_lmm_processed", 
    target_fps=25,              # Paper's frame rate
    target_resolution=(224, 224), # Paper's input resolution
    phase_taxonomy="13_phase"    # Full Cataract-LMM taxonomy
)

# Process Farabi hospital video (training data)
result = preprocessor.process_video(
    "farabi_cataract_001.mp4",
    hospital_center="farabi",
    split_assignment="train",     # 80% for training
    phase_annotations="annotations_001.txt"
)

print(f"Processed for training: {result['success']}")
print(f"Frames extracted: {result['frame_count']}")
print(f"Phase labels: {result['phase_count']}")
```

### Multi-Center Batch Processing
```python
# Process all videos following paper protocol
def process_cataract_lmm_dataset(raw_video_dir, annotations_dir):
    """
    Process complete Cataract-LMM dataset following paper methodology.
    Expected: 150 videos, 28.6 hours total duration.
    """
    
    results = {
        'farabi_train': [],
        'farabi_test': [], 
        'noor_test': []
    }
    
    for video_file in os.listdir(raw_video_dir):
        # Determine hospital center and split
        center = extract_hospital_from_filename(video_file)
        split = determine_split_assignment(video_file, center)
        
        # Process with appropriate configuration
        if center == "farabi":
            config = farabi_processor_config
        elif center == "noor":
            config = noor_processor_config
            
        result = preprocessor.process_video(
            video_file,
            config=config,
            hospital_center=center,
            split_assignment=split
        )
        
        results[f"{center}_{split}"].append(result)
    
    return results

# Execute full dataset processing
dataset_results = process_cataract_lmm_dataset(
    "./raw_cataract_videos",
    "./phase_annotations"
)

# Validate against paper statistics
validate_dataset_statistics(dataset_results)
```

### Domain Adaptation Preparation
```python
# Prepare data for domain adaptation evaluation (Paper Section 4.5)
def prepare_domain_adaptation_splits():
    """
    Prepare train/test splits for domain adaptation following paper protocol.
    
    Paper Method:
    - Train: All Farabi training videos
    - In-domain test: Farabi testing videos  
    - Out-of-domain test: All Noor videos
    """
    
    splits = {
        'train': load_farabi_train_videos(),          # Source domain training
        'test_in_domain': load_farabi_test_videos(),   # Source domain testing
        'test_out_domain': load_noor_test_videos()     # Target domain testing
    }
    
    # Expected domain gap: ~22% F1-score drop (paper result)
    return splits
```

## 📊 Dataset Validation & Statistics

### Paper Statistics Verification
```python
# Validate processed dataset matches paper specifications
def validate_cataract_lmm_statistics(processed_data_dir):
    """
    Verify processed dataset matches Cataract-LMM paper statistics.
    
    Expected (Paper Section 3):
    - Total videos: 150
    - Total duration: 28.6 hours  
    - Farabi videos: 120 (80%)
    - Noor videos: 30 (20%)
    - Average video length: ~11.4 minutes
    """
    
    stats = calculate_dataset_statistics(processed_data_dir)
    
    paper_benchmarks = {
        'total_videos': 150,
        'total_duration_hours': 28.6,
        'farabi_percentage': 0.80,
        'noor_percentage': 0.20,
        'avg_duration_minutes': 11.4
    }
    
    # Validate against paper
    for metric, expected in paper_benchmarks.items():
        actual = stats[metric]
        if abs(actual - expected) > 0.05 * expected:  # 5% tolerance
            print(f"⚠️  {metric}: {actual} vs {expected} (paper)")
        else:
            print(f"✅ {metric}: {actual} matches paper")
    
    return stats
```

## � Integration Points

- **📊 Data Pipeline**: Feeds into [`../data/`](../data/) dataset classes
- **🔧 Transforms**: Compatible with [`../transform.py`](../transform.py) preprocessing
- **🧠 Models**: Provides input for all [`../models/`](../models/) architectures  
- **📓 Notebooks**: Used by research notebooks in [`../notebooks/`](../notebooks/)

## 📖 Research Compliance

### Paper Reproduction Checklist
- ✅ **Frame Rate**: 25 FPS extraction (Section 3.1)
- ✅ **Resolution**: 224×224 pixel normalization
- ✅ **Multi-Center**: Farabi/Noor separation protocol
- ✅ **Phase Taxonomy**: 13-phase surgical workflow
- ✅ **Quality Standards**: Surgical video validation criteria

---

> **📖 Paper Reference**: All preprocessing follows the exact specifications from "Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis" for reproducible research.
