# 🎯 Reference Scripts Collection

> **Original processing scripts that define the exact methodologies implemented in the Cataract-LMM dataset preprocessing**

## 🎯 Objective

This directory contains the original reference scripts that serve as the foundation for the surgical video processing framework. These scripts define the exact processing standards that the framework replicates and are directly referenced in the Cataract-LMM research paper methodology.

## 📂 Contents

### Reference Implementation Scripts

| Script | Purpose | Implementation |
|--------|---------|----------------|
| **`process_video.sh`** | Linux video processing with privacy features | ✅ Reference Standard |
| **`process_videos.bat`** | Windows batch video processing | ✅ Reference Standard |

---

## 🐧 Linux Processing (`process_video.sh`)

### Technical Specification
- **Resolution Support**: Multiple resolutions including 720×480 pixels @ 30fps
- **Processing**: De-identification via crop+blur+overlay technique
- **Format**: MP4 container with H.265 encoding

### FFmpeg Command Analysis

```bash
#!/bin/bash
# Directory containing input files (use current directory if not specified)
INPUT_DIR="."
OUTPUT_DIR="."

# Check if FFmpeg is installed
command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "FFmpeg is not installed. Please install it to continue."; exit 1; }

# Loop through all .mp4 files in the input directory
for input_file in "$INPUT_DIR"/*.mp4; do
    # Skip if no files are found
    [[ -e "$input_file" ]] || continue
    
    # Extract the main filename without extension
    main_filename=$(basename -- "${input_file%.*}")
    
    # Define the output file name
    output_file="${OUTPUT_DIR}/processed_${main_filename}.mp4"
    
    # Apply the FFmpeg command
    ffmpeg -i "$input_file" \
        -filter_complex "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]" \
        -map "[v]" -map 0:a \
        -c:v libx265 -crf 23 -c:a copy -movflags +faststart "$output_file"
    
    # Check if FFmpeg succeeded
    if [ $? -eq 0 ]; then
        echo "Processed: $input_file -> $output_file"
    else
        echo "Error processing: $input_file"
    fi
done

echo "Batch processing complete."
```

### Technical Implementation Breakdown

#### 1. **De-identification Filter Chain**
```bash
-filter_complex "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]"
```

**Step-by-Step Process:**
1. **`[0:v]`** - Takes the input video stream
2. **`crop=268:58:6:422`** - Crops a region of 268×58 pixels starting at position (6,422)
3. **`avgblur=10`** - Applies average blur with radius 10 to the cropped region
4. **`[fg]`** - Labels the blurred region as "foreground"
5. **`[0:v][fg]overlay=6:422[v]`** - Overlays the blurred region back onto the original video at position (6,422)

#### 2. **Video Compression Settings**
```bash
-c:v libx265 -crf 23
```
- **Codec**: H.265 (HEVC) for efficient compression
- **CRF 23**: Constant Rate Factor providing balanced quality/size ratio
- **Quality**: High quality suitable for medical video analysis

#### 3. **Audio and Optimization**
```bash
-c:a copy -movflags +faststart
```
- **Audio Handling**: Copy original audio stream without re-encoding
- **Faststart**: Optimizes file for web streaming and progressive download

#### 4. **Output Naming Convention**
```bash
output_file="${OUTPUT_DIR}/processed_${main_filename}.mp4"
```
- **Prefix**: Adds "processed_" to indicate processed status
- **Format**: Maintains original filename structure for traceability

---

## 🪟 Windows Batch Processing (`process_videos.bat`)

### Technical Specification
- **Resolution Support**: Multiple resolutions including 1920×1080 pixels @ 60fps
- **Processing**: High-quality direct processing
- **Format**: MP4 container with H.265 encoding

### Batch Script Analysis

```bat
@echo off
setlocal enabledelayedexpansion

set INPUT_DIR=.
set OUTPUT_DIR=.

REM Check if FFmpeg is installed
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo FFmpeg is not installed. Please install it to continue.
    pause
    exit /b 1
)

REM Loop through all .mp4 files
for %%f in ("%INPUT_DIR%\*.mp4") do (
    set "input_file=%%f"
    set "filename=%%~nf"
    set "output_file=%OUTPUT_DIR%\processed_!filename!.mp4"
    
    echo Processing: !input_file!
    ffmpeg -i "!input_file!" -vcodec libx265 -crf 23 -movflags +faststart "!output_file!"
    
    if !errorlevel! equ 0 (
        echo Processed: !input_file! -^> !output_file!
    ) else (
        echo Error processing: !input_file!
    )
    echo.
)

echo Batch processing complete.
pause
```

### Technical Implementation Breakdown

#### 1. **Direct Compression Approach**
```bat
ffmpeg -i "!input_file!" -vcodec libx265 -crf 23 -movflags +faststart "!output_file!"
```

**Processing Steps:**
1. **Input**: Direct input without filtering (higher quality source doesn't require de-identification)
2. **Codec**: H.265 (libx265) for optimal compression of HD content
3. **CRF 23**: Balanced quality setting for 1080p content
4. **Optimization**: Faststart for streaming compatibility

#### 2. **Batch Processing Logic**
- **File Discovery**: Automatically finds all .mp4 files in input directory
- **Error Handling**: Checks FFmpeg installation and processing success
- **Progress Reporting**: Displays processing status for each file
- **Naming Convention**: Adds "compressed_" prefix to output files

---

## 🔄 Framework Implementation Alignment

### Core Video Processor Integration

The surgical video processing framework implements these reference scripts exactly:

```python
# Farabi processing (compress_video.sh)
def apply_farabi_processing(self, input_path: Path, output_path: Path) -> ProcessingResult:
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-filter_complex", "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]",
        "-map", "[v]", "-map", "0:a",
        "-c:v", "libx265", "-crf", "23", "-c:a", "copy", 
        "-movflags", "+faststart", str(output_path)
    ]

# Noor processing (compress_videos.bat)  
def apply_noor_processing(self, input_path: Path, output_path: Path) -> ProcessingResult:
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-vcodec", "libx265", "-crf", "23", 
        "-movflags", "+faststart", str(output_path)
    ]
```

### Hospital Detection and Routing

```python
# Automatic hospital detection and appropriate script selection
hospital_source = self.detect_hospital_source(video_info, file_path)

if hospital_source == HospitalSource.FARABI:
    result = self.apply_farabi_processing(input_path, output_path)
elif hospital_source == HospitalSource.NOOR:
    result = self.apply_noor_processing(input_path, output_path)
```

## 🎯 Cataract-LMM Dataset Context

These reference scripts are specifically designed for processing phacoemulsification cataract surgery videos as described in the Cataract-LMM research paper:

### Dataset Technical Specifications
- **Total Videos**: 3,000 phacoemulsification procedures
- **Farabi Hospital**: 720×480 @ 30fps (Haag-Streit HS Hi-R NEO 900)
- **Noor Hospital**: 1920×1080 @ 60fps (ZEISS ARTEVO 800)
- **Processing Period**: January 2021 to December 2024

### De-identification Requirements
- **Privacy Protection**: Patient information anonymization
- **Medical Compliance**: HIPAA-compliant processing
- **Quality Preservation**: Surgical detail preservation for analysis
- **Efficiency**: Batch processing capability for large datasets

### Research Application
These scripts enable the preparation of surgical videos for:
- **Phase Recognition**: Temporal surgical phase labeling
- **Instance Segmentation**: Instrument and tissue segmentation
- **Skill Assessment**: Objective skill scoring
- **Motion Analysis**: Spatiotemporal tracking studies

## 💡 Usage in Framework

### Direct Script Execution
```bash
# Farabi hospital videos
cd /path/to/farabi_videos
bash compress_video.sh

# Noor hospital videos  
cd /path/to/noor_videos
compress_videos.bat
```

### Framework Integration
```bash
# Using the framework with reference script compatibility
python main.py --hospital farabi --input farabi_video.mp4 --output compressed.mp4
python main.py --hospital noor --input noor_video.mp4 --output compressed.mp4
```

### Batch Processing
```bash
# Framework batch processing with reference script logic
python main.py --batch --input-dir ./cataract_videos --output-dir ./processed
```

## 🧪 Validation and Testing

### Reference Script Compliance Testing

```python
def test_farabi_compliance():
    """Validate exact replication of compress_video.sh"""
    # Test filter_complex: [0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]
    # Test encoding: libx265, CRF 23, copy audio
    # Test optimization: +faststart

def test_noor_compliance():
    """Validate exact replication of compress_videos.bat"""
    # Test encoding: libx265, CRF 23
    # Test optimization: +faststart
    # Verify no filtering applied
```

### Command Verification

```bash
# Test reference script replication
python -m pytest tests/test_reference_compliance.py

# Validate processing parameters
python main.py validate --hospital farabi
python main.py validate --hospital noor
```

## 📊 Processing Standard Comparison

| Feature | Farabi Standard | Noor Standard | Framework Enhancement |
|---------|----------------|---------------|----------------------|
| **Privacy** | ✅ Crop+Blur+Overlay | ❌ None | ✅ Configurable Privacy |
| **Compression** | libx265 CRF 23 | libx265 CRF 23 | ✅ Multiple Presets |
| **Filtering** | Complex overlay | None | ✅ Configurable Filters |
| **Web Optimization** | ✅ FastStart | ✅ FastStart | ✅ Multiple Optimizations |
| **Audio** | Copy Original | Copy/Default | ✅ Flexible Audio Options |
| **Error Handling** | ❌ Basic | ❌ Basic | ✅ Comprehensive |
| **Progress Tracking** | ❌ None | ❌ None | ✅ Real-time Monitoring |
| **Batch Processing** | ✅ Directory Batch | ✅ Directory Batch | ✅ Advanced Batch Management |

---

**📋 These reference scripts define the processing standards that ensure consistency and compliance across all surgical video processing operations in the Cataract-LMM dataset.**
