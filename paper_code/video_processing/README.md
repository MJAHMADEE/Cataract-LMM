# 🎬 Video Processing

## Scripts Overview

This directory contains the core scripts and notebooks used for video processing and compression in the Cataract-LMM dataset preparation.

### **📁 Files**

| File | Type | Description |
|------|------|-------------|
| `compress_video.ipynb` | Notebook | Video compression and optimization pipeline |
| `process_video.sh` | Shell Script | Linux/macOS batch video processing |
| `process_videos.bat` | Batch Script | Windows batch video processing |

### **🎯 Video Processing Pipeline**

**Compression Workflow:**
- **Input**: Raw surgical video files (various formats)
- **Processing**: Standardization, compression, quality optimization
- **Output**: Standardized video files suitable for AI training

**Key Operations:**
- **Format standardization**: Convert to consistent video format (MP4/H.264)
- **Resolution optimization**: Standardize resolution while preserving quality
- **Compression**: Reduce file size while maintaining clinical quality
- **Batch processing**: Handle large numbers of video files efficiently

### **⚙️ Technical Specifications**

**Video Parameters:**
- **Format**: MP4 with H.264 codec
- **Resolution**: Optimized for AI training (typically 640x480 or higher)
- **Frame Rate**: Consistent across dataset
- **Quality**: Balance between file size and visual fidelity

**Processing Features:**
- **Batch processing**: Handle hundreds of videos automatically
- **Quality control**: Automated checks for processing errors
- **Progress monitoring**: Track processing status and completion
- **Error handling**: Robust processing with failure recovery

### **🚀 Usage Instructions**

#### **Jupyter Notebook (compress_video.ipynb)**
```python
# Basic usage example
video_compressor = VideoCompressor(
    input_dir="path/to/raw/videos",
    output_dir="path/to/compressed/videos",
    target_quality="high",
    batch_size=10
)

video_compressor.process_batch()
```

#### **Shell Script (Linux/macOS)**
```bash
chmod +x process_video.sh
./process_video.sh input_directory output_directory
```

#### **Batch Script (Windows)**
```batch
process_videos.bat "C:\input\videos" "C:\output\videos"
```

### **📊 Processing Metrics**

**Quality Metrics:**
- **Compression ratio**: Original size vs. compressed size
- **PSNR**: Peak Signal-to-Noise Ratio for quality assessment
- **SSIM**: Structural Similarity Index for perceptual quality
- **Processing time**: Throughput and efficiency metrics

**Batch Processing Results:**
- **Success rate**: Percentage of successfully processed videos
- **Error analysis**: Common failure modes and resolutions
- **Storage savings**: Overall reduction in dataset size
- **Quality preservation**: Maintenance of clinical utility

### **🔧 Dependencies**

**Required Tools:**
```bash
# FFmpeg for video processing
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS

# Python dependencies
pip install opencv-python moviepy tqdm pandas numpy
```

**System Requirements:**
- **CPU**: Multi-core processor for efficient batch processing
- **RAM**: 8GB+ recommended for large video files
- **Storage**: Sufficient space for input + output videos
- **OS**: Cross-platform compatibility (Linux, macOS, Windows)

### **⚡ Performance Optimization**

**Processing Speed:**
- **Parallel processing**: Multi-threaded video encoding
- **GPU acceleration**: Hardware-accelerated encoding when available
- **Batch optimization**: Efficient handling of multiple files
- **Memory management**: Optimized for large video datasets

**Quality vs. Size:**
- **Adaptive bitrate**: Quality-based compression settings
- **Profile optimization**: H.264 profiles for different use cases
- **Lossless options**: Preserve maximum quality when needed
- **Custom presets**: Tailored settings for surgical video content

### **💡 Best Practices**

**Video Processing:**
1. **Backup original videos** before processing
2. **Test settings** on small subset before full batch
3. **Monitor quality** throughout processing pipeline
4. **Validate outputs** for clinical usability
5. **Document processing parameters** for reproducibility

**Clinical Considerations:**
- **Maintain diagnostic quality** for medical applications
- **Preserve temporal resolution** for motion analysis
- **Ensure consistent processing** across all videos
- **Validate with medical experts** for quality acceptance
