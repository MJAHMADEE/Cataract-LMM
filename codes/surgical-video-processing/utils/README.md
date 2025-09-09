# 🛠️ Utility Infrastructure System

> **Enterprise-grade utility modules supporting Cataract-LMM surgical video processing operations**

## 🎯 Objective

This directory contains the foundational utility infrastructure that powers the surgical video processing framework. These modules provide essential services including configuration management, logging, performance monitoring, file operations, and helper functions optimized for medical video processing workflows.

## 📂 Utility Module Inventory

### Core Infrastructure Modules

| Module | Purpose | Functionality | Status |
|--------|---------|---------------|--------|
| **`config_manager.py`** | ⚙️ Configuration System | Workflow configs, YAML management | ✅ Production Ready |
| **`logging_config.py`** | 📝 Logging Infrastructure | Structured logging, audit trails | ✅ Production Ready |
| **`performance_monitor.py`** | 📊 Performance Analytics | System monitoring, metrics collection | ✅ Production Ready |
| **`file_utils.py`** | 📁 File Operations | Secure I/O, validation, backup management | ✅ Production Ready |
| **`helpers.py`** | 🔨 Core Utilities | Video metadata, system info, FFmpeg integration | ✅ Production Ready |

---

## ⚙️ Configuration Management System (`config_manager.py`)

### Primary Functions
- **Workflow Configuration Management**: Automatic loading of processing presets
- **YAML Processing**: Comprehensive YAML configuration file handling
- **Environment Integration**: Environment variable overrides and detection
- **Validation Framework**: Configuration consistency and compatibility validation
- **Dynamic Updates**: Runtime configuration modification and hot-reloading

### Core Configuration Manager

```python
class ConfigManager:
    """
    Configuration management for Cataract-LMM dataset processing
    
    Supports:
    - Standard processing configurations
    - Custom workflows with validation
    - Environment-specific settings
    """
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file"""
        
    def get_standard_config(self) -> Dict[str, Any]:
        """Get standard processing configuration"""
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
```

### Usage Examples

```python
from utils.config_manager import ConfigManager

# Initialize configuration manager
config_mgr = ConfigManager()

# Load standard configurations
standard_config = config_mgr.get_standard_config()

# Load custom configuration with validation
custom_config = config_mgr.load_config("custom_workflow.yaml")

# Environment-aware configuration
config = config_mgr.load_config("default.yaml", env_override=True)

# Configuration merging for complex setups
merged_config = config_mgr.merge_configs([
    "base.yaml",
    "workflow_override.yaml",
    "production_settings.yaml"
])
```

### Standard Configuration Example

```python
# Standard Processing Configuration
standard_config = {
    "workflow": {
        "name": "standard",
        "resolution": "1920x1080",
        "framerate": 30
    },
    "processing": {
        "privacy_protection": True,
        "video_codec": "libx265",
        "crf": 23,
        "audio_codec": "copy"
    }
}
```

---

## 📝 Advanced Logging Infrastructure (`logging_config.py`)

### Comprehensive Logging Features
- **Structured JSON Logging**: Machine-readable log formats for analysis
- **Multi-level Log Management**: DEBUG, INFO, WARNING, ERROR, CRITICAL levels
- **Medical Compliance Logging**: HIPAA-compliant audit trail generation
- **Performance Context Logging**: Automatic operation timing and resource tracking
- **File Rotation Management**: Automated log file rotation and archival

### Professional Logging Setup

```python
class SurgicalVideoLogger:
    """
    Professional logging system for surgical video processing
    
    Features:
    - HIPAA-compliant audit trails
    - Performance metrics integration
    - Multi-format output (console, file, JSON)
    - Automatic context tracking
    """
    
    def setup_logging(self, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
        """Configure logging with medical compliance standards"""
        
    def get_performance_logger(self) -> logging.Logger:
        """Get specialized performance metrics logger"""
        
    def create_audit_logger(self, operation: str) -> logging.Logger:
        """Create audit trail logger for specific operations"""
```

### Context-Aware Logging

```python
from utils.logging_config import setup_logging, LoggingContext, AuditLogger

# Professional logging setup
logger = setup_logging(
    level="INFO",
    log_file="surgical_processing.log",
    json_format=True,
    rotation_size="10MB"
)

# Context-aware operation logging
with LoggingContext("farabi_video_processing", {"hospital": "Farabi", "patient_id": "anonymous"}):
    logger.info("Starting de-identification processing")
    
    # Processing operations automatically logged with context
    result = processor.apply_farabi_processing(input_video, output_video)
    
    logger.info(f"Processing completed: {result.status}")

# Audit trail for compliance
audit_logger = AuditLogger("patient_data_processing")
audit_logger.log_processing_start("video_001.mp4", "privacy_protection")
audit_logger.log_processing_complete("video_001.mp4", "success", {"privacy_verified": True})
```

### Medical Compliance Features

```python
# HIPAA-compliant logging
class HIPAALogger:
    """HIPAA-compliant logging for medical video processing"""
    
    def log_access(self, user: str, resource: str, action: str) -> None:
        """Log access to patient data"""
        
    def log_deidentification(self, video_id: str, method: str, success: bool) -> None:
        """Log de-identification operations"""
        
    def log_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log auditable events for compliance"""
```

---

## 📊 Performance Monitoring System (`performance_monitor.py`)

### Real-time Performance Analytics
- **System Resource Monitoring**: CPU, memory, disk I/O, GPU utilization tracking
- **Processing Metrics Collection**: FPS, compression ratios, quality scores
- **Performance Benchmarking**: Baseline performance measurement and comparison
- **Resource Optimization**: Automatic resource usage optimization suggestions
- **Historical Performance Analysis**: Trend analysis and performance degradation detection

### Performance Monitoring Classes

```python
class SystemMonitor:
    """
    Real-time system performance monitoring for surgical video processing
    
    Monitors:
    - CPU utilization and temperature
    - Memory usage and availability  
    - Disk I/O performance
    - GPU utilization (if available)
    - Network throughput
    """
    
    def start_monitoring(self) -> None:
        """Start real-time system monitoring"""
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
    def optimize_resources(self) -> List[str]:
        """Get resource optimization recommendations"""

class ProcessingMetrics:
    """
    Specialized metrics for surgical video processing operations
    
    Tracks:
    - Processing efficiency and ratios
    - Processing speed (FPS)
    - Quality retention metrics
    - De-identification success rates
    """
    
    def track_processing(self, input_size: int, output_size: int) -> float:
        """Track processing ratio metrics"""
        
    def track_quality(self, quality_score: float) -> None:
        """Track video quality metrics"""
        
    def track_processing_speed(self, frames: int, duration: float) -> float:
        """Track processing speed metrics"""
```

### Performance Analysis Tools

```python
from utils.performance_monitor import SystemMonitor, ProcessingMetrics, PerformanceProfiler

# Comprehensive performance monitoring
monitor = SystemMonitor()
metrics = ProcessingMetrics()

# Start monitoring before processing
monitor.start_monitoring()

# Profile specific operations
with PerformanceProfiler("video_processing") as profiler:
    # Video processing operation
    result = processor.apply_processing(input_video, output_video)
    
    # Track metrics
    metrics.track_processing(input_size, output_size)
    metrics.track_quality(result.quality_score)
    metrics.track_processing_speed(result.frames_processed, profiler.duration)

# Generate performance report
performance_report = monitor.get_performance_report()
optimization_suggestions = monitor.optimize_resources()

# Log performance data
logger.info("Performance Report", extra={
    "cpu_usage": performance_report["cpu_usage"],
    "memory_usage": performance_report["memory_usage"],
    "processing_fps": metrics.get_average_fps(),
    "processing_ratio": metrics.get_average_processing_ratio()
})
```

---

## 📁 Secure File Operations (`file_utils.py`)

### Enterprise File Management
- **Atomic File Operations**: Ensure data integrity during file operations
- **Secure Path Validation**: Prevent path traversal and injection attacks
- **Medical File Validation**: Specialized validation for surgical video files
- **Backup Management**: Automated backup creation and verification
- **Concurrent Access Control**: Safe multi-process file access management

### File Operation Classes

```python
class SecureFileManager:
    """
    Secure file operations for medical video processing
    
    Features:
    - Atomic file operations
    - Path traversal protection
    - Medical compliance validation
    - Automatic backup creation
    """
    
    def validate_video_file(self, file_path: Path) -> bool:
        """Comprehensive video file validation"""
        
    def create_secure_backup(self, source: Path, backup_dir: Path) -> Path:
        """Create verified backup with integrity checking"""
        
    def atomic_move(self, source: Path, destination: Path) -> bool:
        """Atomic file move operation"""

class MedicalVideoValidator:
    """
    Specialized validation for surgical video files
    
    Validates:
    - Video format compatibility
    - Resolution and framerate standards
    - Audio track integrity
    - Metadata completeness
    """
    
    def validate_farabi_standard(self, video_path: Path) -> bool:
        """Validate against Farabi hospital standards"""
        
    def validate_noor_standard(self, video_path: Path) -> bool:
        """Validate against Noor hospital standards"""
        
    def check_privacy_compliance(self, video_path: Path) -> bool:
        """Verify privacy protection compliance"""
```

### Secure File Operations

```python
from utils.file_utils import SecureFileManager, MedicalVideoValidator, BackupManager

# Secure file management
file_mgr = SecureFileManager()
validator = MedicalVideoValidator()
backup_mgr = BackupManager("/secure/backups")

# Validate input video
if validator.validate_farabi_standard(input_video):
    # Create secure backup before processing
    backup_path = backup_mgr.create_verified_backup(input_video)
    
    # Process with atomic operations
    temp_output = file_mgr.create_temp_file(suffix=".mp4")
    
    # Processing...
    result = processor.process_video(input_video, temp_output)
    
    if result.success:
        # Atomic move to final destination
        file_mgr.atomic_move(temp_output, output_video)
        
        # Verify integrity
        if validator.validate_processed_video(output_video):
            logger.info("Processing completed successfully")
        else:
            # Restore from backup
            file_mgr.restore_from_backup(backup_path, output_video)
            logger.error("Processing failed validation, restored from backup")
```

---

## 🔨 Core Helper Functions (`helpers.py`)

### Comprehensive Utility Functions
- **Video Metadata Extraction**: Complete FFprobe integration for video analysis
- **System Information Detection**: Hardware and software capability detection
- **FFmpeg Command Utilities**: Safe FFmpeg command construction and execution
- **Format Conversion Tools**: Video format and codec conversion utilities
- **Cataract-LMM Specific Helpers**: Dataset-specific utility functions

### Helper Function Categories

```python
class VideoInfoExtractor:
    """
    Comprehensive video metadata extraction using FFprobe
    
    Extracts:
    - Resolution, framerate, duration
    - Codec information
    - Audio stream details
    - Metadata and tags
    """
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Extract complete video information"""
        
    def detect_hospital_source(self, video_info: Dict[str, Any]) -> str:
        """Detect hospital source from video characteristics"""
        
    def validate_cataract_video(self, video_path: Path) -> bool:
        """Validate video meets Cataract-LMM standards"""

class SystemCapabilities:
    """
    System capability detection and optimization
    
    Detects:
    - CPU cores and architecture
    - Available memory
    - GPU acceleration support
    - FFmpeg capabilities
    """
    
    def get_system_info(self) -> Dict[str, Any]:
        """Comprehensive system information"""
        
    def detect_gpu_support(self) -> bool:
        """Detect GPU acceleration availability"""
        
    def optimize_ffmpeg_threads(self) -> int:
        """Calculate optimal FFmpeg thread count"""
```

### Practical Helper Usage

```python
from utils.helpers import get_video_info, get_system_info, detect_hospital_source

# Video analysis
video_info = get_video_info("cataract_surgery.mp4")
hospital = detect_hospital_source(video_info)

print(f"Video: {video_info['width']}x{video_info['height']} @ {video_info['fps']}fps")
print(f"Duration: {video_info['duration']}s")
print(f"Hospital Source: {hospital}")

# System optimization
system_info = get_system_info()
optimal_threads = system_info["optimal_ffmpeg_threads"]
gpu_available = system_info["gpu_acceleration"]

print(f"System: {system_info['cpu_cores']} cores, {system_info['memory_gb']}GB RAM")
print(f"GPU Acceleration: {gpu_available}")
print(f"Optimal Threads: {optimal_threads}")

# Hospital-specific processing decisions
if hospital == "Farabi":
    config = config_mgr.get_farabi_config()
    processor.apply_farabi_processing(input_video, output_video)
elif hospital == "Noor":
    config = config_mgr.get_noor_config()
    processor.apply_noor_processing(input_video, output_video)
```

---

## 🔄 Integrated Utility Usage

### Complete Workflow Example

```python
from utils import (
    ConfigManager, setup_logging, SystemMonitor, 
    SecureFileManager, ProcessingMetrics, get_video_info
)

# Initialize all utilities
logger = setup_logging(level="INFO", log_file="surgical_processing.log")
config_mgr = ConfigManager()
monitor = SystemMonitor()
file_mgr = SecureFileManager()
metrics = ProcessingMetrics()

# Processing workflow with full utility integration
def process_surgical_video(input_path: str, output_path: str) -> bool:
    """Complete surgical video processing with full utility support"""
    
    try:
        # Video analysis and hospital detection
        video_info = get_video_info(input_path)
        
        # Load appropriate configuration
        config = config_mgr.load_config("default.yaml")
        
        # Security and backup
        if file_mgr.validate_video_file(input_path):
            backup_path = file_mgr.create_secure_backup(input_path)
            
            # Performance monitoring
            monitor.start_monitoring()
            
            with LoggingContext("surgical_video_processing"):
                # Process video with metrics
                with ProcessingTimer() as timer:
                    result = processor.process_video(input_path, output_path, config)
                
                # Track metrics
                metrics.track_processing_speed(video_info["frame_count"], timer.duration)
                metrics.track_processing(
                    file_mgr.get_file_size(input_path),
                    file_mgr.get_file_size(output_path)
                )
                
                # Generate report
                performance_report = monitor.get_performance_report()
                
                logger.info("Processing completed", extra={
                    "processing_time": timer.duration,
                    "processing_ratio": metrics.get_last_processing_ratio(),
                    "system_performance": performance_report
                })
                
                return result.success
                
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False
```

---

## 🧪 Testing and Validation

### Comprehensive Test Suite

```python
# Test configuration management
def test_hospital_configs():
    """Test hospital configuration loading and validation"""
    config_mgr = ConfigManager()
    
    farabi_config = config_mgr.get_farabi_config()
    assert farabi_config["hospital"]["name"] == "Farabi"
    assert farabi_config["processing"]["filter_complex"] == "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]"
    
    noor_config = config_mgr.get_noor_config()
    assert noor_config["hospital"]["name"] == "Noor"
    assert noor_config["processing"]["video_codec"] == "libx265"

# Test performance monitoring
def test_performance_monitoring():
    """Test system monitoring and metrics collection"""
    monitor = SystemMonitor()
    metrics = ProcessingMetrics()
    
    monitor.start_monitoring()
    
    # Simulate processing
    metrics.track_processing(1000000, 500000)  # 50% processing
    
    report = monitor.get_performance_report()
    assert "cpu_usage" in report
    assert "memory_usage" in report
    
    processing_ratio = metrics.get_average_processing_ratio()
    assert processing_ratio == 0.5

# Test file operations
def test_secure_file_operations():
    """Test secure file management"""
    file_mgr = SecureFileManager()
    validator = MedicalVideoValidator()
    
    # Test video validation
    assert validator.validate_video_file("test_video.mp4")
    
    # Test backup creation
    backup_path = file_mgr.create_secure_backup("test_video.mp4")
    assert backup_path.exists()
```

### Running Tests

```bash
# Run complete utility test suite
python -m pytest tests/test_utils/ -v

# Run specific utility tests
python -m pytest tests/test_utils/test_config_manager.py
python -m pytest tests/test_utils/test_performance_monitor.py
python -m pytest tests/test_utils/test_file_utils.py
python -m pytest tests/test_utils/test_helpers.py

# Performance benchmarking
python scripts/benchmark_utilities.py
```

---

## 📊 Performance Characteristics

| Utility Module | Memory Usage | CPU Impact | I/O Impact | Performance Notes |
|----------------|-------------|------------|------------|-------------------|
| **Config Manager** | Low (< 10MB) | Minimal | Low | Cached configurations |
| **Logging System** | Medium (< 50MB) | Low | Medium | Async logging available |
| **Performance Monitor** | Medium (< 100MB) | Low-Medium | Low | Background monitoring |
| **File Utils** | Low (< 20MB) | Low | High | I/O intensive operations |
| **Helpers** | Low (< 15MB) | Variable | Medium | Depends on FFprobe calls |

---

**🛠️ Professional utility infrastructure ensuring reliable, secure, and efficient surgical video processing operations for the Cataract-LMM dataset.**
