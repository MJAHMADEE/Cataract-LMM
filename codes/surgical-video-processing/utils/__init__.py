"""
Comprehensive Utility Modules for Surgical Video Processing

This package provides production-ready utility modules for surgical video
processing operations, including logging, performance monitoring, file operations,
and configuration management.
"""

# Configuration management
from .config_manager import (
    BackupConfig,
    ConfigManager,
    FFmpegConfig,
    HospitalConfig,
    HospitalType,
    NotificationConfig,
    PerformanceConfig,
    ProcessingConfig,
    ProcessingMode,
    QualityControlConfig,
    SecurityConfig,
    create_sample_config,
    get_config_manager,
    get_current_config,
    load_config,
)

# File operations
from .file_utils import (
    BackupManager,
    FileInfo,
    FileOperations,
    FileValidator,
    PathSanitizer,
    VideoFileExtensions,
    atomic_file_operation,
    create_directory_structure,
    file_lock,
    find_video_files,
    get_file_info,
)

# Core utilities
from .helpers import *

# Logging and monitoring
from .logging_config import (
    ColoredFormatter,
    JSONFormatter,
    LoggingContext,
    ProcessingLogFilter,
    create_performance_logger,
    get_compression_logger,
    get_core_logger,
    get_main_logger,
    get_metadata_logger,
    get_pipeline_logger,
    get_processing_logger,
    get_quality_logger,
    log_error_with_context,
    log_performance_metrics,
    log_processing_end,
    log_processing_start,
    setup_logging,
)

# Performance monitoring
from .performance_monitor import (
    MetricsAggregator,
    MetricsCollector,
    PerformanceMetrics,
    ProcessingTimer,
    SystemMonitor,
    add_metrics_to_global,
    get_global_aggregator,
    performance_monitor,
)

# Version information
__version__ = "1.0.0"
__author__ = "Surgical Video Processing Team"
__description__ = "Professional utilities for surgical video processing"

# Module exports
__all__ = [
    # Core helpers (from helpers.py)
    "format_duration",
    "format_file_size",
    "safe_filename",
    "validate_video_file",
    "get_video_info",
    "hospital_from_filename",
    "ensure_directory",
    "cleanup_temp_files",
    "is_ffmpeg_available",
    "get_system_info",
    "estimate_processing_time",
    "VideoProcessor",
    # Logging
    "setup_logging",
    "get_processing_logger",
    "log_processing_start",
    "log_processing_end",
    "log_error_with_context",
    "create_performance_logger",
    "log_performance_metrics",
    "LoggingContext",
    "get_main_logger",
    "get_core_logger",
    "get_pipeline_logger",
    "get_quality_logger",
    "get_compression_logger",
    "get_metadata_logger",
    "ColoredFormatter",
    "JSONFormatter",
    "ProcessingLogFilter",
    # Performance monitoring
    "PerformanceMetrics",
    "SystemMonitor",
    "ProcessingTimer",
    "MetricsCollector",
    "MetricsAggregator",
    "performance_monitor",
    "get_global_aggregator",
    "add_metrics_to_global",
    # File operations
    "FileInfo",
    "VideoFileExtensions",
    "PathSanitizer",
    "FileValidator",
    "FileOperations",
    "BackupManager",
    "atomic_file_operation",
    "file_lock",
    "get_file_info",
    "find_video_files",
    "create_directory_structure",
    # Configuration
    "ProcessingMode",
    "HospitalType",
    "FFmpegConfig",
    "HospitalConfig",
    "QualityControlConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "BackupConfig",
    "NotificationConfig",
    "ProcessingConfig",
    "ConfigManager",
    "get_config_manager",
    "load_config",
    "get_current_config",
    "create_sample_config",
    # Version info
    "__version__",
    "__author__",
    "__description__",
]


def initialize_logging(level="INFO", log_file=None, console_output=True):
    """
    Initialize logging for the entire framework

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console

    Returns:
        Configured logger
    """
    import logging

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return setup_logging(
        level=level_map.get(level.upper(), logging.INFO),
        log_file=log_file,
        console_output=console_output,
        json_output=bool(log_file),
        include_modules=["surgical_video_processing"],
    )


def create_default_config(config_file="config.yaml"):
    """
    Create a default configuration file

    Args:
        config_file: Path to configuration file

    Returns:
        True if successful
    """
    return create_sample_config(config_file)


def validate_environment():
    """
    Validate the processing environment

    Returns:
        Dictionary with validation results
    """
    from .file_utils import FileValidator
    from .helpers import get_system_info, is_ffmpeg_available

    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "system_info": get_system_info(),
    }

    # Check FFmpeg availability
    if not is_ffmpeg_available():
        results["valid"] = False
        results["errors"].append("FFmpeg not found in PATH")

    # Check system requirements
    system_info = results["system_info"]

    if system_info.get("memory_total_gb", 0) < 2:
        results["warnings"].append("Low system memory (< 2GB)")

    if system_info.get("cpu_count", 0) < 2:
        results["warnings"].append("Low CPU count (< 2 cores)")

    if system_info.get("disk_free_gb", 0) < 10:
        results["warnings"].append("Low disk space (< 10GB)")

    return results


def get_framework_info():
    """
    Get comprehensive framework information

    Returns:
        Dictionary with framework information
    """
    import platform
    import sys
    from pathlib import Path

    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "installation_path": str(Path(__file__).parent.parent),
        "modules": {
            "logging_config": "Production logging system",
            "performance_monitor": "Real-time performance monitoring",
            "file_utils": "Safe file operations and validation",
            "config_manager": "Configuration management system",
            "helpers": "Core utility functions",
        },
    }


# Initialize module-level logging
_module_logger = get_processing_logger(__name__)
_module_logger.info(f"Surgical Video Processing Utils v{__version__} initialized")

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def setup_basic_logging(
    level: int = logging.INFO, log_file: Optional[str] = None
) -> None:
    """
    Set up basic logging configuration for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers, force=True)


def validate_paths(*paths: str) -> bool:
    """
    Validate that all provided paths exist.

    Args:
        *paths: Variable number of path strings

    Returns:
        True if all paths exist, False otherwise
    """
    for path in paths:
        if not Path(path).exists():
            return False
    return True


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    return Path(file_path).stat().st_size / (1024 * 1024)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def generate_timestamp() -> str:
    """
    Generate timestamp string for file naming.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove multiple consecutive underscores
    while "__" in filename:
        filename = filename.replace("__", "_")

    # Remove leading/trailing underscores
    filename = filename.strip("_")

    return filename


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file safely.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    ensure_directory(Path(file_path).parent)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Load YAML file safely.

    Args:
        file_path: Path to YAML file

    Returns:
        Loaded YAML data

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to YAML file.

    Args:
        data: Data to save
        file_path: Output file path
    """
    ensure_directory(Path(file_path).parent)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def get_video_extensions() -> List[str]:
    """
    Get list of supported video file extensions.

    Returns:
        List of video file extensions
    """
    return [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"]


def is_video_file(file_path: str) -> bool:
    """
    Check if file is a supported video format.

    Args:
        file_path: Path to file

    Returns:
        True if file is a video, False otherwise
    """
    return Path(file_path).suffix.lower() in get_video_extensions()


def find_video_files(directory: str, recursive: bool = True) -> List[Path]:
    """
    Find all video files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of video file paths
    """
    dir_path = Path(directory)
    video_files = []

    if not dir_path.exists():
        return video_files

    pattern = "**/*" if recursive else "*"

    for ext in get_video_extensions():
        video_files.extend(dir_path.glob(f"{pattern}{ext}"))
        video_files.extend(dir_path.glob(f"{pattern}{ext.upper()}"))

    return sorted(video_files)


def create_progress_callback(total_items: int, description: str = "Processing"):
    """
    Create a simple progress callback function.

    Args:
        total_items: Total number of items to process
        description: Description for progress display

    Returns:
        Progress callback function
    """

    def progress_callback(current: int):
        percentage = (current / total_items) * 100
        print(f"\r{description}: {current}/{total_items} ({percentage:.1f}%)", end="")
        if current == total_items:
            print()  # New line when complete

    return progress_callback


def calculate_compression_ratio(original_size: float, compressed_size: float) -> float:
    """
    Calculate compression ratio.

    Args:
        original_size: Original file size in bytes
        compressed_size: Compressed file size in bytes

    Returns:
        Compression ratio
    """
    if original_size == 0:
        return 0.0
    return compressed_size / original_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"


def validate_config_paths(config: Dict[str, Any]) -> List[str]:
    """
    Validate that all required paths in configuration exist.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation errors
    """
    errors = []

    # Check temp directory
    temp_dir = config.get("environment", {}).get("temp_directory")
    if temp_dir and not Path(temp_dir).parent.exists():
        errors.append(f"Temp directory parent does not exist: {temp_dir}")

    # Check log directory
    log_dir = config.get("environment", {}).get("log_directory")
    if log_dir and not Path(log_dir).parent.exists():
        errors.append(f"Log directory parent does not exist: {log_dir}")

    return errors


def cleanup_temp_files(temp_directory: str, max_age_hours: int = 24) -> None:
    """
    Clean up old temporary files.

    Args:
        temp_directory: Temporary directory path
        max_age_hours: Maximum age of files to keep in hours
    """
    import time

    temp_path = Path(temp_directory)
    if not temp_path.exists():
        return

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for file_path in temp_path.rglob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                except OSError:
                    pass  # Ignore errors when deleting temp files


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    ENDC = "\033[0m"  # End color
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_colored(text: str, color: str = Colors.WHITE) -> None:
    """
    Print text in color if terminal supports it.

    Args:
        text: Text to print
        color: Color code from Colors class
    """
    if os.isatty(sys.stdout.fileno()):  # Check if output is a terminal
        print(f"{color}{text}{Colors.ENDC}")
    else:
        print(text)


def print_success(text: str) -> None:
    """Print success message in green."""
    print_colored(f"✓ {text}", Colors.GREEN)


def print_error(text: str) -> None:
    """Print error message in red."""
    print_colored(f"✗ {text}", Colors.RED)


def print_warning(text: str) -> None:
    """Print warning message in yellow."""
    print_colored(f"⚠ {text}", Colors.YELLOW)


def print_info(text: str) -> None:
    """Print info message in blue."""
    print_colored(f"ℹ {text}", Colors.BLUE)
