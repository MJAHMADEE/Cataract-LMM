"""
Core Helper Functions for Surgical Video Processing

This module provides essential helper functions and utilities used
throughout the surgical video processing framework.
"""

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _safe_eval_fraction(fraction_str):
    """Safely evaluate a fraction string like '30/1' without using eval()"""
    try:
        if "/" in fraction_str:
            num, denom = fraction_str.split("/", 1)
            return float(num) / float(denom) if float(denom) != 0 else 0.0
        else:
            return float(fraction_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


import psutil

# Module logger
logger = logging.getLogger("surgical_video_processing.utils.helpers")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.1f}s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def format_file_size(bytes_size: Union[int, float]) -> str:
    """
    Format file size in bytes to human-readable string

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted file size string
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"

    for unit in ["KB", "MB", "GB", "TB"]:
        bytes_size /= 1024
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"

    return f"{bytes_size:.1f} PB"


def safe_filename(filename: str, replacement: str = "_", max_length: int = 200) -> str:
    """
    Create a safe filename by removing invalid characters

    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with
        max_length: Maximum filename length

    Returns:
        Safe filename
    """
    # Remove invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    safe_name = re.sub(invalid_chars, replacement, filename)

    # Remove leading/trailing whitespace and dots
    safe_name = safe_name.strip(" .")

    # Handle empty name
    if not safe_name:
        safe_name = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Truncate if too long
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[: max_length - len(ext)] + ext

    return safe_name


def validate_video_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a video file

    Args:
        file_path: Path to video file

    Returns:
        Validation results dictionary
    """
    path = Path(file_path)
    result = {
        "valid": False,
        "exists": False,
        "readable": False,
        "is_video": False,
        "size_mb": 0,
        "errors": [],
        "warnings": [],
    }

    # Check existence
    if not path.exists():
        result["errors"].append(f"File does not exist: {path}")
        return result

    result["exists"] = True

    # Check if it's a file
    if not path.is_file():
        result["errors"].append(f"Path is not a file: {path}")
        return result

    # Check readability
    try:
        with open(path, "rb") as f:
            f.read(1)
        result["readable"] = True
    except Exception as e:
        result["errors"].append(f"Cannot read file: {e}")
        return result

    # Check file size
    try:
        size_bytes = path.stat().st_size
        result["size_mb"] = size_bytes / (1024 * 1024)

        if size_bytes == 0:
            result["errors"].append("File is empty")
            return result
    except Exception as e:
        result["errors"].append(f"Cannot get file size: {e}")
        return result

    # Check video format
    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".mxf",
        ".dv",
        ".3gp",
        ".m4v",
        ".ts",
        ".vob",
    }

    if path.suffix.lower() not in video_extensions:
        result["errors"].append(f"Unsupported video format: {path.suffix}")
        return result

    result["is_video"] = True

    # All checks passed
    if not result["errors"]:
        result["valid"] = True

    return result


def get_video_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get video file information using ffprobe

    Args:
        file_path: Path to video file

    Returns:
        Video information dictionary
    """
    if not is_ffmpeg_available():
        return {"error": "FFmpeg/ffprobe not available"}

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return {"error": f"ffprobe failed: {result.stderr}"}

        data = json.loads(result.stdout)

        # Extract useful information
        info = {
            "format": data.get("format", {}),
            "video_streams": [],
            "audio_streams": [],
            "duration": None,
            "size_bytes": None,
            "bitrate": None,
        }

        # Process format information
        format_info = data.get("format", {})
        if "duration" in format_info:
            info["duration"] = float(format_info["duration"])
        if "size" in format_info:
            info["size_bytes"] = int(format_info["size"])
        if "bit_rate" in format_info:
            info["bitrate"] = int(format_info["bit_rate"])

        # Process streams
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_info = {
                    "codec": stream.get("codec_name"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "fps": (
                        _safe_eval_fraction(stream.get("r_frame_rate", "0/1"))
                        if "/" in stream.get("r_frame_rate", "")
                        else None
                    ),
                    "bitrate": stream.get("bit_rate"),
                    "pixel_format": stream.get("pix_fmt"),
                }
                info["video_streams"].append(video_info)

            elif stream.get("codec_type") == "audio":
                audio_info = {
                    "codec": stream.get("codec_name"),
                    "sample_rate": stream.get("sample_rate"),
                    "channels": stream.get("channels"),
                    "bitrate": stream.get("bit_rate"),
                }
                info["audio_streams"].append(audio_info)

        return info

    except subprocess.TimeoutExpired:
        return {"error": "ffprobe timeout"}
    except json.JSONDecodeError:
        return {"error": "Invalid ffprobe output"}
    except Exception as e:
        return {"error": str(e)}


def hospital_from_filename(filename: str) -> str:
    """
    Detect hospital type from filename

    Args:
        filename: Video filename

    Returns:
        Hospital type ('farabi', 'noor', or 'general')
    """
    filename_lower = filename.lower()

    # Hospital detection patterns
    if any(keyword in filename_lower for keyword in ["farabi", "frb", "f_"]):
        return "farabi"
    elif any(keyword in filename_lower for keyword in ["noor", "nur", "n_"]):
        return "noor"
    else:
        return "general"


def ensure_directory(directory: Union[str, Path]) -> bool:
    """
    Ensure directory exists, create if necessary

    Args:
        directory: Directory path

    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False


def cleanup_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files

    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age in hours for files to keep

    Returns:
        Number of files deleted
    """
    if not Path(temp_dir).exists():
        return 0

    cutoff_time = time.time() - (max_age_hours * 3600)
    deleted_count = 0

    try:
        for file_path in Path(temp_dir).rglob("*"):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")

    return deleted_count


def is_ffmpeg_available() -> bool:
    """
    Check if FFmpeg is available in the system

    Returns:
        True if FFmpeg is available
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True, timeout=10
        )
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def get_system_info() -> Dict[str, Any]:
    """
    Get system information

    Returns:
        System information dictionary
    """
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100,
            "ffmpeg_available": is_ffmpeg_available(),
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "error": str(e),
        }


def estimate_processing_time(
    file_size_mb: float, processing_speed_mbps: float = 5.0
) -> float:
    """
    Estimate video processing time

    Args:
        file_size_mb: File size in MB
        processing_speed_mbps: Processing speed in MB per second

    Returns:
        Estimated processing time in seconds
    """
    if file_size_mb <= 0 or processing_speed_mbps <= 0:
        return 0.0

    return file_size_mb / processing_speed_mbps


def create_temp_file(suffix: str = ".tmp", prefix: str = "svp_") -> str:
    """
    Create a temporary file

    Args:
        suffix: File suffix
        prefix: File prefix

    Returns:
        Temporary file path
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)  # Close the file descriptor
    return temp_path


def create_temp_directory(prefix: str = "svp_") -> str:
    """
    Create a temporary directory

    Args:
        prefix: Directory prefix

    Returns:
        Temporary directory path
    """
    return tempfile.mkdtemp(prefix=prefix)


def retry_operation(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,),
):
    """
    Retry an operation with exponential backoff

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry on

    Returns:
        Function result or raises last exception
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= backoff
            else:
                raise last_exception


class VideoProcessor:
    """Simple video processor for basic operations"""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def get_duration(self, file_path: Union[str, Path]) -> Optional[float]:
        """Get video duration in seconds"""
        info = get_video_info(file_path)
        return info.get("duration")

    def get_resolution(self, file_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
        """Get video resolution (width, height)"""
        info = get_video_info(file_path)
        video_streams = info.get("video_streams", [])
        if video_streams:
            stream = video_streams[0]
            width = stream.get("width")
            height = stream.get("height")
            if width and height:
                return (width, height)
        return None

    def extract_frame(
        self,
        file_path: Union[str, Path],
        output_path: Union[str, Path],
        timestamp: float = 0.0,
    ) -> bool:
        """Extract a frame at given timestamp"""
        try:
            cmd = [
                self.ffmpeg_path,
                "-y",  # Overwrite output
                "-i",
                str(file_path),
                "-ss",
                str(timestamp),
                "-vframes",
                "1",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to extract frame: {e}")
            return False


# Module-level instances
default_video_processor = VideoProcessor()


# Export functions that might be used as standalone utilities
def quick_video_info(file_path: Union[str, Path]) -> str:
    """
    Get quick video information as formatted string

    Args:
        file_path: Path to video file

    Returns:
        Formatted video information string
    """
    info = get_video_info(file_path)

    if "error" in info:
        return f"Error: {info['error']}"

    lines = []
    lines.append(f"File: {Path(file_path).name}")

    if info.get("duration"):
        lines.append(f"Duration: {format_duration(info['duration'])}")

    if info.get("size_bytes"):
        lines.append(f"Size: {format_file_size(info['size_bytes'])}")

    video_streams = info.get("video_streams", [])
    if video_streams:
        stream = video_streams[0]
        if stream.get("width") and stream.get("height"):
            lines.append(f"Resolution: {stream['width']}x{stream['height']}")
        if stream.get("codec"):
            lines.append(f"Video Codec: {stream['codec']}")
        if stream.get("fps"):
            lines.append(f"FPS: {stream['fps']:.1f}")

    audio_streams = info.get("audio_streams", [])
    if audio_streams:
        stream = audio_streams[0]
        if stream.get("codec"):
            lines.append(f"Audio Codec: {stream['codec']}")

    return "\n".join(lines)


def batch_validate_videos(
    directory: Union[str, Path], recursive: bool = True
) -> Dict[str, Any]:
    """
    Validate all video files in a directory

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        Validation results for all files
    """
    results = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "errors": [],
        "files": {},
    }

    pattern = "**/*" if recursive else "*"
    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".mxf",
        ".dv",
        ".3gp",
        ".m4v",
        ".ts",
        ".vob",
    }

    for file_path in Path(directory).glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            results["total_files"] += 1

            validation = validate_video_file(file_path)
            results["files"][str(file_path)] = validation

            if validation["valid"]:
                results["valid_files"] += 1
            else:
                results["invalid_files"] += 1
                results["errors"].extend(validation["errors"])

    return results
