"""
Production-Ready File and Path Utilities

This module provides comprehensive file and path handling utilities
specifically designed for surgical video processing operations,
including validation, sanitization, backup management, and secure operations.
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from .logging_config import get_processing_logger

logger = get_processing_logger(__name__)


@dataclass
class FileInfo:
    """Comprehensive file information container"""

    path: Path
    size_bytes: int
    size_mb: float
    created_time: datetime
    modified_time: datetime
    extension: str
    is_video: bool
    is_readable: bool
    is_writable: bool
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    mime_type: Optional[str] = None
    video_info: Optional[Dict[str, Any]] = None


class VideoFileExtensions:
    """Supported video file extensions"""

    # Common video formats
    COMMON = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    # Professional/medical formats
    PROFESSIONAL = {".mxf", ".dv", ".3gp", ".m4v", ".ts", ".vob"}

    # All supported formats
    ALL = COMMON | PROFESSIONAL

    @classmethod
    def is_video_file(cls, file_path: Union[str, Path]) -> bool:
        """Check if file is a supported video format"""
        return Path(file_path).suffix.lower() in cls.ALL

    @classmethod
    def get_format_info(cls, file_path: Union[str, Path]) -> Dict[str, str]:
        """Get format information for a video file"""
        extension = Path(file_path).suffix.lower()

        format_map = {
            ".mp4": {"container": "MP4", "codecs": "H.264/H.265", "quality": "High"},
            ".avi": {"container": "AVI", "codecs": "Various", "quality": "Variable"},
            ".mov": {"container": "QuickTime", "codecs": "Various", "quality": "High"},
            ".mkv": {"container": "Matroska", "codecs": "Various", "quality": "High"},
            ".wmv": {"container": "WMV", "codecs": "WMV/VC-1", "quality": "Medium"},
            ".webm": {"container": "WebM", "codecs": "VP8/VP9", "quality": "Medium"},
            ".mxf": {
                "container": "MXF",
                "codecs": "Professional",
                "quality": "Professional",
            },
        }

        return format_map.get(
            extension,
            {"container": "Unknown", "codecs": "Unknown", "quality": "Unknown"},
        )


class PathSanitizer:
    """Safe path handling and sanitization"""

    # Invalid characters for filenames
    INVALID_CHARS = r'[<>:"/\\|?*]'

    # Reserved names (Windows)
    RESERVED_NAMES = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    @classmethod
    def sanitize_filename(cls, filename: str, replacement: str = "_") -> str:
        """
        Sanitize filename by removing invalid characters

        Args:
            filename: Original filename
            replacement: Character to replace invalid chars with

        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        sanitized = re.sub(cls.INVALID_CHARS, replacement, filename)

        # Handle reserved names
        name_part = sanitized.split(".")[0].upper()
        if name_part in cls.RESERVED_NAMES:
            sanitized = f"{replacement}{sanitized}"

        # Trim whitespace and dots
        sanitized = sanitized.strip(" .")

        # Ensure not empty
        if not sanitized:
            sanitized = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Limit length (Windows has 255 char limit)
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[: 200 - len(ext)] + ext

        return sanitized

    @classmethod
    def create_safe_path(
        cls, base_dir: Union[str, Path], filename: str, create_dirs: bool = True
    ) -> Path:
        """
        Create a safe file path

        Args:
            base_dir: Base directory
            filename: Desired filename
            create_dirs: Whether to create directories

        Returns:
            Safe file path
        """
        base_path = Path(base_dir)
        safe_filename = cls.sanitize_filename(filename)
        full_path = base_path / safe_filename

        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)

        return full_path

    @classmethod
    def resolve_conflicts(cls, file_path: Path, strategy: str = "increment") -> Path:
        """
        Resolve filename conflicts

        Args:
            file_path: Desired file path
            strategy: 'increment', 'timestamp', or 'overwrite'

        Returns:
            Resolved file path
        """
        if not file_path.exists() or strategy == "overwrite":
            return file_path

        if strategy == "timestamp":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = file_path.stem
            suffix = file_path.suffix
            new_name = f"{stem}_{timestamp}{suffix}"
            return file_path.parent / new_name

        elif strategy == "increment":
            counter = 1
            while True:
                stem = file_path.stem
                suffix = file_path.suffix
                new_name = f"{stem}_{counter:03d}{suffix}"
                new_path = file_path.parent / new_name

                if not new_path.exists():
                    return new_path

                counter += 1
                if counter > 999:  # Safety limit
                    # Fall back to timestamp
                    return cls.resolve_conflicts(file_path, "timestamp")

        return file_path


class FileValidator:
    """Comprehensive file validation"""

    @staticmethod
    def validate_video_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a video file

        Args:
            file_path: Path to video file

        Returns:
            Validation results
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

        # Check if it's a file (not directory)
        if not path.is_file():
            result["errors"].append(f"Path is not a file: {path}")
            return result

        # Check readability
        try:
            with open(path, "rb"):
                pass
            result["readable"] = True
        except PermissionError:
            result["errors"].append(f"No read permission: {path}")
            return result
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

            if result["size_mb"] > 10000:  # > 10GB
                result["warnings"].append(
                    f"Very large file: {result['size_mb']:.1f} MB"
                )

        except Exception as e:
            result["errors"].append(f"Cannot get file size: {e}")
            return result

        # Check video format
        if not VideoFileExtensions.is_video_file(path):
            result["errors"].append(f"Unsupported video format: {path.suffix}")
            return result

        result["is_video"] = True

        # All checks passed
        if not result["errors"]:
            result["valid"] = True

        return result

    @staticmethod
    def validate_directory(
        dir_path: Union[str, Path], create_if_missing: bool = False
    ) -> Dict[str, Any]:
        """
        Validate a directory

        Args:
            dir_path: Path to directory
            create_if_missing: Whether to create directory if missing

        Returns:
            Validation results
        """
        path = Path(dir_path)
        result = {
            "valid": False,
            "exists": False,
            "writable": False,
            "errors": [],
            "warnings": [],
        }

        # Check existence
        if not path.exists():
            if create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    result["exists"] = True
                    logger.info(f"Created directory: {path}")
                except Exception as e:
                    result["errors"].append(f"Cannot create directory: {e}")
                    return result
            else:
                result["errors"].append(f"Directory does not exist: {path}")
                return result
        else:
            result["exists"] = True

        # Check if it's a directory
        if not path.is_dir():
            result["errors"].append(f"Path is not a directory: {path}")
            return result

        # Check writability
        try:
            test_file = (
                path / f"test_write_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tmp"
            )
            test_file.touch()
            test_file.unlink()
            result["writable"] = True
        except Exception as e:
            result["errors"].append(f"Directory not writable: {e}")
            return result

        # All checks passed
        if not result["errors"]:
            result["valid"] = True

        return result


class FileOperations:
    """Safe file operations with error handling"""

    @staticmethod
    def copy_file(
        src: Union[str, Path],
        dst: Union[str, Path],
        preserve_metadata: bool = True,
        verify_copy: bool = True,
    ) -> bool:
        """
        Safely copy a file

        Args:
            src: Source file path
            dst: Destination file path
            preserve_metadata: Whether to preserve file metadata
            verify_copy: Whether to verify the copy

        Returns:
            True if successful
        """
        src_path = Path(src)
        dst_path = Path(dst)

        try:
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if preserve_metadata:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)

            # Verify copy if requested
            if verify_copy:
                if not FileOperations.verify_file_integrity(src_path, dst_path):
                    dst_path.unlink()  # Remove failed copy
                    return False

            logger.info(f"Successfully copied: {src_path} -> {dst_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False

    @staticmethod
    def move_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """
        Safely move a file

        Args:
            src: Source file path
            dst: Destination file path

        Returns:
            True if successful
        """
        src_path = Path(src)
        dst_path = Path(dst)

        try:
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file
            shutil.move(str(src_path), str(dst_path))

            logger.info(f"Successfully moved: {src_path} -> {dst_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to move file: {e}")
            return False

    @staticmethod
    def delete_file(file_path: Union[str, Path], secure: bool = False) -> bool:
        """
        Safely delete a file

        Args:
            file_path: Path to file to delete
            secure: Whether to perform secure deletion

        Returns:
            True if successful
        """
        path = Path(file_path)

        try:
            if not path.exists():
                logger.warning(f"File does not exist: {path}")
                return True

            if secure:
                # Overwrite file with random data before deletion
                with open(path, "r+b") as f:
                    size = f.seek(0, 2)  # Get file size
                    f.seek(0)
                    f.write(os.urandom(size))
                    f.flush()
                    os.fsync(f.fileno())

            path.unlink()
            logger.info(f"Successfully deleted: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    @staticmethod
    def verify_file_integrity(file1: Union[str, Path], file2: Union[str, Path]) -> bool:
        """
        Verify file integrity by comparing checksums

        Args:
            file1: First file path
            file2: Second file path

        Returns:
            True if files are identical
        """
        try:
            hash1 = FileOperations.calculate_checksum(file1)
            hash2 = FileOperations.calculate_checksum(file2)
            return hash1 == hash2
        except Exception:
            return False

    @staticmethod
    def calculate_checksum(file_path: Union[str, Path], algorithm: str = "md5") -> str:
        """
        Calculate file checksum

        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            Hex digest of checksum
        """
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha1":
            hasher = hashlib.sha1()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        return hasher.hexdigest()


class BackupManager:
    """Comprehensive backup management"""

    def __init__(self, backup_dir: Union[str, Path]):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(
        self, file_path: Union[str, Path], backup_name: Optional[str] = None
    ) -> Path:
        """
        Create a backup of a file

        Args:
            file_path: Path to file to backup
            backup_name: Custom backup name

        Returns:
            Path to backup file
        """
        src_path = Path(file_path)

        if backup_name:
            backup_file = self.backup_dir / backup_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = (
                self.backup_dir / f"{src_path.stem}_{timestamp}{src_path.suffix}"
            )

        # Resolve conflicts
        backup_file = PathSanitizer.resolve_conflicts(backup_file, "increment")

        # Create backup
        if FileOperations.copy_file(src_path, backup_file, verify_copy=True):
            logger.info(f"Created backup: {backup_file}")
            return backup_file
        else:
            raise RuntimeError(f"Failed to create backup for {src_path}")

    def restore_backup(
        self, backup_path: Union[str, Path], restore_path: Union[str, Path]
    ) -> bool:
        """
        Restore a file from backup

        Args:
            backup_path: Path to backup file
            restore_path: Path to restore to

        Returns:
            True if successful
        """
        return FileOperations.copy_file(backup_path, restore_path, verify_copy=True)

    def list_backups(self, pattern: str = "*") -> List[Path]:
        """
        List available backups

        Args:
            pattern: File pattern to match

        Returns:
            List of backup file paths
        """
        return list(self.backup_dir.glob(pattern))

    def cleanup_old_backups(self, max_age_days: int = 30, max_count: int = 100) -> int:
        """
        Clean up old backup files

        Args:
            max_age_days: Maximum age of backups to keep
            max_count: Maximum number of backups to keep

        Returns:
            Number of files deleted
        """
        now = datetime.now()
        cutoff_time = now - timedelta(days=max_age_days)

        # Get all backup files sorted by modification time
        backup_files = sorted(
            self.backup_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        deleted_count = 0

        for i, backup_file in enumerate(backup_files):
            should_delete = False

            # Delete if too old
            file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_time < cutoff_time:
                should_delete = True

            # Delete if exceeds count limit
            if i >= max_count:
                should_delete = True

            if should_delete and backup_file.is_file():
                try:
                    backup_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old backup: {backup_file}")
                except Exception as e:
                    logger.error(f"Failed to delete backup {backup_file}: {e}")

        return deleted_count


@contextmanager
def atomic_file_operation(file_path: Union[str, Path]):
    """
    Context manager for atomic file operations

    Ensures that file operations are atomic by using a temporary file
    and moving it to the final location only on success.
    """
    final_path = Path(file_path)
    temp_path = final_path.with_suffix(f"{final_path.suffix}.tmp")

    try:
        yield temp_path

        # Move temp file to final location (atomic operation)
        temp_path.replace(final_path)

    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


@contextmanager
def file_lock(file_path: Union[str, Path], mode: str = "r"):
    """
    Context manager for file locking (Unix systems)

    Args:
        file_path: Path to file to lock
        mode: File open mode
    """
    path = Path(file_path)

    with open(path, mode) as f:
        try:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield f
        except BlockingIOError:
            raise RuntimeError(f"Could not acquire lock on {path}")
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_file_info(
    file_path: Union[str, Path], calculate_checksums: bool = False
) -> FileInfo:
    """
    Get comprehensive file information

    Args:
        file_path: Path to file
        calculate_checksums: Whether to calculate checksums

    Returns:
        FileInfo object with comprehensive file information
    """
    path = Path(file_path)
    stat = path.stat()

    info = FileInfo(
        path=path,
        size_bytes=stat.st_size,
        size_mb=stat.st_size / (1024 * 1024),
        created_time=datetime.fromtimestamp(stat.st_ctime),
        modified_time=datetime.fromtimestamp(stat.st_mtime),
        extension=path.suffix.lower(),
        is_video=VideoFileExtensions.is_video_file(path),
        is_readable=os.access(path, os.R_OK),
        is_writable=os.access(path, os.W_OK),
    )

    if calculate_checksums:
        try:
            info.checksum_md5 = FileOperations.calculate_checksum(path, "md5")
            info.checksum_sha256 = FileOperations.calculate_checksum(path, "sha256")
        except Exception as e:
            logger.warning(f"Could not calculate checksums for {path}: {e}")

    return info


def find_video_files(
    directory: Union[str, Path], recursive: bool = True, min_size_mb: float = 0.1
) -> Generator[Path, None, None]:
    """
    Find video files in a directory

    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        min_size_mb: Minimum file size in MB

    Yields:
        Video file paths
    """
    search_path = Path(directory)
    pattern = "**/*" if recursive else "*"

    for file_path in search_path.glob(pattern):
        if file_path.is_file() and VideoFileExtensions.is_video_file(file_path):

            # Check minimum size
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb >= min_size_mb:
                    yield file_path
            except Exception:
                continue


def create_directory_structure(
    base_dir: Union[str, Path], structure: Dict[str, Any]
) -> bool:
    """
    Create a directory structure from a dictionary

    Args:
        base_dir: Base directory
        structure: Dictionary defining the structure

    Returns:
        True if successful
    """
    try:
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        for name, content in structure.items():
            current_path = base_path / name

            if isinstance(content, dict):
                # It's a subdirectory
                create_directory_structure(current_path, content)
            else:
                # It's a file
                current_path.parent.mkdir(parents=True, exist_ok=True)
                if content is not None:
                    current_path.write_text(str(content))
                else:
                    current_path.touch()

        return True

    except Exception as e:
        logger.error(f"Failed to create directory structure: {e}")
        return False
