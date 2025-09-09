"""
Production Configuration Management System

This module provides comprehensive configuration management for surgical
video processing operations, including environment-specific settings,
validation, and dynamic configuration updates.
"""

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml

from .file_utils import FileValidator, atomic_file_operation
from .logging_config import get_processing_logger

logger = get_processing_logger(__name__)


class ProcessingMode(Enum):
    """Processing mode enumeration"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    BENCHMARK = "benchmark"


class HospitalType(Enum):
    """Hospital type enumeration"""

    FARABI = "farabi"
    NOOR = "noor"
    GENERAL = "general"


@dataclass
class FFmpegConfig:
    """FFmpeg-specific configuration"""

    executable_path: str = "ffmpeg"
    threads: int = 0  # 0 = auto
    preset: str = "medium"
    crf: int = 23
    max_muxing_queue_size: int = 1024
    timeout_seconds: int = 3600
    extra_input_args: List[str] = field(default_factory=list)
    extra_output_args: List[str] = field(default_factory=list)
    hwaccel: Optional[str] = None  # cuda, vaapi, etc.
    hwaccel_device: Optional[str] = None


@dataclass
class HospitalConfig:
    """Hospital-specific configuration"""

    name: str
    type: HospitalType

    # Video specifications
    target_resolution: str = "1920x1080"
    target_fps: int = 30
    target_bitrate: str = "2M"

    # Processing settings
    enable_cropping: bool = False
    crop_filter: str = ""
    enable_blur: bool = False
    blur_filter: str = ""
    enable_deidentification: bool = True

    # Encoding settings
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    container_format: str = "mp4"

    # Quality control
    min_quality_score: float = 0.7
    enable_quality_check: bool = True

    # Custom processing pipeline
    custom_filters: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)


@dataclass
class QualityControlConfig:
    """Quality control configuration"""

    enable_validation: bool = True
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 14400.0  # 4 hours
    min_resolution: str = "320x240"
    max_file_size_gb: float = 20.0
    allowed_codecs: List[str] = field(default_factory=lambda: ["h264", "h265", "mpeg4"])
    enable_content_analysis: bool = False
    quality_threshold: float = 0.6
    enable_audio_validation: bool = True


@dataclass
class SecurityConfig:
    """Security and privacy configuration"""

    enable_encryption: bool = False
    encryption_algorithm: str = "AES-256"
    enable_watermarking: bool = False
    watermark_text: str = ""
    enable_audit_logging: bool = True
    secure_deletion: bool = False
    anonymization_level: str = "basic"  # basic, standard, strict
    phi_removal_enabled: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""

    max_concurrent_jobs: int = 4
    memory_limit_gb: float = 8.0
    temp_dir: str = ""  # Empty string will use tempfile.gettempdir()
    enable_gpu_acceleration: bool = False
    gpu_device_id: int = 0
    batch_size: int = 1
    enable_progress_tracking: bool = True
    checkpoint_interval: int = 100
    enable_performance_monitoring: bool = True


@dataclass
class BackupConfig:
    """Backup and recovery configuration"""

    enable_backups: bool = True
    backup_directory: str = "backups"
    max_backup_age_days: int = 30
    max_backup_count: int = 100
    backup_compression: bool = True
    enable_versioning: bool = True
    backup_verification: bool = True


@dataclass
class NotificationConfig:
    """Notification and alerting configuration"""

    enable_notifications: bool = False
    email_enabled: bool = False
    slack_enabled: bool = False
    webhook_enabled: bool = False
    notification_level: str = "error"  # info, warning, error, critical
    email_settings: Dict[str, str] = field(default_factory=dict)
    slack_settings: Dict[str, str] = field(default_factory=dict)
    webhook_settings: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """Main processing configuration"""

    # Basic settings
    mode: ProcessingMode = ProcessingMode.PRODUCTION
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Component configurations
    ffmpeg: FFmpegConfig = field(default_factory=FFmpegConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    # Hospital configurations
    hospitals: Dict[str, HospitalConfig] = field(default_factory=dict)
    default_hospital: str = "general"

    # Paths
    input_directory: str = "input"
    output_directory: str = "output"
    log_directory: str = "logs"
    temp_directory: str = "temp"
    config_directory: str = "configs"

    # Processing settings
    enable_batch_processing: bool = True
    max_batch_size: int = 10
    enable_resume: bool = True
    enable_dry_run: bool = False

    # Advanced settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Configuration management system"""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file) if config_file else None
        self._config: Optional[ProcessingConfig] = None
        self._config_cache: Dict[str, Any] = {}
        self._watchers: List[callable] = []

    def load_config(
        self, config_file: Optional[Union[str, Path]] = None
    ) -> ProcessingConfig:
        """
        Load configuration from file

        Args:
            config_file: Path to configuration file

        Returns:
            Loaded configuration
        """
        if config_file:
            self.config_file = Path(config_file)

        if not self.config_file or not self.config_file.exists():
            logger.info("No configuration file found, using defaults")
            self._config = ProcessingConfig()
            self._create_default_hospitals()
            return self._config

        try:
            with open(self.config_file, "r") as f:
                if self.config_file.suffix.lower() in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Convert data to ProcessingConfig
            self._config = self._dict_to_config(data)
            logger.info(f"Loaded configuration from {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            self._config = ProcessingConfig()
            self._create_default_hospitals()

        return self._config

    def save_config(self, config_file: Optional[Union[str, Path]] = None) -> bool:
        """
        Save configuration to file

        Args:
            config_file: Path to save configuration to

        Returns:
            True if successful
        """
        if not self._config:
            logger.error("No configuration to save")
            return False

        save_path = Path(config_file) if config_file else self.config_file
        if not save_path:
            logger.error("No configuration file specified")
            return False

        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert config to dict
            config_dict = asdict(self._config)

            # Use atomic file operation
            with atomic_file_operation(save_path) as temp_path:
                with open(temp_path, "w") as f:
                    if save_path.suffix.lower() in [".yaml", ".yml"]:
                        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                    else:
                        json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Saved configuration to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def get_config(self) -> ProcessingConfig:
        """Get current configuration"""
        if not self._config:
            return self.load_config()
        return self._config

    def update_config(self, **kwargs) -> bool:
        """
        Update configuration with new values

        Args:
            **kwargs: Configuration values to update

        Returns:
            True if successful
        """
        if not self._config:
            self.load_config()

        try:
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                else:
                    self._config.custom_settings[key] = value

            # Notify watchers
            for watcher in self._watchers:
                try:
                    watcher(self._config)
                except Exception as e:
                    logger.error(f"Configuration watcher error: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def get_hospital_config(self, hospital_name: str) -> HospitalConfig:
        """
        Get hospital-specific configuration

        Args:
            hospital_name: Name of hospital

        Returns:
            Hospital configuration
        """
        config = self.get_config()

        if hospital_name in config.hospitals:
            return config.hospitals[hospital_name]

        if config.default_hospital in config.hospitals:
            logger.warning(f"Hospital '{hospital_name}' not found, using default")
            return config.hospitals[config.default_hospital]

        logger.warning(
            f"No configuration found for hospital '{hospital_name}', using general"
        )
        return self._create_general_hospital_config()

    def add_hospital_config(self, hospital_config: HospitalConfig) -> bool:
        """
        Add hospital configuration

        Args:
            hospital_config: Hospital configuration to add

        Returns:
            True if successful
        """
        try:
            config = self.get_config()
            config.hospitals[hospital_config.name] = hospital_config
            logger.info(f"Added hospital configuration: {hospital_config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add hospital configuration: {e}")
            return False

    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate current configuration

        Returns:
            Dictionary with validation errors and warnings
        """
        config = self.get_config()
        errors = []
        warnings = []

        # Validate paths
        for path_attr in [
            "input_directory",
            "output_directory",
            "log_directory",
            "temp_directory",
        ]:
            path_value = getattr(config, path_attr)
            if not path_value:
                errors.append(f"Missing {path_attr}")
            else:
                # Check if path is absolute or create it relative to config
                if not os.path.isabs(path_value) and self.config_file:
                    full_path = self.config_file.parent / path_value
                else:
                    full_path = Path(path_value)

                validation = FileValidator.validate_directory(
                    full_path, create_if_missing=True
                )
                if not validation["valid"]:
                    errors.extend(validation["errors"])

        # Validate FFmpeg
        ffmpeg_path = shutil.which(config.ffmpeg.executable_path)
        if not ffmpeg_path:
            errors.append(f"FFmpeg not found: {config.ffmpeg.executable_path}")

        # Validate performance settings
        if config.performance.max_concurrent_jobs < 1:
            errors.append("max_concurrent_jobs must be >= 1")

        if config.performance.memory_limit_gb < 0.5:
            warnings.append("memory_limit_gb is very low")

        # Validate hospitals
        if not config.hospitals:
            warnings.append("No hospital configurations defined")

        for hospital_name, hospital_config in config.hospitals.items():
            # Validate resolution format
            if not re.match(r"^\d+x\d+$", hospital_config.target_resolution):
                errors.append(
                    f"Invalid resolution format for {hospital_name}: {hospital_config.target_resolution}"
                )

            # Validate FPS
            if hospital_config.target_fps < 1 or hospital_config.target_fps > 120:
                warnings.append(
                    f"Unusual FPS for {hospital_name}: {hospital_config.target_fps}"
                )

        return {"errors": errors, "warnings": warnings, "valid": len(errors) == 0}

    def add_config_watcher(self, callback: callable):
        """
        Add a configuration change watcher

        Args:
            callback: Function to call when configuration changes
        """
        self._watchers.append(callback)

    def remove_config_watcher(self, callback: callable):
        """
        Remove a configuration change watcher

        Args:
            callback: Function to remove from watchers
        """
        if callback in self._watchers:
            self._watchers.remove(callback)

    def _dict_to_config(self, data: Dict[str, Any]) -> ProcessingConfig:
        """Convert dictionary to ProcessingConfig"""
        # Handle nested configurations
        if "ffmpeg" in data:
            data["ffmpeg"] = FFmpegConfig(**data["ffmpeg"])

        if "quality_control" in data:
            data["quality_control"] = QualityControlConfig(**data["quality_control"])

        if "security" in data:
            data["security"] = SecurityConfig(**data["security"])

        if "performance" in data:
            data["performance"] = PerformanceConfig(**data["performance"])

        if "backup" in data:
            data["backup"] = BackupConfig(**data["backup"])

        if "notifications" in data:
            data["notifications"] = NotificationConfig(**data["notifications"])

        # Handle hospitals
        if "hospitals" in data:
            hospitals = {}
            for name, hospital_data in data["hospitals"].items():
                hospital_data["type"] = HospitalType(
                    hospital_data.get("type", "general")
                )
                hospitals[name] = HospitalConfig(**hospital_data)
            data["hospitals"] = hospitals

        # Handle enums
        if "mode" in data:
            data["mode"] = ProcessingMode(data["mode"])

        return ProcessingConfig(**data)

    def _create_default_hospitals(self):
        """Create default hospital configurations"""
        # Farabi Hospital (matches compress_video.sh)
        farabi_config = HospitalConfig(
            name="farabi",
            type=HospitalType.FARABI,
            target_resolution="720x480",
            target_fps=30,
            target_bitrate="1M",
            enable_cropping=True,
            crop_filter="crop=720:480:100:50",
            enable_blur=True,
            blur_filter="boxblur=10:1",
            video_codec="libx265",
            audio_codec="aac",
            container_format="mp4",
        )

        # Noor Hospital (matches compress_videos.bat)
        noor_config = HospitalConfig(
            name="noor",
            type=HospitalType.NOOR,
            target_resolution="1920x1080",
            target_fps=60,
            target_bitrate="2M",
            enable_cropping=False,
            enable_blur=False,
            video_codec="libx265",
            audio_codec="aac",
            container_format="mp4",
            custom_filters=["-movflags", "+faststart"],
        )

        # General hospital
        general_config = self._create_general_hospital_config()

        self._config.hospitals = {
            "farabi": farabi_config,
            "noor": noor_config,
            "general": general_config,
        }
        self._config.default_hospital = "general"

    def _create_general_hospital_config(self) -> HospitalConfig:
        """Create general hospital configuration"""
        return HospitalConfig(
            name="general",
            type=HospitalType.GENERAL,
            target_resolution="1920x1080",
            target_fps=30,
            target_bitrate="2M",
            enable_cropping=False,
            enable_blur=False,
            video_codec="libx264",
            audio_codec="aac",
            container_format="mp4",
        )


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Get global configuration manager

    Args:
        config_file: Configuration file path (only used on first call)

    Returns:
        Configuration manager instance
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_file)

    return _config_manager


def load_config(config_file: Optional[Union[str, Path]] = None) -> ProcessingConfig:
    """
    Load configuration (convenience function)

    Args:
        config_file: Configuration file path

    Returns:
        Loaded configuration
    """
    return get_config_manager(config_file).load_config(config_file)


def get_current_config() -> ProcessingConfig:
    """
    Get current configuration (convenience function)

    Returns:
        Current configuration
    """
    return get_config_manager().get_config()


def create_sample_config(output_file: Union[str, Path]) -> bool:
    """
    Create a sample configuration file

    Args:
        output_file: Path to output configuration file

    Returns:
        True if successful
    """
    try:
        config = ProcessingConfig()

        # Create default hospitals
        manager = ConfigManager()
        manager._config = config
        manager._create_default_hospitals()

        # Save sample config
        return manager.save_config(output_file)

    except Exception as e:
        logger.error(f"Failed to create sample configuration: {e}")
        return False


# Import regex for validation
import re
