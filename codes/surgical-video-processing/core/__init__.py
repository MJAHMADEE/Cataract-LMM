"""
Core Video Processing Module

This module contains the fundamental classes and interfaces for surgical video processing.
It provides the base architecture for video handling, processing pipeline coordination,
and error management throughout the surgical video processing framework.

Classes:
    BaseVideoProcessor: Abstract base class for all video processors
    VideoMetadata: Data class for video metadata management
    ProcessingResult: Container for processing operation results
    ProcessingConfig: Configuration management for processing operations
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enumeration of possible processing statuses"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VideoFormat(Enum):
    """Supported video formats"""

    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"


class HospitalSource(Enum):
    """Hospital sources for video data"""

    FARABI = "farabi"
    NOOR = "noor"
    UNKNOWN = "unknown"


@dataclass
class VideoMetadata:
    """
    Comprehensive metadata container for surgical videos

    Attributes:
        file_path: Path to the video file
        original_filename: Original filename before processing
        file_size_bytes: File size in bytes
        duration_seconds: Video duration in seconds
        resolution: Video resolution as (width, height)
        fps: Frames per second
        codec: Video codec information
        recording_date: Date of recording
        surgeon_level: Experience level (resident, fellow, attending)
        procedure_complete: Whether procedure was completed
        quality_score: Technical quality score (0-100)
        has_audio: Whether video contains audio track
        processing_history: List of processing operations applied
        metadata_extracted: Timestamp of metadata extraction
    """

    file_path: Path
    original_filename: str
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    codec: str = ""
    recording_date: Optional[datetime] = None
    surgeon_level: Optional[str] = None
    procedure_complete: bool = True
    quality_score: float = 0.0
    has_audio: bool = False
    processing_history: List[str] = field(default_factory=list)
    metadata_extracted: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization"""
        return {
            "file_path": str(self.file_path),
            "original_filename": self.original_filename,
            "file_size_bytes": self.file_size_bytes,
            "duration_seconds": self.duration_seconds,
            "resolution": self.resolution,
            "fps": self.fps,
            "codec": self.codec,
            "recording_date": (
                self.recording_date.isoformat() if self.recording_date else None
            ),
            "surgeon_level": self.surgeon_level,
            "procedure_complete": self.procedure_complete,
            "quality_score": self.quality_score,
            "has_audio": self.has_audio,
            "processing_history": self.processing_history,
            "metadata_extracted": self.metadata_extracted.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMetadata":
        """Create VideoMetadata instance from dictionary"""
        # Convert string dates back to datetime objects
        if data.get("recording_date"):
            data["recording_date"] = datetime.fromisoformat(data["recording_date"])
        if data.get("metadata_extracted"):
            data["metadata_extracted"] = datetime.fromisoformat(
                data["metadata_extracted"]
            )

        return cls(**data)


@dataclass
class ProcessingResult:
    """
    Container for processing operation results

    Attributes:
        status: Processing status
        input_path: Path to input file
        output_path: Path to output file (if successful)
        metadata: Video metadata
        processing_time: Time taken for processing in seconds
        error_message: Error description (if failed)
        warnings: List of warning messages
        metrics: Dictionary of processing metrics
        timestamp: When processing completed
    """

    status: ProcessingStatus
    input_path: Path
    output_path: Optional[Path] = None
    metadata: Optional[VideoMetadata] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "status": self.status.value,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path) if self.output_path else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProcessingConfig:
    """
    Configuration container for processing operations

    Attributes:
        quality_check_enabled: Enable quality control checks
        deidentification_enabled: Enable de-identification
        processing_enabled: Enable video processing
        metadata_extraction_enabled: Enable metadata extraction
        parallel_processing: Enable parallel processing
        max_workers: Maximum number of worker threads
        output_format: Desired output video format
        quality_threshold: Minimum quality score threshold
        processing_crf: Constant Rate Factor for processing (0-51)
        target_resolution: Target resolution for standardization
        target_fps: Target frame rate for standardization
        preserve_audio: Whether to preserve audio tracks
        backup_originals: Whether to backup original files
        log_level: Logging level
    """

    quality_check_enabled: bool = True
    deidentification_enabled: bool = True
    processing_enabled: bool = True
    metadata_extraction_enabled: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    output_format: VideoFormat = VideoFormat.MP4
    quality_threshold: float = 50.0
    processing_crf: int = 23
    target_resolution: Optional[Tuple[int, int]] = None
    target_fps: Optional[float] = None
    preserve_audio: bool = False
    backup_originals: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "quality_check_enabled": self.quality_check_enabled,
            "deidentification_enabled": self.deidentification_enabled,
            "processing_enabled": self.processing_enabled,
            "metadata_extraction_enabled": self.metadata_extraction_enabled,
            "parallel_processing": self.parallel_processing,
            "max_workers": self.max_workers,
            "output_format": self.output_format.value,
            "quality_threshold": self.quality_threshold,
            "processing_crf": self.processing_crf,
            "target_resolution": self.target_resolution,
            "target_fps": self.target_fps,
            "preserve_audio": self.preserve_audio,
            "backup_originals": self.backup_originals,
            "log_level": self.log_level,
        }


class BaseVideoProcessor(ABC):
    """
    Abstract base class for all video processors

    This class defines the common interface for video processing operations
    in the surgical video processing framework. All specific processors
    (processing, de-identification, quality control, etc.) should inherit
    from this class.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize the base video processor

        Args:
            config: Processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, config.log_level))

    @abstractmethod
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Process a single video file

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Optional video metadata

        Returns:
            ProcessingResult containing operation results
        """
        pass

    def validate_input(self, input_path: Union[str, Path]) -> bool:
        """
        Validate input file exists and is accessible

        Args:
            input_path: Path to input file

        Returns:
            True if file is valid, False otherwise
        """
        path = Path(input_path)
        if not path.exists():
            self.logger.error(f"Input file does not exist: {input_path}")
            return False

        if not path.is_file():
            self.logger.error(f"Input path is not a file: {input_path}")
            return False

        return True

    def prepare_output_directory(self, output_path: Union[str, Path]) -> bool:
        """
        Ensure output directory exists

        Args:
            output_path: Path for output file

        Returns:
            True if directory was created/exists, False otherwise
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            return False

    def log_processing_start(self, input_path: Union[str, Path]):
        """Log the start of processing operation"""
        self.logger.info(f"Starting {self.__class__.__name__} processing: {input_path}")

    def log_processing_complete(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        processing_time: float,
    ):
        """Log the completion of processing operation"""
        self.logger.info(
            f"Completed {self.__class__.__name__} processing: "
            f"{input_path} -> {output_path} ({processing_time:.2f}s)"
        )

    def log_processing_error(self, input_path: Union[str, Path], error: str):
        """Log processing errors"""
        self.logger.error(
            f"Failed {self.__class__.__name__} processing: {input_path} - {error}"
        )


class ProcessingPipeline:
    """
    Orchestrates multiple processing steps in sequence

    This class manages the execution of multiple video processing operations
    in a coordinated pipeline, handling dependencies, error recovery, and
    result aggregation.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize processing pipeline

        Args:
            config: Processing configuration
        """
        self.config = config
        self.processors: List[BaseVideoProcessor] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, config.log_level))

    def add_processor(self, processor: BaseVideoProcessor):
        """
        Add a processor to the pipeline

        Args:
            processor: Video processor instance
        """
        self.processors.append(processor)
        self.logger.info(f"Added processor: {processor.__class__.__name__}")

    def process(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> List[ProcessingResult]:
        """
        Execute the complete processing pipeline

        Args:
            input_path: Path to input video file
            output_dir: Directory for output files
            metadata: Optional video metadata

        Returns:
            List of ProcessingResult objects from each processor
        """
        results = []
        current_input = Path(input_path)
        output_dir = Path(output_dir)

        self.logger.info(f"Starting pipeline processing: {input_path}")

        for i, processor in enumerate(self.processors):
            # Generate output path for this processor
            output_filename = f"step_{i:02d}_{processor.__class__.__name__.lower()}_{current_input.name}"
            output_path = output_dir / output_filename

            # Process with current processor
            result = processor.process(current_input, output_path, metadata)
            results.append(result)

            # If processing failed, stop pipeline
            if result.status == ProcessingStatus.FAILED:
                self.logger.error(
                    f"Pipeline stopped due to failure in {processor.__class__.__name__}"
                )
                break

            # Update input for next processor
            if result.output_path and result.output_path.exists():
                current_input = result.output_path
                metadata = result.metadata

        self.logger.info(
            f"Pipeline processing completed: {len(results)} steps executed"
        )
        return results
