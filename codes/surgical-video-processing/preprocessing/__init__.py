"""
Video Preprocessing Module for Surgical Videos

This module handles video preprocessing operations including standardization,
format conversion, resolution adjustment, frame rate normalization, and
enhancement techniques specifically designed for phacoemulsification
cataract surgery videos.

Classes:
    VideoStandardizer: Standardizes video format, resolution, and frame rate
    VideoEnhancer: Applies enhancement techniques for surgical video quality
    FormatConverter: Handles video format conversion and codec optimization
    ResolutionNormalizer: Normalizes video resolution across different sources
    FrameExtractor: Extracts frames for analysis and quality assessment
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..core import (
    BaseVideoProcessor,
    ProcessingConfig,
    ProcessingResult,
    ProcessingStatus,
    VideoMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """
    Configuration for video preprocessing operations

    Attributes:
        standardize_resolution: Whether to standardize video resolution
        target_width: Target width for standardization
        target_height: Target height for standardization
        standardize_fps: Whether to standardize frame rate
        target_fps: Target frame rate for standardization
        enhance_contrast: Whether to apply contrast enhancement
        reduce_noise: Whether to apply noise reduction
        stabilize_video: Whether to apply video stabilization
        crop_margins: Margins to crop (top, right, bottom, left)
        brightness_adjustment: Brightness adjustment factor (-100 to 100)
        contrast_adjustment: Contrast adjustment factor (0.5 to 2.0)
        saturation_adjustment: Saturation adjustment factor (0.0 to 2.0)
        gamma_correction: Gamma correction factor (0.1 to 3.0)
        sharpen_kernel_size: Kernel size for sharpening (odd numbers only)
        noise_reduction_strength: Noise reduction strength (1-10)
    """

    standardize_resolution: bool = True
    target_width: int = 1280
    target_height: int = 720
    standardize_fps: bool = True
    target_fps: float = 30.0
    enhance_contrast: bool = True
    reduce_noise: bool = True
    stabilize_video: bool = False
    crop_margins: Tuple[int, int, int, int] = (0, 0, 0, 0)  # top, right, bottom, left
    brightness_adjustment: int = 0
    contrast_adjustment: float = 1.0
    saturation_adjustment: float = 1.0
    gamma_correction: float = 1.0
    sharpen_kernel_size: int = 3
    noise_reduction_strength: int = 3


class VideoStandardizer(BaseVideoProcessor):
    """
    Standardizes video format, resolution, and frame rate

    This processor ensures consistent video parameters across different
    source hospitals and recording equipment, making downstream processing
    more reliable and efficient.
    """

    def __init__(
        self, config: ProcessingConfig, preprocessing_config: PreprocessingConfig
    ):
        """
        Initialize video standardizer

        Args:
            config: General processing configuration
            preprocessing_config: Preprocessing-specific configuration
        """
        super().__init__(config)
        self.preprocessing_config = preprocessing_config

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Standardize video format, resolution, and frame rate

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Video metadata

        Returns:
            ProcessingResult with standardization results
        """
        start_time = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not self.validate_input(input_path):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message="Input validation failed",
            )

        if not self.prepare_output_directory(output_path):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message="Output directory preparation failed",
            )

        self.log_processing_start(input_path)

        try:
            # Extract current video properties
            video_info = self._get_video_info(input_path)

            # Build FFmpeg command for standardization
            ffmpeg_cmd = self._build_standardization_command(
                input_path, output_path, video_info, metadata
            )

            # Execute FFmpeg command
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                error_msg = f"FFmpeg failed: {result.stderr}"
                self.log_processing_error(input_path, error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    error_message=error_msg,
                )

            # Update metadata with new properties
            if metadata:
                metadata.resolution = (
                    self.preprocessing_config.target_width,
                    self.preprocessing_config.target_height,
                )
                metadata.fps = self.preprocessing_config.target_fps
                metadata.processing_history.append(
                    f"Standardized by {self.__class__.__name__}"
                )

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_path, processing_time)

            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                input_path=input_path,
                output_path=output_path,
                metadata=metadata,
                processing_time=processing_time,
                metrics={
                    "original_resolution": f"{video_info.get('width', 'unknown')}x{video_info.get('height', 'unknown')}",
                    "target_resolution": f"{self.preprocessing_config.target_width}x{self.preprocessing_config.target_height}",
                    "original_fps": video_info.get("fps", "unknown"),
                    "target_fps": self.preprocessing_config.target_fps,
                },
            )

        except Exception as e:
            error_msg = f"Standardization failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Extract video information using FFprobe"""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)
        video_stream = next(
            (s for s in data["streams"] if s["codec_type"] == "video"), {}
        )

        # Safe evaluation of frame rate fraction
        def safe_eval_fraction(fraction_str):
            try:
                if "/" in fraction_str:
                    num, denom = fraction_str.split("/", 1)
                    return float(num) / float(denom) if float(denom) != 0 else 0.0
                else:
                    return float(fraction_str)
            except (ValueError, ZeroDivisionError):
                return 0.0

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": safe_eval_fraction(video_stream.get("r_frame_rate", "0/1")),
            "codec": video_stream.get("codec_name", ""),
            "duration": float(data.get("format", {}).get("duration", 0)),
        }

    def _build_standardization_command(
        self,
        input_path: Path,
        output_path: Path,
        video_info: Dict[str, Any],
        metadata: Optional[VideoMetadata],
    ) -> List[str]:
        """Build FFmpeg command for video standardization"""
        cmd = ["ffmpeg", "-i", str(input_path)]

        # Video filters
        filters = []

        # Resolution standardization
        if self.preprocessing_config.standardize_resolution:
            filters.append(
                f"scale={self.preprocessing_config.target_width}:{self.preprocessing_config.target_height}"
            )

        # Frame rate standardization
        if self.preprocessing_config.standardize_fps:
            filters.append(f"fps={self.preprocessing_config.target_fps}")

        # Cropping if specified
        if any(self.preprocessing_config.crop_margins):
            top, right, bottom, left = self.preprocessing_config.crop_margins
            crop_width = video_info.get("width", 1920) - left - right
            crop_height = video_info.get("height", 1080) - top - bottom
            filters.append(f"crop={crop_width}:{crop_height}:{left}:{top}")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        # Codec and quality settings
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(self.config.compression_crf),
                "-pix_fmt",
                "yuv420p",
            ]
        )

        # Audio handling
        if self.config.preserve_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-an"])  # Remove audio

        # Output settings
        cmd.extend(["-movflags", "+faststart", str(output_path)])

        return cmd


class VideoEnhancer(BaseVideoProcessor):
    """
    Applies enhancement techniques for surgical video quality

    This processor improves video quality through contrast enhancement,
    noise reduction, sharpening, and other techniques specifically
    optimized for surgical microscopy videos.
    """

    def __init__(
        self, config: ProcessingConfig, preprocessing_config: PreprocessingConfig
    ):
        """
        Initialize video enhancer

        Args:
            config: General processing configuration
            preprocessing_config: Preprocessing-specific configuration
        """
        super().__init__(config)
        self.preprocessing_config = preprocessing_config

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Apply video enhancement techniques

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Video metadata

        Returns:
            ProcessingResult with enhancement results
        """
        start_time = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not self.validate_input(input_path):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message="Input validation failed",
            )

        if not self.prepare_output_directory(output_path):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message="Output directory preparation failed",
            )

        self.log_processing_start(input_path)

        try:
            # Build FFmpeg command for enhancement
            ffmpeg_cmd = self._build_enhancement_command(input_path, output_path)

            # Execute FFmpeg command
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                error_msg = f"FFmpeg enhancement failed: {result.stderr}"
                self.log_processing_error(input_path, error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    error_message=error_msg,
                )

            # Update metadata
            if metadata:
                metadata.processing_history.append(
                    f"Enhanced by {self.__class__.__name__}"
                )

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_path, processing_time)

            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                input_path=input_path,
                output_path=output_path,
                metadata=metadata,
                processing_time=processing_time,
                metrics={
                    "brightness_adjustment": self.preprocessing_config.brightness_adjustment,
                    "contrast_adjustment": self.preprocessing_config.contrast_adjustment,
                    "noise_reduction": self.preprocessing_config.reduce_noise,
                    "contrast_enhancement": self.preprocessing_config.enhance_contrast,
                },
            )

        except Exception as e:
            error_msg = f"Enhancement failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _build_enhancement_command(
        self, input_path: Path, output_path: Path
    ) -> List[str]:
        """Build FFmpeg command for video enhancement"""
        cmd = ["ffmpeg", "-i", str(input_path)]

        # Video filters for enhancement
        filters = []

        # Brightness and contrast adjustment
        if (
            self.preprocessing_config.brightness_adjustment != 0
            or self.preprocessing_config.contrast_adjustment != 1.0
        ):
            brightness = self.preprocessing_config.brightness_adjustment / 100.0
            contrast = self.preprocessing_config.contrast_adjustment
            filters.append(f"eq=brightness={brightness}:contrast={contrast}")

        # Gamma correction
        if self.preprocessing_config.gamma_correction != 1.0:
            filters.append(f"eq=gamma={self.preprocessing_config.gamma_correction}")

        # Saturation adjustment
        if self.preprocessing_config.saturation_adjustment != 1.0:
            filters.append(
                f"eq=saturation={self.preprocessing_config.saturation_adjustment}"
            )

        # Noise reduction
        if self.preprocessing_config.reduce_noise:
            strength = self.preprocessing_config.noise_reduction_strength
            filters.append(f"hqdn3d={strength}:{strength}:{strength}:{strength}")

        # Contrast enhancement using CLAHE
        if self.preprocessing_config.enhance_contrast:
            filters.append("eq=contrast=1.2")

        # Sharpening
        if self.preprocessing_config.sharpen_kernel_size > 1:
            filters.append("unsharp=5:5:1.0:5:5:0.0")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        # Codec settings
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(self.config.compression_crf),
            ]
        )

        # Audio handling
        if self.config.preserve_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-an"])

        cmd.extend(["-movflags", "+faststart", str(output_path)])

        return cmd


class FrameExtractor(BaseVideoProcessor):
    """
    Extracts frames from videos for analysis and quality assessment

    This processor extracts representative frames at specified intervals
    for quality control, thumbnail generation, and sample analysis.
    """

    def __init__(
        self,
        config: ProcessingConfig,
        extraction_interval: float = 5.0,
        max_frames: int = 100,
        output_format: str = "jpg",
    ):
        """
        Initialize frame extractor

        Args:
            config: Processing configuration
            extraction_interval: Interval between extracted frames (seconds)
            max_frames: Maximum number of frames to extract
            output_format: Output image format (jpg, png, bmp)
        """
        super().__init__(config)
        self.extraction_interval = extraction_interval
        self.max_frames = max_frames
        self.output_format = output_format

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Extract frames from video

        Args:
            input_path: Path to input video file
            output_path: Directory path for extracted frames
            metadata: Video metadata

        Returns:
            ProcessingResult with extraction results
        """
        start_time = time.time()
        input_path = Path(input_path)
        output_dir = Path(output_path)

        if not self.validate_input(input_path):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message="Input validation failed",
            )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log_processing_start(input_path)

        try:
            # Open video
            cap = cv2.VideoCapture(str(input_path))

            if not cap.isOpened():
                raise Exception("Failed to open video file")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Calculate frame extraction parameters
            frame_interval = int(fps * self.extraction_interval) if fps > 0 else 30
            frames_to_extract = min(self.max_frames, total_frames // frame_interval)

            extracted_frames = []
            frame_count = 0

            # Extract frames
            for i in range(frames_to_extract):
                frame_number = i * frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame
                timestamp = frame_number / fps if fps > 0 else i
                filename = f"frame_{i:04d}_t{timestamp:.2f}s.{self.output_format}"
                frame_path = output_dir / filename

                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append(str(frame_path))
                frame_count += 1

            cap.release()

            # Update metadata
            if metadata:
                metadata.processing_history.append(
                    f"Frames extracted by {self.__class__.__name__}"
                )

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_dir, processing_time)

            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                input_path=input_path,
                output_path=output_dir,
                metadata=metadata,
                processing_time=processing_time,
                metrics={
                    "frames_extracted": frame_count,
                    "extraction_interval": self.extraction_interval,
                    "video_duration": duration,
                    "video_fps": fps,
                    "extracted_files": extracted_frames,
                },
            )

        except Exception as e:
            error_msg = f"Frame extraction failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )


class HospitalSpecificPreprocessor(BaseVideoProcessor):
    """
    Hospital-specific preprocessing for different recording equipment

    This processor applies hospital-specific adjustments based on the
    recording equipment used at Farabi Eye Hospital (Haag-Streit) and
    Noor Eye Hospital (ZEISS ARTEVO).
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize hospital-specific preprocessor

        Args:
            config: Processing configuration
        """
        super().__init__(config)

        # Reference script-based configurations
        self.processing_configs = {
            "process_video": {
                "crop_filter": "crop=268:58:6:422",
                "blur_filter": "avgblur=10",
                "overlay_position": "6:422",
                "preserve_audio": True,
                "noise_reduction": True,
                "brightness_adjustment": 0,
            },
            "process_videos": {
                "crop_filter": None,
                "blur_filter": None,
                "overlay_position": None,
                "preserve_audio": False,
                "noise_reduction": False,
                "brightness_adjustment": 0,
            },
            "general": {
                "crop_filter": None,
                "blur_filter": None,
                "overlay_position": None,
                "preserve_audio": True,
                "noise_reduction": True,
                "brightness_adjustment": 0,
            },
        }

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        processing_method: str = "general",
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Apply reference script-based preprocessing

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            processing_method: Processing method ('process_video', 'process_videos', 'general')
            metadata: Video metadata

        Returns:
            ProcessingResult with preprocessing results
        """
        start_time = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not self.validate_input(input_path):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message="Input validation failed",
            )

        if not self.prepare_output_directory(output_path):
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message="Output directory preparation failed",
            )

        self.log_processing_start(input_path)

        try:
            # Determine processing method from parameters or filename
            method = self._determine_processing_method(
                input_path, processing_method, metadata
            )

            if method not in self.processing_configs:
                self.logger.warning(
                    f"Unknown processing method {method}, using general settings"
                )
                method = "general"

            # Get reference-based configuration
            config = self.processing_configs[method]

            # Build FFmpeg command with reference script settings
            ffmpeg_cmd = self._build_reference_based_command(
                input_path, output_path, config
            )

            # Execute FFmpeg command
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                error_msg = f"Hospital-specific preprocessing failed: {result.stderr}"
                self.log_processing_error(input_path, error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    error_message=error_msg,
                )

            # Update metadata
            if metadata:
                metadata.processing_history.append(
                    f"Reference-based preprocessing ({method}) by {self.__class__.__name__}"
                )

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_path, processing_time)

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                input_path=input_path,
                output_path=output_path,
                metadata=metadata,
                processing_time=processing_time,
                metrics={"processing_method": method, "applied_config": config},
            )

        except Exception as e:
            error_msg = f"Reference-based preprocessing failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _determine_processing_method(
        self, input_path: Path, method: str, metadata: Optional[VideoMetadata]
    ) -> str:
        """Determine processing method from parameters or filename patterns"""
        # Use specified method if provided
        if method and method in self.processing_configs:
            return method

        # Check filename patterns for hints
        filename_lower = input_path.name.lower()

        if any(keyword in filename_lower for keyword in ["crop", "blur", "overlay"]):
            return "process_video"
        elif any(keyword in filename_lower for keyword in ["simple", "basic"]):
            return "process_videos"

        return "general"

    def _build_reference_based_command(
        self, input_path: Path, output_path: Path, hospital_config: Dict[str, Any]
    ) -> List[str]:
        """Build FFmpeg command with hospital-specific settings"""
        cmd = ["ffmpeg", "-i", str(input_path)]

        filters = []

        # Apply cropping if specified
        if any(hospital_config["crop_margins"]):
            top, right, bottom, left = hospital_config["crop_margins"]
            width, height = hospital_config["source_resolution"]
            crop_width = width - left - right
            crop_height = height - top - bottom
            filters.append(f"crop={crop_width}:{crop_height}:{left}:{top}")

        # Apply brightness and contrast adjustments
        if (
            hospital_config["brightness_adjustment"] != 0
            or hospital_config["contrast_adjustment"] != 1.0
        ):
            brightness = hospital_config["brightness_adjustment"] / 100.0
            contrast = hospital_config["contrast_adjustment"]
            filters.append(f"eq=brightness={brightness}:contrast={contrast}")

        # Apply noise reduction if enabled
        if hospital_config["noise_reduction"]:
            filters.append("hqdn3d=3:3:3:3")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        # Codec settings
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(self.config.compression_crf),
            ]
        )

        # Audio handling
        if self.config.preserve_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-an"])

        cmd.extend(["-movflags", "+faststart", str(output_path)])

        return cmd
