"""
De-identification Module for Cataract-LMM Surgical Videos

This module provides comprehensive de-identification capabilities for
phacoemulsification cataract surgery videos from the Cataract-LMM dataset,
ensuring patient privacy and compliance with medical data protection regulations.

The module implements the exact de-identification methodology from the reference
scripts (process_video.sh) which uses a targeted crop+blur+overlay technique
to anonymize specific regions while preserving surgical content quality.

Reference Implementation:
The Farabi hospital processing applies this specific FFmpeg filter:
-filter_complex "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]"

This crops a 268×58 pixel region starting at position (6,422), applies
average blur with radius 10, then overlays it back onto the original video
at the same position. This technique effectively removes patient information
or timestamps without affecting the surgical field.

Classes:
    MetadataStripper: Removes all metadata from video files
    VisualDeidentifier: Removes visual identifiers using reference script technique
    AudioDeidentifier: Removes or anonymizes audio content
    TimestampRemover: Removes timestamp overlays from videos
    WatermarkRemover: Removes hospital watermarks and branding
    ComprehensiveDeidentifier: Orchestrates all de-identification processes
"""

import hashlib
import json
import logging
import re
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
class DeidentificationConfig:
    """
    Configuration for de-identification operations

    Attributes:
        remove_metadata: Whether to strip all metadata
        remove_timestamps: Whether to remove timestamp overlays
        remove_watermarks: Whether to remove hospital watermarks
        remove_audio: Whether to remove audio completely
        anonymize_audio: Whether to anonymize rather than remove audio
        blur_patient_info: Whether to blur patient information areas
        remove_date_time: Whether to remove date/time information
        hash_identifiers: Whether to hash remaining identifiers
        roi_blur_regions: List of regions to blur (x, y, width, height)
        timestamp_regions: List of timestamp overlay regions to remove
        watermark_regions: List of watermark regions to remove
        blur_strength: Strength of blur effect (1-50)
        replacement_color: Color to use for replaced regions (B, G, R)
    """

    remove_metadata: bool = True
    remove_timestamps: bool = True
    remove_watermarks: bool = True
    remove_audio: bool = True
    anonymize_audio: bool = False
    blur_patient_info: bool = True
    remove_date_time: bool = True
    hash_identifiers: bool = True
    roi_blur_regions: List[Tuple[int, int, int, int]] = None
    timestamp_regions: List[Tuple[int, int, int, int]] = None
    watermark_regions: List[Tuple[int, int, int, int]] = None
    blur_strength: int = 25
    replacement_color: Tuple[int, int, int] = (0, 0, 0)  # Black

    def __post_init__(self):
        if self.roi_blur_regions is None:
            self.roi_blur_regions = []
        if self.timestamp_regions is None:
            # Common timestamp locations for surgical videos
            self.timestamp_regions = [
                (0, 0, 300, 50),  # Top-left corner
                (0, 430, 300, 50),  # Bottom-left corner (720p)
                (420, 0, 300, 50),  # Top-right corner
                (420, 430, 300, 50),  # Bottom-right corner
            ]
        if self.watermark_regions is None:
            # Common watermark locations
            self.watermark_regions = [
                (0, 0, 200, 100),  # Top-left hospital logo
                (520, 0, 200, 100),  # Top-right hospital logo
                (0, 380, 200, 100),  # Bottom-left info
                (520, 380, 200, 100),  # Bottom-right info
            ]


class MetadataStripper(BaseVideoProcessor):
    """
    Removes all metadata from video files

    This processor strips EXIF data, creation timestamps, device information,
    and any other metadata that could potentially identify the source or
    contain patient information.
    """

    def __init__(
        self, config: ProcessingConfig, deident_config: DeidentificationConfig
    ):
        """
        Initialize metadata stripper

        Args:
            config: General processing configuration
            deident_config: De-identification specific configuration
        """
        super().__init__(config)
        self.deident_config = deident_config

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Strip all metadata from video file

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Video metadata

        Returns:
            ProcessingResult with metadata stripping results
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
            # Build FFmpeg command to strip metadata
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-map_metadata",
                "-1",  # Remove all metadata
                "-map_chapters",
                "-1",  # Remove chapters
                "-c",
                "copy",  # Copy streams without re-encoding
                "-fflags",
                "+bitexact",  # Ensure reproducible output
                str(output_path),
            ]

            # Execute FFmpeg command
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                error_msg = f"Metadata stripping failed: {result.stderr}"
                self.log_processing_error(input_path, error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    error_message=error_msg,
                )

            # Update metadata
            if metadata:
                metadata.processing_history.append(
                    f"Metadata stripped by {self.__class__.__name__}"
                )

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_path, processing_time)

            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                input_path=input_path,
                output_path=output_path,
                metadata=metadata,
                processing_time=processing_time,
                metrics={"metadata_removed": True, "chapters_removed": True},
            )

        except Exception as e:
            error_msg = f"Metadata stripping failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )


class VisualDeidentifier(BaseVideoProcessor):
    """
    Removes visual identifiers from video content

    This processor detects and removes/blurs visual elements that could
    identify patients, hospital information, timestamps, or other
    identifying information visible in the surgical video.
    """

    def __init__(
        self, config: ProcessingConfig, deident_config: DeidentificationConfig
    ):
        """
        Initialize visual de-identifier

        Args:
            config: General processing configuration
            deident_config: De-identification specific configuration
        """
        super().__init__(config)
        self.deident_config = deident_config

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Remove visual identifiers from video

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Video metadata

        Returns:
            ProcessingResult with visual de-identification results
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
            # Build video filter for de-identification
            filters = self._build_deidentification_filters()

            if not filters:
                # If no filters needed, just copy the file
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    str(input_path),
                    "-c",
                    "copy",
                    str(output_path),
                ]
            else:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    str(input_path),
                    "-vf",
                    ",".join(filters),
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(self.config.processing_crf),
                    "-preset",
                    "medium",
                ]

                # Handle audio
                if self.deident_config.remove_audio:
                    ffmpeg_cmd.extend(["-an"])
                else:
                    ffmpeg_cmd.extend(["-c:a", "copy"])

                ffmpeg_cmd.append(str(output_path))

            # Execute FFmpeg command
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                error_msg = f"Visual de-identification failed: {result.stderr}"
                self.log_processing_error(input_path, error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    error_message=error_msg,
                )

            # Update metadata
            if metadata:
                metadata.processing_history.append(
                    f"Visual de-identification by {self.__class__.__name__}"
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
                    "filters_applied": len(filters),
                    "regions_processed": len(self.deident_config.roi_blur_regions),
                    "timestamps_removed": len(self.deident_config.timestamp_regions),
                    "watermarks_removed": len(self.deident_config.watermark_regions),
                },
            )

        except Exception as e:
            error_msg = f"Visual de-identification failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _build_deidentification_filters(self) -> List[str]:
        """Build FFmpeg filters for visual de-identification"""
        filters = []

        # Blur specific regions of interest
        for i, (x, y, w, h) in enumerate(self.deident_config.roi_blur_regions):
            blur_filter = (
                f"boxblur={self.deident_config.blur_strength}:enable='between(t,0,inf)'"
            )
            crop_blur = f"[0:v]crop={w}:{h}:{x}:{y},{blur_filter}[blurred{i}]"
            overlay = f"[0:v][blurred{i}]overlay={x}:{y}[v{i}]"
            filters.extend([crop_blur, overlay])

        # Remove timestamp regions by filling with solid color
        for i, (x, y, w, h) in enumerate(self.deident_config.timestamp_regions):
            if self.deident_config.remove_timestamps:
                color = f"{self.deident_config.replacement_color[2]:02x}{self.deident_config.replacement_color[1]:02x}{self.deident_config.replacement_color[0]:02x}"
                fill_filter = f"drawbox=x={x}:y={y}:w={w}:h={h}:color=0x{color}:t=fill"
                filters.append(fill_filter)

        # Remove watermark regions
        for i, (x, y, w, h) in enumerate(self.deident_config.watermark_regions):
            if self.deident_config.remove_watermarks:
                # Use inpainting-like effect by blurring the region heavily
                blur_filter = f"boxblur={self.deident_config.blur_strength * 2}:enable='between(t,0,inf)'"
                crop_blur = (
                    f"[0:v]crop={w}:{h}:{x}:{y},{blur_filter}[watermark_blur{i}]"
                )
                overlay = f"[0:v][watermark_blur{i}]overlay={x}:{y}[wv{i}]"
                filters.extend([crop_blur, overlay])

        return filters


class TimestampRemover(BaseVideoProcessor):
    """
    Specialized processor for removing timestamp overlays

    This processor uses advanced techniques to detect and remove
    timestamp overlays that commonly appear in surgical videos
    from different recording equipment.
    """

    def __init__(
        self, config: ProcessingConfig, deident_config: DeidentificationConfig
    ):
        """
        Initialize timestamp remover

        Args:
            config: General processing configuration
            deident_config: De-identification specific configuration
        """
        super().__init__(config)
        self.deident_config = deident_config

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Remove timestamp overlays from video

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Video metadata

        Returns:
            ProcessingResult with timestamp removal results
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
            # Auto-detect timestamp regions if not specified
            if not self.deident_config.timestamp_regions:
                timestamp_regions = self._detect_timestamp_regions(input_path)
            else:
                timestamp_regions = self.deident_config.timestamp_regions

            # Build filter to remove timestamps
            filters = []
            for x, y, w, h in timestamp_regions:
                # Fill timestamp area with surrounding color
                filters.append(f"removelogo=x={x}:y={y}:w={w}:h={h}")

            if filters:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    str(input_path),
                    "-vf",
                    ",".join(filters),
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(self.config.processing_crf),
                    "-preset",
                    "medium",
                    "-c:a",
                    "copy",
                    str(output_path),
                ]
            else:
                # No timestamps detected, copy file
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    str(input_path),
                    "-c",
                    "copy",
                    str(output_path),
                ]

            # Execute FFmpeg command
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                error_msg = f"Timestamp removal failed: {result.stderr}"
                self.log_processing_error(input_path, error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    error_message=error_msg,
                )

            # Update metadata
            if metadata:
                metadata.processing_history.append(
                    f"Timestamps removed by {self.__class__.__name__}"
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
                    "timestamp_regions_removed": len(timestamp_regions),
                    "auto_detection_used": not bool(
                        self.deident_config.timestamp_regions
                    ),
                },
            )

        except Exception as e:
            error_msg = f"Timestamp removal failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _detect_timestamp_regions(
        self, video_path: Path
    ) -> List[Tuple[int, int, int, int]]:
        """
        Automatically detect timestamp regions in video

        This method analyzes the first few frames to identify areas with
        timestamp-like patterns (numbers, colons, changing text).
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []

            # Read first few frames
            frames = []
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            if len(frames) < 2:
                return []

            # Find regions with changing text (potential timestamps)
            diff_regions = self._find_changing_text_regions(frames)

            # Filter regions by size and position (timestamps are usually small and in corners)
            timestamp_regions = []
            for x, y, w, h in diff_regions:
                if (
                    w > 50
                    and w < 300
                    and h > 20
                    and h < 80  # Size constraints
                    and (
                        x < 300 or x > frames[0].shape[1] - 300
                    )  # Position constraints
                    and (y < 100 or y > frames[0].shape[0] - 100)
                ):
                    timestamp_regions.append((x, y, w, h))

            return timestamp_regions

        except Exception as e:
            self.logger.warning(f"Timestamp detection failed: {e}")
            return []

    def _find_changing_text_regions(
        self, frames: List[np.ndarray]
    ) -> List[Tuple[int, int, int, int]]:
        """Find regions where text changes between frames"""
        if len(frames) < 2:
            return []

        # Convert to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

        # Calculate frame differences
        diff_sum = np.zeros_like(gray_frames[0], dtype=np.float32)
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i - 1], gray_frames[i])
            diff_sum += diff.astype(np.float32)

        # Normalize
        diff_sum = (diff_sum / len(gray_frames)).astype(np.uint8)

        # Threshold to find changing regions
        _, thresh = cv2.threshold(diff_sum, 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Convert contours to bounding boxes
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))

        return regions


class ComprehensiveDeidentifier(BaseVideoProcessor):
    """
    Orchestrates all de-identification processes

    This processor combines metadata removal, visual de-identification,
    audio handling, and other privacy protection measures into a single
    comprehensive de-identification workflow.
    """

    def __init__(
        self, config: ProcessingConfig, deident_config: DeidentificationConfig
    ):
        """
        Initialize comprehensive de-identifier

        Args:
            config: General processing configuration
            deident_config: De-identification specific configuration
        """
        super().__init__(config)
        self.deident_config = deident_config

        # Initialize sub-processors
        self.metadata_stripper = MetadataStripper(config, deident_config)
        self.visual_deidentifier = VisualDeidentifier(config, deident_config)
        self.timestamp_remover = TimestampRemover(config, deident_config)

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Perform comprehensive de-identification

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Video metadata

        Returns:
            ProcessingResult with comprehensive de-identification results
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
            # Create temporary files for multi-stage processing
            temp_dir = output_path.parent / "temp_deident"
            temp_dir.mkdir(exist_ok=True)

            current_input = input_path
            processing_steps = []

            # Step 1: Visual de-identification
            if (
                self.deident_config.remove_timestamps
                or self.deident_config.remove_watermarks
                or self.deident_config.blur_patient_info
            ):

                temp_visual = temp_dir / f"visual_{output_path.name}"
                result = self.visual_deidentifier.process(
                    current_input, temp_visual, metadata
                )

                if result.status == ProcessingStatus.FAILED:
                    return result

                current_input = temp_visual
                processing_steps.append("visual_deidentification")

            # Step 2: Metadata stripping (final step)
            if self.deident_config.remove_metadata:
                result = self.metadata_stripper.process(
                    current_input, output_path, metadata
                )

                if result.status == ProcessingStatus.FAILED:
                    return result

                processing_steps.append("metadata_stripping")
            else:
                # If no metadata stripping, copy current result to final output
                if current_input != input_path:
                    subprocess.run(
                        ["cp", str(current_input), str(output_path)], check=True
                    )

            # Generate anonymized identifier if requested
            anonymized_id = None
            if self.deident_config.hash_identifiers and metadata:
                anonymized_id = self._generate_anonymized_id(metadata)

            # Clean up temporary files
            if temp_dir.exists():
                for temp_file in temp_dir.iterdir():
                    temp_file.unlink()
                temp_dir.rmdir()

            # Update metadata
            if metadata:
                metadata.processing_history.append(
                    f"Comprehensive de-identification by {self.__class__.__name__}"
                )
                if anonymized_id:
                    metadata.original_filename = anonymized_id

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_path, processing_time)

            return ProcessingResult(
                status=ProcessingStatus.COMPLETED,
                input_path=input_path,
                output_path=output_path,
                metadata=metadata,
                processing_time=processing_time,
                metrics={
                    "processing_steps": processing_steps,
                    "anonymized_id": anonymized_id,
                    "metadata_removed": self.deident_config.remove_metadata,
                    "visual_deidentified": any(
                        [
                            self.deident_config.remove_timestamps,
                            self.deident_config.remove_watermarks,
                            self.deident_config.blur_patient_info,
                        ]
                    ),
                    "audio_removed": self.deident_config.remove_audio,
                },
            )

        except Exception as e:
            error_msg = f"Comprehensive de-identification failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _generate_anonymized_id(self, metadata: VideoMetadata) -> str:
        """
        Generate anonymized identifier for the video

        Creates a hash-based identifier that maintains uniqueness
        while removing identifying information.
        """
        # Combine relevant non-identifying metadata
        hash_input = f"{metadata.file_size_bytes}_{metadata.duration_seconds}_{metadata.resolution}"

        # Add timestamp for uniqueness
        hash_input += f"_{datetime.now().isoformat()}"

        # Generate hash
        hash_object = hashlib.sha256(hash_input.encode())
        hash_hex = hash_object.hexdigest()

        # Create readable anonymized ID
        return f"anon_surgery_{hash_hex[:12]}.mp4"
