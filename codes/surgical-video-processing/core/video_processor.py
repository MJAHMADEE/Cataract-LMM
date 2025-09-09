"""
Core Video Processing Engine for Cataract-LMM Dataset

This module provides the main video processing engine that handles FFmpeg
integration and core video manipulation functionality, directly aligned with
the methodology described in the Cataract-LMM research paper for processing
phacoemulsification cataract surgery videos.

The processing engine implements the exact techniques from the reference scripts
(process_video.sh and process_videos.bat) to ensure consistency with the dataset
preprocessing methodology described in the academic publication.

Key Features:
- Direct implementation of reference script FFmpeg commands
- Exact replication of reference script processing parameters
- Comprehensive metadata extraction aligned with dataset specifications
- Batch processing capabilities for large-scale video processing

References:
    Cataract-LMM: Large-Scale, Multi-Center, Multi-Task Benchmark
    for Deep Learning in Surgical Video Analysis
"""

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import ProcessingResult, ProcessingStatus, VideoMetadata

logger = logging.getLogger(__name__)


@dataclass
class FFmpegConfig:
    """Configuration for FFmpeg processing operations"""

    binary_path: str = "ffmpeg"
    global_options: List[str] = None
    input_options: List[str] = None
    output_options: List[str] = None
    hardware_acceleration: bool = True
    thread_count: Optional[int] = None
    memory_limit: Optional[str] = None

    def __post_init__(self):
        if self.global_options is None:
            self.global_options = ["-y", "-hide_banner", "-loglevel", "warning"]
        if self.input_options is None:
            self.input_options = []
        if self.output_options is None:
            self.output_options = []


class CoreVideoProcessor:
    """
    Core video processing engine using FFmpeg

    This class provides the fundamental video processing capabilities
    that align with the original process_video.sh and process_videos.bat
    scripts, while adding enhanced error handling and monitoring.
    """

    def __init__(self, ffmpeg_config: FFmpegConfig = None):
        """Initialize core video processor"""
        self.ffmpeg_config = ffmpeg_config or FFmpegConfig()
        self.logger = logging.getLogger(__name__)

        # Verify FFmpeg installation
        self._verify_ffmpeg_installation()

    def _verify_ffmpeg_installation(self) -> bool:
        """Verify FFmpeg is installed and accessible"""
        try:
            result = subprocess.run(
                [self.ffmpeg_config.binary_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError("FFmpeg not found or not working properly")

            # Extract version information
            version_line = result.stdout.split("\n")[0]
            self.logger.info(f"FFmpeg detected: {version_line}")
            return True

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            raise RuntimeError(f"FFmpeg not available: {e}")

    def get_video_info(self, input_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive video information using FFprobe

        Args:
            input_path: Path to input video file

        Returns:
            Dictionary containing video metadata
        """
        try:
            # Use ffprobe to get detailed video information
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(input_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to get video info: {result.stderr}")

            probe_data = json.loads(result.stdout)

            # Extract relevant information
            video_stream = None
            audio_stream = None

            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = stream

            if not video_stream:
                raise ValueError("No video stream found in file")

            # Build metadata dictionary
            info = {
                "duration": float(probe_data.get("format", {}).get("duration", 0)),
                "file_size": int(probe_data.get("format", {}).get("size", 0)),
                "format_name": probe_data.get("format", {}).get("format_name", ""),
                "video": {
                    "codec": video_stream.get("codec_name", ""),
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "fps": self._parse_fps(video_stream.get("r_frame_rate", "0/1")),
                    "bitrate": int(video_stream.get("bit_rate", 0)),
                    "pixel_format": video_stream.get("pix_fmt", ""),
                },
                "audio": (
                    {
                        "has_audio": audio_stream is not None,
                        "codec": (
                            audio_stream.get("codec_name", "") if audio_stream else ""
                        ),
                        "sample_rate": (
                            int(audio_stream.get("sample_rate", 0))
                            if audio_stream
                            else 0
                        ),
                        "channels": (
                            int(audio_stream.get("channels", 0)) if audio_stream else 0
                        ),
                    }
                    if audio_stream
                    else {"has_audio": False}
                ),
            }

            return info

        except Exception as e:
            self.logger.error(f"Error getting video info for {input_path}: {e}")
            raise

    def _parse_fps(self, fps_string: str) -> float:
        """Parse frame rate from FFprobe format (e.g., '30/1')"""
        try:
            if "/" in fps_string:
                num, den = fps_string.split("/")
                return float(num) / float(den) if float(den) != 0 else 0.0
            return float(fps_string)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def process_video_with_reference_method(
        self, input_path: Path, output_path: Path, method: str = "process_video"
    ) -> ProcessingResult:
        """
        Process video using reference script methodology

        Args:
            input_path: Path to input video file
            output_path: Path to output processed video file
            method: Processing method ("process_video" for .sh script or "process_videos" for .bat script)

        Returns:
            ProcessingResult with status and metadata
        """
        try:
            if method == "process_video":
                # Apply process_video.sh methodology
                cmd = [
                    self.ffmpeg_config.binary_path,
                    *self.ffmpeg_config.global_options,
                    "-i",
                    str(input_path),
                    "-filter_complex",
                    "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]",
                    "-map",
                    "[v]",
                    "-map",
                    "0:a",
                    "-c:v",
                    "libx265",
                    "-crf",
                    "23",
                    "-c:a",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            else:  # process_videos method
                # Apply process_videos.bat methodology
                cmd = [
                    self.ffmpeg_config.binary_path,
                    *self.ffmpeg_config.global_options,
                    "-i",
                    str(input_path),
                    "-vcodec",
                    "libx265",
                    "-crf",
                    "23",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]

            self.logger.info(
                f"Processing video with {method} methodology: {input_path.name}"
            )

            # Execute FFmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False,
            )

            if result.returncode == 0:
                self.logger.info(f"Successfully processed: {output_path}")
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    input_path=input_path,
                    output_path=output_path,
                    file_size_reduction=self._calculate_size_reduction(
                        input_path, output_path
                    ),
                    processing_time=0.0,  # Would need timing logic
                    error_message=None,
                )
            else:
                error_msg = f"FFmpeg processing failed: {result.stderr}"
                self.logger.error(error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    output_path=output_path,
                    error_message=error_msg,
                )

        except subprocess.TimeoutExpired:
            error_msg = f"Processing timeout for {input_path}"
            self.logger.error(error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                output_path=output_path,
                error_message=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error processing {input_path}: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                output_path=output_path,
                error_message=error_msg,
            )

    def _calculate_size_reduction(self, input_path: Path, output_path: Path) -> float:
        """
        Calculate file size reduction percentage

        Args:
            input_path: Path to original file
            output_path: Path to processed file

        Returns:
            Size reduction percentage (0.0 to 100.0)
        """
        try:
            if not output_path.exists():
                return 0.0

            input_size = input_path.stat().st_size
            output_size = output_path.stat().st_size

            if input_size == 0:
                return 0.0

            reduction = ((input_size - output_size) / input_size) * 100
            return max(0.0, reduction)  # Ensure non-negative

        except Exception as e:
            self.logger.warning(f"Could not calculate size reduction: {e}")
            return 0.0

    def apply_generic_processing(
        self, input_path: Path, output_path: Path, crf: int = 23, codec: str = "libx265"
    ) -> ProcessingResult:
        """
        Apply generic processing for videos using reference script methodology

        Args:
            input_path: Input video file path
            output_path: Output video file path
            crf: Constant Rate Factor (lower = higher quality)
            codec: Video codec to use

        Returns:
            Processing result
        """
        try:
            cmd = [
                self.ffmpeg_config.binary_path,
                *self.ffmpeg_config.global_options,
                "-i",
                str(input_path),
                "-c:v",
                codec,
                "-crf",
                str(crf),
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            self.logger.info(f"Processing video: {input_path.name}")

            # Execute FFmpeg command
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    input_path=input_path,
                    output_path=output_path,
                    processing_time=0.0,
                    file_size_reduction=self._calculate_size_reduction(
                        input_path, output_path
                    ),
                    metrics={
                        "processing_method": "generic",
                        "codec": codec,
                        "crf": crf,
                    },
                )
            else:
                error_msg = f"FFmpeg failed: {result.stderr}"
                self.logger.error(error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    error_message=error_msg,
                )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
            )

    def create_video_metadata(self, input_path: Path) -> VideoMetadata:
        """
        Create comprehensive video metadata

        Args:
            input_path: Path to video file

        Returns:
            VideoMetadata object
        """
        try:
            # Get video information
            video_info = self.get_video_info(input_path)

            # Create metadata object without hospital source detection
            metadata = VideoMetadata(
                file_path=input_path,
                original_filename=input_path.name,
                file_size_bytes=video_info.get("file_size", 0),
                duration_seconds=video_info.get("duration", 0.0),
                resolution=(
                    video_info.get("video", {}).get("width", 0),
                    video_info.get("video", {}).get("height", 0),
                ),
                fps=video_info.get("video", {}).get("fps", 0.0),
                codec=video_info.get("video", {}).get("codec", ""),
                has_audio=video_info.get("audio", {}).get("has_audio", False),
            )

            return metadata

        except Exception as e:
            self.logger.error(f"Error creating metadata for {input_path}: {e}")
            # Return minimal metadata
            return VideoMetadata(
                file_path=input_path, original_filename=input_path.name
            )


class BatchVideoProcessor:
    """
    Batch processing functionality for multiple videos

    Handles parallel processing, progress tracking, and error recovery
    for large-scale video processing operations.
    """

    def __init__(self, core_processor: CoreVideoProcessor, max_workers: int = 4):
        """Initialize batch processor"""
        self.core_processor = core_processor
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> List[ProcessingResult]:
        """
        Process all videos in a directory following Cataract-LMM dataset conventions

        Processes surgical videos according to the methodology described in the Cataract-LMM
        research paper. Output files follow the dataset naming convention:

        Input files may follow patterns such as:
        - RV_<RawVideoID>_S<Site>.mp4 (Raw videos)
        - PH_<ClipID>_<RawVideoID>_S<Site>.mp4 (Phase recognition subset)
        - SE_<ClipID>_<RawVideoID>_S<Site>.mp4 (Instance segmentation subset)

        Where:
        - S1 = Farabi Hospital (Haag-Streit HS Hi-R NEO 900, 720×480, 30fps)
        - S2 = Noor Hospital (ZEISS ARTEVO 800, 1920×1080, 60fps)

        Output files are prefixed with "processed_" to indicate processed status.

        Args:
            input_dir: Directory containing input videos (supports .mp4, .avi, .mov, .mkv)
            output_dir: Directory for output videos
            progress_callback: Optional callback for progress updates

        Returns:
            List of processing results for each video
        """
        # Find all video files
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        video_files = []

        for ext in video_extensions:
            video_files.extend(input_dir.glob(f"*{ext}"))
            video_files.extend(input_dir.glob(f"*{ext.upper()}"))

        video_files = sorted(video_files)

        if not video_files:
            self.logger.warning(f"No video files found in {input_dir}")
            return []

        self.logger.info(f"Found {len(video_files)} video files to process")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process videos
        results = []

        for i, video_file in enumerate(video_files):
            try:
                # Create output filename with correct terminology
                output_filename = f"processed_{video_file.stem}.mp4"
                output_path = output_dir / output_filename

                # Get video metadata
                metadata = self.core_processor.create_video_metadata(video_file)

                # Apply reference script processing methodology
                # Default to process_video.sh method unless specifically configured
                processing_method = "process_video"  # Can be configured
                result = self.core_processor.process_video_with_reference_method(
                    video_file, output_path, processing_method
                )

                # Add metadata to result
                result.metadata = metadata
                results.append(result)

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, len(video_files), video_file)

                # Log result
                if result.status == ProcessingStatus.SUCCESS:
                    self.logger.info(f"Successfully processed: {video_file.name}")
                else:
                    self.logger.error(
                        f"Failed to process: {video_file.name} - {result.error_message}"
                    )

            except Exception as e:
                error_msg = f"Unexpected error processing {video_file}: {e}"
                self.logger.error(error_msg)
                result = ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=video_file,
                    error_message=error_msg,
                )
                results.append(result)

        return results
