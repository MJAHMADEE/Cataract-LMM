"""
Video Compression Module for Surgical Videos

This module provides advanced video compression capabilities specifically
optimized for phacoemulsification cataract surgery videos. It balances
file size reduction with preservation of surgical detail quality essential
for medical analysis and training purposes.

Classes:
    AdaptiveCompressor: Intelligent compression based on content analysis
    QualityPreservingCompressor: Compression with surgical detail preservation
    BatchCompressor: High-throughput batch compression
    CompressionOptimizer: Optimization for different use cases
    CompressionAnalyzer: Analysis of compression quality and efficiency
"""

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import psutil

from ..core import (
    BaseVideoProcessor,
    ProcessingConfig,
    ProcessingResult,
    ProcessingStatus,
    VideoMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """
    Configuration for video compression operations

    Attributes:
        target_quality: Target quality level (0-100, higher is better)
        max_file_size_mb: Maximum output file size in MB
        preserve_surgical_detail: Whether to preserve fine surgical details
        compression_speed: Speed vs quality trade-off (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        output_format: Output video format (mp4, mkv, avi)
        video_codec: Video codec (libx264, libx265, libvpx-vp9)
        audio_codec: Audio codec (aac, mp3, copy, none)
        crf_value: Constant Rate Factor (0-51, lower is higher quality)
        bitrate_mode: Bitrate control mode (crf, cbr, vbr, abr)
        target_bitrate_mbps: Target bitrate in Mbps for CBR/ABR
        max_bitrate_mbps: Maximum bitrate in Mbps for VBR
        two_pass_encoding: Whether to use two-pass encoding
        adaptive_bitrate: Whether to use adaptive bitrate based on content
        roi_encoding: Whether to use region-of-interest encoding
        surgical_field_roi: ROI for surgical field (x, y, width, height)
        denoising: Whether to apply denoising during compression
        sharpening: Whether to apply sharpening during compression
        color_space: Target color space (yuv420p, yuv422p, yuv444p)
        pixel_format: Target pixel format
        hardware_acceleration: Hardware acceleration method (none, nvenc, qsv, vaapi)
    """

    target_quality: int = 75
    max_file_size_mb: Optional[float] = None
    preserve_surgical_detail: bool = True
    compression_speed: str = "medium"
    output_format: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf_value: int = 23
    bitrate_mode: str = "crf"
    target_bitrate_mbps: Optional[float] = None
    max_bitrate_mbps: Optional[float] = None
    two_pass_encoding: bool = False
    adaptive_bitrate: bool = True
    roi_encoding: bool = True
    surgical_field_roi: Optional[Tuple[int, int, int, int]] = None
    denoising: bool = False
    sharpening: bool = False
    color_space: str = "yuv420p"
    pixel_format: str = "yuv420p"
    hardware_acceleration: str = "none"


class QualityPreservingCompressor(BaseVideoProcessor):
    """
    Compression with surgical detail preservation

    This compressor uses advanced encoding techniques to maintain
    surgical detail quality while achieving significant file size reduction.
    It employs region-of-interest encoding, adaptive quality settings,
    and surgical-specific optimization techniques.
    """

    def __init__(self, config: ProcessingConfig, compression_config: CompressionConfig):
        """
        Initialize quality-preserving compressor

        Args:
            config: General processing configuration
            compression_config: Compression-specific configuration
        """
        super().__init__(config)
        self.compression_config = compression_config

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Compress video while preserving surgical detail quality

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Video metadata

        Returns:
            ProcessingResult with compression results
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
            # Get input video information
            input_info = self._get_video_info(input_path)

            # Analyze content for optimal compression settings
            content_analysis = self._analyze_video_content(input_path)

            # Build optimized FFmpeg command
            ffmpeg_cmd = self._build_compression_command(
                input_path, output_path, input_info, content_analysis, metadata
            )

            # Execute compression
            if self.compression_config.two_pass_encoding:
                result = self._execute_two_pass_encoding(
                    ffmpeg_cmd, input_path, output_path
                )
            else:
                result = self._execute_single_pass_encoding(ffmpeg_cmd)

            if result.returncode != 0:
                error_msg = f"Video compression failed: {result.stderr}"
                self.log_processing_error(input_path, error_msg)
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=input_path,
                    error_message=error_msg,
                )

            # Analyze compression results
            output_info = self._get_video_info(output_path)
            compression_metrics = self._calculate_compression_metrics(
                input_info, output_info
            )

            # Update metadata
            if metadata:
                metadata.file_size_bytes = output_info.get("size", 0)
                metadata.processing_history.append(
                    f"Compressed by {self.__class__.__name__}"
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
                    "compression_ratio": compression_metrics["compression_ratio"],
                    "size_reduction_percentage": compression_metrics[
                        "size_reduction_percentage"
                    ],
                    "bitrate_reduction": compression_metrics["bitrate_reduction"],
                    "quality_score": compression_metrics.get("quality_score", 0),
                    "input_size_mb": input_info.get("size", 0) / (1024 * 1024),
                    "output_size_mb": output_info.get("size", 0) / (1024 * 1024),
                    "content_analysis": content_analysis,
                    "compression_settings": {
                        "crf": self.compression_config.crf_value,
                        "preset": self.compression_config.compression_speed,
                        "codec": self.compression_config.video_codec,
                    },
                },
            )

        except Exception as e:
            error_msg = f"Video compression failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get comprehensive video information using FFprobe"""
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
        format_info = data.get("format", {})

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
            "duration": float(format_info.get("duration", 0)),
            "bitrate": int(format_info.get("bit_rate", 0)),
            "size": int(format_info.get("size", 0)),
            "codec": video_stream.get("codec_name", ""),
            "pixel_format": video_stream.get("pix_fmt", ""),
            "total_frames": int(video_stream.get("nb_frames", 0)),
        }

    def _analyze_video_content(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze video content for compression optimization

        This method analyzes the video content to determine optimal
        compression parameters based on motion, detail, and complexity.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return {
                "complexity": "medium",
                "motion_level": "medium",
                "detail_level": "medium",
            }

        # Sample frames for analysis
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = min(20, total_frames // 10)

        motion_scores = []
        detail_scores = []
        brightness_values = []

        prev_frame = None

        for i in range(sample_count):
            frame_pos = i * (total_frames // sample_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if not ret:
                continue

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Analyze detail level using Laplacian variance
            detail_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            detail_scores.append(detail_score)

            # Analyze brightness
            brightness = np.mean(gray)
            brightness_values.append(brightness)

            # Analyze motion if we have a previous frame
            if prev_frame is not None:
                # Calculate optical flow magnitude
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_frame,
                    gray,
                    np.array(
                        [
                            [x, y]
                            for y in range(0, gray.shape[0], 20)
                            for x in range(0, gray.shape[1], 20)
                        ],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2),
                    None,
                )[0]

                if flow is not None:
                    motion_magnitude = np.mean(
                        np.linalg.norm(flow.reshape(-1, 2), axis=1)
                    )
                    motion_scores.append(motion_magnitude)

            prev_frame = gray

        cap.release()

        # Categorize content characteristics
        avg_detail = np.mean(detail_scores) if detail_scores else 100
        avg_motion = np.mean(motion_scores) if motion_scores else 1
        avg_brightness = np.mean(brightness_values) if brightness_values else 128

        detail_level = (
            "high" if avg_detail > 200 else "medium" if avg_detail > 100 else "low"
        )
        motion_level = (
            "high" if avg_motion > 3 else "medium" if avg_motion > 1 else "low"
        )
        complexity = (
            "high"
            if detail_level == "high" and motion_level == "high"
            else "low" if detail_level == "low" and motion_level == "low" else "medium"
        )

        return {
            "complexity": complexity,
            "motion_level": motion_level,
            "detail_level": detail_level,
            "avg_detail_score": avg_detail,
            "avg_motion_score": avg_motion,
            "avg_brightness": avg_brightness,
            "brightness_consistency": (
                np.std(brightness_values) if brightness_values else 0
            ),
        }

    def _build_compression_command(
        self,
        input_path: Path,
        output_path: Path,
        input_info: Dict[str, Any],
        content_analysis: Dict[str, Any],
        metadata: Optional[VideoMetadata],
    ) -> List[str]:
        """Build optimized FFmpeg compression command"""
        cmd = ["ffmpeg", "-i", str(input_path)]

        # Hardware acceleration
        if self.compression_config.hardware_acceleration != "none":
            if self.compression_config.hardware_acceleration == "nvenc":
                cmd.extend(["-hwaccel", "cuda"])
            elif self.compression_config.hardware_acceleration == "qsv":
                cmd.extend(["-hwaccel", "qsv"])
            elif self.compression_config.hardware_acceleration == "vaapi":
                cmd.extend(["-hwaccel", "vaapi"])

        # Video codec selection
        codec = self.compression_config.video_codec
        if self.compression_config.hardware_acceleration == "nvenc":
            if "x264" in codec:
                codec = "h264_nvenc"
            elif "x265" in codec:
                codec = "hevc_nvenc"

        cmd.extend(["-c:v", codec])

        # Compression quality settings
        if self.compression_config.bitrate_mode == "crf":
            # Adjust CRF based on content analysis
            crf = self.compression_config.crf_value
            if content_analysis["complexity"] == "high":
                crf = max(18, crf - 2)  # Higher quality for complex content
            elif content_analysis["complexity"] == "low":
                crf = min(28, crf + 2)  # Lower quality for simple content

            cmd.extend(["-crf", str(crf)])

        elif self.compression_config.bitrate_mode == "cbr":
            if self.compression_config.target_bitrate_mbps:
                bitrate = f"{self.compression_config.target_bitrate_mbps}M"
                cmd.extend(["-b:v", bitrate, "-minrate", bitrate, "-maxrate", bitrate])

        elif self.compression_config.bitrate_mode == "vbr":
            if self.compression_config.target_bitrate_mbps:
                cmd.extend(["-b:v", f"{self.compression_config.target_bitrate_mbps}M"])
            if self.compression_config.max_bitrate_mbps:
                cmd.extend(["-maxrate", f"{self.compression_config.max_bitrate_mbps}M"])

        # Encoding speed preset
        if "nvenc" not in codec:
            cmd.extend(["-preset", self.compression_config.compression_speed])
        else:
            # NVENC presets
            nvenc_presets = {
                "ultrafast": "fast",
                "superfast": "fast",
                "veryfast": "fast",
                "faster": "medium",
                "fast": "medium",
                "medium": "medium",
                "slow": "slow",
                "slower": "slow",
                "veryslow": "slow",
            }
            cmd.extend(
                [
                    "-preset",
                    nvenc_presets.get(
                        self.compression_config.compression_speed, "medium"
                    ),
                ]
            )

        # Pixel format
        cmd.extend(["-pix_fmt", self.compression_config.pixel_format])

        # Video filters
        filters = []

        # Region of Interest (ROI) encoding for surgical field
        if (
            self.compression_config.roi_encoding
            and self.compression_config.surgical_field_roi
            and "x264" in codec
        ):
            x, y, w, h = self.compression_config.surgical_field_roi
            # Create ROI map for higher quality in surgical field
            roi_filter = f"addroi=x={x}:y={y}:w={w}:h={h}:qoffset=-2"
            filters.append(roi_filter)

        # Denoising
        if self.compression_config.denoising:
            if content_analysis["motion_level"] == "low":
                filters.append("hqdn3d=2:2:2:2")  # Light denoising for low motion
            else:
                filters.append("hqdn3d=4:4:4:4")  # Stronger denoising for high motion

        # Sharpening
        if (
            self.compression_config.sharpening
            and content_analysis["detail_level"] == "high"
        ):
            filters.append("unsharp=5:5:0.8:3:3:0.4")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        # Audio settings
        if self.compression_config.audio_codec == "none":
            cmd.extend(["-an"])
        elif self.compression_config.audio_codec == "copy":
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-c:a", self.compression_config.audio_codec])
            if self.compression_config.audio_codec == "aac":
                cmd.extend(["-b:a", "128k"])

        # Output format optimization
        if self.compression_config.output_format == "mp4":
            cmd.extend(["-movflags", "+faststart"])  # Optimize for streaming

        # File size constraint
        if self.compression_config.max_file_size_mb:
            target_size_bits = (
                self.compression_config.max_file_size_mb * 8 * 1024 * 1024
            )
            duration = input_info.get("duration", 1)
            max_bitrate = target_size_bits / duration
            cmd.extend(["-fs", str(int(target_size_bits / 8))])

        cmd.append(str(output_path))

        return cmd

    def _execute_single_pass_encoding(
        self, ffmpeg_cmd: List[str]
    ) -> subprocess.CompletedProcess:
        """Execute single-pass encoding"""
        return subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)

    def _execute_two_pass_encoding(
        self, ffmpeg_cmd: List[str], input_path: Path, output_path: Path
    ) -> subprocess.CompletedProcess:
        """Execute two-pass encoding for better quality"""
        # Pass 1: Analysis
        pass1_cmd = ffmpeg_cmd.copy()

        # Remove output file and add pass 1 parameters
        pass1_cmd = pass1_cmd[:-1]  # Remove output file
        pass1_cmd.extend(["-pass", "1", "-f", "null", "/dev/null"])

        result1 = subprocess.run(pass1_cmd, capture_output=True, text=True, check=False)

        if result1.returncode != 0:
            return result1

        # Pass 2: Encoding
        pass2_cmd = ffmpeg_cmd.copy()
        pass2_cmd = pass2_cmd[:-1] + ["-pass", "2"] + [pass2_cmd[-1]]

        result2 = subprocess.run(pass2_cmd, capture_output=True, text=True, check=False)

        # Clean up pass files
        for pass_file in Path(".").glob("ffmpeg2pass-*"):
            pass_file.unlink()

        return result2

    def _calculate_compression_metrics(
        self, input_info: Dict, output_info: Dict
    ) -> Dict[str, float]:
        """Calculate compression performance metrics"""
        input_size = input_info.get("size", 1)
        output_size = output_info.get("size", 1)
        input_bitrate = input_info.get("bitrate", 1)
        output_bitrate = output_info.get("bitrate", 1)

        compression_ratio = input_size / output_size if output_size > 0 else 1
        size_reduction_percentage = (
            ((input_size - output_size) / input_size) * 100 if input_size > 0 else 0
        )
        bitrate_reduction = (
            ((input_bitrate - output_bitrate) / input_bitrate) * 100
            if input_bitrate > 0
            else 0
        )

        return {
            "compression_ratio": compression_ratio,
            "size_reduction_percentage": size_reduction_percentage,
            "bitrate_reduction": bitrate_reduction,
            "space_saved_mb": (input_size - output_size) / (1024 * 1024),
        }


class AdaptiveCompressor(BaseVideoProcessor):
    """
    Intelligent compression based on content analysis

    This compressor automatically adjusts compression parameters based on
    video content analysis, quality requirements, and target constraints.
    """

    def __init__(self, config: ProcessingConfig, compression_config: CompressionConfig):
        """
        Initialize adaptive compressor

        Args:
            config: General processing configuration
            compression_config: Compression-specific configuration
        """
        super().__init__(config)
        self.compression_config = compression_config
        self.quality_compressor = QualityPreservingCompressor(
            config, compression_config
        )

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Adaptively compress video based on content analysis

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            metadata: Video metadata

        Returns:
            ProcessingResult with adaptive compression results
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

        self.log_processing_start(input_path)

        try:
            # Analyze video for optimal compression strategy
            video_info = self.quality_compressor._get_video_info(input_path)
            content_analysis = self.quality_compressor._analyze_video_content(
                input_path
            )

            # Adapt compression settings based on analysis
            adapted_config = self._adapt_compression_settings(
                self.compression_config, video_info, content_analysis, metadata
            )

            # Create adapted compressor
            adapted_compressor = QualityPreservingCompressor(
                self.config, adapted_config
            )

            # Perform compression with adapted settings
            result = adapted_compressor.process(input_path, output_path, metadata)

            # Add adaptation information to metrics
            if result.status == ProcessingStatus.COMPLETED:
                result.metrics["adaptation_info"] = {
                    "original_crf": self.compression_config.crf_value,
                    "adapted_crf": adapted_config.crf_value,
                    "original_preset": self.compression_config.compression_speed,
                    "adapted_preset": adapted_config.compression_speed,
                    "content_complexity": content_analysis["complexity"],
                    "adaptation_reason": self._get_adaptation_reason(content_analysis),
                }

            return result

        except Exception as e:
            error_msg = f"Adaptive compression failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _adapt_compression_settings(
        self,
        config: CompressionConfig,
        video_info: Dict[str, Any],
        content_analysis: Dict[str, Any],
        metadata: Optional[VideoMetadata],
    ) -> CompressionConfig:
        """Adapt compression settings based on content analysis"""
        adapted_config = CompressionConfig(**config.__dict__)

        # Adapt CRF based on content complexity
        if content_analysis["complexity"] == "high":
            adapted_config.crf_value = max(18, config.crf_value - 3)
            adapted_config.compression_speed = (
                "slower"  # Better quality for complex content
            )
        elif content_analysis["complexity"] == "low":
            adapted_config.crf_value = min(28, config.crf_value + 3)
            adapted_config.compression_speed = (
                "faster"  # Faster encoding for simple content
            )

        # Adapt based on motion level
        if content_analysis["motion_level"] == "high":
            adapted_config.denoising = True
            adapted_config.crf_value = max(20, adapted_config.crf_value - 1)
        elif content_analysis["motion_level"] == "low":
            adapted_config.sharpening = True

        # Adapt based on detail level
        if content_analysis["detail_level"] == "high":
            adapted_config.preserve_surgical_detail = True
            adapted_config.crf_value = max(18, adapted_config.crf_value - 2)

        # Adapt based on video resolution
        resolution = video_info.get("width", 0) * video_info.get("height", 0)
        if resolution > 1920 * 1080:  # 4K or higher
            adapted_config.crf_value = max(20, adapted_config.crf_value - 2)
            adapted_config.compression_speed = "slow"
        elif resolution < 720 * 480:  # Low resolution
            adapted_config.crf_value = min(26, adapted_config.crf_value + 2)

        # Adapt based on hospital source
        if metadata and hasattr(metadata, "hospital_source"):
            if metadata.hospital_source.value == "farabi":  # Lower resolution source
                adapted_config.sharpening = True
                adapted_config.crf_value = max(20, adapted_config.crf_value - 1)
            elif metadata.hospital_source.value == "noor":  # Higher resolution source
                adapted_config.preserve_surgical_detail = True

        # File size constraints
        if config.max_file_size_mb:
            file_size_mb = video_info.get("size", 0) / (1024 * 1024)
            if file_size_mb > config.max_file_size_mb * 2:
                adapted_config.crf_value = min(30, adapted_config.crf_value + 4)
                adapted_config.two_pass_encoding = True

        return adapted_config

    def _get_adaptation_reason(self, content_analysis: Dict[str, Any]) -> str:
        """Generate explanation for adaptation decisions"""
        reasons = []

        if content_analysis["complexity"] == "high":
            reasons.append("High content complexity detected - increased quality")
        elif content_analysis["complexity"] == "low":
            reasons.append("Low content complexity detected - optimized for speed")

        if content_analysis["motion_level"] == "high":
            reasons.append("High motion detected - enabled denoising")
        elif content_analysis["motion_level"] == "low":
            reasons.append("Low motion detected - enabled sharpening")

        if content_analysis["detail_level"] == "high":
            reasons.append("High detail level detected - preserved surgical details")

        return "; ".join(reasons) if reasons else "No adaptation required"


class BatchCompressor(BaseVideoProcessor):
    """
    High-throughput batch compression for multiple videos

    This processor handles batch compression of multiple surgical videos
    with parallel processing, progress tracking, and resource management.
    """

    def __init__(self, config: ProcessingConfig, compression_config: CompressionConfig):
        """
        Initialize batch compressor

        Args:
            config: General processing configuration
            compression_config: Compression-specific configuration
        """
        super().__init__(config)
        self.compression_config = compression_config
        self.max_workers = min(config.max_workers, psutil.cpu_count())

    def process_batch(
        self,
        input_directory: Union[str, Path],
        output_directory: Union[str, Path],
        file_pattern: str = "*.mp4",
    ) -> List[ProcessingResult]:
        """
        Process multiple videos in batch

        Args:
            input_directory: Directory containing input videos
            output_directory: Directory for output videos
            file_pattern: Pattern to match video files

        Returns:
            List of ProcessingResult objects
        """
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all video files
        video_files = list(input_dir.glob(file_pattern))

        if not video_files:
            self.logger.warning(
                f"No video files found matching pattern: {file_pattern}"
            )
            return []

        self.logger.info(f"Starting batch compression of {len(video_files)} videos")

        results = []

        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_file = {}
            for video_file in video_files:
                output_file = output_dir / video_file.name

                # Create compressor for this video
                if self.compression_config.adaptive_bitrate:
                    compressor = AdaptiveCompressor(
                        self.config, self.compression_config
                    )
                else:
                    compressor = QualityPreservingCompressor(
                        self.config, self.compression_config
                    )

                future = executor.submit(compressor.process, video_file, output_file)
                future_to_file[future] = video_file

            # Collect results as they complete
            for future in as_completed(future_to_file):
                video_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.status == ProcessingStatus.COMPLETED:
                        self.logger.info(f"Completed: {video_file.name}")
                    else:
                        self.logger.error(
                            f"Failed: {video_file.name} - {result.error_message}"
                        )

                except Exception as e:
                    self.logger.error(f"Exception processing {video_file.name}: {e}")
                    results.append(
                        ProcessingResult(
                            status=ProcessingStatus.FAILED,
                            input_path=video_file,
                            error_message=str(e),
                        )
                    )

        # Generate batch summary
        self._generate_batch_summary(results, output_dir)

        return results

    def _generate_batch_summary(
        self, results: List[ProcessingResult], output_dir: Path
    ):
        """Generate batch processing summary report"""
        successful = [r for r in results if r.status == ProcessingStatus.COMPLETED]
        failed = [r for r in results if r.status == ProcessingStatus.FAILED]

        summary = {
            "total_videos": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "total_processing_time": sum(r.processing_time for r in results),
            "average_processing_time": (
                sum(r.processing_time for r in results) / len(results) if results else 0
            ),
            "total_size_reduction_mb": sum(
                r.metrics.get("input_size_mb", 0) - r.metrics.get("output_size_mb", 0)
                for r in successful
            ),
            "average_compression_ratio": (
                sum(r.metrics.get("compression_ratio", 1) for r in successful)
                / len(successful)
                if successful
                else 1
            ),
            "failed_files": [str(r.input_path) for r in failed],
        }

        # Save summary
        with open(output_dir / "batch_compression_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(
            f"Batch compression completed: {len(successful)}/{len(results)} successful"
        )
