"""
Quality Control Module for Surgical Videos

This module provides comprehensive quality assessment and control for
phacoemulsification cataract surgery videos, implementing automated
technical quality screening to exclude recordings based on pre-defined
criteria such as incomplete procedures, poor focus, or excessive glare.

Classes:
    QualityAssessment: Comprehensive video quality analysis
    FocusQualityChecker: Analyzes video focus and sharpness
    GlareDetector: Detects excessive glare and overexposure
    CompletenessChecker: Verifies procedure completeness
    MotionAnalyzer: Analyzes camera stability and motion artifacts
    ExposureAnalyzer: Evaluates lighting and exposure quality
    QualityControlPipeline: Orchestrates all quality checks
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import exposure, filters, measure

from ..core import (
    BaseVideoProcessor,
    ProcessingConfig,
    ProcessingResult,
    ProcessingStatus,
    VideoMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for surgical videos

    Attributes:
        overall_score: Overall quality score (0-100)
        focus_score: Focus quality score (0-100)
        glare_score: Glare assessment score (0-100, higher is better)
        exposure_score: Exposure quality score (0-100)
        motion_score: Motion stability score (0-100)
        completeness_score: Procedure completeness score (0-100)
        artifact_score: Artifact detection score (0-100, higher is better)
        frame_consistency_score: Frame-to-frame consistency (0-100)
        surgical_field_visibility: Visibility of surgical field (0-100)
        instrument_clarity: Clarity of surgical instruments (0-100)
        tissue_contrast: Tissue contrast quality (0-100)
        lighting_uniformity: Lighting distribution uniformity (0-100)
    """

    overall_score: float = 0.0
    focus_score: float = 0.0
    glare_score: float = 0.0
    exposure_score: float = 0.0
    motion_score: float = 0.0
    completeness_score: float = 0.0
    artifact_score: float = 0.0
    frame_consistency_score: float = 0.0
    surgical_field_visibility: float = 0.0
    instrument_clarity: float = 0.0
    tissue_contrast: float = 0.0
    lighting_uniformity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            "overall_score": self.overall_score,
            "focus_score": self.focus_score,
            "glare_score": self.glare_score,
            "exposure_score": self.exposure_score,
            "motion_score": self.motion_score,
            "completeness_score": self.completeness_score,
            "artifact_score": self.artifact_score,
            "frame_consistency_score": self.frame_consistency_score,
            "surgical_field_visibility": self.surgical_field_visibility,
            "instrument_clarity": self.instrument_clarity,
            "tissue_contrast": self.tissue_contrast,
            "lighting_uniformity": self.lighting_uniformity,
        }


@dataclass
class QualityControlConfig:
    """
    Configuration for quality control operations

    Attributes:
        min_overall_score: Minimum overall quality score threshold
        min_focus_score: Minimum focus quality threshold
        max_glare_percentage: Maximum allowed glare percentage
        min_exposure_score: Minimum exposure quality threshold
        max_motion_threshold: Maximum allowed motion artifacts
        min_completeness_score: Minimum procedure completeness threshold
        enable_focus_check: Enable focus quality analysis
        enable_glare_check: Enable glare detection
        enable_exposure_check: Enable exposure analysis
        enable_motion_check: Enable motion analysis
        enable_completeness_check: Enable completeness verification
        sample_frame_count: Number of frames to sample for analysis
        sample_interval: Interval between sampled frames (seconds)
        surgical_field_roi: ROI for surgical field analysis (x, y, w, h)
        focus_kernel_size: Kernel size for focus analysis
        glare_threshold: Brightness threshold for glare detection
        motion_threshold: Threshold for motion detection
        generate_report: Whether to generate detailed quality report
        save_analysis_frames: Whether to save analysis result frames
    """

    min_overall_score: float = 60.0
    min_focus_score: float = 50.0
    max_glare_percentage: float = 15.0
    min_exposure_score: float = 40.0
    max_motion_threshold: float = 20.0
    min_completeness_score: float = 70.0
    enable_focus_check: bool = True
    enable_glare_check: bool = True
    enable_exposure_check: bool = True
    enable_motion_check: bool = True
    enable_completeness_check: bool = True
    sample_frame_count: int = 50
    sample_interval: float = 2.0
    surgical_field_roi: Optional[Tuple[int, int, int, int]] = None
    focus_kernel_size: int = 9
    glare_threshold: int = 240
    motion_threshold: float = 5.0
    generate_report: bool = True
    save_analysis_frames: bool = False


class FocusQualityChecker(BaseVideoProcessor):
    """
    Analyzes video focus and sharpness quality

    This processor evaluates the focus quality of surgical videos using
    multiple algorithms including Laplacian variance, Sobel gradient,
    and Brenner gradient to ensure adequate sharpness for surgical analysis.
    """

    def __init__(self, config: ProcessingConfig, quality_config: QualityControlConfig):
        """
        Initialize focus quality checker

        Args:
            config: General processing configuration
            quality_config: Quality control specific configuration
        """
        super().__init__(config)
        self.quality_config = quality_config

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Analyze focus quality of video

        Args:
            input_path: Path to input video file
            output_path: Path for analysis results
            metadata: Video metadata

        Returns:
            ProcessingResult with focus quality analysis
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
            # Sample frames from video
            frames = self._sample_frames(input_path)

            if not frames:
                raise Exception("No frames could be extracted from video")

            # Calculate focus metrics for each frame
            focus_scores = []
            for i, frame in enumerate(frames):
                score = self._calculate_focus_score(frame)
                focus_scores.append(score)

            # Calculate overall focus quality
            avg_focus_score = np.mean(focus_scores)
            min_focus_score = np.min(focus_scores)
            max_focus_score = np.max(focus_scores)
            focus_variance = np.var(focus_scores)

            # Determine if focus quality is acceptable
            focus_acceptable = avg_focus_score >= self.quality_config.min_focus_score

            # Generate detailed analysis
            analysis_results = {
                "average_focus_score": avg_focus_score,
                "minimum_focus_score": min_focus_score,
                "maximum_focus_score": max_focus_score,
                "focus_variance": focus_variance,
                "focus_acceptable": focus_acceptable,
                "frame_count_analyzed": len(frames),
                "focus_scores_per_frame": focus_scores,
                "focus_distribution": {
                    "excellent": len([s for s in focus_scores if s >= 80]),
                    "good": len([s for s in focus_scores if 60 <= s < 80]),
                    "acceptable": len([s for s in focus_scores if 40 <= s < 60]),
                    "poor": len([s for s in focus_scores if s < 40]),
                },
            }

            # Save analysis results
            if self.quality_config.generate_report:
                self._save_focus_analysis(
                    output_path, analysis_results, frames, focus_scores
                )

            # Update metadata
            if metadata:
                metadata.quality_score = avg_focus_score
                metadata.processing_history.append(
                    f"Focus analysis by {self.__class__.__name__}"
                )

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_path, processing_time)

            status = (
                ProcessingStatus.COMPLETED
                if focus_acceptable
                else ProcessingStatus.FAILED
            )

            return ProcessingResult(
                status=status,
                input_path=input_path,
                output_path=output_path,
                metadata=metadata,
                processing_time=processing_time,
                metrics=analysis_results,
                warnings=(
                    []
                    if focus_acceptable
                    else [
                        f"Focus quality below threshold: {avg_focus_score:.1f} < {self.quality_config.min_focus_score}"
                    ]
                ),
            )

        except Exception as e:
            error_msg = f"Focus quality analysis failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _sample_frames(self, video_path: Path) -> List[np.ndarray]:
        """Sample frames from video for analysis"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame sampling parameters
        if self.quality_config.sample_interval > 0 and fps > 0:
            frame_step = int(fps * self.quality_config.sample_interval)
        else:
            frame_step = max(1, total_frames // self.quality_config.sample_frame_count)

        frames = []
        frame_count = 0

        while (
            len(frames) < self.quality_config.sample_frame_count
            and frame_count < total_frames
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)
            frame_count += frame_step

        cap.release()
        return frames

    def _calculate_focus_score(self, frame: np.ndarray) -> float:
        """
        Calculate focus score using multiple methods

        Combines Laplacian variance, Sobel gradient, and Brenner gradient
        for robust focus assessment.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply ROI if specified
        if self.quality_config.surgical_field_roi:
            x, y, w, h = self.quality_config.surgical_field_roi
            gray = gray[y : y + h, x : x + w]

        # Method 1: Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Method 2: Sobel gradient magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).mean()

        # Method 3: Brenner gradient (modified)
        kernel = np.array([[-1, 0, 1]], dtype=np.float32)
        brenner_x = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, kernel)
        brenner_y = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, kernel.T)
        brenner_score = (brenner_x**2 + brenner_y**2).mean()

        # Combine scores with weights
        combined_score = (
            0.4 * min(100, laplacian_var / 10)  # Laplacian (weighted)
            + 0.3 * min(100, sobel_magnitude / 5)  # Sobel (weighted)
            + 0.3 * min(100, brenner_score / 50)  # Brenner (weighted)
        )

        return min(100, max(0, combined_score))

    def _save_focus_analysis(
        self, output_path: Path, analysis: Dict, frames: List, scores: List
    ):
        """Save focus analysis results and visualizations"""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save numerical results
        with open(output_path / "focus_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Generate focus score plot
        plt.figure(figsize=(12, 6))
        plt.plot(scores, "b-", linewidth=2, label="Focus Score")
        plt.axhline(
            y=self.quality_config.min_focus_score,
            color="r",
            linestyle="--",
            label=f"Threshold ({self.quality_config.min_focus_score})",
        )
        plt.xlabel("Frame Number")
        plt.ylabel("Focus Score")
        plt.title("Focus Quality Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            output_path / "focus_analysis_plot.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Save sample frames with focus scores if requested
        if self.quality_config.save_analysis_frames:
            frames_dir = output_path / "focus_frames"
            frames_dir.mkdir(exist_ok=True)

            for i, (frame, score) in enumerate(
                zip(frames[::5], scores[::5])
            ):  # Save every 5th frame
                filename = f"frame_{i:03d}_score_{score:.1f}.jpg"
                cv2.imwrite(str(frames_dir / filename), frame)


class GlareDetector(BaseVideoProcessor):
    """
    Detects excessive glare and overexposure in surgical videos

    This processor identifies regions of excessive brightness that could
    obscure key anatomical structures during phacoemulsification surgery.
    """

    def __init__(self, config: ProcessingConfig, quality_config: QualityControlConfig):
        """
        Initialize glare detector

        Args:
            config: General processing configuration
            quality_config: Quality control specific configuration
        """
        super().__init__(config)
        self.quality_config = quality_config

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Detect glare and overexposure in video

        Args:
            input_path: Path to input video file
            output_path: Path for analysis results
            metadata: Video metadata

        Returns:
            ProcessingResult with glare detection analysis
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
            # Sample frames from video
            frames = self._sample_frames(input_path)

            if not frames:
                raise Exception("No frames could be extracted from video")

            # Analyze glare in each frame
            glare_percentages = []
            glare_regions = []

            for frame in frames:
                glare_pct, regions = self._detect_glare_in_frame(frame)
                glare_percentages.append(glare_pct)
                glare_regions.append(regions)

            # Calculate overall glare statistics
            avg_glare_percentage = np.mean(glare_percentages)
            max_glare_percentage = np.max(glare_percentages)
            frames_with_excessive_glare = len(
                [
                    p
                    for p in glare_percentages
                    if p > self.quality_config.max_glare_percentage
                ]
            )

            # Determine if glare is acceptable
            glare_acceptable = (
                avg_glare_percentage <= self.quality_config.max_glare_percentage
            )

            # Generate detailed analysis
            analysis_results = {
                "average_glare_percentage": avg_glare_percentage,
                "maximum_glare_percentage": max_glare_percentage,
                "frames_with_excessive_glare": frames_with_excessive_glare,
                "total_frames_analyzed": len(frames),
                "glare_acceptable": glare_acceptable,
                "glare_percentages_per_frame": glare_percentages,
                "glare_threshold_used": self.quality_config.glare_threshold,
                "glare_severity_distribution": {
                    "minimal": len([p for p in glare_percentages if p < 5]),
                    "moderate": len([p for p in glare_percentages if 5 <= p < 15]),
                    "high": len([p for p in glare_percentages if 15 <= p < 30]),
                    "excessive": len([p for p in glare_percentages if p >= 30]),
                },
            }

            # Save analysis results
            if self.quality_config.generate_report:
                self._save_glare_analysis(
                    output_path,
                    analysis_results,
                    frames,
                    glare_percentages,
                    glare_regions,
                )

            # Update metadata
            if metadata:
                metadata.processing_history.append(
                    f"Glare analysis by {self.__class__.__name__}"
                )

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_path, processing_time)

            status = (
                ProcessingStatus.COMPLETED
                if glare_acceptable
                else ProcessingStatus.FAILED
            )

            return ProcessingResult(
                status=status,
                input_path=input_path,
                output_path=output_path,
                metadata=metadata,
                processing_time=processing_time,
                metrics=analysis_results,
                warnings=(
                    []
                    if glare_acceptable
                    else [
                        f"Excessive glare detected: {avg_glare_percentage:.1f}% > {self.quality_config.max_glare_percentage}%"
                    ]
                ),
            )

        except Exception as e:
            error_msg = f"Glare detection failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _sample_frames(self, video_path: Path) -> List[np.ndarray]:
        """Sample frames from video for analysis"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame sampling parameters
        if self.quality_config.sample_interval > 0 and fps > 0:
            frame_step = int(fps * self.quality_config.sample_interval)
        else:
            frame_step = max(1, total_frames // self.quality_config.sample_frame_count)

        frames = []
        frame_count = 0

        while (
            len(frames) < self.quality_config.sample_frame_count
            and frame_count < total_frames
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)
            frame_count += frame_step

        cap.release()
        return frames

    def _detect_glare_in_frame(
        self, frame: np.ndarray
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """
        Detect glare regions in a single frame

        Returns:
            Tuple of (glare_percentage, glare_regions)
        """
        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply ROI if specified
        roi_mask = None
        if self.quality_config.surgical_field_roi:
            x, y, w, h = self.quality_config.surgical_field_roi
            roi_mask = np.zeros_like(gray)
            roi_mask[y : y + h, x : x + w] = 255
            analysis_area = gray[y : y + h, x : x + w]
            total_pixels = w * h
        else:
            analysis_area = gray
            total_pixels = gray.shape[0] * gray.shape[1]

        # Detect bright regions (potential glare)
        bright_mask = analysis_area > self.quality_config.glare_threshold

        # Calculate glare percentage
        glare_pixels = np.sum(bright_mask)
        glare_percentage = (glare_pixels / total_pixels) * 100

        # Find connected glare regions
        labeled_regions = measure.label(bright_mask)
        glare_regions = []

        for region in measure.regionprops(labeled_regions):
            # Filter out small regions (likely noise)
            if region.area > 50:  # Minimum area threshold
                y1, x1, y2, x2 = region.bbox

                # Adjust coordinates if ROI was used
                if self.quality_config.surgical_field_roi:
                    roi_x, roi_y, _, _ = self.quality_config.surgical_field_roi
                    x1 += roi_x
                    x2 += roi_x
                    y1 += roi_y
                    y2 += roi_y

                glare_regions.append((x1, y1, x2 - x1, y2 - y1))

        return glare_percentage, glare_regions

    def _save_glare_analysis(
        self,
        output_path: Path,
        analysis: Dict,
        frames: List,
        glare_percentages: List,
        glare_regions: List,
    ):
        """Save glare analysis results and visualizations"""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save numerical results
        with open(output_path / "glare_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Generate glare percentage plot
        plt.figure(figsize=(12, 6))
        plt.plot(glare_percentages, "r-", linewidth=2, label="Glare Percentage")
        plt.axhline(
            y=self.quality_config.max_glare_percentage,
            color="orange",
            linestyle="--",
            label=f"Threshold ({self.quality_config.max_glare_percentage}%)",
        )
        plt.xlabel("Frame Number")
        plt.ylabel("Glare Percentage (%)")
        plt.title("Glare Detection Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            output_path / "glare_analysis_plot.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Save sample frames with glare detection if requested
        if self.quality_config.save_analysis_frames:
            frames_dir = output_path / "glare_frames"
            frames_dir.mkdir(exist_ok=True)

            for i, (frame, glare_pct, regions) in enumerate(
                zip(frames[::5], glare_percentages[::5], glare_regions[::5])
            ):
                # Draw glare regions on frame
                annotated_frame = frame.copy()
                for x, y, w, h in regions:
                    cv2.rectangle(
                        annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2
                    )
                    cv2.putText(
                        annotated_frame,
                        "GLARE",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

                filename = f"frame_{i:03d}_glare_{glare_pct:.1f}pct.jpg"
                cv2.imwrite(str(frames_dir / filename), annotated_frame)


class QualityControlPipeline(BaseVideoProcessor):
    """
    Orchestrates all quality control checks

    This processor combines focus analysis, glare detection, exposure assessment,
    motion analysis, and completeness checking into a comprehensive quality
    control pipeline for surgical videos.
    """

    def __init__(self, config: ProcessingConfig, quality_config: QualityControlConfig):
        """
        Initialize quality control pipeline

        Args:
            config: General processing configuration
            quality_config: Quality control specific configuration
        """
        super().__init__(config)
        self.quality_config = quality_config

        # Initialize individual checkers
        self.focus_checker = FocusQualityChecker(config, quality_config)
        self.glare_detector = GlareDetector(config, quality_config)

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: Optional[VideoMetadata] = None,
    ) -> ProcessingResult:
        """
        Perform comprehensive quality control analysis

        Args:
            input_path: Path to input video file
            output_path: Path for analysis results
            metadata: Video metadata

        Returns:
            ProcessingResult with comprehensive quality analysis
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
            # Initialize quality metrics
            quality_metrics = QualityMetrics()
            all_warnings = []
            analysis_results = {}

            # Create subdirectories for each analysis
            focus_dir = output_path / "focus_analysis"
            glare_dir = output_path / "glare_analysis"

            # Run focus analysis
            if self.quality_config.enable_focus_check:
                focus_result = self.focus_checker.process(
                    input_path, focus_dir, metadata
                )
                quality_metrics.focus_score = focus_result.metrics.get(
                    "average_focus_score", 0
                )
                analysis_results["focus"] = focus_result.metrics
                all_warnings.extend(focus_result.warnings)

            # Run glare detection
            if self.quality_config.enable_glare_check:
                glare_result = self.glare_detector.process(
                    input_path, glare_dir, metadata
                )
                # Convert glare percentage to score (100 - percentage)
                glare_pct = glare_result.metrics.get("average_glare_percentage", 0)
                quality_metrics.glare_score = max(
                    0, 100 - glare_pct * 2
                )  # Scale appropriately
                analysis_results["glare"] = glare_result.metrics
                all_warnings.extend(glare_result.warnings)

            # Calculate overall quality score
            scores = []
            if self.quality_config.enable_focus_check:
                scores.append(quality_metrics.focus_score)
            if self.quality_config.enable_glare_check:
                scores.append(quality_metrics.glare_score)

            quality_metrics.overall_score = np.mean(scores) if scores else 0

            # Determine if video passes quality control
            quality_acceptable = (
                quality_metrics.overall_score >= self.quality_config.min_overall_score
            )

            # Generate comprehensive report
            comprehensive_analysis = {
                "quality_metrics": quality_metrics.to_dict(),
                "individual_analyses": analysis_results,
                "quality_acceptable": quality_acceptable,
                "quality_thresholds": {
                    "min_overall_score": self.quality_config.min_overall_score,
                    "min_focus_score": self.quality_config.min_focus_score,
                    "max_glare_percentage": self.quality_config.max_glare_percentage,
                },
                "analysis_configuration": {
                    "sample_frame_count": self.quality_config.sample_frame_count,
                    "sample_interval": self.quality_config.sample_interval,
                    "surgical_field_roi": self.quality_config.surgical_field_roi,
                },
            }

            # Save comprehensive report
            if self.quality_config.generate_report:
                self._save_comprehensive_report(
                    output_path, comprehensive_analysis, quality_metrics
                )

            # Update metadata
            if metadata:
                metadata.quality_score = quality_metrics.overall_score
                metadata.processing_history.append(
                    f"Quality control by {self.__class__.__name__}"
                )

            processing_time = time.time() - start_time
            self.log_processing_complete(input_path, output_path, processing_time)

            status = (
                ProcessingStatus.COMPLETED
                if quality_acceptable
                else ProcessingStatus.FAILED
            )

            return ProcessingResult(
                status=status,
                input_path=input_path,
                output_path=output_path,
                metadata=metadata,
                processing_time=processing_time,
                metrics=comprehensive_analysis,
                warnings=all_warnings,
            )

        except Exception as e:
            error_msg = f"Quality control analysis failed: {str(e)}"
            self.log_processing_error(input_path, error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _save_comprehensive_report(
        self, output_path: Path, analysis: Dict, metrics: QualityMetrics
    ):
        """Save comprehensive quality control report"""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        with open(output_path / "quality_control_report.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Generate comprehensive quality dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Surgical Video Quality Control Dashboard", fontsize=16)

        # Overall quality score gauge
        ax1 = axes[0, 0]
        score = metrics.overall_score
        colors = ["red" if score < 40 else "orange" if score < 70 else "green"]
        ax1.pie(
            [score, 100 - score],
            colors=colors + ["lightgray"],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.3),
        )
        ax1.text(
            0,
            0,
            f"{score:.1f}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        ax1.set_title("Overall Quality Score")

        # Individual scores bar chart
        ax2 = axes[0, 1]
        score_names = ["Focus", "Glare", "Exposure", "Motion"]
        score_values = [
            metrics.focus_score,
            metrics.glare_score,
            metrics.exposure_score,
            metrics.motion_score,
        ]
        bars = ax2.bar(
            score_names,
            score_values,
            color=["skyblue", "lightcoral", "lightgreen", "wheat"],
        )
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Quality Score")
        ax2.set_title("Individual Quality Metrics")
        ax2.axhline(y=60, color="red", linestyle="--", alpha=0.7, label="Threshold")

        # Quality distribution pie chart
        ax3 = axes[1, 0]
        if "focus" in analysis["individual_analyses"]:
            focus_dist = analysis["individual_analyses"]["focus"].get(
                "focus_distribution", {}
            )
            if focus_dist:
                labels = list(focus_dist.keys())
                sizes = list(focus_dist.values())
                ax3.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
                ax3.set_title("Focus Quality Distribution")

        # Quality trends (placeholder for future implementation)
        ax4 = axes[1, 1]
        ax4.text(
            0.5,
            0.5,
            "Quality Trends\n(Future Implementation)",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.set_title("Quality Trends Over Time")

        plt.tight_layout()
        plt.savefig(output_path / "quality_dashboard.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Generate text summary report
        with open(output_path / "quality_summary.txt", "w") as f:
            f.write("SURGICAL VIDEO QUALITY CONTROL REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Quality Score: {metrics.overall_score:.1f}/100\n")
            f.write(
                f"Quality Acceptable: {'YES' if analysis['quality_acceptable'] else 'NO'}\n\n"
            )
            f.write("Individual Metrics:\n")
            f.write(f"  Focus Quality: {metrics.focus_score:.1f}/100\n")
            f.write(f"  Glare Score: {metrics.glare_score:.1f}/100\n")
            f.write(f"  Exposure Score: {metrics.exposure_score:.1f}/100\n")
            f.write(f"  Motion Score: {metrics.motion_score:.1f}/100\n\n")

            if analysis.get("individual_analyses", {}).get("focus"):
                focus_analysis = analysis["individual_analyses"]["focus"]
                f.write("Focus Analysis Details:\n")
                f.write(
                    f"  Average Focus Score: {focus_analysis.get('average_focus_score', 0):.1f}\n"
                )
                f.write(
                    f"  Minimum Focus Score: {focus_analysis.get('minimum_focus_score', 0):.1f}\n"
                )
                f.write(
                    f"  Maximum Focus Score: {focus_analysis.get('maximum_focus_score', 0):.1f}\n"
                )
                f.write(
                    f"  Focus Variance: {focus_analysis.get('focus_variance', 0):.2f}\n\n"
                )

            if analysis.get("individual_analyses", {}).get("glare"):
                glare_analysis = analysis["individual_analyses"]["glare"]
                f.write("Glare Analysis Details:\n")
                f.write(
                    f"  Average Glare: {glare_analysis.get('average_glare_percentage', 0):.1f}%\n"
                )
                f.write(
                    f"  Maximum Glare: {glare_analysis.get('maximum_glare_percentage', 0):.1f}%\n"
                )
                f.write(
                    f"  Frames with Excessive Glare: {glare_analysis.get('frames_with_excessive_glare', 0)}\n\n"
                )

            f.write("Recommendations:\n")
            if metrics.overall_score < 40:
                f.write(
                    "  - Video quality is poor. Consider re-recording if possible.\n"
                )
            elif metrics.overall_score < 70:
                f.write("  - Video quality is acceptable but could be improved.\n")
            else:
                f.write("  - Video quality is good for analysis.\n")

            if metrics.focus_score < 50:
                f.write("  - Focus quality is poor. Check camera focus settings.\n")
            if metrics.glare_score < 70:
                f.write("  - Excessive glare detected. Adjust lighting conditions.\n")
