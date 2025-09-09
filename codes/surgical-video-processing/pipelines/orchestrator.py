"""
Pipeline Configuration and Orchestration

This module provides pipeline configuration and orchestration classes
that handle the complete workflow management for surgical video processing.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..core import ProcessingResult, ProcessingStatus, VideoMetadata
from . import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Container for processing performance metrics"""

    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    total_input_size_mb: float = 0.0
    total_output_size_mb: float = 0.0
    processing_ratio: float = 0.0
    quality_scores: List[float] = None

    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []


class PipelineOrchestrator:
    """
    Main pipeline orchestrator that coordinates all processing steps

    This class manages the complete workflow from input validation
    through final output generation, with comprehensive error handling
    and progress tracking.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline orchestrator"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = ProcessingMetrics()

        # Initialize processing components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all processing components"""
        from ..core.video_processor import BatchVideoProcessor, CoreVideoProcessor
        from ..deidentification import DeidentificationPipeline
        from ..metadata import MetadataManager
        from ..quality_control import QualityControlPipeline

        # Core components
        self.core_processor = CoreVideoProcessor()
        self.batch_processor = BatchVideoProcessor(
            self.core_processor, self.config.max_workers
        )

        # Optional components based on configuration
        if self.config.quality_control_enabled:
            self.quality_controller = QualityControlPipeline(
                self.config.quality_control_config
            )

        if self.config.metadata_extraction_enabled:
            self.metadata_manager = MetadataManager()

        if self.config.deidentification_enabled:
            self.deidentifier = DeidentificationPipeline(
                self.config.deidentification_config
            )

    def process_single_video(
        self, input_path: Path, output_dir: Path
    ) -> ProcessingResult:
        """
        Process a single video through the complete pipeline

        Args:
            input_path: Path to input video file
            output_dir: Directory for output files

        Returns:
            Processing result
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting processing of {input_path.name}")

            # Validate input
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Extract metadata
            metadata = self.core_processor.create_video_metadata(input_path)
            self.logger.debug(f"Extracted metadata for {input_path.name}")

            # Step 2: Quality control (if enabled)
            if self.config.quality_control_enabled and hasattr(
                self, "quality_controller"
            ):
                quality_result = self.quality_controller.analyze_video(str(input_path))

                if quality_result.overall_score < self.config.quality_threshold:
                    warning_msg = f"Video quality below threshold: {quality_result.overall_score:.1f}"
                    self.logger.warning(warning_msg)

                    if self.config.skip_low_quality:
                        return ProcessingResult(
                            status=ProcessingStatus.SKIPPED,
                            input_path=input_path,
                            metadata=metadata,
                            warnings=[warning_msg],
                            processing_time=time.time() - start_time,
                        )

                # Update metadata with quality score
                metadata.quality_score = quality_result.overall_score

            # Step 3: De-identification (if enabled)
            if self.config.deidentification_enabled and hasattr(self, "deidentifier"):
                # Apply de-identification before processing
                deident_result = self.deidentifier.process_video(input_path, output_dir)
                if deident_result.status != ProcessingStatus.COMPLETED:
                    return deident_result

                # Use de-identified video as input for processing
                processing_input = deident_result.output_path
            else:
                processing_input = input_path

            # Step 4: Apply processing based on reference script methodology
            output_filename = f"processed_{input_path.stem}.mp4"
            output_path = output_dir / output_filename

            # Use reference script processing methodology
            processing_method = self.config.processing.get(
                "reference_method", "process_video"
            )
            result = self.core_processor.process_video_with_reference_method(
                processing_input, output_path, processing_method
            )

            # Update result with metadata and processing time
            result.metadata = metadata
            result.processing_time = time.time() - start_time

            # Step 5: Generate metadata report (if enabled)
            if self.config.generate_metadata_reports and hasattr(
                self, "metadata_manager"
            ):
                metadata_file = output_dir / f"{input_path.stem}_metadata.json"
                self.metadata_manager.save_metadata(metadata, metadata_file)

            # Clean up temporary files if de-identification was used
            if (
                self.config.deidentification_enabled
                and hasattr(self, "deidentifier")
                and processing_input != input_path
            ):
                try:
                    processing_input.unlink()  # Remove temporary de-identified file
                except Exception as e:
                    self.logger.warning(
                        f"Could not remove temporary file {processing_input}: {e}"
                    )

            self.logger.info(
                f"Completed processing of {input_path.name} in {result.processing_time:.2f}s"
            )
            return result

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                input_path=input_path,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def process_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> List[ProcessingResult]:
        """
        Process multiple videos in batch

        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for output videos
            progress_callback: Optional callback for progress updates

        Returns:
            List of processing results
        """
        start_time = time.time()

        # Find all video files
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
        video_files = []

        for ext in video_extensions:
            video_files.extend(input_dir.glob(f"*{ext}"))
            video_files.extend(input_dir.glob(f"*{ext.upper()}"))

        video_files = sorted(video_files)

        if not video_files:
            self.logger.warning(f"No video files found in {input_dir}")
            return []

        self.logger.info(f"Processing batch of {len(video_files)} videos")

        # Initialize metrics
        self.metrics = ProcessingMetrics()
        self.metrics.total_files = len(video_files)

        # Process videos
        results = []

        for i, video_file in enumerate(video_files):
            try:
                # Process single video
                result = self.process_single_video(video_file, output_dir)
                results.append(result)

                # Update metrics
                self._update_metrics(result, video_file)

                # Call progress callback
                if progress_callback:
                    progress_callback(i + 1, len(video_files), video_file, result)

            except KeyboardInterrupt:
                self.logger.info("Batch processing interrupted by user")
                break
            except Exception as e:
                error_msg = f"Unexpected error processing {video_file}: {e}"
                self.logger.error(error_msg)
                result = ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    input_path=video_file,
                    error_message=error_msg,
                )
                results.append(result)
                self.metrics.failed_files += 1

        # Finalize metrics
        self.metrics.total_processing_time = time.time() - start_time
        if self.metrics.successful_files > 0:
            self.metrics.average_processing_time = (
                self.metrics.total_processing_time / self.metrics.successful_files
            )

        if self.metrics.total_input_size_mb > 0:
            self.metrics.processing_ratio = (
                self.metrics.total_input_size_mb / self.metrics.total_output_size_mb
            )

        self.logger.info(
            f"Batch processing completed: {self.metrics.successful_files}/{self.metrics.total_files} successful"
        )

        # Generate batch report if enabled
        if self.config.generate_batch_reports:
            self._generate_batch_report(results, output_dir)

        return results

    def _update_metrics(self, result: ProcessingResult, input_file: Path):
        """Update processing metrics with result"""
        if result.status == ProcessingStatus.COMPLETED:
            self.metrics.successful_files += 1

            # File size metrics
            input_size_mb = input_file.stat().st_size / (1024 * 1024)
            self.metrics.total_input_size_mb += input_size_mb

            if result.output_path and result.output_path.exists():
                output_size_mb = result.output_path.stat().st_size / (1024 * 1024)
                self.metrics.total_output_size_mb += output_size_mb

            # Quality score metrics
            if result.metadata and result.metadata.quality_score > 0:
                self.metrics.quality_scores.append(result.metadata.quality_score)

        elif result.status == ProcessingStatus.FAILED:
            self.metrics.failed_files += 1

    def _generate_batch_report(self, results: List[ProcessingResult], output_dir: Path):
        """Generate comprehensive batch processing report"""
        try:
            report = {
                "batch_summary": {
                    "total_files": self.metrics.total_files,
                    "successful_files": self.metrics.successful_files,
                    "failed_files": self.metrics.failed_files,
                    "success_rate": self.metrics.successful_files
                    / self.metrics.total_files
                    * 100,
                    "total_processing_time": self.metrics.total_processing_time,
                    "average_processing_time": self.metrics.average_processing_time,
                    "total_input_size_mb": self.metrics.total_input_size_mb,
                    "total_output_size_mb": self.metrics.total_output_size_mb,
                    "processing_ratio": self.metrics.processing_ratio,
                    "space_saved_mb": self.metrics.total_input_size_mb
                    - self.metrics.total_output_size_mb,
                    "space_saved_percentage": (
                        (
                            (
                                self.metrics.total_input_size_mb
                                - self.metrics.total_output_size_mb
                            )
                            / self.metrics.total_input_size_mb
                            * 100
                        )
                        if self.metrics.total_input_size_mb > 0
                        else 0
                    ),
                },
                "quality_statistics": {
                    "average_quality_score": (
                        sum(self.metrics.quality_scores)
                        / len(self.metrics.quality_scores)
                        if self.metrics.quality_scores
                        else 0
                    ),
                    "min_quality_score": (
                        min(self.metrics.quality_scores)
                        if self.metrics.quality_scores
                        else 0
                    ),
                    "max_quality_score": (
                        max(self.metrics.quality_scores)
                        if self.metrics.quality_scores
                        else 0
                    ),
                },
                "hospital_distribution": self._analyze_hospital_distribution(results),
                "processing_details": [
                    {
                        "input_file": str(result.input_path),
                        "output_file": (
                            str(result.output_path) if result.output_path else None
                        ),
                        "status": result.status.value,
                        "processing_time": result.processing_time,
                        "error_message": result.error_message,
                        "hospital_source": (
                            result.metadata.hospital_source.value
                            if result.metadata
                            else "unknown"
                        ),
                        "quality_score": (
                            result.metadata.quality_score if result.metadata else 0
                        ),
                    }
                    for result in results
                ],
                "report_generated": datetime.now().isoformat(),
            }

            # Save report
            report_file = (
                output_dir
                / f"batch_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Batch report saved to {report_file}")

        except Exception as e:
            self.logger.error(f"Error generating batch report: {e}")

    def _analyze_processing_distribution(
        self, results: List[ProcessingResult]
    ) -> Dict[str, int]:
        """Analyze distribution of processing results by method"""
        distribution = {"successful": 0, "failed": 0, "total": len(results)}

        for result in results:
            if result.status == ProcessingStatus.SUCCESS:
                distribution["successful"] += 1
            else:
                distribution["failed"] += 1

        return distribution

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get current processing metrics summary"""
        return {
            "total_files": self.metrics.total_files,
            "successful_files": self.metrics.successful_files,
            "failed_files": self.metrics.failed_files,
            "success_rate": (
                (self.metrics.successful_files / self.metrics.total_files * 100)
                if self.metrics.total_files > 0
                else 0
            ),
            "total_processing_time": self.metrics.total_processing_time,
            "average_processing_time": self.metrics.average_processing_time,
            "processing_ratio": self.metrics.processing_ratio,
            "space_saved_mb": self.metrics.total_input_size_mb
            - self.metrics.total_output_size_mb,
            "average_quality_score": (
                sum(self.metrics.quality_scores) / len(self.metrics.quality_scores)
                if self.metrics.quality_scores
                else 0
            ),
        }
