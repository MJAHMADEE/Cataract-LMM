"""
Processing Pipelines for Surgical Video Processing
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for processing pipelines"""

    hospital_type: str = "general"
    quality_preset: str = "balanced"
    enable_gpu: bool = False
    show_progress: bool = True
    dry_run: bool = False
    batch_mode: bool = False
    resume_processing: bool = False
    enable_backup: bool = False
    skip_quality_check: bool = False
    skip_deidentification: bool = False
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class SurgicalVideoProcessor:
    """Complete end-to-end processing pipeline for surgical videos"""

    def __init__(self, config: Union[Dict[str, Any], PipelineConfig] = None):
        """Initialize surgical video processor"""
        if isinstance(config, dict):
            valid_fields = set(PipelineConfig.__dataclass_fields__.keys())
            filtered_config = {k: v for k, v in config.items() if k in valid_fields}
            self.config = PipelineConfig(**filtered_config)
        elif config is None:
            self.config = PipelineConfig()
        else:
            self.config = config

        self.logger = logging.getLogger(__name__)

        # Initialize orchestrator
        try:
            from .orchestrator import PipelineOrchestrator

            self.orchestrator = PipelineOrchestrator(self.config)
        except ImportError:
            self.orchestrator = None

    def process_video(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Process a single video"""
        if self.orchestrator:
            return self.orchestrator.process_single_video(
                Path(input_path), Path(output_path)
            )
        else:
            return {"success": False, "error": "Orchestrator not available"}

    def process_batch(
        self,
        input_directory: Union[str, Path],
        output_directory: Union[str, Path],
        resume: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Process multiple videos in batch"""
        if self.orchestrator:
            return self.orchestrator.process_batch(
                Path(input_directory), Path(output_directory), resume, progress_callback
            )
        else:
            return {"success": False, "error": "Orchestrator not available"}


# Legacy compatibility classes
class QualityControlPipeline(SurgicalVideoProcessor):
    """Quality-focused processing workflow"""

    def __init__(self, config=None):
        if config is None:
            config = PipelineConfig()
        elif isinstance(config, dict):
            config = PipelineConfig(**config)
        config.skip_quality_check = False
        config.quality_preset = "high"
        super().__init__(config)


class AnalysisPreparationPipeline(SurgicalVideoProcessor):
    """Pipeline for analysis-ready video preparation"""

    def __init__(self, config=None):
        if config is None:
            config = PipelineConfig()
        elif isinstance(config, dict):
            config = PipelineConfig(**config)
        config.skip_deidentification = False
        config.quality_preset = "balanced"
        super().__init__(config)


class ArchivalPipeline(SurgicalVideoProcessor):
    """Pipeline for long-term storage preparation"""

    def __init__(self, config=None):
        if config is None:
            config = PipelineConfig()
        elif isinstance(config, dict):
            config = PipelineConfig(**config)
        config.enable_backup = True
        config.quality_preset = "high"
        super().__init__(config)


# Import orchestrator if available
try:
    from .orchestrator import PipelineOrchestrator, ProcessingMetrics

    orchestrator_available = True
except ImportError:
    PipelineOrchestrator = None
    ProcessingMetrics = None
    orchestrator_available = False

# Export main classes
__all__ = [
    "SurgicalVideoProcessor",
    "PipelineConfig",
    "QualityControlPipeline",
    "AnalysisPreparationPipeline",
    "ArchivalPipeline",
]

if orchestrator_available:
    __all__.extend(["PipelineOrchestrator", "ProcessingMetrics"])
