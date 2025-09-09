"""
Surgical Video Processing Framework

This package provides comprehensive video processing capabilities for surgical videos,
with focus on cataract surgery analysis as described in the Cataract-LMM research paper.

The framework includes:
- Core video processing engine
- Hospital-specific configurations (Farabi S1, Noor S2)
- Quality control and validation
- Metadata extraction and management
- Compression and preprocessing pipelines
- Real-time processing capabilities

Author: Cataract-LMM Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Cataract-LMM Team"

# Import main components for easy access
try:
    from .core import ProcessingConfig, ProcessingResult, VideoMetadata
    from .pipelines import PipelineConfig, SurgicalVideoProcessor
    from .utils import ConfigManager, setup_logging

    # Compatibility aliases
    PipelineOrchestrator = SurgicalVideoProcessor
    BatchProcessor = SurgicalVideoProcessor

    __all__ = [
        "ProcessingConfig",
        "VideoMetadata",
        "ProcessingResult",
        "SurgicalVideoProcessor",
        "PipelineConfig",
        "PipelineOrchestrator",  # Legacy alias
        "BatchProcessor",  # Legacy alias
        "ConfigManager",
        "setup_logging",
    ]
except ImportError as e:
    # Fallback imports for testing
    ProcessingConfig = None
    VideoMetadata = None
    ProcessingResult = None
    SurgicalVideoProcessor = None
    PipelineConfig = None
    PipelineOrchestrator = None
    BatchProcessor = None
    ConfigManager = None
    setup_logging = None

    __all__ = []
