#!/usr/bin/env python3
"""
Surgical Video Processing - Main Entry Point

This is the main entry point for the surgical video processing framework,
providing comprehensive command-line interface and programmatic access to all
processing capabilities with full integration of the enhanced framework.

Usage:
    python main.py --input video.mp4 --output processed.mp4
    python main.py --batch --input-dir /path/to/videos --output-dir /path/to/output
    python main.py --config config.yaml --input video.mp4
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

# Import enhanced framework components
from pipelines import PipelineConfig, SurgicalVideoProcessor
from utils import (
    VideoFileExtensions,
    create_default_config,
    format_duration,
    format_file_size,
    get_current_config,
    get_framework_info,
    get_global_aggregator,
    initialize_logging,
    performance_monitor,
    validate_environment,
    validate_video_file,
)


class SurgicalVideoProcessingCLI:
    """Enhanced command-line interface for surgical video processing"""

    def __init__(self):
        self.processor = None
        self.interrupted = False
        self.start_time = None

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print("\n⚠️  Interrupt received. Shutting down gracefully...")
        self.interrupted = True


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Surgical Video Processing Framework - Enhanced Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video with reference script methodology
  python main.py --method process_video --input video.mp4 --output processed.mp4

  # Batch process directory
  python main.py --batch --input-dir ./videos --output-dir ./processed

  # Process with custom configuration
  python main.py --config custom.yaml --input video.mp4 --output processed.mp4

  # Quick processing for large batches
  python main.py --quick --batch --input-dir ./videos --output-dir ./quick_output

  # High quality processing for research
  python main.py --high-quality --input video.mp4 --output processed.mp4
  
  # Validate environment and create sample config
  python main.py --validate-env
  python main.py --create-config sample.yaml
        """,
    )

    # Input/Output arguments
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument("--input", "-i", help="Input video file path")
    io_group.add_argument("--output", "-o", help="Output video file path")
    io_group.add_argument("--input-dir", help="Input directory for batch processing")
    io_group.add_argument("--output-dir", help="Output directory for batch processing")

    # Configuration arguments (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config", "-c", help="Path to custom configuration YAML file"
    )
    config_group.add_argument(
        "--method",
        choices=["process_video", "process_videos", "general"],
        help="Use reference script processing method",
    )
    config_group.add_argument(
        "--quick", action="store_true", help="Use quick processing configuration"
    )
    config_group.add_argument(
        "--high-quality",
        action="store_true",
        help="Use high quality processing configuration",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--batch", action="store_true", help="Enable batch processing mode"
    )
    proc_group.add_argument(
        "--parallel", action="store_true", help="Enable parallel processing"
    )
    proc_group.add_argument("--workers", type=int, help="Number of worker processes")
    proc_group.add_argument(
        "--resume", action="store_true", help="Resume interrupted batch processing"
    )
    proc_group.add_argument(
        "--dry-run", action="store_true", help="Validate inputs without processing"
    )

    # Quality and performance options
    quality_group = parser.add_argument_group("Quality & Performance")
    quality_group.add_argument(
        "--quality",
        choices=["fast", "balanced", "high"],
        default="balanced",
        help="Processing quality preset",
    )
    quality_group.add_argument(
        "--skip-quality-check", action="store_true", help="Skip quality control checks"
    )
    quality_group.add_argument(
        "--skip-deidentification",
        action="store_true",
        help="Skip de-identification (NOT recommended for production)",
    )
    quality_group.add_argument(
        "--gpu", action="store_true", help="Enable GPU acceleration"
    )
    quality_group.add_argument(
        "--preset",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="FFmpeg encoding preset",
    )

    # Backup and safety options
    safety_group = parser.add_argument_group("Backup & Safety")
    safety_group.add_argument(
        "--backup", action="store_true", help="Create backup of original videos"
    )
    safety_group.add_argument("--backup-dir", help="Custom backup directory")

    # Logging options
    log_group = parser.add_argument_group("Logging & Output")
    log_group.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug)",
    )
    log_group.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress all output except errors"
    )
    log_group.add_argument("--log-file", help="Path to log file")
    log_group.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show progress information",
    )
    log_group.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable progress information",
    )

    # Utility options
    util_group = parser.add_argument_group("Utility Options")
    util_group.add_argument("--create-config", help="Create sample configuration file")
    util_group.add_argument(
        "--validate-env", action="store_true", help="Validate processing environment"
    )
    util_group.add_argument(
        "--info", action="store_true", help="Show framework information"
    )
    util_group.add_argument(
        "--version",
        action="version",
        version=f"Surgical Video Processing {get_framework_info()['version']}",
    )

    return parser


def determine_config_file(args) -> Optional[str]:
    """Determine which configuration file to use based on arguments."""
    if args.config:
        return args.config
    elif args.hospital:
        return f"configs/{args.hospital}.yaml"
    elif args.quick:
        return "configs/fast_training.yaml"
    elif args.high_quality:
        return "configs/high_accuracy.yaml"
    else:
        return "configs/default.yaml"


def setup_enhanced_logging(args) -> logging.Logger:
    """Set up enhanced logging based on command-line arguments"""
    # Determine log level
    if args.quiet:
        level = "ERROR"
    elif args.verbose >= 2:
        level = "DEBUG"
    elif args.verbose >= 1:
        level = "INFO"
    else:
        level = "WARNING"

    # Initialize logging
    logger = initialize_logging(
        level=level, log_file=args.log_file, console_output=not args.quiet
    )

    return logger


def validate_arguments(args) -> List[str]:
    """Validate command-line arguments and return list of errors"""
    errors = []

    # Check for required arguments
    if not any(
        [args.input, args.input_dir, args.create_config, args.validate_env, args.info]
    ):
        errors.append("Must specify input file, input directory, or utility option")

    # Validate input/output combinations
    if args.input and not args.output and not args.dry_run:
        errors.append("Output file required when processing single input")

    if args.batch and not (args.input_dir and args.output_dir):
        errors.append("Input and output directories required for batch processing")

    # Validate file paths
    if args.input and not Path(args.input).exists():
        errors.append(f"Input file does not exist: {args.input}")

    if args.input_dir and not Path(args.input_dir).exists():
        errors.append(f"Input directory does not exist: {args.input_dir}")

    if args.config and not Path(args.config).exists():
        errors.append(f"Configuration file does not exist: {args.config}")

    return errors


def find_video_files(input_path: Path) -> List[Path]:
    """Find all video files in the input path using enhanced detection."""
    if input_path.is_file():
        if VideoFileExtensions.is_video_file(input_path):
            return [input_path]
        else:
            raise ValueError(f"Input file {input_path} is not a supported video format")

    elif input_path.is_dir():
        video_files = []
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and VideoFileExtensions.is_video_file(file_path):
                video_files.append(file_path)

        return sorted(video_files)

    else:
        raise ValueError(f"Input path {input_path} does not exist")


def create_processor_config(args) -> PipelineConfig:
    """Create processor configuration from arguments"""
    # Determine quality preset
    if args.quick:
        quality = "fast"
    elif args.high_quality:
        quality = "high"
    else:
        quality = args.quality

    # Create config
    config = PipelineConfig(
        hospital_type=args.hospital or "general",
        quality_preset=quality,
        enable_gpu=args.gpu,
        show_progress=args.progress and not args.quiet,
        dry_run=args.dry_run,
        batch_mode=args.batch,
        resume_processing=args.resume,
        enable_backup=args.backup,
        skip_quality_check=args.skip_quality_check,
        skip_deidentification=args.skip_deidentification,
    )

    # Apply custom settings
    if args.preset:
        config.custom_settings["ffmpeg_preset"] = args.preset
    if args.workers:
        config.custom_settings["max_workers"] = args.workers
    if args.backup_dir:
        config.custom_settings["backup_directory"] = args.backup_dir
    if args.parallel:
        config.custom_settings["parallel_processing"] = True

    return config


def process_single_video(args, logger) -> bool:
    """Process a single video file"""
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"🎬 Processing video: {input_path.name}")

    # Validate input file
    validation = validate_video_file(input_path)
    if not validation["valid"]:
        print(f"❌ Invalid input file:")
        for error in validation["errors"]:
            print(f"   - {error}")
        return False

    print(f"📁 Input size: {format_file_size(validation['size_mb'] * 1024 * 1024)}")

    if args.dry_run:
        print(f"✅ Dry run: Input validation passed")
        return True

    # Create processor configuration
    config = create_processor_config(args)

    # Process video
    try:
        processor = SurgicalVideoProcessor(config)

        with performance_monitor("single_video_processing", str(input_path)) as monitor:
            monitor.add_custom_metric("hospital_type", config.hospital_type)
            monitor.add_custom_metric("quality_preset", config.quality_preset)

            result = processor.process_video(
                input_path=str(input_path), output_path=str(output_path)
            )

            if result["success"]:
                monitor.set_output_file(str(output_path))

                print(f"✅ Processing completed successfully")
                print(f"📁 Output: {output_path}")
                print(
                    f"📊 Processing time: {format_duration(result.get('duration', 0))}"
                )

                if output_path.exists():
                    output_size = output_path.stat().st_size
                    print(f"📁 Output size: {format_file_size(output_size)}")

                    if validation["size_mb"] > 0:
                        processing_ratio = (
                            validation["size_mb"] * 1024 * 1024
                        ) / output_size
                        print(f"📉 Processing ratio: {processing_ratio:.2f}x")

                return True
            else:
                print(f"❌ Processing failed:")
                for error in result.get("errors", []):
                    print(f"   - {error}")
                return False

    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Processing error: {e}")
        print(f"❌ Processing error: {e}")
        return False


def process_batch_videos(args, logger) -> bool:
    """Process multiple videos in batch mode"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"📁 Batch processing: {input_dir} -> {output_dir}")

    # Create processor configuration
    config = create_processor_config(args)

    # Process batch
    try:
        processor = SurgicalVideoProcessor(config)

        with performance_monitor("batch_processing", str(input_dir)) as monitor:
            monitor.add_custom_metric("hospital_type", config.hospital_type)
            monitor.add_custom_metric("quality_preset", config.quality_preset)

            result = processor.process_batch(
                input_directory=str(input_dir),
                output_directory=str(output_dir),
                resume=args.resume,
            )

            if result["success"]:
                print(f"✅ Batch processing completed")
                print(f"📊 Processed: {result.get('processed_count', 0)} videos")
                print(f"❌ Failed: {result.get('failed_count', 0)} videos")
                print(
                    f"📊 Total time: {format_duration(result.get('total_duration', 0))}"
                )

                return True
            else:
                print(f"❌ Batch processing failed:")
                for error in result.get("errors", []):
                    print(f"   - {error}")
                return False

    except KeyboardInterrupt:
        print(f"\n⚠️  Batch processing interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        print(f"❌ Batch processing error: {e}")
        return False


def show_performance_summary(args):
    """Show performance summary"""
    if not args.progress or args.quiet:
        return

    aggregator = get_global_aggregator()
    summary = aggregator.get_overall_summary()

    if summary and summary.get("total_operations", 0) > 0:
        print(f"\n📊 Performance Summary:")
        print(f"   Total operations: {summary.get('total_operations', 0)}")
        print(
            f"   Total processing time: {format_duration(summary.get('total_duration_hours', 0) * 3600)}"
        )
        print(f"   Total input data: {summary.get('total_input_size_gb', 0):.1f} GB")
        print(f"   Total output data: {summary.get('total_output_size_gb', 0):.1f} GB")
        if summary.get("overall_processing_ratio"):
            print(
                f"   Average processing ratio: {summary['overall_processing_ratio']:.2f}x"
            )
        print(f"   Success rate: {summary.get('overall_success_rate', 0) * 100:.1f}%")


def main():
    """Enhanced main entry point."""
    cli = SurgicalVideoProcessingCLI()
    cli.start_time = time.time()

    parser = setup_argument_parser()
    args = parser.parse_args()

    # Set up enhanced logging
    logger = setup_enhanced_logging(args)

    try:
        # Handle utility commands first
        if args.info:
            info = get_framework_info()
            print(f"🏥 Surgical Video Processing Framework")
            print(f"   Version: {info['version']}")
            print(f"   Author: {info['author']}")
            print(f"   Description: {info['description']}")
            return 0

        if args.validate_env:
            print("🔍 Validating processing environment...")
            validation = validate_environment()

            if validation["errors"]:
                print(f"\n❌ Environment validation failed:")
                for error in validation["errors"]:
                    print(f"   - {error}")
                return 1
            else:
                print(f"\n✅ Environment validation passed")
                if validation["warnings"]:
                    print(f"\n⚠️  Warnings:")
                    for warning in validation["warnings"]:
                        print(f"   - {warning}")
                return 0

        if args.create_config:
            success = create_default_config(args.create_config)
            if success:
                print(f"✅ Created sample configuration: {args.create_config}")
                return 0
            else:
                print(f"❌ Failed to create configuration file")
                return 1

        # Validate arguments
        errors = validate_arguments(args)
        if errors:
            for error in errors:
                print(f"❌ Error: {error}")
            return 1

        # Load configuration if specified
        if args.config:
            from utils import get_config_manager

            config_manager = get_config_manager(args.config)
            config_manager.load_config()
            logger.info(f"Loaded configuration: {args.config}")

        # Process videos
        success = False

        if args.batch:
            success = process_batch_videos(args, logger)
        else:
            success = process_single_video(args, logger)

        # Show performance summary
        show_performance_summary(args)

        # Show elapsed time
        if cli.start_time and args.progress and not args.quiet:
            elapsed = time.time() - cli.start_time
            print(f"⏱️  Total elapsed time: {format_duration(elapsed)}")

        return 0 if success else 1

    except KeyboardInterrupt:
        print(f"\n⚠️  Application interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose >= 2:
            import traceback

            traceback.print_exc()
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
