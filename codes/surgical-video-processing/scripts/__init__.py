"""
Utility scripts for surgical video processing framework.

This module contains helper scripts for common tasks such as batch processing,
data migration, system setup, and maintenance operations.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from surgical_video_processing.configs import ConfigManager
from surgical_video_processing.metadata import MetadataExtractor, MetadataManager
from surgical_video_processing.pipelines import BatchProcessor, SurgicalVideoProcessor
from surgical_video_processing.utils import find_video_files, setup_logging


def setup_system():
    """
    System setup script to verify dependencies and create necessary directories.
    """
    print("Setting up surgical video processing environment...")

    # Check FFmpeg installation
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg is installed and accessible")
        else:
            print("✗ FFmpeg is not working properly")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg is not installed or not in PATH")
        print("Please install FFmpeg before proceeding")
        return False

    # Check Python dependencies
    try:
        import cv2
        import numpy as np
        import yaml

        print("✓ Python dependencies are installed")
    except ImportError as e:
        print(f"✗ Missing Python dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

    # Create necessary directories
    directories = ["./logs", "./metadata", "./temp", "./output", "./backup"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    print("✓ System setup complete!")
    return True


def migrate_legacy_scripts():
    """
    Migrate from legacy compression scripts to new framework.
    """
    print("Migrating from legacy scripts...")

    # Check for legacy scripts
    legacy_dir = Path("../Video Processing")
    if not legacy_dir.exists():
        print("No legacy scripts found to migrate")
        return

    # Create legacy scripts backup
    backup_dir = Path("./legacy_scripts")
    backup_dir.mkdir(exist_ok=True)

    for script_file in legacy_dir.glob("*"):
        if script_file.is_file():
            shutil.copy2(script_file, backup_dir / script_file.name)
            print(f"✓ Backed up: {script_file.name}")

    # Generate migration configuration
    migration_config = {
        "migration_info": {
            "date": "2024-08-30",
            "source": "Legacy Video Processing scripts",
            "target": "Surgical Video Processing Framework",
        },
        "legacy_parameters": {
            "crop_filter": "crop=268:58:6:422,avgblur=10",
            "codec": "libx265",
            "crf": 23,
            "copy_audio": True,
        },
        "equivalent_config": "farabi_config.yaml",
    }

    with open("migration_info.json", "w") as f:
        json.dump(migration_config, f, indent=2)

    print("✓ Migration complete!")
    print("Legacy scripts backed up to ./legacy_scripts/")
    print("Use farabi_config.yaml for equivalent functionality")


def batch_extract_metadata():
    """
    Extract metadata from all videos in a directory.
    """
    parser = argparse.ArgumentParser(description="Batch metadata extraction")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", default="./metadata", help="Output directory")
    parser.add_argument(
        "--format", choices=["json", "yaml"], default="json", help="Output format"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = find_video_files(str(input_dir))
    print(f"Found {len(video_files)} video files")

    # Extract metadata
    extractor = MetadataExtractor()
    metadata_list = []

    for i, video_file in enumerate(video_files, 1):
        print(f"Processing {i}/{len(video_files)}: {video_file.name}")

        try:
            metadata = extractor.extract_metadata(str(video_file))
            metadata_list.append(metadata)

            # Save individual metadata file
            output_file = output_dir / f"{video_file.stem}_metadata.{args.format}"

            if args.format == "json":
                metadata.save_to_file(str(output_file))
            else:  # yaml
                with open(output_file, "w") as f:
                    import yaml

                    yaml.dump(metadata.to_dict(), f, default_flow_style=False)

        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")

    # Generate summary report
    manager = MetadataManager(str(output_dir))
    manager.generate_batch_report(metadata_list, str(output_dir / "batch_report.json"))

    print(f"✓ Extracted metadata for {len(metadata_list)} videos")
    print(f"✓ Saved to: {output_dir}")


def validate_configuration():
    """
    Validate all configuration files.
    """
    print("Validating configuration files...")

    config_dir = Path("./configs")
    config_files = list(config_dir.glob("*.yaml"))

    for config_file in config_files:
        print(f"Validating {config_file.name}...")

        try:
            config = ConfigManager.load_config(str(config_file))
            ConfigManager.validate_config(config)
            print(f"✓ {config_file.name} is valid")
        except Exception as e:
            print(f"✗ {config_file.name} is invalid: {e}")

    print("Configuration validation complete!")


def analyze_dataset():
    """
    Analyze video dataset and generate comprehensive report.
    """
    parser = argparse.ArgumentParser(description="Dataset analysis")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument(
        "--output", "-o", default="./dataset_analysis.json", help="Output report file"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    video_files = find_video_files(str(input_dir))

    print(f"Analyzing {len(video_files)} videos...")

    # Extract metadata for analysis
    extractor = MetadataExtractor()
    analysis_data = {
        "total_videos": len(video_files),
        "hospitals": {"farabi": 0, "noor": 0, "unknown": 0},
        "resolutions": {},
        "fps_distribution": {},
        "duration_stats": {"total": 0, "mean": 0, "min": 0, "max": 0},
        "file_sizes": {"total_gb": 0, "mean_mb": 0},
        "codecs": {},
        "quality_distribution": {},
    }

    durations = []
    file_sizes = []

    for i, video_file in enumerate(video_files, 1):
        if i % 100 == 0:
            print(f"Processed {i}/{len(video_files)} videos...")

        try:
            metadata = extractor.extract_metadata(str(video_file))

            # Hospital distribution
            hospital = metadata.hospital_source or "unknown"
            analysis_data["hospitals"][hospital] += 1

            # Resolution distribution
            resolution = f"{metadata.width}x{metadata.height}"
            if resolution not in analysis_data["resolutions"]:
                analysis_data["resolutions"][resolution] = 0
            analysis_data["resolutions"][resolution] += 1

            # FPS distribution
            fps_range = (
                f"{int(metadata.fps//10)*10}-{int(metadata.fps//10)*10+9}"
                if metadata.fps > 0
                else "unknown"
            )
            if fps_range not in analysis_data["fps_distribution"]:
                analysis_data["fps_distribution"][fps_range] = 0
            analysis_data["fps_distribution"][fps_range] += 1

            # Duration and file size stats
            if metadata.duration_seconds > 0:
                durations.append(metadata.duration_seconds)
            file_sizes.append(metadata.file_size_bytes)

            # Codec distribution
            codec = metadata.codec or "unknown"
            if codec not in analysis_data["codecs"]:
                analysis_data["codecs"][codec] = 0
            analysis_data["codecs"][codec] += 1

        except Exception as e:
            print(f"Error analyzing {video_file.name}: {e}")

    # Calculate statistics
    if durations:
        analysis_data["duration_stats"] = {
            "total": sum(durations),
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
        }

    if file_sizes:
        total_gb = sum(file_sizes) / (1024**3)
        mean_mb = (sum(file_sizes) / len(file_sizes)) / (1024**2)
        analysis_data["file_sizes"] = {
            "total_gb": round(total_gb, 2),
            "mean_mb": round(mean_mb, 2),
        }

    # Save analysis report
    with open(args.output, "w") as f:
        json.dump(analysis_data, f, indent=2)

    print(f"✓ Dataset analysis complete!")
    print(f"✓ Report saved to: {args.output}")

    # Print summary
    print("\nDataset Summary:")
    print(f"Total videos: {analysis_data['total_videos']}")
    print(f"Hospitals: {analysis_data['hospitals']}")
    print(f"Total duration: {analysis_data['duration_stats']['total']:.1f} hours")
    print(f"Total size: {analysis_data['file_sizes']['total_gb']:.1f} GB")


def cleanup_temp_files():
    """
    Clean up temporary files and directories.
    """
    print("Cleaning up temporary files...")

    temp_directories = [
        "./temp",
        "./logs",
        "./__pycache__",
        "./surgical_video_processing/__pycache__",
    ]

    for temp_dir in temp_directories:
        temp_path = Path(temp_dir)
        if temp_path.exists():
            try:
                shutil.rmtree(temp_path)
                print(f"✓ Cleaned: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean {temp_dir}: {e}")

    # Recreate necessary directories
    Path("./temp").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)

    print("✓ Cleanup complete!")


def benchmark_performance():
    """
    Benchmark processing performance with different configurations.
    """
    parser = argparse.ArgumentParser(description="Performance benchmarking")
    parser.add_argument("--test-video", required=True, help="Test video file")
    parser.add_argument(
        "--output", default="./benchmark_results.json", help="Results file"
    )

    args = parser.parse_args()

    if not Path(args.test_video).exists():
        print(f"Test video not found: {args.test_video}")
        return

    print("Running performance benchmarks...")

    # Test configurations
    test_configs = [
        ("quick_processing.yaml", "Quick Processing"),
        ("default.yaml", "Default Processing"),
        ("high_quality.yaml", "High Quality"),
        ("farabi_config.yaml", "Farabi Hospital"),
        ("noor_config.yaml", "Noor Hospital"),
    ]

    results = {
        "test_video": args.test_video,
        "timestamp": "2024-08-30T15:30:00",
        "benchmarks": [],
    }

    for config_file, config_name in test_configs:
        print(f"Testing {config_name}...")

        try:
            import time

            config = ConfigManager.load_config(config_file)
            processor = SurgicalVideoProcessor(config)

            start_time = time.time()
            result = processor.process_video(args.test_video, "./temp_benchmark")
            end_time = time.time()

            benchmark_result = {
                "config_name": config_name,
                "config_file": config_file,
                "processing_time": end_time - start_time,
                "success": result.success,
                "output_size": (
                    Path(result.output_path).stat().st_size if result.success else 0
                ),
                "quality_score": result.quality_score if result.success else None,
            }

            results["benchmarks"].append(benchmark_result)
            print(f"✓ {config_name}: {benchmark_result['processing_time']:.2f}s")

        except Exception as e:
            print(f"✗ {config_name} failed: {e}")
            results["benchmarks"].append(
                {
                    "config_name": config_name,
                    "config_file": config_file,
                    "error": str(e),
                    "success": False,
                }
            )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Benchmark results saved to: {args.output}")

    # Cleanup
    temp_dir = Path("./temp_benchmark")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def main():
    """Main script entry point with command selection."""
    if len(sys.argv) < 2:
        print("Available utility scripts:")
        print("  setup           - Setup system and verify dependencies")
        print("  migrate         - Migrate from legacy scripts")
        print("  extract-metadata - Batch extract video metadata")
        print("  validate-config - Validate configuration files")
        print("  analyze-dataset - Analyze video dataset")
        print("  cleanup         - Clean temporary files")
        print("  benchmark       - Performance benchmarking")
        print("\nUsage: python scripts/utilities.py <command> [options]")
        return

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove command from argv for argparse

    if command == "setup":
        setup_system()
    elif command == "migrate":
        migrate_legacy_scripts()
    elif command == "extract-metadata":
        batch_extract_metadata()
    elif command == "validate-config":
        validate_configuration()
    elif command == "analyze-dataset":
        analyze_dataset()
    elif command == "cleanup":
        cleanup_temp_files()
    elif command == "benchmark":
        benchmark_performance()
    else:
        print(f"Unknown command: {command}")
        print("Run without arguments to see available commands")


if __name__ == "__main__":
    main()
