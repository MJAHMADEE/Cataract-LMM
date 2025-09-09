#!/usr/bin/env python3
"""
Cataract-LMM Project Master Setup Script

This script provides comprehensive setup and validation for the entire
Cataract-LMM project, ensuring all modules are properly configured and
dependencies are installed according to the research paper specifications.

Usage:
    python setup.py                    # Full setup
    python setup.py --validate-only    # Only validate existing setup
    python setup.py --module skill     # Setup specific module only
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_section(title: str, symbol: str = "🔧"):
    """Print a formatted section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{symbol} {title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * (len(title) + 4)}{Colors.ENDC}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ️  {message}{Colors.ENDC}")


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ is required")
        return False
    print_success(
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected"
    )
    return True


def check_gpu_availability():
    """Check GPU availability and provide recommendations."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print_success(f"GPU {i}: {gpu_name}")
            return True
        else:
            print_warning("No CUDA-capable GPU detected. Performance may be limited.")
            print_info("For best performance, use a CUDA-capable GPU")
            return False
    except ImportError:
        print_warning("PyTorch not installed, cannot check GPU availability")
        return False


def install_unified_requirements() -> bool:
    """Install unified project requirements."""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print_error("Unified requirements.txt not found!")
        return False

    try:
        print_info("Installing unified project dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def setup_module(module_name: str) -> bool:
    """Setup individual module."""
    module_path = Path(__file__).parent / f"surgical-{module_name}"
    setup_script = module_path / "setup.py"

    if not module_path.exists():
        print_error(f"Module directory not found: {module_path}")
        return False

    if setup_script.exists():
        try:
            print_info(f"Running setup for {module_name} module...")
            subprocess.check_call(
                [sys.executable, str(setup_script)], cwd=str(module_path)
            )
            print_success(f"{module_name} module setup completed")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to setup {module_name}: {e}")
            return False
    else:
        print_warning(f"No setup.py found for {module_name}, skipping")
        return True


def validate_module_imports() -> Dict[str, bool]:
    """Validate that all modules can be imported correctly."""
    modules = {
        "video-processing": ["utils", "core", "pipelines"],
        "instance-segmentation": ["models", "training", "inference"],
        "phase-recognition": ["models", "data", "validation"],
        "skill-assessment": ["models", "data", "engine", "utils"],
    }

    results = {}

    for module_name, submodules in modules.items():
        module_path = Path(__file__).parent / f"surgical-{module_name}"
        if not module_path.exists():
            results[module_name] = False
            continue

        # Test basic import by checking if key files exist
        success = True
        for submodule in submodules:
            submodule_path = module_path / submodule / "__init__.py"
            if not submodule_path.exists():
                success = False
                break

        results[module_name] = success

    return results


def create_project_structure():
    """Create any missing project directories."""
    base_path = Path(__file__).parent
    directories = [
        "logs",
        "outputs",
        "checkpoints",
        "evaluation_results",
        "test_data",
        ".model_cache",
        "experiments",
    ]

    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(exist_ok=True)
        print_info(f"Ensured directory exists: {directory}")


def validate_configurations() -> bool:
    """Validate that all configuration files are present and valid."""
    config_files = [
        "surgical-video-processing/configs/default.yaml",
        "surgical-instance-segmentation/configs/dataset_config.yaml",
        "surgical-phase-recognition/configs/default.yaml",
        "surgical-skill-assessment/configs/config.yaml",
    ]

    base_path = Path(__file__).parent
    all_valid = True

    for config_file in config_files:
        config_path = base_path / config_file
        if config_path.exists():
            try:
                import yaml

                with open(config_path, "r") as f:
                    yaml.safe_load(f)
                print_success(f"Valid config: {config_file}")
            except yaml.YAMLError as e:
                print_error(f"Invalid YAML in {config_file}: {e}")
                all_valid = False
            except ImportError:
                print_warning(f"Cannot validate {config_file} (PyYAML not installed)")
        else:
            print_warning(f"Config file not found: {config_file}")
            all_valid = False

    return all_valid


def check_ffmpeg():
    """Check if FFmpeg is available for video processing."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print_success("FFmpeg is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("FFmpeg not found - required for video processing")
        print_info("Install with: sudo apt update && sudo apt install ffmpeg")
        return False


def generate_summary_report(validation_results: Dict[str, bool]) -> None:
    """Generate and display summary report."""
    print_section("Setup Summary Report", "📊")

    print(f"\n{Colors.BOLD}Module Status:{Colors.ENDC}")
    for module, status in validation_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {module}")

    total_modules = len(validation_results)
    working_modules = sum(validation_results.values())

    print(f"\n{Colors.BOLD}Overall Status:{Colors.ENDC}")
    print(f"  Working modules: {working_modules}/{total_modules}")

    if working_modules == total_modules:
        print_success("All modules are properly configured!")
    elif working_modules > 0:
        print_warning(f"{total_modules - working_modules} modules need attention")
    else:
        print_error("No modules are working - check installation")

    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("  1. Review module-specific README files")
    print("  2. Configure data paths in .env file")
    print("  3. Download required datasets")
    print("  4. Run module-specific tests")


def main():
    """Main setup orchestration."""
    parser = argparse.ArgumentParser(description="Cataract-LMM Project Setup")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing setup"
    )
    parser.add_argument(
        "--module",
        choices=["video", "instrument", "phase", "skill"],
        help="Setup specific module only",
    )
    args = parser.parse_args()

    print_section("Cataract-LMM Project Setup", "🏥")
    print("Comprehensive setup for multi-task surgical video analysis")

    if not check_python_version():
        sys.exit(1)

    if not args.validate_only:
        # Full setup process
        if args.module:
            # Setup specific module
            module_mapping = {
                "video": "video-processing",
                "instrument": "instance-segmentation",
                "phase": "phase-recognition",
                "skill": "skill-assessment",
            }
            setup_module(module_mapping[args.module])
        else:
            # Full setup
            print_section("Installing Dependencies", "📦")
            if not install_unified_requirements():
                sys.exit(1)

            print_section("Setting Up Modules", "🔧")
            modules = [
                "video-processing",
                "instance-segmentation",
                "phase-recognition",
                "skill-assessment",
            ]
            for module in modules:
                setup_module(module)

            print_section("Creating Project Structure", "📁")
            create_project_structure()

    # Validation phase
    print_section("Validation", "🔍")

    check_gpu_availability()
    check_ffmpeg()

    print_info("Validating configurations...")
    validate_configurations()

    print_info("Validating module imports...")
    validation_results = validate_module_imports()

    generate_summary_report(validation_results)

    print_section("Setup Complete", "🎉")


if __name__ == "__main__":
    main()
