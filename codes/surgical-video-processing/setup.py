#!/usr/bin/env python3
"""
Quick setup script for surgical video processing framework.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and accessible")
            return True
        else:
            print("❌ FFmpeg is not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install FFmpeg first.")
        return False


def install_dependencies():
    """Install Python dependencies."""
    try:
        print("📦 Installing Python dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["logs", "metadata", "temp", "output", "backup"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def validate_configs():
    """Validate configuration files."""
    try:
        import yaml

        config_dir = Path("configs")
        for config_file in config_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                yaml.safe_load(f)
            print(f"✅ Validated: {config_file.name}")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🏥 Surgical Video Processing Framework Setup")
    print("=" * 50)

    # Check prerequisites
    if not check_python_version():
        sys.exit(1)

    if not check_ffmpeg():
        print("\n📋 FFmpeg Installation Instructions:")
        print("  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Create directories
    create_directories()

    # Validate configurations
    if not validate_configs():
        sys.exit(1)

    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print(
        "  1. Test with sample video: python main.py --dry-run --input sample.mp4 --output ./test"
    )
    print("  2. Read documentation: README.md")
    print("  3. Configure for your hospital: configs/README.md")


if __name__ == "__main__":
    main()
