#!/usr/bin/env python3
"""
Setup script for Surgical Phase Recognition module.
Installs dependencies and validates the environment.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install Python dependencies."""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print("⚠️ No requirements.txt found, creating basic one...")
        basic_requirements = [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "pytorchvideo>=0.1.5",
            "timm>=0.6.0",
            "transformers>=4.20.0",
            "numpy>=1.21.0",
            "pandas>=1.4.0",
            "scikit-learn>=1.1.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "PyYAML>=6.0",
            "tqdm>=4.64.0",
            "einops>=0.4.0",
        ]
        with open(requirements_file, "w") as f:
            f.write("\n".join(basic_requirements))

    try:
        print("📦 Installing dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    base_dir = Path(__file__).parent
    directories = [
        "outputs",
        "logs",
        "data/processed",
        "models/checkpoints",
        "experiments",
        "tests/test_data",
    ]

    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {full_path}")


def validate_installation():
    """Validate the installation."""
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import torch
        import torchvision
        import yaml

        print("✅ All core dependencies imported successfully")

        # Check for GPU
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA not available, using CPU")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main setup function."""
    print("🔧 Surgical Phase Recognition - Setup")
    print("=" * 50)

    if not check_python_version():
        sys.exit(1)

    if not install_dependencies():
        sys.exit(1)

    create_directories()

    if not validate_installation():
        sys.exit(1)

    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("  1. Prepare your phase-annotated dataset")
    print("  2. Configure training: edit configs/default.yaml")
    print(
        "  3. Run training: python -m validation.training_framework --config configs/default.yaml"
    )
    print(
        "  4. Evaluate model: python -m validation.evaluate --model checkpoints/best.pth"
    )


if __name__ == "__main__":
    main()
