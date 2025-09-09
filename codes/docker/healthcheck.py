#!/usr/bin/env python3
"""
Health Check Script for Cataract-LMM Docker Container

This script validates that the container is running properly and all
critical components are functional.

Author: Senior Principal Engineer
Version: 1.0.0
"""

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path


def check_python_environment():
    """Check Python environment and critical imports."""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            return False, f"Python version {sys.version_info} is too old"

        # Test critical imports
        critical_modules = [
            "torch",
            "torchvision",
            "numpy",
            "cv2",
            "yaml",
            "pandas",
            "matplotlib",
            "sklearn",
        ]

        for module in critical_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                return False, f"Failed to import {module}: {e}"

        return True, "Python environment OK"
    except Exception as e:
        return False, f"Python environment check failed: {e}"


def check_gpu_availability():
    """Check GPU availability."""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            return True, f"GPU available: {device_count} device(s)"
        else:
            return True, "GPU not available (CPU mode)"
    except Exception as e:
        return False, f"GPU check failed: {e}"


def check_ffmpeg():
    """Check FFmpeg installation."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return True, "FFmpeg available"
        else:
            return False, "FFmpeg not working"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg check timed out"
    except FileNotFoundError:
        return False, "FFmpeg not found"
    except Exception as e:
        return False, f"FFmpeg check failed: {e}"


def check_file_permissions():
    """Check file system permissions."""
    try:
        test_file = Path("/app/logs/healthcheck.tmp")
        test_file.parent.mkdir(parents=True, exist_ok=True)

        # Test write permission
        test_file.write_text("health check")
        content = test_file.read_text()
        test_file.unlink()

        if content == "health check":
            return True, "File permissions OK"
        else:
            return False, "File read/write test failed"
    except Exception as e:
        return False, f"File permissions check failed: {e}"


def check_network():
    """Basic network connectivity check."""
    try:
        import urllib.parse
        import urllib.request

        # Use HTTPS for security and validate URL scheme
        url = "https://www.google.com"
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return True, "Network connectivity not required"

        urllib.request.urlopen(url, timeout=3)
        return True, "Network connectivity OK"
    except Exception:
        # Network connectivity is optional for container operation
        return True, "Network connectivity not required"


def main():
    """Main health check routine."""
    checks = [
        ("Python Environment", check_python_environment),
        ("GPU Availability", check_gpu_availability),
        ("FFmpeg", check_ffmpeg),
        ("File Permissions", check_file_permissions),
        ("Network", check_network),
    ]

    results = {}
    all_passed = True

    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            results[check_name] = {"passed": passed, "message": message}
            if not passed:
                all_passed = False
        except Exception as e:
            results[check_name] = {"passed": False, "message": f"Exception: {e}"}
            all_passed = False

    # Generate health check report
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "timestamp": timestamp,
        "overall_status": "HEALTHY" if all_passed else "UNHEALTHY",
        "checks": results,
    }

    # Output results
    if all_passed:
        print("✅ Container is HEALTHY")
        print(json.dumps(report, indent=2))
        sys.exit(0)
    else:
        print("❌ Container is UNHEALTHY")
        print(json.dumps(report, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
