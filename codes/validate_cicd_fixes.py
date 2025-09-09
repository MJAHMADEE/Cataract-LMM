#!/usr/bin/env python3
"""
CI/CD Validation Script

This script validates the fixes applied to resolve CI/CD failures:
1. Black formatting compliance
2. Test import resolution
3. Basic test functionality
4. Performance test setup

Run this to verify fixes before committing.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"🚨 EXCEPTION: {description} - {e}")
        return False


def main():
    """Main validation function."""
    print("🚀 Starting CI/CD Fix Validation")
    print("=" * 50)

    # Change to codes directory
    codes_dir = Path(__file__).parent
    if codes_dir.name != "codes":
        codes_dir = Path.cwd() / "codes"

    if not codes_dir.exists():
        print("❌ FATAL: codes directory not found")
        return False

    import os

    os.chdir(codes_dir)
    print(f"📁 Working directory: {codes_dir}")

    results = []

    # Test 1: Black formatting check
    results.append(
        run_command(
            [
                "python",
                "-m",
                "black",
                "--check",
                "--diff",
                "surgical-skill-assessment/utils/helpers.py",
            ],
            "Black formatting check on previously failing file",
        )
    )

    # Test 2: Unit test collection
    results.append(
        run_command(
            [
                "python",
                "-m",
                "pytest",
                "surgical-instance-segmentation/tests",
                "--collect-only",
                "-m",
                "unit",
            ],
            "Unit test collection for surgical-instance-segmentation",
        )
    )

    # Test 3: Run a sample unit test
    results.append(
        run_command(
            [
                "python",
                "-m",
                "pytest",
                "surgical-instance-segmentation/tests/test_framework_validation.py::TestFrameworkIntegration::test_01_model_imports",
                "-v",
                "--tb=short",
            ],
            "Sample unit test execution",
        )
    )

    # Test 4: Performance test setup check
    results.append(
        run_command(
            [
                "python",
                "-c",
                "import pytest_benchmark; print('pytest-benchmark available')",
            ],
            "Performance test dependency check",
        )
    )

    # Test 5: Phase classification test
    results.append(
        run_command(
            [
                "python",
                "-m",
                "pytest",
                "surgical-phase-recognition/tests",
                "--collect-only",
                "-m",
                "unit",
            ],
            "Phase classification unit test collection",
        )
    )

    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0

    print(f"✅ Passed: {passed}/{total} ({success_rate:.1f}%)")

    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ CI/CD fixes are working correctly")
        return True
    else:
        print(f"⚠️ {total - passed} validation(s) failed")
        print("🔧 Additional fixes may be needed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
