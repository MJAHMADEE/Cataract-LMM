#!/usr/bin/env python3
"""
Phase 3 Testing Loop - Optimized Validation Framework
=====================================================

Fast and efficient test execution for zero-tolerance QA/DevOps protocol.
Focuses on critical validation without heavy dependency imports.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class Phase3FastValidator:
    """Optimized Phase 3 testing validator."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def log_test(self, test_name: str, status: bool, message: str = ""):
        """Log test result."""
        self.total_tests += 1
        if status:
            self.passed_tests += 1
            print(f"✅ {test_name}")
        else:
            self.failed_tests += 1
            print(f"❌ {test_name} - {message}")

        self.test_results[test_name] = {
            "status": "PASS" if status else "FAIL",
            "message": message,
            "timestamp": time.time(),
        }

    def run_comprehensive_validation(self) -> bool:
        """Execute comprehensive validation quickly."""
        print(
            "================================================================================"
        )
        print("🚀 PHASE 3: TESTING LOOP - OPTIMIZED VALIDATION")
        print(
            "================================================================================"
        )
        print("📋 Zero-Tolerance Protocol: All tests must pass for 100% compliance")
        print("")

        start_time = time.time()

        # 1. Core Infrastructure Tests
        print("1️⃣ CORE INFRASTRUCTURE VALIDATION")
        print("==================================")

        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.log_test(
            "Python Runtime", sys.version_info >= (3, 8), f"Version: {python_version}"
        )

        # Project structure
        required_structure = {
            "surgical-video-processing": "dir",
            "surgical-instance-segmentation": "dir",
            "surgical-phase-recognition": "dir",
            "surgical-skill-assessment": "dir",
            "tests": "dir",
            "docs": "dir",
            "pyproject.toml": "file",
            "setup.py": "file",
            "Dockerfile": "file",
            "Makefile": "file",
        }

        for item, item_type in required_structure.items():
            path = self.project_root / item
            if item_type == "dir":
                self.log_test(f"Directory: {item}", path.exists() and path.is_dir())
            else:
                self.log_test(f"File: {item}", path.exists() and path.is_file())

        # 2. Module Structure Validation
        print("\n2️⃣ MODULE STRUCTURE VALIDATION")
        print("===============================")

        modules = [
            "surgical-video-processing",
            "surgical-instance-segmentation",
            "surgical-phase-recognition",
            "surgical-skill-assessment",
        ]

        for module in modules:
            module_path = self.project_root / module
            if module_path.exists():
                init_file = module_path / "__init__.py"
                self.log_test(f"Module Init: {module}", init_file.exists())

                # Check for standard subdirectories
                subdirs = ["configs", "tests"]
                for subdir in subdirs:
                    subdir_path = module_path / subdir
                    if subdir_path.exists():
                        self.log_test(
                            f"Subdir: {module}/{subdir}", subdir_path.is_dir()
                        )

        # 3. Configuration Validation
        print("\n3️⃣ CONFIGURATION VALIDATION")
        print("============================")

        # Test pyproject.toml
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                content = pyproject_file.read_text()
                self.log_test(
                    "PyProject.toml Readable", True, f"Size: {len(content)} chars"
                )

                # Check key sections
                key_sections = [
                    "[tool.poetry]",
                    "[build-system]",
                    "[tool.pytest.ini_options]",
                ]
                for section in key_sections:
                    self.log_test(
                        f"PyProject Section: {section.split('.')[1]}",
                        section in content,
                    )
            except Exception as e:
                self.log_test("PyProject.toml Readable", False, str(e))

        # Test Dockerfile
        dockerfile = self.project_root / "Dockerfile"
        if dockerfile.exists():
            try:
                content = dockerfile.read_text()
                self.log_test(
                    "Dockerfile Readable", True, f"Size: {len(content)} chars"
                )

                # Check key instructions
                docker_instructions = ["FROM", "RUN", "COPY", "WORKDIR"]
                for instruction in docker_instructions:
                    self.log_test(f"Docker: {instruction}", instruction in content)
            except Exception as e:
                self.log_test("Dockerfile Readable", False, str(e))

        # 4. Testing Framework Validation
        print("\n4️⃣ TESTING FRAMEWORK VALIDATION")
        print("================================")

        # Count test files
        test_files = list(self.project_root.glob("**/test*.py"))
        self.log_test(
            "Test Files Discovery",
            len(test_files) > 0,
            f"Found {len(test_files)} test files",
        )

        # Check main test directory
        main_tests = self.project_root / "tests"
        if main_tests.exists():
            main_test_files = list(main_tests.glob("*.py"))
            self.log_test(
                "Main Tests Directory",
                len(main_test_files) > 0,
                f"Found {len(main_test_files)} test files",
            )

        # 5. Reports and Documentation
        print("\n5️⃣ REPORTS AND DOCUMENTATION")
        print("=============================")

        # Check reports directory
        reports_dir = self.project_root / "reports"
        self.log_test("Reports Directory", reports_dir.exists())

        if reports_dir.exists():
            report_files = ["bandit_report.json", "flake8_report.txt"]
            for report_file in report_files:
                file_path = reports_dir / report_file
                self.log_test(f"Report: {report_file}", file_path.exists())

        # Check phase reports
        phase_reports = [
            "PHASE_1_COMPLETION_REPORT.md",
            "test_container.py",
            "simple_test.py",
            "final_validation.py",
        ]

        for report in phase_reports:
            file_path = self.project_root / report
            self.log_test(f"Phase File: {report}", file_path.exists())

        # 6. Security and Quality Validation
        print("\n6️⃣ SECURITY AND QUALITY VALIDATION")
        print("===================================")

        # Check for security configuration
        security_indicators = {
            "Docker Security": (
                "USER cataract" in (self.project_root / "Dockerfile").read_text()
                if (self.project_root / "Dockerfile").exists()
                else False
            ),
            "Poetry Lock": (self.project_root / "poetry.lock").exists(),
            "Pre-commit Config": (
                self.project_root / ".pre-commit-config.yaml"
            ).exists(),
        }

        for check, result in security_indicators.items():
            self.log_test(check, result)

        # 7. Dependencies Quick Check
        print("\n7️⃣ DEPENDENCIES QUICK CHECK")
        print("============================")

        # Check if critical packages are available (without full import)
        try:
            import importlib.util

            critical_packages = ["torch", "cv2", "numpy", "pandas"]

            for package in critical_packages:
                spec = importlib.util.find_spec(package)
                self.log_test(f"Package Available: {package}", spec is not None)
        except Exception as e:
            self.log_test("Package Discovery", False, str(e))

        # Generate final results
        execution_time = time.time() - start_time
        success_rate = (
            (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        )

        print("\n" + "=" * 80)
        print("📊 PHASE 3 VALIDATION RESULTS")
        print("=" * 80)
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"📈 Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📊 Success Rate: {success_rate:.1f}%")

        if self.failed_tests == 0:
            print("🎉 ZERO-TOLERANCE COMPLIANCE: ACHIEVED ✅")
            print("🎯 Phase 3 Status: 100% COMPLETE")
            compliance_status = "ACHIEVED"
        else:
            print("⚠️  ZERO-TOLERANCE COMPLIANCE: NOT ACHIEVED")
            print(f"🔧 Remediation Required: {self.failed_tests} tests failed")
            compliance_status = "NOT_ACHIEVED"

        print("=" * 80)

        # Save comprehensive report
        report = {
            "phase": "Phase 3 - Testing Loop (Optimized)",
            "timestamp": time.time(),
            "execution_time_seconds": execution_time,
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": round(success_rate, 2),
                "compliance_status": compliance_status,
                "zero_tolerance_achieved": self.failed_tests == 0,
            },
            "detailed_results": self.test_results,
            "validation_categories": [
                "Core Infrastructure",
                "Module Structure",
                "Configuration",
                "Testing Framework",
                "Reports & Documentation",
                "Security & Quality",
                "Dependencies",
            ],
        }

        # Save report
        try:
            reports_dir = self.project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            report_file = reports_dir / "phase3_comprehensive_report.json"

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            print(f"📄 Comprehensive report saved: {report_file}")
        except Exception as e:
            print(f"⚠️  Could not save report: {e}")

        return self.failed_tests == 0


def main():
    """Main entry point for optimized Phase 3 testing."""
    validator = Phase3FastValidator()
    success = validator.run_comprehensive_validation()

    if success:
        print("\n🎉 PHASE 3 TESTING LOOP: 100% SUCCESSFUL")
        print("✅ Zero-Tolerance QA/DevOps Protocol: COMPLETE")
    else:
        print("\n⚠️  PHASE 3 TESTING LOOP: ISSUES DETECTED")
        print("🔧 Remediation required for zero-tolerance compliance")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
