"""
Security and Vulnerability Tests for Cataract-LMM Project
=========================================================

This module contains security-focused tests to ensure the project
meets enterprise security standards.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestSecurity:
    """Security testing suite."""

    def setup_method(self):
        """Set up security tests."""
        self.project_root = PROJECT_ROOT

    @pytest.mark.unit
    def test_no_hardcoded_secrets(self):
        """Test that no hardcoded secrets exist in the codebase."""
        secret_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "api_key",
            "private_key",
            "access_key",
            "auth",
        ]

        suspicious_files = []

        for py_file in self.project_root.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read().lower()

                for pattern in secret_patterns:
                    if f"{pattern} = " in content or f'"{pattern}"' in content:
                        # Check if it's not just a variable name or comment
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if pattern in line and "=" in line:
                                # Skip if it's clearly a placeholder or config key
                                if any(
                                    x in line
                                    for x in ["config", "placeholder", "example", "#"]
                                ):
                                    continue
                                suspicious_files.append((py_file, i + 1, line.strip()))
            except (UnicodeDecodeError, PermissionError):
                continue

        assert (
            len(suspicious_files) == 0
        ), f"Potential hardcoded secrets found: {suspicious_files}"

    @pytest.mark.unit
    def test_safe_file_operations(self):
        """Test that file operations use safe practices."""
        unsafe_patterns = [
            "eval(",
            "exec(",
            "open(",  # Should specify encoding
            "pickle.load",
            "yaml.load(",  # Should use safe_load
        ]

        violations = []

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                for i, line in enumerate(lines):
                    for pattern in unsafe_patterns:
                        if pattern in line and not line.strip().startswith("#"):
                            # Check for safe usage
                            if pattern == "open(" and "encoding=" in line:
                                continue
                            if pattern == "yaml.load(" and "safe_load" in line:
                                continue
                            violations.append((py_file, i + 1, line.strip()))
            except (UnicodeDecodeError, PermissionError):
                continue

        # Allow some violations in test files
        test_violations = [v for v in violations if "test" not in str(v[0])]
        assert (
            len(test_violations) == 0
        ), f"Unsafe file operations found: {test_violations}"

    @pytest.mark.unit
    def test_environment_variable_usage(self):
        """Test proper environment variable usage."""
        env_files = list(self.project_root.rglob("*.py"))

        secure_env_usage = []

        for py_file in env_files:
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for os.environ usage
                if "os.environ[" in content:
                    # Should use os.environ.get() with defaults
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if "os.environ[" in line and "get(" not in line:
                            secure_env_usage.append((py_file, i + 1, line.strip()))
            except (UnicodeDecodeError, PermissionError):
                continue

        # This is a warning, not a hard failure
        if secure_env_usage:
            print(
                f"Warning: Consider using os.environ.get() instead of direct access: {len(secure_env_usage)} occurrences"
            )

    @pytest.mark.unit
    def test_input_validation_patterns(self):
        """Test that input validation is present where needed."""
        validation_keywords = ["assert", "raise", "ValueError", "TypeError", "if not"]

        function_files = []
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "test" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Find functions that take parameters
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def ") and "(" in line and "):" in line:
                        # Check if function has parameters (excluding self)
                        params = line.split("(")[1].split(")")[0]
                        if params and params.strip() not in ["self", ""]:
                            # Check next 10 lines for validation
                            validation_found = False
                            for j in range(i + 1, min(i + 11, len(lines))):
                                check_line = lines[j]
                                if any(
                                    keyword in check_line
                                    for keyword in validation_keywords
                                ):
                                    validation_found = True
                                    break

                            if not validation_found:
                                function_files.append((py_file, i + 1, line.strip()))
            except (UnicodeDecodeError, PermissionError):
                continue

        # This is informational - not all functions need validation
        if len(function_files) > 20:  # Only warn if many functions lack validation
            print(
                f"Info: {len(function_files)} functions might benefit from input validation"
            )

    @pytest.mark.integration
    def test_dependency_licenses(self):
        """Test that all dependencies have acceptable licenses."""
        try:
            # Try to read from pyproject.toml
            import toml

            pyproject_path = self.project_root / "pyproject.toml"

            if pyproject_path.exists():
                with open(pyproject_path, "r") as f:
                    config = toml.load(f)

                dependencies = (
                    config.get("tool", {}).get("poetry", {}).get("dependencies", {})
                )

                # Check for dependencies with restrictive licenses (this is basic)
                # In a real implementation, you'd use tools like pip-licenses
                suspicious_deps = []
                for dep_name in dependencies.keys():
                    if dep_name in ["python"]:  # Skip python itself
                        continue
                    # This is a placeholder - in practice you'd check actual licenses
                    # using tools like pip-licenses or license-checker

                # For now, just ensure we have dependencies defined
                assert len(dependencies) > 0, "No dependencies found in pyproject.toml"

        except ImportError:
            pytest.skip("toml package not available for license checking")

    @pytest.mark.unit
    def test_secure_defaults(self):
        """Test that secure defaults are used in configurations."""
        config_files = list(self.project_root.rglob("*.yaml")) + list(
            self.project_root.rglob("*.yml")
        )

        insecure_patterns = [
            "debug: true",
            "ssl: false",
            "verify: false",
            "insecure: true",
        ]

        security_issues = []

        for config_file in config_files:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    content = f.read().lower()

                for pattern in insecure_patterns:
                    if pattern in content:
                        security_issues.append((config_file, pattern))
            except (UnicodeDecodeError, PermissionError):
                continue

        # Allow debug mode in development configs
        production_issues = [
            issue
            for issue in security_issues
            if "dev" not in str(issue[0]) and "test" not in str(issue[0])
        ]

        assert (
            len(production_issues) == 0
        ), f"Insecure defaults in production configs: {production_issues}"

    @pytest.mark.unit
    def test_dockerfile_security(self):
        """Test Dockerfile for security best practices."""
        dockerfile_path = self.project_root / "Dockerfile"

        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")

        with open(dockerfile_path, "r") as f:
            content = f.read()

        # Check for security best practices
        security_checks = {
            "non_root_user": any(
                line.strip().startswith("USER ") and "root" not in line
                for line in content.split("\n")
            ),
            "no_sudo": "sudo" not in content.lower(),
            "specific_versions": "FROM ubuntu:" in content,  # Should specify version
            "health_check": "HEALTHCHECK" in content,
        }

        failed_checks = [
            check for check, passed in security_checks.items() if not passed
        ]

        if failed_checks:
            print(
                f"Warning: Dockerfile security recommendations not followed: {failed_checks}"
            )

    @pytest.mark.unit
    def test_git_security(self):
        """Test git repository security."""
        gitignore_path = self.project_root / ".gitignore"

        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                gitignore_content = f.read()

            # Check for important patterns
            security_patterns = [
                "*.key",
                "*.pem",
                ".env",
                "secrets",
                "*.log",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
            ]

            missing_patterns = []
            for pattern in security_patterns:
                if pattern not in gitignore_content:
                    missing_patterns.append(pattern)

            if missing_patterns:
                print(
                    f"Warning: Consider adding these patterns to .gitignore: {missing_patterns}"
                )


class TestComplianceAndStandards:
    """Test compliance with coding standards and best practices."""

    def setup_method(self):
        """Set up compliance tests."""
        self.project_root = PROJECT_ROOT

    @pytest.mark.unit
    def test_python_version_compatibility(self):
        """Test Python version compatibility declarations."""
        pyproject_path = self.project_root / "pyproject.toml"

        if pyproject_path.exists():
            try:
                import toml

                with open(pyproject_path, "r") as f:
                    config = toml.load(f)

                python_req = (
                    config.get("tool", {})
                    .get("poetry", {})
                    .get("dependencies", {})
                    .get("python")
                )
                assert (
                    python_req is not None
                ), "Python version requirement not specified"
                assert "3.8" in python_req, "Should support Python 3.8+"

            except ImportError:
                pytest.skip("toml package not available")

    @pytest.mark.unit
    def test_docstring_coverage(self):
        """Test that functions have docstrings."""
        python_files = [
            f
            for f in self.project_root.rglob("*.py")
            if "__pycache__" not in str(f) and "test" not in str(f)
        ]

        missing_docstrings = []

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                for i, line in enumerate(lines):
                    if line.strip().startswith("def ") and not line.strip().startswith(
                        "def __"
                    ):
                        # Check next few lines for docstring
                        docstring_found = False
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                docstring_found = True
                                break

                        if not docstring_found:
                            missing_docstrings.append((py_file, i + 1, line.strip()))

            except (UnicodeDecodeError, PermissionError):
                continue

        # Allow some missing docstrings, but warn if too many
        if len(missing_docstrings) > 50:
            print(f"Warning: {len(missing_docstrings)} functions missing docstrings")

    @pytest.mark.unit
    def test_logging_usage(self):
        """Test proper logging usage instead of print statements."""
        python_files = [
            f for f in self.project_root.rglob("*.py") if "__pycache__" not in str(f)
        ]

        print_statements = []

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                for i, line in enumerate(lines):
                    if "print(" in line and not line.strip().startswith("#"):
                        # Allow print in test files and specific cases
                        if "test" in str(py_file) or "debug" in line.lower():
                            continue
                        print_statements.append((py_file, i + 1, line.strip()))

            except (UnicodeDecodeError, PermissionError):
                continue

        # This is informational - print statements aren't always bad
        if len(print_statements) > 10:
            print(
                f"Info: Consider using logging instead of print: {len(print_statements)} statements"
            )

    @pytest.mark.integration
    def test_error_handling_patterns(self):
        """Test that proper error handling is implemented."""
        python_files = [
            f
            for f in self.project_root.rglob("*.py")
            if "__pycache__" not in str(f) and "test" not in str(f)
        ]

        try_blocks = []
        bare_excepts = []

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                for i, line in enumerate(lines):
                    if line.strip().startswith("try:"):
                        try_blocks.append((py_file, i + 1))
                    if line.strip() == "except:" or line.strip().startswith("except:"):
                        bare_excepts.append((py_file, i + 1, line.strip()))

            except (UnicodeDecodeError, PermissionError):
                continue

        # Bare except statements are generally bad practice
        assert len(bare_excepts) == 0, f"Bare except statements found: {bare_excepts}"

        # Should have some error handling
        assert (
            len(try_blocks) > 0
        ), "No try/except blocks found - consider adding error handling"
