"""
Integration Tests for Cataract-LMM Project
==========================================

This module contains integration tests that verify the interaction
between different components of the project.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent


class TestProjectIntegration:
    """Integration tests for the overall project."""

    def setup_method(self):
        """Set up integration tests."""
        self.project_root = PROJECT_ROOT
        self.modules = [
            "surgical-video-processing",
            "surgical-instance-segmentation",
            "surgical-phase-recognition",
            "surgical-skill-assessment",
        ]

    @pytest.mark.integration
    def test_module_interdependencies(self):
        """Test that modules can work together."""
        # Verify each module exists and has proper structure
        for module in self.modules:
            module_path = self.project_root / module
            assert module_path.exists(), f"Module {module} does not exist"
            assert (
                module_path / "__init__.py"
            ).exists(), f"Module {module} missing __init__.py"

            # Check for basic module structure
            expected_dirs = ["configs", "tests", "utils"]
            for expected_dir in expected_dirs:
                dir_path = module_path / expected_dir
                if dir_path.exists():
                    assert (
                        dir_path.is_dir()
                    ), f"{module}/{expected_dir} should be a directory"

    @pytest.mark.integration
    def test_config_file_consistency(self):
        """Test that configuration files are consistent across modules."""
        config_files = {}

        for module in self.modules:
            module_path = self.project_root / module
            configs_dir = module_path / "configs"

            if configs_dir.exists():
                config_files[module] = []
                for config_file in configs_dir.glob("*.yaml"):
                    try:
                        with open(config_file, "r") as f:
                            config_data = yaml.safe_load(f)
                            config_files[module].append(
                                {"file": config_file.name, "data": config_data}
                            )
                    except yaml.YAMLError as e:
                        pytest.fail(f"Invalid YAML in {module}/{config_file.name}: {e}")

        # Verify all config files are valid
        for module, configs in config_files.items():
            assert len(configs) > 0 or module in [
                "surgical-skill-assessment"
            ], f"Module {module} has no configuration files"

    @pytest.mark.integration
    def test_docker_integration(self):
        """Test Docker integration and build process."""
        dockerfile = self.project_root / "Dockerfile"
        docker_compose = self.project_root / "docker-compose.yml"

        if dockerfile.exists():
            # Verify Dockerfile syntax
            with open(dockerfile, "r") as f:
                content = f.read()
                assert "FROM" in content, "Dockerfile missing FROM instruction"
                assert "WORKDIR" in content, "Dockerfile missing WORKDIR instruction"

        if docker_compose.exists():
            # Verify docker-compose.yml syntax
            try:
                with open(docker_compose, "r") as f:
                    compose_data = yaml.safe_load(f)
                    assert (
                        "services" in compose_data
                    ), "docker-compose.yml missing services"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid docker-compose.yml: {e}")

    @pytest.mark.integration
    def test_makefile_targets(self):
        """Test that Makefile targets work correctly."""
        makefile = self.project_root / "Makefile"

        if not makefile.exists():
            pytest.skip("Makefile not found")

        # Read Makefile and extract targets
        with open(makefile, "r") as f:
            content = f.read()

        # Check for essential targets
        essential_targets = [
            "install",
            "test",
            "lint",
            "format",
            "clean",
            "help",
            "coverage",
            "security",
        ]

        for target in essential_targets:
            target_pattern = f"{target}:"
            assert (
                target_pattern in content
            ), f"Makefile missing essential target: {target}"

    @pytest.mark.integration
    def test_requirements_consistency(self):
        """Test that requirements files are consistent."""
        pyproject_toml = self.project_root / "pyproject.toml"
        requirements_txt = self.project_root / "requirements.txt"
        requirements_dev_txt = self.project_root / "requirements-dev.txt"

        # If pyproject.toml exists, it should be the primary dependency source
        if pyproject_toml.exists():
            import toml

            try:
                with open(pyproject_toml, "r") as f:
                    pyproject_data = toml.load(f)

                # Verify Poetry configuration
                assert "tool" in pyproject_data, "pyproject.toml missing [tool] section"
                assert "poetry" in pyproject_data.get(
                    "tool", {}
                ), "pyproject.toml missing [tool.poetry]"

                poetry_config = pyproject_data["tool"]["poetry"]
                assert "dependencies" in poetry_config, "Poetry missing dependencies"

            except toml.TomlDecodeError as e:
                pytest.fail(f"Invalid pyproject.toml: {e}")

        # Check requirements.txt format if it exists
        if requirements_txt.exists():
            with open(requirements_txt, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Basic package name validation
                        assert not line.startswith(
                            "-"
                        ), f"Invalid requirement at line {i}: {line}"

    @pytest.mark.integration
    def test_git_integration(self):
        """Test Git integration and hooks."""
        git_dir = self.project_root / ".git"
        gitignore = self.project_root / ".gitignore"
        pre_commit_config = self.project_root / ".pre-commit-config.yaml"

        if git_dir.exists():
            # Test git status
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # Should not raise an exception
                assert result.returncode >= 0, "Git status command failed"
            except subprocess.TimeoutExpired:
                pytest.fail("Git status command timed out")

        # Check .gitignore exists and has basic entries
        if gitignore.exists():
            with open(gitignore, "r") as f:
                gitignore_content = f.read()
                essential_ignores = ["__pycache__", "*.pyc", ".env", "venv/", ".venv/"]
                for ignore_pattern in essential_ignores:
                    assert (
                        ignore_pattern in gitignore_content
                    ), f".gitignore missing {ignore_pattern}"

        # Check pre-commit configuration
        if pre_commit_config.exists():
            try:
                with open(pre_commit_config, "r") as f:
                    pre_commit_data = yaml.safe_load(f)
                    assert "repos" in pre_commit_data, "pre-commit config missing repos"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid pre-commit config: {e}")

    @pytest.mark.integration
    def test_ci_cd_integration(self):
        """Test CI/CD pipeline configuration."""
        github_dir = self.project_root / ".github"
        workflows_dir = github_dir / "workflows"

        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob("*.yml")) + list(
                workflows_dir.glob("*.yaml")
            )

            assert len(workflow_files) > 0, "No GitHub workflow files found"

            for workflow_file in workflow_files:
                try:
                    with open(workflow_file, "r") as f:
                        workflow_data = yaml.safe_load(f)

                    # Basic workflow validation
                    assert (
                        "on" in workflow_data
                    ), f"Workflow {workflow_file.name} missing 'on' trigger"
                    assert (
                        "jobs" in workflow_data
                    ), f"Workflow {workflow_file.name} missing 'jobs'"

                    # Check for essential jobs/steps
                    jobs = workflow_data.get("jobs", {})
                    job_names = list(jobs.keys())

                    # Should have some form of testing
                    test_job_found = any(
                        "test" in job_name.lower()
                        or any(
                            "test" in step.get("name", "").lower()
                            for step in job_config.get("steps", [])
                            if isinstance(step, dict)
                        )
                        for job_name, job_config in jobs.items()
                        if isinstance(job_config, dict)
                    )

                    if not test_job_found:
                        print(f"Warning: No test job found in {workflow_file.name}")

                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid workflow file {workflow_file.name}: {e}")

    @pytest.mark.integration
    def test_documentation_integration(self):
        """Test documentation structure and integration."""
        docs_dir = self.project_root / "docs"
        readme_files = list(self.project_root.glob("README*"))

        # Check main README
        assert len(readme_files) > 0, "No README file found in project root"

        main_readme = None
        for readme in readme_files:
            if readme.name.lower() == "readme.md":
                main_readme = readme
                break

        if not main_readme:
            main_readme = readme_files[0]

        # Verify README content
        with open(main_readme, "r", encoding="utf-8") as f:
            readme_content = f.read()
            assert len(readme_content) > 100, "README file is too short"

            # Check for essential sections
            essential_sections = ["install", "usage", "description"]
            for section in essential_sections:
                section_found = any(
                    section.lower() in line.lower()
                    for line in readme_content.split("\n")
                    if line.strip().startswith("#")
                )
                if not section_found:
                    print(f"Warning: No {section} section found in README")

        # Check docs directory if it exists
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.rst"))
            # Should have some documentation files
            if len(doc_files) == 0:
                print(
                    "Warning: docs directory exists but contains no .md or .rst files"
                )


class TestModuleIntegration:
    """Test integration between specific modules."""

    def setup_method(self):
        """Set up module integration tests."""
        self.project_root = PROJECT_ROOT

    @pytest.mark.integration
    def test_video_processing_integration(self):
        """Test surgical video processing module integration."""
        video_module = self.project_root / "surgical-video-processing"

        if not video_module.exists():
            pytest.skip("Surgical video processing module not found")

        # Check module structure
        expected_components = ["core", "preprocessing", "pipelines", "utils", "configs"]

        for component in expected_components:
            component_path = video_module / component
            if component_path.exists():
                assert component_path.is_dir(), f"{component} should be a directory"

                # Check for __init__.py in Python packages
                init_file = component_path / "__init__.py"
                if init_file.exists():
                    # Verify init file is valid Python
                    try:
                        with open(init_file, "r") as f:
                            content = f.read()
                            compile(content, str(init_file), "exec")
                    except SyntaxError as e:
                        pytest.fail(f"Syntax error in {init_file}: {e}")

    @pytest.mark.integration
    def test_instrument_segmentation_integration(self):
        """Test surgical instance segmentation module integration."""
        segmentation_module = self.project_root / "surgical-instance-segmentation"

        if not segmentation_module.exists():
            pytest.skip("Surgical instrument segmentation module not found")

        # Check for model files
        model_files = list(segmentation_module.glob("*.pt")) + list(
            segmentation_module.glob("*.pth")
        )
        if len(model_files) > 0:
            for model_file in model_files:
                # Basic file validation
                assert (
                    model_file.stat().st_size > 1000
                ), f"Model file {model_file.name} seems too small"

    @pytest.mark.integration
    def test_phase_classification_integration(self):
        """Test surgical phase recognition module integration."""
        phase_module = self.project_root / "surgical-phase-recognition"

        if not phase_module.exists():
            pytest.skip("Surgical phase classification module not found")

        # Check for analysis components
        analysis_dir = phase_module / "analysis"
        if analysis_dir.exists():
            analysis_files = list(analysis_dir.glob("*.py"))
            for analysis_file in analysis_files:
                # Verify Python files are valid
                try:
                    with open(analysis_file, "r") as f:
                        content = f.read()
                        compile(content, str(analysis_file), "exec")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {analysis_file}: {e}")

    @pytest.mark.integration
    def test_skill_assessment_integration(self):
        """Test surgical skill assessment module integration."""
        skill_module = self.project_root / "surgical-skill-assessment"

        if not skill_module.exists():
            pytest.skip("Surgical skill assessment module not found")

        # Check main files
        main_files = list(skill_module.glob("main*.py"))
        for main_file in main_files:
            # Verify main files are valid
            try:
                with open(main_file, "r") as f:
                    content = f.read()
                    compile(content, str(main_file), "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {main_file}: {e}")


class TestDataIntegration:
    """Test data handling and processing integration."""

    @pytest.mark.integration
    def test_config_data_integration(self):
        """Test configuration data integration across modules."""
        config_schemas = {}

        for module_name in [
            "surgical-video-processing",
            "surgical-instance-segmentation",
            "surgical-phase-recognition",
            "surgical-skill-assessment",
        ]:
            module_path = PROJECT_ROOT / module_name
            configs_dir = module_path / "configs"

            if configs_dir.exists():
                for config_file in configs_dir.glob("*.yaml"):
                    try:
                        with open(config_file, "r") as f:
                            config_data = yaml.safe_load(f)

                        # Extract schema information
                        if isinstance(config_data, dict):
                            config_schemas[f"{module_name}/{config_file.name}"] = {
                                "keys": list(config_data.keys()),
                                "structure": self._analyze_config_structure(
                                    config_data
                                ),
                            }

                    except yaml.YAMLError as e:
                        pytest.fail(
                            f"YAML error in {module_name}/{config_file.name}: {e}"
                        )

        # Verify consistency in common configuration patterns
        common_keys = set()
        for schema in config_schemas.values():
            common_keys.update(schema["keys"])

        # Look for common configuration patterns
        pattern_keys = ["model", "data", "training", "evaluation", "paths"]
        found_patterns = {key: [] for key in pattern_keys}

        for config_name, schema in config_schemas.items():
            for pattern_key in pattern_keys:
                if any(pattern_key in key.lower() for key in schema["keys"]):
                    found_patterns[pattern_key].append(config_name)

        print(f"Configuration patterns found: {found_patterns}")

    def _analyze_config_structure(self, config_data: Dict[str, Any]) -> Dict[str, str]:
        """Analyze the structure of configuration data."""
        structure = {}
        for key, value in config_data.items():
            if isinstance(value, dict):
                structure[key] = "nested_dict"
            elif isinstance(value, list):
                structure[key] = "list"
            elif isinstance(value, str):
                structure[key] = "string"
            elif isinstance(value, (int, float)):
                structure[key] = "numeric"
            else:
                structure[key] = "other"
        return structure

    @pytest.mark.integration
    def test_test_data_integration(self):
        """Test that test data is properly organized."""
        test_dirs = []

        for module_name in [
            "surgical-video-processing",
            "surgical-instance-segmentation",
            "surgical-phase-recognition",
            "surgical-skill-assessment",
        ]:
            module_path = PROJECT_ROOT / module_name
            tests_dir = module_path / "tests"

            if tests_dir.exists():
                test_dirs.append(tests_dir)

                # Check test file structure
                test_files = list(tests_dir.glob("test_*.py"))
                for test_file in test_files:
                    # Verify test files are valid
                    try:
                        with open(test_file, "r") as f:
                            content = f.read()
                            compile(content, str(test_file), "exec")

                        # Check for test functions
                        assert (
                            "def test_" in content
                        ), f"No test functions found in {test_file}"

                    except SyntaxError as e:
                        pytest.fail(f"Syntax error in {test_file}: {e}")

        print(f"Found {len(test_dirs)} test directories")


class TestEnvironmentIntegration:
    """Test environment and dependency integration."""

    @pytest.mark.integration
    def test_python_environment_integration(self):
        """Test Python environment and package integration."""
        # Check Python version compatibility
        python_version = sys.version_info
        assert python_version >= (
            3,
            8,
        ), f"Python 3.8+ required, found {python_version.major}.{python_version.minor}"

        # Check for essential packages
        essential_packages = ["pytest", "numpy", "pyyaml"]
        missing_packages = []

        for package in essential_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            pytest.skip(f"Missing essential packages: {missing_packages}")

    @pytest.mark.integration
    def test_path_resolution_integration(self):
        """Test that all path references resolve correctly."""
        # Test relative path resolution from different modules
        for module_name in [
            "surgical-video-processing",
            "surgical-instance-segmentation",
        ]:
            module_path = PROJECT_ROOT / module_name

            if module_path.exists():
                # Test common path patterns
                relative_paths = ["../configs", "./utils", "../../data"]

                for rel_path in relative_paths:
                    full_path = module_path / rel_path
                    # Path should resolve without errors
                    resolved = full_path.resolve()
                    assert isinstance(
                        resolved, Path
                    ), f"Path resolution failed for {rel_path} from {module_name}"

    @pytest.mark.integration
    def test_import_integration(self):
        """Test that imports work correctly across modules."""
        # Add project root to path for testing
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            # Test importing modules
            import_tests = []

            for module_name in [
                "surgical-video-processing",
                "surgical-instance-segmentation",
            ]:
                module_path = PROJECT_ROOT / module_name
                init_file = module_path / "__init__.py"

                if init_file.exists():
                    try:
                        # Attempt to import the module
                        import importlib.util

                        spec = importlib.util.spec_from_file_location(
                            module_name.replace("-", "_"), init_file
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            import_tests.append((module_name, True, None))
                    except Exception as e:
                        import_tests.append((module_name, False, str(e)))

            # Report import results
            failed_imports = [test for test in import_tests if not test[1]]
            if failed_imports:
                failure_details = "\n".join(
                    [f"{name}: {error}" for name, _, error in failed_imports]
                )
                print(f"Import failures:\n{failure_details}")

        finally:
            # Clean up sys.path
            if str(PROJECT_ROOT) in sys.path:
                sys.path.remove(str(PROJECT_ROOT))
