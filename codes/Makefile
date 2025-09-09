# ============================================================================
# Cataract-LMM Unified Development Interface
# 
# A comprehensive Makefile providing standardized commands for all 
# development tasks in the Cataract-LMM project
# ============================================================================

.PHONY: help install install-dev setup format lint test test-unit test-integration test-performance
.PHONY: coverage coverage-report security build clean docker-build docker-run validate
.PHONY: docs docs-serve pre-commit update check-deps audit release

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# Configuration Variables
# ============================================================================
PROJECT_NAME := cataract-lmm
PYTHON_VERSION := 3.8
DOCKER_IMAGE := $(PROJECT_NAME):latest
DOCKER_DEV_IMAGE := $(PROJECT_NAME):dev
POETRY := poetry
PYTEST := pytest
COVERAGE := coverage
REPORTS_DIR := reports
DOCS_DIR := docs

# Colors for terminal output
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
BLUE := \033[34m
RESET := \033[0m

# ============================================================================
# Help Target
# ============================================================================
help: ## Display this help message
	@echo "$(BOLD)$(BLUE)🏥 Cataract-LMM Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Setup & Installation:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; category=""} /^## / {category=$$1; gsub(/^## /, "", category); print "\n$(BOLD)" category "$(RESET)"} /^[a-zA-Z_-]+:.*?##/ {if(category=="") print "$(BOLD)General:$(RESET)"; printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## Setup & Installation
install: ## Install production dependencies using Poetry
	@echo "$(BOLD)$(BLUE)📦 Installing production dependencies...$(RESET)"
	$(POETRY) install --only=main
	@echo "$(GREEN)✅ Production dependencies installed successfully$(RESET)"

install-dev: ## Install all dependencies including dev tools
	@echo "$(BOLD)$(BLUE)📦 Installing all dependencies...$(RESET)"
	$(POETRY) install --with=dev
	@echo "$(GREEN)✅ All dependencies installed successfully$(RESET)"

setup: install-dev ## Complete project setup (install deps + pre-commit)
	@echo "$(BOLD)$(BLUE)🔧 Setting up development environment...$(RESET)"
	$(POETRY) run pre-commit install
	@mkdir -p $(REPORTS_DIR)/coverage
	@mkdir -p logs
	@mkdir -p checkpoints
	@echo "$(GREEN)✅ Development environment setup complete$(RESET)"

## Code Quality & Formatting
format: ## Format all Python code using black and isort
	@echo "$(BOLD)$(BLUE)🎨 Formatting code...$(RESET)"
	$(POETRY) run black .
	$(POETRY) run isort .
	@echo "$(GREEN)✅ Code formatting complete$(RESET)"

lint: ## Run all linting checks (flake8, mypy, bandit)
	@echo "$(BOLD)$(BLUE)🔍 Running linting checks...$(RESET)"
	@echo "$(YELLOW)Running flake8...$(RESET)"
	$(POETRY) run flake8 .
	@echo "$(YELLOW)Running mypy...$(RESET)"
	$(POETRY) run mypy codes --ignore-missing-imports
	@echo "$(YELLOW)Running bandit security check...$(RESET)"
	$(POETRY) run bandit -r codes -f json -o $(REPORTS_DIR)/bandit_report.json || true
	@echo "$(GREEN)✅ Linting checks complete$(RESET)"

pre-commit: ## Run all pre-commit hooks manually
	@echo "$(BOLD)$(BLUE)🔒 Running pre-commit hooks...$(RESET)"
	$(POETRY) run pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit hooks executed$(RESET)"

## Testing
test: ## Run all tests with coverage
	@echo "$(BOLD)$(BLUE)🧪 Running all tests...$(RESET)"
	$(POETRY) run $(PYTEST) --maxfail=5 --tb=short
	@echo "$(GREEN)✅ All tests completed$(RESET)"

test-unit: ## Run only unit tests
	@echo "$(BOLD)$(BLUE)🧪 Running unit tests...$(RESET)"
	$(POETRY) run $(PYTEST) -m "unit" --maxfail=5
	@echo "$(GREEN)✅ Unit tests completed$(RESET)"

test-integration: ## Run only integration tests
	@echo "$(BOLD)$(BLUE)🧪 Running integration tests...$(RESET)"
	$(POETRY) run $(PYTEST) -m "integration" --maxfail=5
	@echo "$(GREEN)✅ Integration tests completed$(RESET)"

test-performance: ## Run performance tests
	@echo "$(BOLD)$(BLUE)🧪 Running performance tests...$(RESET)"
	$(POETRY) run $(PYTEST) -m "performance" --maxfail=5
	@echo "$(GREEN)✅ Performance tests completed$(RESET)"

test-parallel: ## Run tests in parallel for faster execution
	@echo "$(BOLD)$(BLUE)🧪 Running tests in parallel...$(RESET)"
	$(POETRY) run $(PYTEST) -n auto --maxfail=5
	@echo "$(GREEN)✅ Parallel tests completed$(RESET)"

## Coverage
coverage: ## Generate test coverage report
	@echo "$(BOLD)$(BLUE)📊 Generating coverage report...$(RESET)"
	$(POETRY) run $(PYTEST) --cov=codes --cov-report=html:$(REPORTS_DIR)/coverage --cov-report=term-missing --cov-report=xml:$(REPORTS_DIR)/coverage.xml
	@echo "$(GREEN)✅ Coverage report generated$(RESET)"
	@echo "$(BLUE)📄 HTML report: $(REPORTS_DIR)/coverage/index.html$(RESET)"

coverage-report: coverage ## Generate and open coverage report
	@echo "$(BOLD)$(BLUE)📊 Opening coverage report...$(RESET)"
	@python -c "import webbrowser; webbrowser.open('file://$(PWD)/$(REPORTS_DIR)/coverage/index.html')" 2>/dev/null || echo "$(YELLOW)Please open $(REPORTS_DIR)/coverage/index.html manually$(RESET)"

## Security & Auditing
security: ## Run security scans (bandit + dependency audit)
	@echo "$(BOLD)$(BLUE)🔒 Running security scans...$(RESET)"
	@echo "$(YELLOW)Running Bandit security scan...$(RESET)"
	$(POETRY) run bandit -r codes -f json -o $(REPORTS_DIR)/bandit_report.json
	@echo "$(YELLOW)Running dependency vulnerability scan...$(RESET)"
	$(POETRY) export -f requirements.txt --output $(REPORTS_DIR)/requirements_export.txt
	$(POETRY) run pip-audit -r $(REPORTS_DIR)/requirements_export.txt --format=json --output=$(REPORTS_DIR)/pip_audit_report.json || true
	@echo "$(GREEN)✅ Security scans completed$(RESET)"
	@echo "$(BLUE)📄 Reports: $(REPORTS_DIR)/bandit_report.json, $(REPORTS_DIR)/pip_audit_report.json$(RESET)"

audit: security ## Alias for security scan

## Docker
docker-build: ## Build Docker container
	@echo "$(BOLD)$(BLUE)🐳 Building Docker container...$(RESET)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✅ Docker container built: $(DOCKER_IMAGE)$(RESET)"

docker-dev-build: ## Build development Docker container
	@echo "$(BOLD)$(BLUE)🐳 Building development Docker container...$(RESET)"
	docker build -f Dockerfile.dev -t $(DOCKER_DEV_IMAGE) . || docker build -t $(DOCKER_DEV_IMAGE) .
	@echo "$(GREEN)✅ Development Docker container built: $(DOCKER_DEV_IMAGE)$(RESET)"

docker-run: ## Run Docker container interactively
	@echo "$(BOLD)$(BLUE)🐳 Running Docker container...$(RESET)"
	docker run -it --rm -v $(PWD):/workspace $(DOCKER_IMAGE) /bin/bash

docker-scan: ## Scan Docker image for vulnerabilities using Trivy
	@echo "$(BOLD)$(BLUE)🔍 Scanning Docker image for vulnerabilities...$(RESET)"
	@command -v trivy >/dev/null 2>&1 || { echo "$(YELLOW)Trivy not found. Installing...$(RESET)"; curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin; }
	trivy image --format json --output $(REPORTS_DIR)/trivy_report.json $(DOCKER_IMAGE) || true
	@echo "$(GREEN)✅ Docker vulnerability scan completed$(RESET)"

## Build & Distribution
build: ## Build the package
	@echo "$(BOLD)$(BLUE)📦 Building package...$(RESET)"
	$(POETRY) build
	@echo "$(GREEN)✅ Package built successfully$(RESET)"

## Environment Management
update: ## Update all dependencies
	@echo "$(BOLD)$(BLUE)⬆️  Updating dependencies...$(RESET)"
	$(POETRY) update
	$(POETRY) run pre-commit autoupdate
	@echo "$(GREEN)✅ Dependencies updated$(RESET)"

check-deps: ## Check for dependency issues
	@echo "$(BOLD)$(BLUE)🔍 Checking dependencies...$(RESET)"
	$(POETRY) check
	$(POETRY) run pip check
	@echo "$(GREEN)✅ Dependencies check completed$(RESET)"

## Validation & CI
validate: lint test coverage security ## Run complete validation pipeline
	@echo "$(BOLD)$(GREEN)🎉 Complete validation pipeline passed!$(RESET)"

ci: ## Run CI pipeline locally
	@echo "$(BOLD)$(BLUE)🔄 Running CI pipeline locally...$(RESET)"
	$(MAKE) install-dev
	$(MAKE) validate
	$(MAKE) docker-build
	$(MAKE) docker-scan
	@echo "$(BOLD)$(GREEN)🎉 CI pipeline completed successfully!$(RESET)"

## Documentation
docs: ## Build documentation
	@echo "$(BOLD)$(BLUE)📚 Building documentation...$(RESET)"
	$(POETRY) run sphinx-build -b html $(DOCS_DIR) $(REPORTS_DIR)/docs
	@echo "$(GREEN)✅ Documentation built$(RESET)"
	@echo "$(BLUE)📄 Documentation: $(REPORTS_DIR)/docs/index.html$(RESET)"

docs-serve: docs ## Build and serve documentation locally
	@echo "$(BOLD)$(BLUE)📚 Serving documentation...$(RESET)"
	@python -c "import webbrowser; webbrowser.open('file://$(PWD)/$(REPORTS_DIR)/docs/index.html')" 2>/dev/null || echo "$(YELLOW)Please open $(REPORTS_DIR)/docs/index.html manually$(RESET)"

## Cleanup
clean: ## Clean up build artifacts and cache files
	@echo "$(BOLD)$(BLUE)🧹 Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -rf .tox/
	rm -rf htmlcov/
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

clean-all: clean ## Complete cleanup including reports and logs
	@echo "$(BOLD)$(BLUE)🧹 Complete cleanup...$(RESET)"
	rm -rf $(REPORTS_DIR)/
	rm -rf logs/
	rm -rf checkpoints/
	@echo "$(GREEN)✅ Complete cleanup finished$(RESET)"

## Release
release: ## Prepare and validate for release
	@echo "$(BOLD)$(BLUE)🚀 Preparing for release...$(RESET)"
	$(MAKE) clean
	$(MAKE) validate
	$(MAKE) build
	$(MAKE) docker-build
	@echo "$(BOLD)$(GREEN)🎉 Release preparation completed!$(RESET)"

## Module-specific commands
train-segmentation: ## Train instrument segmentation model
	@echo "$(BOLD)$(BLUE)🤖 Training segmentation model...$(RESET)"
	$(POETRY) run python -m codes.surgical-instance-segmentation.training.training_manager

train-phase: ## Train phase classification model
	@echo "$(BOLD)$(BLUE)🤖 Training phase classification model...$(RESET)"
	$(POETRY) run python -m codes.surgical-phase-recognition.training.main

train-skill: ## Train skill assessment model
	@echo "$(BOLD)$(BLUE)🤖 Training skill assessment model...$(RESET)"
	$(POETRY) run python -m codes.surgical-skill-assessment.main

process-videos: ## Process videos
	@echo "$(BOLD)$(BLUE)🎬 Processing videos...$(RESET)"
	$(POETRY) run python -m codes.surgical-video-processing.main

## Development utilities
dev-install: ## Install development tools globally
	@echo "$(BOLD)$(BLUE)🛠️  Installing development tools...$(RESET)"
	pip install pre-commit poetry twine wheel
	@echo "$(GREEN)✅ Development tools installed$(RESET)"

env-info: ## Display environment information
	@echo "$(BOLD)$(BLUE)ℹ️  Environment Information$(RESET)"
	@echo "Python version: $$(python --version)"
	@echo "Poetry version: $$(poetry --version)"
	@echo "Working directory: $$(pwd)"
	@echo "Virtual environment: $$(poetry env info --path)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git status: $$(git status --porcelain 2>/dev/null | wc -l) modified files"

# ============================================================================
# Advanced Targets
# ============================================================================

stress-test: ## Run stress tests with large datasets
	@echo "$(BOLD)$(BLUE)💪 Running stress tests...$(RESET)"
	$(POETRY) run $(PYTEST) -m "slow or performance" --maxfail=1 --tb=line

benchmark: ## Run performance benchmarks
	@echo "$(BOLD)$(BLUE)⏱️  Running benchmarks...$(RESET)"
	$(POETRY) run python -m pytest benchmarks/ -v --benchmark-only

profile: ## Profile the application
	@echo "$(BOLD)$(BLUE)📊 Profiling application...$(RESET)"
	$(POETRY) run python -m cProfile -o $(REPORTS_DIR)/profile.stats -m codes.main
	@echo "$(GREEN)✅ Profile saved to $(REPORTS_DIR)/profile.stats$(RESET)"

memory-profile: ## Run memory profiling
	@echo "$(BOLD)$(BLUE)🧠 Running memory profiling...$(RESET)"
	$(POETRY) run mprof run python -m codes.main
	$(POETRY) run mprof plot -o $(REPORTS_DIR)/memory_profile.png
	@echo "$(GREEN)✅ Memory profile saved to $(REPORTS_DIR)/memory_profile.png$(RESET)"

install-gpu: ## Install GPU-specific packages
	@echo "$(BOLD)$(BLUE)🎮 Installing GPU packages...$(RESET)"
	$(POETRY) install --extras gpu
	@echo "$(GREEN)✅ GPU packages installed$(RESET)"

install-all: ## Install all optional dependencies
	@echo "$(BOLD)$(BLUE)📦 Installing all optional dependencies...$(RESET)"
	$(POETRY) install --extras all
	@echo "$(GREEN)✅ All optional dependencies installed$(RESET)"
