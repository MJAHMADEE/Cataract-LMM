# 🤝 Contributing to Cataract-LMM

Thank you for your interest in contributing to the Cataract-LMM project! This document provides comprehensive guidelines for contributing to our surgical AI research framework.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

---

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inspiring community for all.

### Our Pledge
- **Respectful**: Treat all participants with respect and courtesy
- **Inclusive**: Welcome contributors from diverse backgrounds and experience levels
- **Collaborative**: Foster open communication and constructive feedback
- **Professional**: Maintain high standards of professional conduct
- **Medical Ethics**: Uphold ethical standards in medical AI research

---

## 🚀 Getting Started

### Prerequisites
Before contributing, ensure you have:
- Python 3.8+ installed
- Git configured with your username and email
- Familiarity with PyTorch and computer vision concepts
- Understanding of medical imaging and surgical workflows (helpful but not required)

### Areas for Contribution

We welcome contributions in several areas:

#### 🧠 **AI/ML Models**
- New model architectures for surgical analysis
- Performance optimizations and improvements
- Novel training strategies and techniques
- Model compression and quantization

#### 🛠️ **Engineering & Infrastructure**
- CI/CD pipeline improvements
- Docker and containerization enhancements
- Performance monitoring and observability
- Security and compliance features

#### 📚 **Documentation & Education**
- Tutorials and educational content
- API documentation improvements
- Code examples and use cases
- Translation to other languages

#### 🧪 **Testing & Quality Assurance**
- Unit and integration tests
- Performance benchmarks
- Security testing
- Edge case validation

#### 🎯 **Data & Preprocessing**
- Data augmentation techniques
- Preprocessing optimizations
- Dataset utilities and tools
- Quality control mechanisms

---

## 🔧 Development Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Cataract_LMM.git
cd Cataract_LMM

# Add the original repository as upstream
git remote add upstream https://github.com/MJAHMADEE/Cataract_LMM.git
```

### 2. Set Up Development Environment

```bash
# Navigate to the codes directory
cd codes

# Option A: Using Poetry (Recommended)
poetry install --extras "dev docs"
poetry shell

# Option B: Using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 3. Install Development Tools

```bash
# Install pre-commit hooks
pre-commit install

# Verify installation
make dev-check
```

### 4. Run Initial Tests

```bash
# Run the test suite to ensure everything works
pytest tests/ -v

# Run validation script
python setup.py --validate-only
```

---

## 🔄 Contribution Workflow

### Branching Strategy

We follow **Git Flow** branching model:

```
main                    # Production-ready code
├── develop            # Integration branch for features
├── feature/           # Feature development branches
├── bugfix/           # Bug fix branches
├── hotfix/           # Critical production fixes
└── release/          # Release preparation branches
```

### Step-by-Step Workflow

#### 1. **Sync with Upstream**
```bash
git checkout develop
git pull upstream develop
```

#### 2. **Create Feature Branch**
```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Examples:
git checkout -b feature/yolo-v11-integration
git checkout -b feature/phase-recognition-optimization
git checkout -b bugfix/memory-leak-training
```

#### 3. **Make Changes**
- Write clean, well-documented code
- Follow our coding standards
- Add appropriate tests
- Update documentation if needed

#### 4. **Test Your Changes**
```bash
# Run pre-commit checks
pre-commit run --all-files

# Run tests
make test

# Run specific test categories
make test-unit
make test-integration

# Check code quality
make lint
make type-check
```

#### 5. **Commit Changes**
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add YOLOv11 integration for surgical instrument detection

- Implement YOLOv11 model wrapper
- Add configuration support for v11 models
- Update training pipeline for new architecture
- Add comprehensive tests for new functionality

Closes #123"
```

#### 6. **Push and Create Pull Request**
```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
gh pr create --title "feat: YOLOv11 integration" --body "Description of changes..."
```

---

## 📝 Coding Standards

### Code Style

We enforce consistent code style using automated tools:

#### **Python Code Formatting**
```bash
# Black formatter (line length: 88)
black .

# Import sorting with isort
isort .

# Run both
make format
```

#### **Linting and Type Checking**
```bash
# Flake8 linting
flake8 .

# MyPy type checking
mypy .

# Run all checks
make lint
```

### Naming Conventions

#### **Files and Directories**
```python
# Snake case for Python files
model_factory.py
data_utils.py
training_pipeline.py

# Kebab case for directories
surgical-instance-segmentation/
surgical-phase-recognition/
```

#### **Python Code**
```python
# Variables and functions: snake_case
def process_surgical_video(input_path: str) -> VideoMetadata:
    frame_count = 0
    processing_time = time.time()

# Classes: PascalCase
class SurgicalPhaseClassifier:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_name = model_config["name"]

# Constants: SCREAMING_SNAKE_CASE
MAX_VIDEO_LENGTH = 3600
DEFAULT_BATCH_SIZE = 16
SUPPORTED_FORMATS = [".mp4", ".avi", ".mov"]
```

#### **Configuration Keys**
```yaml
# Use snake_case in YAML configurations
model:
  architecture_name: "resnet50"
  learning_rate: 0.001
  batch_size: 32
  
training:
  max_epochs: 100
  early_stopping_patience: 10
```

### Documentation Standards

#### **Docstring Format**
We use Google-style docstrings:

```python
def train_segmentation_model(
    model: torch.nn.Module,
    dataset: Dataset,
    config: Dict[str, Any]
) -> TrainingResults:
    """Train a surgical instance segmentation model.
    
    This function implements the complete training pipeline for surgical
    instrument segmentation, including data loading, model training,
    validation, and checkpointing.
    
    Args:
        model: PyTorch model to train (YOLOv8, Mask R-CNN, etc.)
        dataset: Training dataset with surgical images and annotations
        config: Training configuration dictionary containing:
            - learning_rate: Learning rate for optimizer
            - batch_size: Training batch size
            - epochs: Number of training epochs
            - device: Training device ('cuda' or 'cpu')
    
    Returns:
        TrainingResults object containing:
            - best_model_path: Path to best model checkpoint
            - training_metrics: Dictionary of training metrics
            - validation_metrics: Dictionary of validation metrics
            - training_time: Total training time in seconds
    
    Raises:
        ValueError: If invalid configuration parameters are provided
        RuntimeError: If GPU is required but not available
        
    Examples:
        >>> model = YOLOv8(num_classes=4)
        >>> dataset = SurgicalDataset("data/train")
        >>> config = {"learning_rate": 0.001, "batch_size": 16, "epochs": 100}
        >>> results = train_segmentation_model(model, dataset, config)
        >>> print(f"Best mAP: {results.validation_metrics['map_50']:.3f}")
    """
```

#### **Type Annotations**
Always include type annotations for functions:

```python
from typing import Dict, List, Optional, Tuple, Union
import torch
from pathlib import Path

def load_model_weights(
    model: torch.nn.Module,
    weights_path: Union[str, Path],
    device: str = "cpu",
    strict: bool = True
) -> Dict[str, torch.Tensor]:
    """Load model weights with proper error handling."""
    ...

def process_batch(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Process a batch of images through the model."""
    ...
```

---

## 🧪 Testing Guidelines

### Test Structure

Our testing strategy follows the testing pyramid:

```
tests/
├── unit/                  # Fast, isolated unit tests (80% of tests)
│   ├── models/           
│   ├── data/
│   ├── training/
│   └── utils/
├── integration/          # Module interaction tests (15% of tests)  
│   ├── pipelines/
│   ├── workflows/
│   └── api/
├── e2e/                 # End-to-end workflow tests (5% of tests)
│   ├── full_pipeline/
│   └── user_scenarios/
├── performance/         # Performance and benchmark tests
├── security/           # Security and vulnerability tests
└── fixtures/           # Test data and fixtures
```

### Writing Tests

#### **Unit Tests**
```python
import pytest
import torch
from unittest.mock import Mock, patch
from surgical_instance_segmentation.models import YOLOv8

class TestYOLOv8Model:
    """Test suite for YOLOv8 model implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "model_size": "medium",
            "num_classes": 4,
            "pretrained": False
        }
        self.model = YOLOv8(self.config)
    
    def test_model_initialization(self):
        """Test model initializes with correct architecture."""
        assert self.model.num_classes == 4
        assert isinstance(self.model.backbone, torch.nn.Module)
        assert self.model.training is True
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        batch_size, channels, height, width = 2, 3, 640, 640
        input_tensor = torch.randn(batch_size, channels, height, width)
        
        output = self.model(input_tensor)
        
        assert len(output) == 3  # Three detection layers
        for detection_layer in output:
            assert detection_layer.dim() == 3
            assert detection_layer.shape[0] == batch_size
    
    @pytest.mark.gpu
    def test_gpu_inference(self):
        """Test model works correctly on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        device = torch.device("cuda")
        model = self.model.to(device)
        input_tensor = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
        assert all(out.device.type == "cuda" for out in output)
```

#### **Integration Tests**
```python
class TestTrainingPipeline:
    """Test complete training pipeline integration."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a small sample dataset for testing."""
        return create_sample_surgical_dataset(num_samples=50)
    
    def test_full_training_loop(self, sample_dataset):
        """Test complete training pipeline runs without errors."""
        config = {
            "model": {"architecture": "yolov8", "size": "nano"},
            "training": {"epochs": 2, "batch_size": 4},
            "data": {"num_workers": 0}  # Disable multiprocessing for tests
        }
        
        trainer = TrainingPipeline(config)
        results = trainer.train(sample_dataset)
        
        assert results.completed_epochs == 2
        assert "loss" in results.metrics
        assert results.best_model_path.exists()
```

### Test Markers

Use pytest markers to categorize tests:

```python
# Slow running tests
@pytest.mark.slow
def test_full_dataset_processing():
    """Test processing of complete dataset (runs in CI nightly only)."""
    
# GPU required tests  
@pytest.mark.gpu
def test_multi_gpu_training():
    """Test distributed training across multiple GPUs."""
    
# Integration tests
@pytest.mark.integration
def test_model_pipeline_integration():
    """Test integration between model and data pipeline."""
    
# Security tests
@pytest.mark.security
def test_input_validation_against_malicious_data():
    """Test security against malicious input data."""
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific categories
pytest -m "not slow"           # Skip slow tests
pytest -m "unit"              # Run only unit tests
pytest -m "integration"       # Run only integration tests
pytest -m "gpu" --gpu-id=0   # Run GPU tests on specific GPU

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/models/test_yolo.py -v

# Run specific test
pytest tests/unit/models/test_yolo.py::TestYOLOv8Model::test_forward_pass_shape
```

---

## 📖 Documentation

### Documentation Types

#### **Code Documentation**
- Comprehensive docstrings for all public functions and classes
- Type annotations for all function parameters and return values
- Inline comments for complex algorithms and medical domain logic

#### **User Documentation**
- README updates for new features
- Tutorial notebooks for new capabilities
- API reference updates
- Configuration documentation

#### **Developer Documentation**
- Architecture decision records (ADRs)
- Technical design documents
- Contribution guidelines updates
- Development environment setup

### Building Documentation

```bash
# Install documentation dependencies
poetry install --extras "docs"

# Build HTML documentation
cd docs
make html

# Build and serve documentation locally
make livehtml

# Generate API documentation
make apidoc

# Build PDF documentation
make latexpdf
```

---

## 📋 Pull Request Process

### Before Submitting

**Checklist:**
- [ ] Code follows style guidelines (run `make format lint`)
- [ ] All tests pass (run `make test`)
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages follow conventional commit format
- [ ] Pre-commit hooks pass

### Pull Request Template

When creating a pull request, use this template:

```markdown
## 🎯 Description
Brief description of the changes made.

## 🔗 Related Issues
Closes #123
Addresses #456

## 🧪 Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## 🧪 Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] Performance impact assessed

## 📋 Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Corresponding changes to documentation made
- [ ] No new warnings introduced
- [ ] Tests added that prove fix is effective or feature works
- [ ] New and existing unit tests pass locally

## 🖼️ Screenshots (if applicable)
Add screenshots or videos demonstrating the changes.

## 🚀 Deployment Notes
Any special considerations for deployment or configuration.
```

### Review Process

1. **Automated Checks**: All CI/CD checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Manual testing by reviewers if applicable
4. **Documentation**: Verify documentation updates
5. **Approval**: Maintainer approval required for merge

### Merge Strategy

- **Feature branches**: Squash and merge to develop
- **Bug fixes**: Squash and merge to develop
- **Hotfixes**: Merge to main and cherry-pick to develop
- **Releases**: Merge develop to main

---

## 🐛 Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for solutions
3. **Try the latest version** to see if issue persists
4. **Gather relevant information** (system specs, error logs, etc.)

### Issue Types

#### **🐛 Bug Report**
```markdown
**Describe the Bug**
Clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 1.12.0]
- CUDA version: [e.g., 11.6]
- GPU: [e.g., RTX 3090]

**Additional Context**
Add any other context about the problem.
```

#### **💡 Feature Request**
```markdown
**Is your feature request related to a problem?**
Clear description of the problem.

**Describe the solution you'd like**
Clear description of desired solution.

**Describe alternatives considered**
Other solutions or features considered.

**Additional context**
Any other context or screenshots.
```

#### **📖 Documentation Issue**
```markdown
**Documentation Section**
Which section needs improvement?

**Issue Description**
What is unclear or missing?

**Suggested Improvement**
How should it be improved?
```

---

## 🌟 Community

### Communication Channels

- **GitHub Discussions**: General discussions, questions, and announcements
- **GitHub Issues**: Bug reports and feature requests
- **Email**: Direct contact with maintainers (mjahmadee@kntu.ac.ir)

### Recognition

We recognize contributors through:
- **Contributors Graph**: Automatic GitHub recognition
- **Release Notes**: Contributors mentioned in releases
- **Hall of Fame**: Special recognition for significant contributions
- **Conference Presentations**: Opportunity to co-present research

### Mentorship Program

We offer mentorship for:
- **New Contributors**: Getting started with the codebase
- **Students**: Academic projects and research collaboration
- **Industry Professionals**: Applying surgical AI in practice
- **Open Source Beginners**: Learning open source contribution

---

## 📚 Resources

### Learning Resources
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Computer Vision**: CS231n Stanford Course
- **Medical Imaging**: Medical Image Analysis fundamentals
- **Git Workflow**: Atlassian Git tutorials

### Medical AI Ethics
- **HIPAA Compliance**: Understanding medical data privacy
- **AI Ethics in Healthcare**: Responsible AI development
- **Bias in Medical AI**: Addressing algorithmic bias
- **Regulatory Considerations**: FDA guidelines for AI/ML

### Development Tools
- **VS Code Extensions**: Python, PyTorch, GitLens
- **Pre-commit Hooks**: Automated code quality checks
- **Docker**: Containerization for reproducible environments
- **Weights & Biases**: Experiment tracking and model management

---

## 🙏 Thank You

Your contributions help advance the field of surgical AI and improve patient outcomes worldwide. Every contribution, no matter how small, makes a difference in building better surgical assistance technologies.

**Happy Contributing! 🚀**

---

*For questions about contributing, please reach out through GitHub Discussions or email the maintainers directly.*
