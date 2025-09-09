# 🧪 Comprehensive Testing Suite

This directory contains the comprehensive test suite for the surgical phase recognition system, ensuring reliability, correctness, and performance across all components with extensive validation coverage.

## 📚 Reference Notebook Integration

All test components validate **full compatibility** with the primary validation notebook [`phase_validation_comprehensive.ipynb`](../../Validation.ipynb), ensuring the framework maintains exact notebook functionality while providing robust testing coverage.

## 📁 Contents

### Core Testing Files

- **`test_all_components.py`** - Comprehensive test suite for all system components
- **`__init__.py`** - Test module interface and test runner utilities

## 🧪 Test Coverage

### Component Testing

#### Model Architecture Tests (`TestModels`)
- **Video Transformers**: Swin3D, MViT architecture validation
- **3D CNNs**: R3D, MC3, Slow, X3D model testing
- **CNN-RNN Hybrids**: ResNet-LSTM, EfficientNet-GRU validation
- **Multi-stage Models**: TeCNO architecture testing
- **Model Factory**: Unified model creation testing

#### Data Pipeline Tests (`TestDataComponents`)
- **Dataset Classes**: PhaseDataset and SequentialPhaseDataset testing
- **Data Utilities**: PhaseMapper and AnnotationProcessor validation
- **Data Loading**: Batch loading and transformation testing
- **Annotation Processing**: Format validation and error handling

#### Preprocessing Tests (`TestPreprocessing`)
- **Video Processing**: Frame extraction and resizing validation
- **Format Conversion**: Video format compatibility testing
- **Quality Control**: Video validation and error detection
- **Batch Processing**: Multi-video processing testing

#### Training Framework Tests (`TestTrainingFramework`)
- **PyTorch Lightning**: Training module validation
- **Loss Functions**: Loss computation and backpropagation testing
- **Optimizers**: Learning rate and weight update validation
- **Metrics**: Accuracy and F1-score calculation testing

#### Configuration Tests (`TestConfigs`)
- **Configuration Loading**: YAML configuration parsing
- **Validation**: Configuration parameter validation
- **Dataclasses**: Configuration object creation and validation
- **Overrides**: Configuration parameter override testing

#### Analysis Tests (`TestAnalysisTools`)
- **Model Analysis**: Performance analysis and visualization testing
- **Error Analysis**: Misclassification pattern detection
- **Report Generation**: Automated report creation testing
- **Visualization**: Plot and chart generation validation

### Integration Testing (`TestIntegration`)
- **End-to-End Workflow**: Complete pipeline testing
- **Model Training**: Training loop validation
- **Inference Pipeline**: Model prediction testing
- **Configuration Integration**: Config-driven workflow testing

### Utility Testing (`TestUtilities`)
- **Helper Functions**: Utility function validation
- **Import Testing**: Module import verification
- **GPU Testing**: CUDA availability and usage testing
- **File Operations**: Data I/O operation testing

## 🔧 Running Tests

### Complete Test Suite
```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=. --cov-report=html

# Run tests with detailed output
python -m pytest tests/ -v --tb=long
```

### Specific Test Categories
```bash
# Test only model architectures
python -m pytest tests/test_all_components.py::TestModels -v

# Test data pipeline
python -m pytest tests/test_all_components.py::TestDataComponents -v

# Test training framework
python -m pytest tests/test_all_components.py::TestTrainingFramework -v

# Test configuration system
python -m pytest tests/test_all_components.py::TestConfigs -v
```

### Programmatic Test Running
```python
from tests import run_all_tests

# Run all tests programmatically
exit_code = run_all_tests()

if exit_code == 0:
    print("All tests passed!")
else:
    print(f"Tests failed with exit code: {exit_code}")
```

## 📊 Test Configuration

### Test Parameters
```python
TEST_CONFIG = {
    "test_data_size": 100,
    "test_batch_size": 4,
    "test_sequence_length": 8,
    "test_num_classes": 11,
    "test_image_size": (224, 224),
    "test_video_frames": 16
}
```

### Mock Data Generation
```python
# Sample test data creation
def create_test_data():
    # Video tensor for testing
    video_tensor = torch.randn(2, 3, 16, 224, 224)
    
    # Labels for testing
    labels = torch.randint(0, 11, (2,))
    
    # Annotations for testing
    annotations = [
        {
            'video_id': 'test_video_1',
            'start_frame': 0,
            'end_frame': 100,
            'phase': 'Incision'
        }
    ]
    
    return video_tensor, labels, annotations
```

## 🎯 Test Categories

### Unit Tests
- **Individual Functions**: Testing isolated function behavior
- **Class Methods**: Testing class method functionality
- **Edge Cases**: Testing boundary conditions and error cases
- **Input Validation**: Testing parameter validation and error handling

### Integration Tests
- **Component Interaction**: Testing component communication
- **Workflow Validation**: Testing complete processing pipelines
- **Configuration Integration**: Testing config-driven functionality
- **End-to-End Testing**: Testing complete system workflows

### Performance Tests
- **Memory Usage**: Testing memory consumption patterns
- **GPU Utilization**: Testing GPU usage and efficiency
- **Processing Speed**: Testing inference and training speed
- **Scalability**: Testing with various data sizes

### Regression Tests
- **Model Output Consistency**: Ensuring reproducible model outputs
- **API Compatibility**: Testing backward compatibility
- **Configuration Changes**: Testing config parameter changes
- **Version Compatibility**: Testing across different library versions

## 🔍 Test Examples

### Model Testing
```python
def test_swin_video_3d():
    """Test Swin3D Video Transformer model."""
    model = SwinVideo3D(num_classes=11, pretrained=False)
    input_tensor = torch.randn(2, 3, 16, 224, 224)
    
    # Test forward pass
    output = model(input_tensor)
    
    # Validate output shape
    assert output.shape == (2, 11)
    assert torch.is_tensor(output)
    
    # Test gradients
    loss = output.sum()
    loss.backward()
    
    # Check that gradients are computed
    assert any(p.grad is not None for p in model.parameters())
```

### Data Pipeline Testing
```python
def test_phase_dataset():
    """Test PhaseDataset functionality."""
    annotations = [
        {'video_id': 'test', 'start_frame': 0, 'end_frame': 100, 'phase': 'Incision'}
    ]
    
    dataset = PhaseDataset(
        video_dir='/dummy/path',
        annotations=annotations,
        phase_names=['Incision', 'Rhexis'],
        transform=None
    )
    
    # Test dataset length
    assert len(dataset) == 1
    
    # Test phase mapping
    assert dataset.phase_mapper.phase_to_id('Incision') == 0
```

### Training Framework Testing
```python
def test_training_step():
    """Test training step functionality."""
    config = {
        'model': {'type': 'swin_video_3d', 'num_classes': 11},
        'training': {'learning_rate': 1e-4}
    }
    
    training_module = PhaseClassificationTraining(config)
    
    # Create mock batch
    batch = (
        torch.randn(2, 3, 16, 224, 224),  # video
        torch.randint(0, 11, (2,))        # labels
    )
    
    # Test training step
    loss = training_module.training_step(batch, 0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
```

## 📈 Test Metrics

### Coverage Goals
- **Line Coverage**: >90% line coverage across all modules
- **Branch Coverage**: >85% branch coverage for critical paths
- **Function Coverage**: 100% function coverage for public APIs
- **Integration Coverage**: Complete workflow coverage

### Performance Benchmarks
- **Test Execution Time**: <5 minutes for complete test suite
- **Memory Usage**: <8GB RAM for full test execution
- **GPU Memory**: <4GB VRAM for GPU-enabled tests
- **CI/CD Integration**: <10 minutes for automated testing

## 🛠️ Test Infrastructure

### Continuous Integration
```yaml
# GitHub Actions workflow example
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python -m pytest tests/ -v --cov=.
```

### Test Environment Setup
```bash
# Set up test environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
python -m pytest tests/ -v
```

### Mock and Fixtures
```python
@pytest.fixture
def sample_model():
    """Provide a sample model for testing."""
    return create_model('swin_video_3d', num_classes=11, pretrained=False)

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return torch.randn(4, 3, 16, 224, 224), torch.randint(0, 11, (4,))
```

## 📖 Usage Examples

See the comprehensive reference notebook: [`../notebooks/phase_validation_comprehensive.ipynb`](../notebooks/phase_validation_comprehensive.ipynb)
