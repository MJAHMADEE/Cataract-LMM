# 🧪 Testing Suite

This directory contains the comprehensive testing suite for the surgical video processing framework, ensuring reliability, performance, and compliance with reference standards.

## 📁 Test Structure

### Test Categories

| Test Type | Purpose | Coverage | Status |
|-----------|---------|----------|--------|
| 🔧 **Unit Tests** | Component-level validation | Core modules, utilities | ✅ Complete |
| 🔗 **Integration Tests** | Cross-component functionality | Pipeline workflows | ✅ Complete |
| 📊 **Performance Tests** | Speed and efficiency metrics | Processing benchmarks | ✅ Complete |
| 🎯 **Compliance Tests** | Reference script alignment | Hospital standards | ✅ Complete |
| 🛡️ **Error Handling Tests** | Failure scenarios and recovery | Error conditions | ✅ Complete |
- Pipeline stage integration
- Configuration loading and validation
- Metadata flow between components
- Error propagation and handling

### 3. End-to-End Tests
**Complete workflow testing:**
- Full video processing pipeline
- Batch processing capabilities
- Hospital-specific configurations
- Output validation and verification

### 4. Performance Tests
**System performance validation:**
- Memory usage optimization
- Processing speed benchmarks
- Parallel processing efficiency
- Large dataset handling

## Running Tests

### Complete Test Suite
```bash
# Run all tests
python -m surgical_video_processing.tests

# Or using pytest (if installed)
pytest surgical_video_processing/tests/
```

### Specific Test Categories
```bash
# Run unit tests only
python -c "from surgical_video_processing.tests import TestConfigManager; import unittest; unittest.main(module=TestConfigManager)"

# Run integration tests
python -c "from surgical_video_processing.tests import TestPipelineIntegration; import unittest; unittest.main(module=TestPipelineIntegration)"

# Run with verbose output
python -m surgical_video_processing.tests -v
```

### Individual Test Classes
```bash
# Test configuration management
python -m unittest surgical_video_processing.tests.TestConfigManager

# Test metadata extraction
python -m unittest surgical_video_processing.tests.TestMetadataExtractor

# Test quality control
python -m unittest surgical_video_processing.tests.TestQualityControl
```

## Test Components

### TestConfigManager
Tests configuration system functionality:
- **Default config loading**: Validates default.yaml structure
- **Hospital configs**: Tests farabi_config.yaml and noor_config.yaml
- **Environment overrides**: Tests environment variable integration
- **Validation**: Tests configuration validation logic
- **Error handling**: Tests malformed configuration handling

```python
def test_hospital_config_generation(self):
    config = ConfigManager.generate_hospital_config(
        hospital_name="farabi",
        resolution=(720, 480),
        fps=30.0
    )
    assert config['processing']['target_resolution'] == [720, 480]
```

### TestMetadataExtractor
Tests metadata extraction capabilities:
- **Video analysis**: Tests OpenCV and FFmpeg integration
- **Hospital detection**: Validates automatic hospital identification
- **Equipment detection**: Tests equipment model recognition
- **Format support**: Tests various video format handling
- **Error resilience**: Tests corrupted file handling

```python
def test_hospital_detection(self):
    # Test Farabi detection (720x480)
    metadata = extractor.extract_metadata("farabi_video.mp4")
    assert metadata.hospital_source == "farabi"
    assert "Haag-Streit" in metadata.equipment_model
```

### TestQualityControl
Tests quality assessment algorithms:
- **Focus analysis**: Tests Laplacian and gradient methods
- **Glare detection**: Tests brightness-based glare identification
- **Exposure analysis**: Tests histogram-based exposure metrics
- **Motion detection**: Tests frame difference analysis
- **ROI processing**: Tests surgical field detection

```python
def test_focus_analysis(self):
    sharp_frame = create_sharp_test_frame()
    blurry_frame = create_blurry_test_frame()
    
    sharp_score = focus_checker.analyze_focus(sharp_frame)
    blurry_score = focus_checker.analyze_focus(blurry_frame)
    
    assert sharp_score > blurry_score
```

### TestVideoProcessing
Tests video processing components:
- **Standardization**: Tests resolution and FPS conversion
- **Enhancement**: Tests contrast and noise reduction
- **De-identification**: Tests metadata and visual anonymization
- **Compression**: Tests quality-preserving compression
- **FFmpeg integration**: Tests command generation and execution

```python
def test_video_standardization(self):
    standardizer = VideoStandardizer(config)
    result = standardizer.standardize_video("input.mp4", "output.mp4")
    assert result.success
    assert result.output_format == "mp4"
```

### TestPipelineIntegration
Tests complete pipeline integration:
- **Multi-stage processing**: Tests sequential stage execution
- **Error propagation**: Tests error handling across stages
- **Configuration flow**: Tests config parameter passing
- **Output validation**: Tests intermediate and final outputs
- **Memory management**: Tests resource cleanup

```python
def test_complete_pipeline(self):
    processor = SurgicalVideoProcessor(config)
    result = processor.process_video("input.mp4", "output_dir")
    
    assert result.success
    assert Path(result.output_path).exists()
    assert result.quality_score is not None
```

### TestBatchProcessing
Tests batch processing capabilities:
- **Directory processing**: Tests batch directory handling
- **Parallel execution**: Tests multi-worker processing
- **Progress tracking**: Tests progress reporting
- **Error aggregation**: Tests batch error handling
- **Resume functionality**: Tests interrupted batch resumption

```python
def test_batch_processing(self):
    batch_processor = BatchProcessor(config)
    results = batch_processor.process_directory("input/", "output/")
    
    assert len(results) == expected_count
    assert all(r.success for r in results)
```

### TestErrorHandling
Tests comprehensive error handling:
- **File errors**: Tests missing and corrupted file handling
- **Configuration errors**: Tests invalid configuration handling
- **Processing errors**: Tests FFmpeg and OpenCV error handling
- **Memory errors**: Tests out-of-memory condition handling
- **Network errors**: Tests network-dependent functionality

```python
def test_corrupted_file_handling(self):
    with pytest.raises(VideoProcessingError):
        processor.process_video("corrupted.mp4", "output/")
```

## Test Data Generation

### Synthetic Video Creation
```python
def create_test_video(width=720, height=480, fps=30, duration=10):
    """Create synthetic test video with known characteristics."""
    video_path = "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for i in range(int(fps * duration)):
        frame = create_test_frame(quality="good")
        out.write(frame)
    
    out.release()
    return video_path
```

### Quality Test Frames
```python
def create_test_frame(quality_type="good"):
    """Create frames with specific quality characteristics."""
    if quality_type == "sharp":
        # High-frequency content for sharpness testing
        frame = create_sharp_pattern()
    elif quality_type == "blurry":
        # Low-frequency content for blur testing
        frame = apply_gaussian_blur()
    elif quality_type == "glare":
        # Bright regions for glare testing
        frame = add_glare_regions()
    
    return frame
```

## Mocking and Test Isolation

### FFmpeg Mocking
```python
@patch('subprocess.run')
def test_video_compression(self, mock_run):
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Success"
    
    result = compressor.compress_video("input.mp4", "output.mp4")
    assert result.success
```

### OpenCV Mocking
```python
@patch('cv2.VideoCapture')
def test_quality_analysis(self, mock_capture):
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, test_frame)
    mock_capture.return_value = mock_cap
    
    result = analyzer.analyze_video("test.mp4")
    assert result.quality_score > 0
```

## Continuous Integration

### Test Automation
```yaml
# .github/workflows/tests.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          apt-get update && apt-get install -y ffmpeg
      - name: Run tests
        run: python -m surgical_video_processing.tests
```

### Coverage Reporting
```bash
# Install coverage tools
pip install coverage pytest-cov

# Run tests with coverage
coverage run -m surgical_video_processing.tests
coverage report
coverage html  # Generate HTML report
```

## Performance Benchmarks

### Processing Speed Tests
```python
def test_processing_performance(self):
    """Test processing speed benchmarks."""
    start_time = time.time()
    result = processor.process_video("benchmark_video.mp4", "output/")
    processing_time = time.time() - start_time
    
    # Should process faster than real-time for quick mode
    assert processing_time < video_duration * 0.5
```

### Memory Usage Tests
```python
def test_memory_usage(self):
    """Test memory consumption during processing."""
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss
    processor.process_video("large_video.mp4", "output/")
    peak_memory = process.memory_info().rss
    
    memory_increase = peak_memory - initial_memory
    assert memory_increase < 2 * 1024 * 1024 * 1024  # < 2GB
```

## Test Configuration

### Test Settings
```yaml
# test_config.yaml
test_settings:
  create_temp_videos: true
  temp_video_duration: 5  # seconds
  skip_slow_tests: false
  mock_ffmpeg: true
  
quality_control:
  min_overall_score: 50.0  # Lower threshold for testing
  sample_frame_count: 5    # Fewer frames for speed
  
processing:
  max_workers: 2  # Limited for test environment
```

### Environment Setup
```bash
# Set up test environment
export SURGICAL_VIDEO_TEST_MODE=true
export SURGICAL_VIDEO_TEST_DATA_DIR=./test_data
export SURGICAL_VIDEO_MOCK_FFMPEG=true
```

## Debugging Failed Tests

### Verbose Output
```bash
# Run with maximum verbosity
python -m surgical_video_processing.tests -v --tb=long

# Run specific failing test
python -m unittest surgical_video_processing.tests.TestQualityControl.test_focus_analysis -v
```

### Test Data Inspection
```python
def debug_test_failure(self):
    """Helper method for debugging test failures."""
    # Save test data for manual inspection
    cv2.imwrite("debug_frame.jpg", test_frame)
    
    # Print detailed analysis
    print(f"Focus score: {focus_score}")
    print(f"Glare percentage: {glare_percentage}")
    
    # Assert with detailed error message
    self.assertGreater(focus_score, threshold, 
                      f"Focus score {focus_score} below threshold {threshold}")
```

## Test Maintenance

### Adding New Tests
1. **Create test method** following naming convention `test_*`
2. **Add appropriate assertions** for expected behavior
3. **Include error cases** for comprehensive coverage
4. **Update documentation** for new test functionality

### Test Data Management
- **Synthetic data**: Use generated test videos when possible
- **Real data**: Include sample real videos for integration testing
- **Privacy**: Ensure no sensitive medical data in test suite
- **Size limits**: Keep test data size reasonable for CI/CD

### Best Practices
- **Test isolation**: Each test should be independent
- **Cleanup**: Always clean up temporary files and resources
- **Mocking**: Mock external dependencies (FFmpeg, file systems)
- **Documentation**: Document complex test scenarios
- **Performance**: Keep tests fast for development workflow
