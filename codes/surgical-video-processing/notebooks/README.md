# Jupyter Notebooks

Interactive notebooks for analysis, visualization, and development of the surgical video processing framework.

## Overview

This directory contains Jupyter notebooks that provide interactive environments for:
- **Data exploration and analysis**
- **Quality control visualization**
- **Algorithm development and testing**
- **Performance analysis and optimization**
- **Research and experimentation**

## Available Notebooks

### 1. Dataset Analysis (`dataset_analysis.ipynb`)
Comprehensive analysis of surgical video datasets with interactive visualizations.

**Content:**
- Hospital distribution analysis
- Video quality statistics
- Technical specification summaries
- Temporal distribution patterns
- File size and duration analysis

**Key Features:**
- Interactive plots with Plotly
- Statistical summaries
- Hospital comparison charts
- Quality metric distributions

### 2. Quality Control Validation (`quality_control_validation.ipynb`)
Validate and visualize quality control algorithms performance.

**Content:**
- Focus detection algorithm testing
- Glare detection validation
- Motion analysis visualization
- Quality threshold optimization
- ROI detection validation

**Key Features:**
- Side-by-side image comparisons
- Quality score distributions
- Algorithm parameter tuning
- Performance benchmarking

### 3. Processing Pipeline Demo (`processing_pipeline_demo.ipynb`)
Interactive demonstration of the complete processing pipeline.

**Content:**
- Step-by-step pipeline execution
- Intermediate result visualization
- Configuration comparison
- Before/after quality comparison
- Processing time analysis

**Key Features:**
- Live processing examples
- Configuration experiments
- Visual result comparison
- Performance profiling

### 4. Hospital Configuration Optimization (`hospital_config_optimization.ipynb`)
Optimize configurations for specific hospital equipment.

**Content:**
- Equipment-specific parameter tuning
- Quality vs. speed trade-offs
- Compression optimization
- Batch processing analysis
- Custom configuration generation

**Key Features:**
- Parameter sensitivity analysis
- Optimization algorithms
- Visual parameter impact
- Configuration export

### 5. Metadata Exploration (`metadata_exploration.ipynb`)
Explore and analyze video metadata patterns.

**Content:**
- Metadata schema exploration
- Hospital identification accuracy
- Equipment detection validation
- Temporal pattern analysis
- Quality correlation analysis

**Key Features:**
- Interactive metadata browser
- Pattern recognition
- Correlation matrices
- Anomaly detection

## Usage Instructions

### Environment Setup
```bash
# Install Jupyter and visualization packages
pip install jupyter matplotlib seaborn plotly pandas

# Start Jupyter server
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Running Notebooks
1. **Navigate to notebooks directory**
2. **Open desired notebook**
3. **Run cells sequentially** or use "Run All"
4. **Modify parameters** in configuration cells as needed

### Data Requirements
Most notebooks expect data in the following structure:
```
data/
├── videos/
│   ├── farabi/
│   │   ├── video_001.mp4
│   │   └── ...
│   └── noor/
│       ├── video_001.mp4
│       └── ...
├── metadata/
│   ├── video_001_metadata.json
│   └── ...
└── processed/
    ├── farabi/
    └── noor/
```

## Notebook Descriptions

### Dataset Analysis Notebook
```python
# Load and analyze video dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load metadata
metadata_df = load_video_metadata("./metadata/")

# Generate hospital distribution
plot_hospital_distribution(metadata_df)

# Analyze quality metrics
plot_quality_distributions(metadata_df)

# Show temporal patterns
plot_temporal_patterns(metadata_df)
```

**Output Examples:**
- Hospital distribution pie chart
- Quality score histograms
- Resolution vs. quality scatter plots
- Processing time comparisons

### Quality Control Validation
```python
# Test focus detection algorithms
from surgical_video_processing.quality_control import FocusQualityChecker

checker = FocusQualityChecker()

# Load test images
sharp_image = load_test_image("sharp_surgery.jpg")
blurry_image = load_test_image("blurry_surgery.jpg")

# Analyze and visualize
sharp_score = checker.analyze_focus(sharp_image)
blurry_score = checker.analyze_focus(blurry_image)

# Display results
plot_focus_comparison(sharp_image, blurry_image, sharp_score, blurry_score)
```

**Visualizations:**
- Before/after image comparisons
- Quality score distributions
- Algorithm performance metrics
- Parameter sensitivity charts

### Processing Pipeline Demo
```python
# Demo complete processing pipeline
from surgical_video_processing.pipelines import SurgicalVideoProcessor
from surgical_video_processing.configs import ConfigManager

# Load configuration
config = ConfigManager.load_config("farabi_config.yaml")
processor = SurgicalVideoProcessor(config)

# Process sample video with visualization
result = processor.process_video_with_visualization(
    "sample_surgery.mp4",
    "./output/",
    show_intermediate=True
)

# Display processing steps
show_processing_timeline(result)
show_quality_progression(result)
```

**Interactive Features:**
- Real-time processing progress
- Intermediate result preview
- Quality metric tracking
- Configuration experimentation

## Development Notebooks

### Algorithm Development (`algorithm_development.ipynb`)
Develop and test new algorithms:

```python
# Develop new quality metric
def new_quality_metric(frame):
    # Algorithm implementation
    return quality_score

# Test on sample data
test_frames = load_test_frames()
scores = [new_quality_metric(frame) for frame in test_frames]

# Visualize results
plot_quality_scores(scores)
validate_against_ground_truth(scores, ground_truth)
```

### Performance Optimization (`performance_optimization.ipynb`)
Optimize processing performance:

```python
# Benchmark different approaches
import time

def benchmark_approach(approach_func, test_data):
    start_time = time.time()
    result = approach_func(test_data)
    end_time = time.time()
    return end_time - start_time, result

# Test multiple approaches
approaches = [approach_1, approach_2, approach_3]
results = [benchmark_approach(app, test_data) for app in approaches]

# Visualize performance comparison
plot_performance_comparison(results)
```

## Visualization Examples

### Quality Distribution Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create quality distribution plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Focus score distribution
sns.histplot(data=metadata_df, x='focus_score', hue='hospital_source', ax=axes[0,0])
axes[0,0].set_title('Focus Score Distribution by Hospital')

# Glare percentage distribution
sns.histplot(data=metadata_df, x='glare_percentage', hue='hospital_source', ax=axes[0,1])
axes[0,1].set_title('Glare Percentage Distribution')

# Overall quality vs. resolution
sns.scatterplot(data=metadata_df, x='width', y='quality_score', 
                hue='hospital_source', ax=axes[1,0])
axes[1,0].set_title('Quality vs. Resolution')

# Processing time analysis
sns.boxplot(data=processing_df, x='hospital_source', y='processing_time', ax=axes[1,1])
axes[1,1].set_title('Processing Time by Hospital')

plt.tight_layout()
plt.show()
```

### Interactive Processing Demonstration
```python
import ipywidgets as widgets
from IPython.display import display

# Create interactive controls
quality_threshold = widgets.FloatSlider(
    value=70.0, min=0.0, max=100.0, step=1.0,
    description='Quality Threshold:'
)

compression_crf = widgets.IntSlider(
    value=23, min=15, max=35, step=1,
    description='Compression CRF:'
)

def update_processing(quality_threshold, compression_crf):
    # Update configuration
    config['quality_control']['min_overall_score'] = quality_threshold
    config['compression']['crf_value'] = compression_crf
    
    # Show impact
    show_config_impact(config)

# Create interactive widget
interactive_widget = widgets.interact(
    update_processing,
    quality_threshold=quality_threshold,
    compression_crf=compression_crf
)

display(interactive_widget)
```

## Research Applications

### Comparative Analysis
```python
# Compare hospital-specific processing results
farabi_results = load_processing_results("farabi")
noor_results = load_processing_results("noor")

# Statistical comparison
from scipy import stats

# Quality score comparison
farabi_quality = [r.quality_score for r in farabi_results]
noor_quality = [r.quality_score for r in noor_results]

# Perform t-test
t_stat, p_value = stats.ttest_ind(farabi_quality, noor_quality)

print(f"Quality difference: t={t_stat:.3f}, p={p_value:.3f}")

# Visualize comparison
plot_hospital_comparison(farabi_quality, noor_quality)
```

### Algorithm Validation
```python
# Validate new algorithm against ground truth
ground_truth = load_ground_truth_data()
algorithm_results = run_algorithm_on_test_set()

# Calculate validation metrics
accuracy = calculate_accuracy(algorithm_results, ground_truth)
precision = calculate_precision(algorithm_results, ground_truth)
recall = calculate_recall(algorithm_results, ground_truth)

# ROC curve analysis
plot_roc_curve(algorithm_results, ground_truth)

# Confusion matrix
plot_confusion_matrix(algorithm_results, ground_truth)
```

## Export and Sharing

### Report Generation
```python
# Generate HTML report
from IPython.display import HTML
import base64

# Create comprehensive report
report_html = generate_analysis_report(
    metadata_df, 
    quality_results, 
    processing_stats
)

# Save as standalone HTML
with open("analysis_report.html", "w") as f:
    f.write(report_html)

# Display in notebook
display(HTML(report_html))
```

### Configuration Export
```python
# Export optimized configurations
optimized_config = optimize_configuration(
    dataset=metadata_df,
    target_quality=85.0,
    max_processing_time=30.0
)

# Save configuration
import yaml
with open("optimized_config.yaml", "w") as f:
    yaml.dump(optimized_config, f)

print("Optimized configuration saved!")
```

## Best Practices

### 1. Data Management
- **Load data efficiently** using pandas and chunking for large datasets
- **Cache results** to avoid recomputation during development
- **Use relative paths** for portability across environments

### 2. Visualization
- **Interactive plots** with Plotly for better exploration
- **Consistent styling** using seaborn themes
- **Clear annotations** and titles for all plots

### 3. Code Organization
- **Modular functions** for reusable analysis components
- **Configuration cells** at the top of notebooks
- **Clear documentation** for complex analysis steps

### 4. Performance
- **Efficient data loading** with appropriate data types
- **Vectorized operations** using NumPy and pandas
- **Progress bars** for long-running operations

### 5. Reproducibility
- **Fixed random seeds** for reproducible results
- **Version documentation** of key packages
- **Clear parameter documentation** for analysis

## Integration with Framework

### Loading Framework Components
```python
# Import framework modules
import sys
sys.path.append('../')

from surgical_video_processing.configs import ConfigManager
from surgical_video_processing.metadata import MetadataManager
from surgical_video_processing.quality_control import QualityControlPipeline
from surgical_video_processing.pipelines import SurgicalVideoProcessor
```

### Using Live Data
```python
# Connect to live processing results
from surgical_video_processing.utils import find_video_files

# Load latest processed videos
processed_videos = find_video_files("../output/")
metadata_files = glob.glob("../metadata/*.json")

# Analyze current processing status
current_stats = analyze_processing_status(processed_videos, metadata_files)
```

This notebook collection provides a comprehensive environment for interactive analysis, development, and validation of the surgical video processing framework, supporting both operational use and research applications.
