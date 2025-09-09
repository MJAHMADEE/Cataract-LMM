# ⚙️ Training & Evaluation Engine

The engine module provides the core computational logic for training, validation, evaluation, and inference in the surgical skill assessment framework. This module implements state-of-the-art training strategies optimized for video-based deep learning models.

## 🎯 Core Components

### 🚀 `trainer.py` - Training & Validation Logic

Advanced training infrastructure with modern deep learning optimizations:

#### **`train_one_epoch`** - Training Execution
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for accelerated training and reduced memory usage
- **Gradient Accumulation**: Support for effective large batch sizes on memory-constrained hardware
- **Dynamic Loss Scaling**: Automatic loss scaling for numerical stability in mixed precision training
- **Comprehensive Metrics**: Real-time tracking of loss, accuracy, precision, recall, and F1-score
- **Progress Monitoring**: Rich console output with training progress visualization

#### **`validate_one_epoch`** - Validation Pipeline
- **Efficient Validation**: Memory-optimized validation loop with no gradient computation
- **Metric Computation**: Comprehensive evaluation metrics aligned with training metrics
- **Early Stopping Support**: Integration with early stopping mechanisms for optimal model selection
- **Statistical Analysis**: Detailed validation statistics for model performance assessment

#### **`collate_fn`** - Batch Processing
- **Variable Length Handling**: Robust batch collation for videos of different lengths
- **Memory Optimization**: Efficient tensor stacking and memory management
- **Error Resilience**: Graceful handling of data loading errors and corrupted samples

### 📊 `evaluator.py` - Model Evaluation Framework

Comprehensive model assessment with detailed analytics:

#### **`evaluate_model`** - Complete Evaluation Pipeline
- **Multi-Metric Assessment**: Precision, recall, F1-score, specificity, and AUC-ROC calculation
- **Confusion Matrix Generation**: Detailed confusion matrix with visualization capabilities
- **Classification Reports**: Professional-grade classification reports with per-class statistics
- **Statistical Significance**: Confidence intervals and statistical significance testing
- **Export Capabilities**: JSON and CSV export for further analysis

#### **Performance Analysis Features**
- **Per-Class Metrics**: Detailed analysis for lower-skilled vs higher-skilled classification
- **Error Analysis**: Identification of misclassified samples with detailed error reports
- **Visualization**: Professional confusion matrix plots and metric visualizations
- **Report Generation**: Comprehensive evaluation reports with actionable insights

### 🔍 `predictor.py` - Inference Engine

Production-ready inference capabilities for real-world deployment:

#### **`run_inference`** - Single Video Prediction
- **End-to-End Processing**: Complete pipeline from raw video to skill classification
- **Confidence Scoring**: Probability distributions and confidence intervals for predictions
- **Preprocessing Integration**: Automatic video preprocessing aligned with training pipeline
- **Error Handling**: Robust error handling for various video formats and quality issues

#### **Inference Features**
- **Real-Time Processing**: Optimized for low-latency inference scenarios
- **Batch Inference**: Support for processing multiple videos simultaneously
- **Model Agnostic**: Compatible with all supported model architectures
- **Output Formatting**: Structured prediction outputs with metadata and confidence scores

## 🚀 Performance Optimizations

### 💾 Memory Management
- **Dynamic Batch Sizing**: Automatic batch size optimization based on available GPU memory
- **Memory Monitoring**: Real-time memory usage tracking and optimization
- **Garbage Collection**: Intelligent memory cleanup during training loops
- **Cache Management**: Efficient caching strategies for improved throughput

### ⚡ Training Acceleration
- **Mixed Precision**: FP16 training for 1.5-2x speed improvements
- **Gradient Accumulation**: Simulated large batch training on limited hardware
- **DataLoader Optimization**: Multi-threaded data loading with prefetching
- **GPU Utilization**: Maximized GPU compute utilization through optimized kernels

### 📈 Monitoring & Logging
- **Rich Console Output**: Beautiful progress bars and real-time metric display
- **Tensorboard Integration**: Comprehensive logging for training visualization
- **Checkpoint Management**: Automatic model checkpointing with best model preservation
- **Metric Tracking**: Detailed metric history with trend analysis

## 🎛️ Configuration Integration

### Training Configuration
```yaml
train:
  epochs: 50
  batch_size: 4
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  mixed_precision: true
  gradient_accumulation_steps: 1
```

### Evaluation Configuration
```yaml
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auc_roc"]
  save_predictions: true
  confusion_matrix: true
  classification_report: true
```

## 🔧 Usage Examples

### Training Integration
```python
from engine.trainer import train_one_epoch, validate_one_epoch

# Training loop
for epoch in range(num_epochs):
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, 
        criterion, scaler, device, config
    )
    
    val_metrics = validate_one_epoch(
        model, val_loader, criterion, device, config
    )
```

### Model Evaluation
```python
from engine.evaluator import evaluate_model

# Comprehensive evaluation
results = evaluate_model(
    model, test_loader, device, 
    class_names, output_dir
)
```

### Inference Processing
```python
from engine.predictor import run_inference

# Single video inference
prediction = run_inference(
    model, video_path, device, 
    class_names, config
)
```

## 🎯 Integration Points

### Model Compatibility
- **Multi-Architecture Support**: Compatible with CNN, CNN-RNN, and Transformer models
- **Transfer Learning**: Optimized for pretrained model fine-tuning
- **Custom Models**: Extensible architecture for custom model integration

### Hardware Optimization
- **Multi-GPU Training**: Distributed training across multiple GPUs
- **CPU Fallback**: Graceful degradation to CPU when GPU is unavailable
- **Memory Adaptation**: Automatic adaptation to available hardware resources

This engine module provides the computational foundation for robust, efficient, and scalable surgical skill assessment model training and deployment.
