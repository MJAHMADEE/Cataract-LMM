# ⚙️ Configuration Framework

**Comprehensive configuration management system** for surgical instance segmentation with hierarchical configurations, environment-aware settings, and production-ready deployment configurations across all model architectures.

## 🏗️ **Architecture Overview**

### **🔧 Hierarchical Configuration System**
```
configs/
├── base/                       # Base configuration templates
│   ├── base_config.yaml       # Foundation configuration schema
│   ├── medical_standards.yaml # Medical compliance standards
│   ├── hardware_configs.yaml  # Hardware-specific configurations
│   └── environment_configs.yaml # Environment setup configurations
├── models/                     # Model-specific configurations
│   ├── mask_rcnn/             # Mask R-CNN configurations
│   │   ├── training.yaml      # Training configuration
│   │   ├── inference.yaml     # Inference configuration
│   │   ├── production.yaml    # Production deployment
│   │   └── research.yaml      # Research experimentation
│   ├── sam/                   # SAM configurations
│   │   ├── adaptation.yaml    # Domain adaptation settings
│   │   ├── inference.yaml     # Inference configuration
│   │   └── interactive.yaml   # Interactive segmentation
│   ├── yolo/                  # YOLO configurations  
│   │   ├── yolov8_training.yaml # YOLOv8 training settings
│   │   ├── yolov11_training.yaml # YOLOv11 training settings
│   │   ├── realtime.yaml      # Real-time inference
│   │   └── optimization.yaml  # Performance optimization
│   └── ensemble/              # Ensemble configurations
│       ├── weighted_voting.yaml # Weighted ensemble
│       ├── adaptive_selection.yaml # Adaptive selection
│       └── fusion_strategies.yaml # Multi-model fusion
├── datasets/                   # Dataset configurations
│   ├── aras_surgical.yaml     # ARAS surgical dataset
│   ├── endovis_2018.yaml      # EndoVis challenge dataset
│   ├── custom_surgical.yaml   # Custom surgical dataset
│   └── data_augmentation.yaml # Augmentation configurations
├── deployment/                 # Deployment configurations
│   ├── cloud/                 # Cloud deployment configs
│   │   ├── aws_production.yaml # AWS production deployment
│   │   ├── azure_clinical.yaml # Azure clinical deployment
│   │   └── gcp_research.yaml  # GCP research deployment
│   ├── edge/                  # Edge deployment configs
│   │   ├── jetson_nano.yaml   # NVIDIA Jetson Nano
│   │   ├── jetson_xavier.yaml # NVIDIA Jetson Xavier
│   │   └── intel_nuc.yaml     # Intel NUC edge devices
│   ├── hospital/              # Hospital deployment configs
│   │   ├── or_realtime.yaml   # Operating room real-time
│   │   ├── training_sim.yaml  # Training simulation
│   │   └── research_lab.yaml  # Research laboratory
│   └── api/                   # API deployment configs
│       ├── rest_api.yaml      # REST API configuration
│       ├── grpc_server.yaml   # gRPC server configuration
│       └── microservices.yaml # Microservices architecture
├── experiments/               # Experiment configurations
│   ├── comparative_study.yaml # Model comparison experiments
│   ├── ablation_studies.yaml # Ablation study configurations
│   ├── hyperparameter_opt.yaml # Hyperparameter optimization
│   └── clinical_validation.yaml # Clinical validation studies
└── validation/                # Validation configurations
    ├── medical_validation.yaml # Medical validation protocols
    ├── regulatory_compliance.yaml # Regulatory requirements
    ├── safety_protocols.yaml  # Safety validation protocols
    └── performance_benchmarks.yaml # Performance benchmarking
```

## 🎯 **Base Configuration Framework**

### **Foundation Configuration Schema**
```yaml
# base/base_config.yaml - Master configuration template
project:
  name: "surgical_instance_segmentation"
  version: "1.0.0"
  description: "Production-ready surgical instance segmentation framework"
  medical_compliance: true
  regulatory_standard: "fda_510k"

# Global settings
global:
  random_seed: 42
  deterministic: true
  benchmark_mode: false
  precision: "mixed"  # fp32, fp16, mixed
  device: "auto"  # auto, cuda, cpu, cuda:0, etc.

# Medical standards compliance
medical:
  patient_privacy: true
  hipaa_compliant: true
  audit_logging: true
  safety_validation: true
  clinical_guidelines: "surgical_ai_standards_2024"
  
# Data handling standards
data:
  quality_validation: true
  annotation_standards: "medical_grade"
  preprocessing_pipeline: "surgical_optimized"
  augmentation_medical_safe: true
  
# Model standards
model:
  architecture_validation: true
  performance_thresholds:
    minimum_accuracy: 0.85
    minimum_precision: 0.90
    minimum_recall: 0.85
    maximum_latency: 50  # milliseconds
  safety_requirements:
    uncertainty_estimation: true
    failure_detection: true
    graceful_degradation: true

# Logging and monitoring
logging:
  level: "INFO"
  format: "structured"
  medical_audit: true
  performance_tracking: true
  error_reporting: true
  
# Security settings
security:
  authentication_required: true
  encryption_at_rest: true
  encryption_in_transit: true
  access_control: "rbac"
  audit_trail: true
```

### **Medical Standards Configuration**
```yaml
# base/medical_standards.yaml - Medical compliance requirements
medical_compliance:
  regulatory_frameworks:
    - "FDA_510K"
    - "CE_MDR"
    - "ISO_13485"
    - "ISO_14155"
    - "IEC_62304"
  
  clinical_validation:
    required: true
    expert_review: true
    multi_site_validation: true
    patient_safety_protocols: true
    
  data_protection:
    hipaa_compliance: true
    gdpr_compliance: true
    anonymization_required: true
    audit_logging_required: true
    
  performance_requirements:
    clinical_accuracy_threshold: 0.90
    safety_margin: 0.05
    false_negative_rate_max: 0.10
    false_positive_rate_max: 0.15
    
  documentation_requirements:
    clinical_evidence: true
    safety_analysis: true
    risk_management: true
    usability_testing: true
    
instrument_classification:
  ontology: "surgical_instrument_ontology_v2.1"
  hierarchy_validation: true
  medical_accuracy_check: true
  
  critical_instruments:
    - "forceps"
    - "scissors" 
    - "needle_holder"
    - "scalpel"
    
  instrument_families:
    grasping:
      - "forceps"
      - "graspers"
      - "clamps"
    cutting:
      - "scissors"
      - "scalpel"
      - "electrocautery"
    suturing:
      - "needle_holder"
      - "suture_scissors"
```

## 🧠 **Model-Specific Configurations**

### **Mask R-CNN Production Configuration**
```yaml
# models/mask_rcnn/production.yaml - Optimized for clinical deployment
model:
  architecture: "mask_rcnn_resnet50_fpn"
  backbone:
    name: "resnet50"
    pretrained: true
    trainable_layers: 3
    norm_layer: "batch_norm"
    
  fpn:
    in_channels: [256, 512, 1024, 2048]
    out_channels: 256
    num_outs: 5
    
  roi_head:
    bbox_roi_extractor:
      type: "single_level"
      roi_layer: "roi_align"
      output_size: 7
      sampling_ratio: 0
    mask_roi_extractor:
      type: "single_level" 
      roi_layer: "roi_align"
      output_size: 14
      sampling_ratio: 0
      
  num_classes: 13  # 12 surgical instruments + background

training:
  optimizer:
    type: "adamw"
    lr: 0.0005
    weight_decay: 0.0005
    betas: [0.9, 0.999]
    eps: 1e-8
    
  lr_scheduler:
    type: "step"
    step_size: 3
    gamma: 0.1
    warmup_epochs: 1
    
  loss_weights:
    classification: 1.0
    bbox_regression: 1.0
    mask: 1.0
    
  training_schedule:
    epochs: 25
    batch_size: 8
    gradient_clipping: 1.0
    mixed_precision: true
    accumulate_grad_batches: 2

inference:
  confidence_threshold: 0.7
  nms_threshold: 0.5
  max_detections: 100
  batch_size: 1
  device: "cuda"
  precision: "fp16"
  
  postprocessing:
    mask_threshold: 0.5
    filter_small_masks: true
    min_mask_area: 100
    smooth_masks: true
    
performance_targets:
  accuracy: 0.915  # mAP@0.5 for YOLOv11-L
  precision: 0.900
  recall: 0.850
  inference_speed: 22  # ms per frame (45 FPS)
  memory_usage: 6.5  # GB GPU memory

medical_validation:
  clinical_accuracy_target: 0.90
  safety_margin: 0.05
  expert_agreement_threshold: 0.85
  false_negative_tolerance: 0.10
```

### **YOLO Real-time Configuration**
```yaml
# models/yolo/realtime.yaml - Optimized for real-time surgical applications
model:
  architecture: "yolov11l-seg"
  task: "segment"
  
  backbone:
    type: "csp_darknet"
    depth_multiple: 1.0
    width_multiple: 1.0
    
  neck:
    type: "pafpn"
    in_channels: [256, 512, 1024]
    out_channels: 256
    
  head:
    type: "yolo_head"
    num_classes: 12
    anchors: 3
    anchor_generator: "auto"

training:
  data_config: "./datasets/aras_surgical.yaml"
  epochs: 80
  batch_size: 20
  imgsz: 640
  device: 0
  workers: 8
  
  optimizer:
    type: "adamw"
    lr0: 0.01
    lrf: 0.01
    momentum: 0.937
    weight_decay: 0.0005
    
  hyperparameters:
    warmup_epochs: 3.0
    warmup_momentum: 0.8
    box_loss_gain: 7.5
    cls_loss_gain: 0.5
    dfl_loss_gain: 1.5
    
  augmentation:
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 0.0
    translate: 0.1
    scale: 0.5
    shear: 0.0
    perspective: 0.0
    flipud: 0.0
    fliplr: 0.5
    mosaic: 1.0
    mixup: 0.0

inference:
  confidence: 0.5
  iou: 0.7
  max_det: 300
  device: "cuda"
  half: true  # FP16 inference
  
  optimization:
    tensorrt: true
    dynamic_batch: true
    workspace_size: 4  # GB
    
realtime_requirements:
  target_fps: 30
  max_latency: 33  # milliseconds (for 30 FPS)
  memory_limit: 4  # GB
  cpu_usage_limit: 50  # percent
  
surgical_specific:
  instrument_priority_weights:
    forceps: 1.2
    scissors: 1.1
    needle_holder: 1.15
    scalpel: 1.3  # Highest priority for safety
  
  temporal_consistency: true
  motion_compensation: true
  occlusion_handling: true
```

### **SAM Interactive Configuration**
```yaml
# models/sam/interactive.yaml - Interactive segmentation configuration
model:
  architecture: "sam_vit_h"
  checkpoint: "sam_vit_h_4b8939.pth"
  
  image_encoder:
    type: "vit_h"
    embed_dim: 1280
    depth: 32
    num_heads: 16
    global_attn_indexes: [7, 15, 23, 31]
    
  prompt_encoder:
    embed_dim: 256
    image_embedding_size: [64, 64]
    input_image_size: [1024, 1024]
    mask_in_chans: 16
    
  mask_decoder:
    num_multimask_outputs: 3
    transformer_dim: 256
    transformer_depth: 2
    transformer_mlp_dim: 2048

adaptation:
  surgical_domain: true
  freeze_image_encoder: true
  freeze_prompt_encoder: false
  
  fine_tuning:
    learning_rate: 1e-4
    optimizer: "adamw"
    weight_decay: 1e-2
    warmup_steps: 250
    max_steps: 5000
    
interactive_settings:
  prompt_types:
    - "bbox"
    - "point"
    - "mask"
    
  bbox_prompts:
    enabled: true
    multimask_output: true
    stability_score_thresh: 0.95
    
  point_prompts:
    enabled: true
    max_points: 10
    positive_label: 1
    negative_label: 0
    
  mask_prompts:
    enabled: true
    mask_input: "previous_prediction"
    iterative_refinement: true

inference:
  image_size: [1024, 1024]
  multimask_output: true
  return_logits: true
  
  postprocessing:
    stability_score_thresh: 0.95
    stability_score_offset: 1.0
    box_nms_thresh: 0.7
    crop_n_layers: 1
    crop_n_points_downscale_factor: 2
    
surgical_workflow:
  annotation_assistance: true
  quality_control: true
  expert_validation: true
  real_time_feedback: true
```

## 📊 **Dataset Configurations**

### **ARAS Surgical Dataset Configuration**
```yaml
# datasets/aras_surgical.yaml - ARAS dataset configuration
dataset:
  name: "aras_surgical_instruments"
  version: "2.1"
  format: "coco"
  
  paths:
    root: "./data/aras_surgical"
    images: "images"
    annotations: "annotations.json"
    
  splits:
    train: 0.7
    val: 0.2
    test: 0.1
    strategy: "surgical_phase_aware"
    
classes:
  num_classes: 12
  names:
    0: "forceps"
    1: "scissors"
    2: "needle_holder"
    3: "grasper"
    4: "clip_applier"
    5: "irrigator"
    6: "suction"
    7: "hook"
    8: "probe"
    9: "dissector"
    10: "retractor"
    11: "other_instrument"
    
  hierarchical_mapping:
    grasping_tools: [0, 3, 7]  # forceps, grasper, hook
    cutting_tools: [1, 9]      # scissors, dissector
    suturing_tools: [2]        # needle_holder
    maintenance_tools: [5, 6]  # irrigator, suction
    
preprocessing:
  image_size: [640, 640]
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  
  surgical_specific:
    lighting_normalization: true
    contrast_enhancement: true
    noise_reduction: true
    
augmentation:
  enabled: true
  medical_safe: true
  
  transforms:
    horizontal_flip: 0.5
    vertical_flip: 0.0  # Not medically realistic
    rotation: 15  # degrees
    brightness: 0.2
    contrast: 0.2
    saturation: 0.1
    hue: 0.05
    
  surgical_augmentations:
    surgical_lighting: 0.3
    smoke_simulation: 0.2
    blood_occlusion: 0.1
    instrument_reflection: 0.15

validation:
  quality_checks: true
  annotation_validation: true
  medical_compliance: true
  inter_annotator_agreement: 0.85
```

## 🚀 **Deployment Configurations**

### **Production Cloud Deployment**
```yaml
# deployment/cloud/aws_production.yaml - AWS production deployment
infrastructure:
  provider: "aws"
  region: "us-east-1"
  availability_zones: ["us-east-1a", "us-east-1b", "us-east-1c"]
  
compute:
  instance_type: "p3.8xlarge"  # 4x V100 GPUs
  min_instances: 2
  max_instances: 10
  auto_scaling: true
  
  container:
    platform: "linux/amd64"
    base_image: "nvidia/pytorch:23.05-py3"
    gpu_support: true
    memory_limit: "60Gi"
    cpu_limit: "32"
    
storage:
  model_storage:
    type: "s3"
    bucket: "surgical-ai-models"
    encryption: "AES256"
    versioning: true
    
  data_storage:
    type: "efs"
    throughput_mode: "provisioned"
    performance_mode: "generalPurpose"
    
networking:
  vpc_cidr: "10.0.0.0/16"
  load_balancer: "application"
  ssl_termination: true
  health_checks: true
  
security:
  encryption_in_transit: true
  encryption_at_rest: true
  authentication: "oauth2"
  authorization: "rbac"
  
  compliance:
    hipaa: true
    sox: true
    pci_dss: false
    
monitoring:
  cloudwatch: true
  custom_metrics: true
  alerting: true
  log_aggregation: true
  
  medical_metrics:
    accuracy_monitoring: true
    safety_alerts: true
    performance_tracking: true
    audit_logging: true

api_configuration:
  type: "rest"
  version: "v1"
  rate_limiting: true
  authentication_required: true
  
  endpoints:
    - path: "/predict/mask_rcnn"
      method: "POST"
      timeout: 30
      max_payload: "50MB"
      
    - path: "/predict/yolo"
      method: "POST" 
      timeout: 5
      max_payload: "20MB"
      
    - path: "/health"
      method: "GET"
      timeout: 5
      authentication: false

performance_targets:
  throughput: 1000  # requests per minute
  latency_p95: 2000  # milliseconds
  availability: 99.9  # percent
  error_rate: 0.1  # percent
```

### **Edge Device Configuration**
```yaml
# deployment/edge/jetson_nano.yaml - NVIDIA Jetson Nano deployment
hardware:
  device: "jetson_nano"
  architecture: "aarch64"
  gpu: "tegra_x1"
  memory: "4GB"
  storage: "64GB"
  
optimization:
  tensorrt: true
  precision: "fp16"
  dynamic_batching: false
  max_workspace_size: "1GB"
  
  model_optimization:
    pruning: true
    quantization: true
    knowledge_distillation: false
    
runtime:
  framework: "tensorrt"
  backend: "cuda"
  memory_pool_size: "2GB"
  
  performance_targets:
    fps: 15
    latency: 66  # milliseconds
    power_consumption: 10  # watts
    
software_stack:
  jetpack: "4.6.1"
  cuda: "10.2"
  tensorrt: "8.2.1"
  opencv: "4.5.0"
  
deployment:
  container: true
  base_image: "nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3"
  auto_update: false
  rollback_capability: true
  
monitoring:
  system_metrics: true
  temperature_monitoring: true
  power_monitoring: true
  performance_tracking: true
  
  alerts:
    temperature_threshold: 80  # celsius
    memory_usage_threshold: 90  # percent
    cpu_usage_threshold: 95  # percent
```

## 🔬 **Experiment Configurations**

### **Comprehensive Model Comparison Study**
```yaml
# experiments/comparative_study.yaml - Multi-model comparison experiment
experiment:
  name: "surgical_segmentation_comparative_study_v1"
  description: "Comprehensive comparison of segmentation models for surgical instruments"
  version: "1.0"
  
models_to_compare:
  - name: "mask_rcnn_resnet50"
    config: "./models/mask_rcnn/production.yaml"
    checkpoint: "./checkpoints/mask_rcnn_best.pth"
    
  - name: "yolov8l_seg"
    config: "./models/yolo/yolov8_training.yaml"
    checkpoint: "./checkpoints/yolov8l_best.pt"
    
  - name: "yolov11l_seg"
    config: "./models/yolo/yolov11_training.yaml"
    checkpoint: "./checkpoints/yolov11l_best.pt"
    
  - name: "sam_vit_h_surgical"
    config: "./models/sam/interactive.yaml"
    checkpoint: "./checkpoints/sam_surgical_adapted.pth"
    
  - name: "ensemble_weighted"
    config: "./models/ensemble/weighted_voting.yaml"
    models: ["mask_rcnn_resnet50", "yolov11l_seg"]

evaluation_metrics:
  segmentation:
    - "map_50"
    - "map_75"
    - "map_50_95"
    - "mean_iou"
    - "mean_dice"
    - "pixel_accuracy"
    
  medical_specific:
    - "clinical_accuracy"
    - "expert_agreement"
    - "safety_score"
    - "critical_miss_rate"
    
  performance:
    - "inference_time"
    - "memory_usage"
    - "throughput"
    - "energy_consumption"

test_datasets:
  - name: "aras_test_set"
    path: "./data/aras_surgical/test"
    
  - name: "endovis_2018_test"
    path: "./data/endovis_2018/test"
    
  - name: "custom_clinical_test"
    path: "./data/clinical_validation/test"

statistical_analysis:
  significance_testing: true
  confidence_level: 0.95
  multiple_testing_correction: "bonferroni"
  bootstrap_iterations: 1000
  
reporting:
  generate_plots: true
  create_tables: true
  statistical_summary: true
  clinical_interpretation: true
  publication_ready: true
```

---

**⚙️ This configuration framework provides comprehensive, hierarchical configuration management for all aspects of surgical instance segmentation with medical-grade standards, production deployment support, and seamless integration with the reference notebook implementations.**
