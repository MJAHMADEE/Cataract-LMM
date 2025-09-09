# Surgical Video Processing Framework - Refactoring Plan

## Phase 1: Analysis & Code Refactoring

### Current Structure Analysis

The current framework has the following structure:
- **Core processing engine**: Well-structured but needs alignment with reference scripts
- **Hospital-specific configurations**: Good foundation but needs refinement
- **De-identification module**: Present but needs alignment with reference script approach
- **Quality control**: Comprehensive but may need optimization
- **Compression**: Advanced but needs to incorporate reference script logic

### Key Issues Identified

1. **Reference Script Alignment**: The current compression logic doesn't match the specific FFmpeg command from `compress_video.sh`
2. **De-identification Implementation**: Needs to incorporate the specific crop+blur+overlay technique from reference
3. **Directory Structure**: Some redundancy in documentation files
4. **Code Comments**: Need to align with academic paper context
5. **Dataset Naming**: Must ensure compliance with Cataract-LMM conventions

### Refactoring Actions Required

#### 1. Core Video Processor Enhancement
- ✅ Align with reference script FFmpeg command structure
- ✅ Implement specific crop+blur+overlay de-identification technique
- ✅ Ensure hospital-specific processing optimizations

#### 2. Directory Structure Optimization
- ✅ Remove redundant README files (keep only main and per-directory)
- ✅ Consolidate documentation
- ✅ Ensure consistent naming conventions

#### 3. Code Enhancement
- ✅ Add detailed comments referencing academic paper methodology
- ✅ Ensure all code aligns with Cataract-LMM paper context
- ✅ Implement proper error handling and monitoring

#### 4. Configuration Refinement
- ✅ Ensure hospital configurations match paper specifications
- ✅ Align with dataset naming conventions
- ✅ Optimize for phacoemulsification video processing

### Implementation Order
1. Clean up directory structure
2. Update core video processor with reference script logic
3. Enhance de-identification to match reference approach
4. Update all documentation and comments
5. Ensure dataset naming convention compliance
6. Comprehensive testing and validation
