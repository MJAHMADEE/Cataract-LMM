# 📚 Documentation

> **Comprehensive documentation framework for the Cataract-LMM surgical video analysis project**

## 📋 Overview

This directory contains the complete **documentation framework** for the Cataract-LMM project, providing detailed technical documentation, API references, user guides, and development resources. Built with **Sphinx** for professional, medical-grade documentation standards.

### 🎯 Documentation Philosophy

Following **medical software documentation standards** with:
- **📖 Comprehensive Coverage**: Every module, API, and workflow documented
- **👩‍⚕️ Medical Context**: Clinical use cases and surgical workflow integration
- **🔧 Developer-Focused**: Clear API references and integration guides
- **🎓 Educational**: Learning resources for AI in surgical video analysis
- **📊 Visual**: Rich diagrams, flowcharts, and example visualizations

## 📁 Contents

| File / Component | Type | Description |
|:----------------|:-----|:------------|
| `conf.py` | ⚙️ **Sphinx Configuration** | Main Sphinx documentation configuration with medical-specific themes and extensions |
| `index.rst` | 📖 **Main Documentation** | Primary reStructuredText documentation index with comprehensive project overview |
| `index.md` | 📝 **Markdown Index** | Alternative Markdown-based documentation entry point for web viewing |
| `index_new.rst` | 🆕 **Updated Documentation** | Latest documentation updates with enhanced medical AI content |
| `STAGE_6_TEST_COVERAGE.md` | 🧪 **Test Coverage Report** | Detailed test coverage analysis and quality metrics |
| `_static/` | 🎨 **Static Assets** | CSS, images, diagrams, and visual assets for documentation rendering |

## 🚀 Building Documentation

### Local Development

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build HTML documentation
cd codes/docs
sphinx-build -b html . _build/html

# Build with automatic reload during development
sphinx-autobuild . _build/html --watch ../

# Open documentation in browser
open _build/html/index.html
```

### Production Build

```bash
# Clean previous builds
make clean

# Build production documentation
make html

# Build PDF documentation (requires LaTeX)
make latexpdf

# Build EPUB for mobile reading
make epub
```

## 📖 Documentation Structure

### 🏗️ Main Documentation (`index.rst`)

**Comprehensive project documentation covering:**

```restructuredtext
Cataract-LMM Documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   getting_started
   api_reference
   modules/index
   tutorials/index
   deployment/index
   contributing
   changelog
```

**Key Sections:**
- 🚀 **Getting Started**: Installation, setup, and first steps
- 🔧 **API Reference**: Complete API documentation for all modules
- 📚 **Module Documentation**: Detailed docs for each core module
- 🎓 **Tutorials**: Step-by-step guides and examples
- 🚀 **Deployment**: Production deployment guides
- 🤝 **Contributing**: Development and contribution guidelines

### 📊 Module Documentation Structure

Each core module has comprehensive documentation:

```
docs/modules/
├── video_processing/
│   ├── api.rst
│   ├── tutorials.rst
│   ├── examples.rst
│   └── configuration.rst
├── instance_segmentation/
│   ├── models.rst
│   ├── training.rst
│   ├── evaluation.rst
│   └── deployment.rst
├── phase_recognition/
│   ├── architectures.rst
│   ├── datasets.rst
│   ├── benchmarks.rst
│   └── clinical_integration.rst
└── skill_assessment/
    ├── methodology.rst
    ├── metrics.rst
    ├── validation.rst
    └── clinical_applications.rst
```

## 🎨 Documentation Themes and Styling

### Medical-Grade Styling (`_static/custom.css`)

```css
/* Professional medical documentation styling */
:root {
    --medical-blue: #2E86AB;
    --surgical-green: #A23B72;
    --clinical-gray: #F18F01;
    --safety-red: #C73E1D;
}

.medical-callout {
    background: var(--medical-blue);
    border-left: 4px solid var(--surgical-green);
    padding: 1rem;
    margin: 1rem 0;
}

.surgical-workflow {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}
```

### Visual Assets

- 📊 **Architecture Diagrams**: System architecture and data flow
- 🎥 **Workflow Illustrations**: Surgical video analysis pipelines
- 📈 **Performance Charts**: Benchmarking and comparison graphs
- 🏥 **Clinical Integration**: Hospital workflow diagrams
- 🎯 **Model Visualizations**: Neural network architecture diagrams

## 📚 Documentation Types

### 🔧 API Documentation

**Automatically generated from docstrings:**

```python
def process_surgical_video(input_path: str, config: Dict) -> ProcessingResult:
    """
    Process surgical video with Cataract-LMM pipeline.
    
    Args:
        input_path: Path to surgical video file
        config: Processing configuration dictionary
        
    Returns:
        ProcessingResult containing all analysis results
        
    Example:
        >>> result = process_surgical_video('surgery.mp4', config)
        >>> print(f"Detected phases: {result.phases}")
        >>> print(f"Skill score: {result.skill_assessment.score}")
    """
```

### 🎓 Tutorial Documentation

**Step-by-step learning guides:**

1. **Getting Started Tutorial**
   - Environment setup
   - First surgical video analysis
   - Understanding results

2. **Advanced Workflows**
   - Multi-task analysis pipelines
   - Custom model training
   - Performance optimization

3. **Clinical Integration**
   - Hospital workflow integration
   - Real-time analysis setup
   - Quality assurance protocols

### 📊 Technical Specifications

**Detailed technical documentation:**

- **Performance Benchmarks**: Speed, accuracy, and resource usage
- **Hardware Requirements**: GPU, memory, and storage specifications  
- **Clinical Validation**: Medical accuracy and safety considerations
- **Compliance Standards**: HIPAA, FDA, and medical device regulations

## 🔧 Sphinx Configuration

### Extensions and Plugins

```python
# conf.py extensions for medical documentation
extensions = [
    'sphinx.ext.autodoc',        # Automatic API documentation
    'sphinx.ext.viewcode',       # Source code linking
    'sphinx.ext.napoleon',       # Google/NumPy docstring support
    'sphinx.ext.intersphinx',    # Cross-project linking
    'sphinx.ext.mathjax',        # Mathematical equations
    'sphinx_rtd_theme',          # Read the Docs theme
    'sphinx.ext.githubpages',    # GitHub Pages deployment
    'sphinx_tabs.tabs',          # Tabbed content
    'sphinx_copybutton',         # Copy code button
    'sphinx_design',             # Modern design elements
    'myst_parser',               # Markdown support
]
```

### Medical Documentation Standards

```python
# Medical software documentation configuration
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Medical compliance settings
html_context = {
    'medical_grade': True,
    'hipaa_compliant': True,
    'clinical_validation': True
}
```

## 📊 Documentation Analytics

### Coverage Metrics

Track documentation completeness:

```bash
# Generate documentation coverage report
sphinx-build -b coverage . _build/coverage

# Check for undocumented APIs
python -m sphinx.ext.coverage _build/coverage

# Validate all cross-references
sphinx-build -b linkcheck . _build/linkcheck
```

### Quality Assurance

- **📝 Docstring Coverage**: 95%+ target for all public APIs
- **🔗 Link Validation**: Automated checking of all external links
- **📊 Medical Accuracy**: Clinical review of all medical content
- **🌐 Accessibility**: WCAG 2.1 compliance for web documentation

## 🚀 Documentation Deployment

### GitHub Pages

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation
on:
  push:
    branches: [main]
jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r docs/requirements.txt
      - name: Build documentation
        run: sphinx-build -b html docs docs/_build/html
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html
```

### Read the Docs Integration

- **📚 Automatic Builds**: Documentation builds on every commit
- **🌍 Multi-Version**: Support for different release versions
- **🔍 Search Integration**: Full-text search across all documentation
- **📱 Mobile-Friendly**: Responsive design for all devices

## 🧪 Documentation Testing

### Automated Testing

```bash
# Test all code examples in documentation
sphinx-build -b doctest . _build/doctest

# Validate documentation structure
doc8 --ignore D001 docs/

# Test documentation builds without warnings
sphinx-build -W -b html . _build/html
```

### Manual Review Process

1. **📖 Content Review**: Medical accuracy and completeness
2. **🎨 Visual Review**: Layout, images, and formatting
3. **🔗 Link Testing**: All external and internal links
4. **📱 Mobile Testing**: Responsive design validation
5. **🔍 Search Testing**: Search functionality and results

## 📚 Contributing to Documentation

### Writing Guidelines

1. **🎯 Audience-Focused**: Write for the intended user (developer, clinician, researcher)
2. **📊 Example-Rich**: Include practical code examples and use cases
3. **🏥 Medical Context**: Provide clinical relevance and applications
4. **🔧 Actionable**: Every guide should lead to concrete outcomes
5. **📏 Consistent**: Follow established style and formatting guidelines

### Documentation Review Process

```bash
# Create documentation branch
git checkout -b docs/new-feature-documentation

# Write documentation following templates
# docs/templates/ contains standard templates

# Build and test locally
sphinx-build -b html . _build/html

# Submit for review
git push origin docs/new-feature-documentation
# Create pull request with documentation label
```

## 📚 References

- **Sphinx Documentation**: https://www.sphinx-doc.org/
- **Medical Documentation Standards**: IEC 62304, ISO 13485
- **reStructuredText Guide**: https://docutils.sourceforge.io/rst.html
- **Clinical AI Guidelines**: FDA Software as Medical Device guidance

---

**💡 Note**: This documentation framework follows medical software documentation standards and ensures the comprehensive coverage needed for clinical deployment of AI-assisted surgical video analysis systems.
