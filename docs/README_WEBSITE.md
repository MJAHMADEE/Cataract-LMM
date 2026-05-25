# 🌐 Cataract-LMM Project Website

This directory contains the source code for the **professional, interactive Github Pages website** for **Cataract-LMM: A Large-Scale, Multi-Source, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis**.

## 📋 Overview
- **Published in**: Nature Scientific Data (May 2026)
- **License**: CC BY-NC-ND 4.0 (Open-Access Dataset)
- **DOI**: [10.1038/s41597-026-07464-0](https://doi.org/10.1038/s41597-026-07464-0)

## 🏗️ Architecture & Stack
- Single-page HTML5 architecture (`index.html`)
- **Tailwind CSS** (via CDN for zero-build-step deployment)
- Vanilla JavaScript for IntersectionObserver scroll animations and number counters
- Custom "Sci-Tech Minimalist" styling with glassmorphism effects
- Dynamic neural network particle background

## 📚 Features

### Content Sections
1. **Hero Section** - Eye-catching introduction with key CTAs
   - GitHub Repository Link
   - Hugging Face Dataset Download
   - Research Paper (Nature Scientific Data)
   - ArXiv Preprint

2. **Data Counters** - Key statistics visualization
   - 3,000+ procedures
   - 1,134+ hours of video
   - 6,094 segmented frames
   - 4 annotation tasks
   - 2 clinical centers

3. **Overview Section** - Problem statement and dataset description
   - Reality gap in surgical AI
   - Multi-center acquisition
   - Hardware heterogeneity

4. **Data Subsets** - Four core annotation tasks
   - Phase Recognition (13-phase surgical workflow)
   - Instance Segmentation (12 instrument/anatomy classes)
   - Object Tracking (spatiotemporal analysis)
   - Skill Assessment (GRASIS/ICO-OSCAR rubric)

5. **Interactive Benchmarks** - Technical validation results
   - Workflow & Phase Recognition
   - Instance Segmentation
   - Spatiotemporal Tracking
   - Objective Skill Assessment

6. **Citation & License Footer**
   - Multiple citation formats (BibTeX, APA, MLA, Chicago, Harvard, IEEE)
   - License information (CC BY-NC-ND 4.0)
   - Publication details (Nature Scientific Data)
   - Contact information

## 🎨 Customization Guide

### 1. Replace Placeholder Assets
The website uses SVG placeholders. Replace with actual figures from your paper:
- `docs/assets/hero-bg.mp4` (Optional background video)
- `docs/assets/teaser-figure.png` (Main methodology figure)
- `docs/assets/task-1-phase.png` (Phase recognition visualization)
- `docs/assets/task-2-seg.png` (Segmentation examples)
- `docs/assets/task-3-track.png` (Tracking visualization)
- `docs/assets/task-4-skill.png` (Skill assessment results)

*Tip: Dark backgrounds or transparent PNGs blend best with the glassmorphism effects.*

### 2. Update Citation Information
Edit the citation section in `index.html` (search for `switchCitationTab`):
- Update author names
- Modify publication year and date
- Update DOI and journal reference
- Adjust BibTeX entry

### 3. Update Links
Key links to update:
- ArXiv: `https://arxiv.org/abs/2510.16371`
- Nature Scientific Data: `https://doi.org/10.1038/s41597-026-07464-0`
- Hugging Face Dataset: `https://huggingface.co/datasets/mjahmadi/Cataract-LMM`
- GitHub Repository: `https://github.com/MJAHMADEE/Cataract-LMM`

### 4. License Compliance
- Update footer license statement if needed
- Update institutional affiliations
- Maintain attribution to all partners

## 🚀 Deployment via GitHub Pages

### Setup
1. Ensure files are in the `/docs` folder of your repository
2. Go to **Repository Settings** → **Pages**
3. Select **Deploy from a branch**
4. Choose `main` branch and `/docs` folder
5. Click **Save**

### Verification
- Website will be available at: `https://<username>.github.io/Cataract-LMM/`
- Deployment typically completes within 2-3 minutes
- Check the **Deployments** tab for status

## 🔧 Theme Support

The website includes automatic light/dark mode support:
- Dark mode: Professional medical visualization theme
- Light mode: High-contrast readable alternative
- User preference is saved to browser localStorage
- Toggle button in header navbar

## 📝 Browser Support
- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers

## 🎯 Performance Optimizations
- Glassmorphism CSS for visual depth without heavy assets
- SVG icons for crisp scaling at any resolution
- Minimal JavaScript for fast interactions
- CDN-hosted Tailwind CSS
- Particle network animation with canvas

## 📞 Support
For website-related questions, contact:
- **Academic**: mjahmadi@email.kntu.ac.ir
- **Personal**: mjahmadee@gmail.com
