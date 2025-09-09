# 🎨 Static Assets

## Overview

This directory contains static assets used by the documentation system, including custom stylesheets, images, and other resources for the Cataract-LMM documentation site.

## 📁 Contents

### **Stylesheets**

| File | Description |
|------|-------------|
| `custom.css` | Custom CSS styles for documentation theming and layout enhancements |

## 🎯 Purpose

### **Documentation Theming**

The static assets in this directory provide:

- **Visual Consistency**: Uniform styling across all documentation pages
- **Brand Identity**: Custom colors, fonts, and layouts reflecting the project's professional appearance
- **Enhanced Readability**: Optimized typography and spacing for technical documentation
- **Responsive Design**: Mobile-friendly layouts for documentation access on various devices

### **Custom Styling Features**

The `custom.css` file includes:

```css
/* Medical AI Theme Colors */
:root {
    --primary-color: #26a69a;     /* Medical AI accent */
    --secondary-color: #3776ab;   /* Python blue */
    --accent-color: #ee4c2c;      /* PyTorch orange */
    --text-color: #333333;        /* Dark gray */
    --background-color: #ffffff;   /* Clean white */
}

/* Enhanced Code Blocks */
.highlight {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1em 0;
}

/* Professional Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
```

## 🛠️ Usage

### **Sphinx Integration**

The static assets are automatically included in Sphinx documentation builds:

```python
# docs/conf.py
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
```

### **Custom Styling Guidelines**

When modifying styles:

1. **Maintain Consistency**: Follow established color schemes and typography
2. **Responsive Design**: Ensure styles work across device sizes
3. **Accessibility**: Use sufficient color contrast and readable fonts
4. **Performance**: Optimize CSS for fast loading times

## 📱 Responsive Design

The custom styles support multiple screen sizes:

- **Desktop** (≥1200px): Full-width layouts with sidebar navigation
- **Tablet** (768px - 1199px): Condensed layouts with collapsible menus
- **Mobile** (≤767px): Single-column layouts with touch-friendly interfaces

## 🎨 Design System

### **Color Palette**

| Color | Usage | Hex Code |
|-------|-------|----------|
| Medical AI Green | Primary accents, links | `#26a69a` |
| Python Blue | Code elements, headers | `#3776ab` |
| PyTorch Orange | Warnings, highlights | `#ee4c2c` |
| Dark Gray | Body text | `#333333` |
| Light Gray | Borders, dividers | `#e0e0e0` |

### **Typography**

- **Headers**: `"Roboto", "Helvetica Neue", Arial, sans-serif`
- **Body Text**: `"Source Sans Pro", Arial, sans-serif`
- **Code**: `"JetBrains Mono", "Consolas", monospace`

### **Component Styling**

```css
/* Professional buttons */
.btn {
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s ease;
}

/* Enhanced admonitions */
.admonition {
    border-left: 4px solid var(--primary-color);
    background: rgba(38, 166, 154, 0.05);
}

/* Code syntax highlighting */
.highlight .k { color: #008000; }  /* Keywords */
.highlight .s { color: #ba2121; }  /* Strings */
.highlight .c { color: #808080; }  /* Comments */
```

## 📊 Asset Management

### **File Organization**

```
_static/
├── custom.css          # Main stylesheet
├── images/            # Documentation images (if added)
├── fonts/             # Custom fonts (if added)
└── js/                # Custom JavaScript (if added)
```

### **Build Process**

Static assets are:

1. **Copied** to the documentation output directory during builds
2. **Minified** in production builds for optimal performance
3. **Cached** by browsers for faster subsequent page loads
4. **Versioned** to ensure updates are properly loaded

## 🔧 Development

### **Local Development**

When working on documentation styles:

```bash
# Serve documentation locally with live reload
cd docs
make livehtml

# Build documentation to test styles
make html
```

### **Style Testing**

Test custom styles across:

- Different browsers (Chrome, Firefox, Safari, Edge)
- Various screen sizes (desktop, tablet, mobile)
- Documentation sections (API reference, tutorials, guides)
- Different themes (light/dark if supported)

## 📚 Best Practices

1. **Semantic CSS**: Use meaningful class names and maintain clear structure
2. **Progressive Enhancement**: Ensure base styles work without custom CSS
3. **Performance**: Minimize CSS size and use efficient selectors
4. **Maintainability**: Comment complex styles and organize logically
5. **Testing**: Verify styles across browsers and devices

## 🎯 Future Enhancements

Planned improvements:

- **Dark Mode**: Toggle between light and dark themes
- **Print Styles**: Optimized layouts for printed documentation
- **Interactive Elements**: Enhanced forms and navigation components
- **Animation**: Subtle transitions and micro-interactions

---

*These static assets ensure the Cataract-LMM documentation maintains a professional, accessible, and visually appealing presentation across all platforms.*
