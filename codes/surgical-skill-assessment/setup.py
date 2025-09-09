"""
Setup script for Surgical Skill Assessment package.
"""

import os

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="surgical-skill-assessment",
    version="1.0.0",
    author="Surgical Skill Assessment Team",
    author_email="your-email@example.com",
    description="Deep learning pipeline for automated surgical skill assessment from video data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/surgical-skill-assessment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "surgical-skill-assessment=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "notebooks/*.ipynb"],
    },
)
