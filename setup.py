#!/usr/bin/env python3
"""Setup script for the Vegas backtesting engine."""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# Define Cython extensions
extensions = [
    Extension(
        "vegas.engine.event_engine",
        ["vegas/engine/event_engine.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="vegas",
    version="0.1.0",
    description="Minimal Vectorized Backtesting Engine",
    author="David Styers",
    author_email="david.styers@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "zstandard>=0.15.0",  # For zstd compression support
        "quantstats>=0.0.59",  # For performance analytics and tearsheets
        "duckdb>=0.9.0",      # For efficient data querying
        "pyarrow>=14.0.0",    # For Parquet support
        "cython>=3.0.0",      # For Cython compilation
    ],
    ext_modules=cythonize(extensions),
    entry_points={
        "console_scripts": [
            "vegas=vegas.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 