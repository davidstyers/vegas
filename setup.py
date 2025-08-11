#!/usr/bin/env python3
"""Setup script for the Vegas backtesting engine."""

from setuptools import find_packages, setup

setup(
    name="vegas",
    version="0.1.0",
    description="Event-Driven Backtesting Engine",
    author="David Styers",
    author_email="david.styers@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "polars>=0.20.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "zstandard>=0.15.0",  # For zstd compression support
        "quantstats>=0.0.59",  # For performance analytics and tearsheets
        "duckdb>=0.9.0",  # For efficient data querying
        "pyarrow>=14.0.0",  # For Parquet support
        "pytz>=2023.3.0",  # For timezone support
    ],
    extras_require={
        "ibkr": [
            "ibapi>=9.81.1",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
        ],
    },
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
