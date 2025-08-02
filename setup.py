#!/usr/bin/env python3
"""
Setup script for the Memento framework.
"""

from setuptools import find_packages, setup

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memento",
    version="0.2.0",
    author="Jaroslaw Nowosad",
    author_email="yarenty@gmail.com",
    description="A Meta-Cognitive Framework for Self-Evolving System Prompts in AI Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yarenty/prompt_learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.0.0",
        "ollama>=0.1.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memento=memento.cli.main:main",
        ],
    },
)
