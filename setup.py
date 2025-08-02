#!/usr/bin/env python3
"""
Setup script for the Memento framework.

This script provides easy installation and development setup for the framework.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_development_environment():
    """Set up the development environment."""
    print("🚀 Setting up Memento development environment...")

    # Create necessary directories
    directories = [
        "data",
        "data/feedback",
        "data/evolution",
        "data/cache",
        "data/datasets",
        "logs",
        "results",
        "docs",
        "tests/data",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Memento Framework Configuration
# Environment
MEMENTO_ENV=development
MEMENTO_DEBUG=true
MEMENTO_LOG_LEVEL=INFO

# Model Configuration
MEMENTO__MODEL__MODEL_TYPE=ollama
MEMENTO__MODEL__MODEL_NAME=codellama
MEMENTO__MODEL__TEMPERATURE=0.7
MEMENTO__MODEL__MAX_TOKENS=2048

# Evaluation Configuration
MEMENTO__EVALUATION__CRITERIA=["correctness", "efficiency", "readability", "maintainability"]
MEMENTO__EVALUATION__BACKEND=llm
MEMENTO__EVALUATION__BATCH_SIZE=10
MEMENTO__EVALUATION__CACHE_RESULTS=true

# Learning Configuration
MEMENTO__LEARNING__MAX_ITERATIONS=50
MEMENTO__LEARNING__CONVERGENCE_THRESHOLD=0.01
MEMENTO__LEARNING__MIN_CONFIDENCE=0.7

# API Keys (uncomment and set as needed)
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
"""
        env_file.write_text(env_content)
        print("✅ Created .env file")

    # Create .gitignore if it doesn't exist
    gitignore_file = Path(".gitignore")
    if not gitignore_file.exists():
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VS Code
.vscode/

# Memento specific
data/cache/*
data/feedback/*
data/evolution/*
logs/*.log
results/*
!data/cache/.gitkeep
!data/feedback/.gitkeep
!data/evolution/.gitkeep
!logs/.gitkeep
!results/.gitkeep

# API keys and secrets
*.key
*.pem
secrets.json
"""
        gitignore_file.write_text(gitignore_content)
        print("✅ Created .gitignore file")

    # Create .gitkeep files for empty directories
    gitkeep_dirs = ["data/cache", "data/feedback", "data/evolution", "logs", "results"]
    for directory in gitkeep_dirs:
        gitkeep_file = Path(directory) / ".gitkeep"
        gitkeep_file.touch(exist_ok=True)

    print("✅ Development environment setup complete!")


def install_pre_commit_hooks():
    """Install pre-commit hooks."""
    try:
        import subprocess

        result = subprocess.run(
            ["pre-commit", "install"], capture_output=True, text=True, check=True
        )
        print("✅ Pre-commit hooks installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not install pre-commit hooks: {e}")
        print("You can install them manually with: pre-commit install")
    except FileNotFoundError:
        print(
            "⚠️  Warning: pre-commit not found. Install it with: pip install pre-commit"
        )


def main():
    """Main setup function."""
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        setup_development_environment()
        install_pre_commit_hooks()
    else:
        print("Usage: python setup.py dev")
        print("This will set up the development environment.")


if __name__ == "__main__":
    main()
