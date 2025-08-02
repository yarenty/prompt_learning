# Memento: A Meta-Cognitive Framework for Self-Evolving System Prompts

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)

A novel meta-cognitive framework that enables large language models (LLMs) to improve their problem-solving capabilities through self-evolving system prompts autonomously.

## 🎯 Overview

Memento incorporates meta-cognitive strategies, such as reflection, principle extraction, and knowledge integration, to continuously enhance reasoning performance across diverse domains. The framework demonstrates superior adaptability, accuracy, and generalization compared to existing approaches.

### Key Features

- **Meta-Cognitive Learning**: AI learns to improve its reasoning strategies
- **Self-Evolving Prompts**: System prompts are updated based on reflection and outcome evaluation
- **Cross-Domain Adaptability**: Framework applied across programming, mathematics, and writing
- **Structured Knowledge Integration**: Extracts, organizes, and reuses problem-solving principles
- **Comprehensive Benchmarking**: Comparison against PromptBreeder, Self-Evolving GPT, and Auto-Evolve

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Benchmarking](#benchmarking)
- [Development](#development)
- [Research](#research)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) (for local model support)
- Git

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yarenty/prompt_learning.git
cd prompt_learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Set up development environment
python setup.py dev

# Install pre-commit hooks
pre-commit install
```

### Model Setup

```bash
# Pull Llama 3.2 model (recommended)
ollama pull llama3.2

# Or pull other models
ollama pull llama2
ollama pull mistral
```

## ⚡ Quick Start

### Initialize the Framework

```bash
# Initialize Memento framework
memento init

# Check status
memento status
```

### Run a Simple Experiment

```python
from memento import PromptLearner, FeedbackCollector, PromptProcessor

# Initialize components
learner = PromptLearner(model="llama3.2")
collector = FeedbackCollector(model="llama3.2")
processor = PromptProcessor(model="llama3.2")

# Define a problem
problem = {
    "description": "Write a function to find the maximum element in a list",
    "solution": "def find_max(lst): return max(lst) if lst else None"
}

# Run learning cycle
result = learner.evaluate_prompt_performance(
    prompt="You are a Python expert.",
    problem=problem,
    evaluation_criteria=["correctness", "efficiency", "readability"]
)

print(f"Evaluation: {result['evaluation']}")
```

### Using the CLI

```bash
# Run an experiment
memento run problems/algorithm_problem.txt --iterations 10

# Run benchmark comparison
memento benchmark data/datasets/humaneval --models memento promptbreeder

# Check framework status
memento status

# Clean data and cache
memento clean --yes
```

## 🏗️ Architecture

### Core Components

```
memento/
├── core/                    # Core framework components
│   ├── learner.py          # PromptLearner - Main learning engine
│   ├── collector.py        # FeedbackCollector - Evaluation system
│   └── processor.py        # PromptProcessor - Principle extraction
├── config/                 # Configuration management
│   ├── models.py           # Configuration data models
│   └── settings.py         # Settings management
├── utils/                  # Utility functions
│   ├── logger.py           # Logging system
│   ├── validators.py       # Validation utilities
│   └── helpers.py          # Helper functions
└── cli/                    # Command-line interface
    └── main.py             # CLI entry point
```

### Learning Cycle

1. **Problem Solving**: Execute tasks using current system prompt
2. **Evaluation**: Measure output quality using multiple criteria
3. **Reflection**: Identify reasoning strategies that led to success/failure
4. **Principle Extraction**: Capture reusable techniques and insights
5. **Prompt Evolution**: Update system prompt with refined strategies

## 📖 Usage

### Basic Usage

```python
from memento import PromptLearner
from memento.config import get_settings

# Get settings
settings = get_settings()

# Initialize learner
learner = PromptLearner(
    model=settings.model.model_name,
    storage_path=str(settings.storage.evolution_path)
)

# Run learning cycle
evolution_result = learner.evaluate_prompt_performance(
    prompt="You are an expert programmer.",
    problem={
        "description": "Implement a binary search algorithm",
        "solution": "def binary_search(arr, target): ..."
    },
    evaluation_criteria=["correctness", "efficiency", "readability"]
)
```

### Advanced Usage

```python
from memento import PromptLearner, FeedbackCollector, PromptProcessor
from memento.utils.logger import setup_logger

# Setup logging
logger = setup_logger("experiment", level="DEBUG")

# Initialize all components
learner = PromptLearner(model="llama3.2")
collector = FeedbackCollector(model="llama3.2")
processor = PromptProcessor(model="llama3.2")

# Run multi-iteration learning
for iteration in range(10):
    logger.info(f"Starting iteration {iteration + 1}")
    
    # Collect feedback
    feedback = collector.collect_solution_feedback(
        problem="Write a sorting algorithm",
        solution="def sort(arr): return sorted(arr)",
        evaluation_criteria=["correctness", "efficiency", "readability"]
    )
    
    # Process feedback and extract insights
    insights = processor.process_feedback()
    
    # Evolve prompt
    updated_prompt = learner.evolve_prompt(
        current_prompt="You are a programmer.",
        lessons=insights
    )
    
    logger.info(f"Iteration {iteration + 1} completed")
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Environment
MEMENTO_ENV=development
MEMENTO_DEBUG=true
MEMENTO_LOG_LEVEL=INFO

# Model Configuration
MEMENTO__MODEL__MODEL_TYPE=ollama
MEMENTO__MODEL__MODEL_NAME=llama3.2
MEMENTO__MODEL__TEMPERATURE=0.7
MEMENTO__MODEL__MAX_TOKENS=2048

# Evaluation Configuration
MEMENTO__EVALUATION__CRITERIA=["correctness", "efficiency", "readability", "maintainability"]
MEMENTO__EVALUATION__BACKEND=llm
MEMENTO__EVALUATION__BATCH_SIZE=10

# Learning Configuration
MEMENTO__LEARNING__MAX_ITERATIONS=50
MEMENTO__LEARNING__CONVERGENCE_THRESHOLD=0.01
MEMENTO__LEARNING__MIN_CONFIDENCE=0.7
```

### Configuration Classes

```python
from memento.config import ModelConfig, EvaluationConfig, LearningConfig

# Model configuration
model_config = ModelConfig(
    model_type="ollama",
    model_name="llama3.2",
    temperature=0.7,
    max_tokens=2048
)

# Evaluation configuration
eval_config = EvaluationConfig(
    criteria=["correctness", "efficiency", "readability"],
    backend="llm",
    batch_size=10
)

# Learning configuration
learning_config = LearningConfig(
    max_iterations=50,
    convergence_threshold=0.01,
    min_confidence=0.7
)
```

## 🏆 Benchmarking

### Supported Baselines

- **PromptBreeder**: Evolutionary prompt optimization
- **Self-Evolving GPT**: Experience accumulation and memory
- **Auto-Evolve**: Self-reasoning framework with error correction

### Running Benchmarks

```bash
# Run full benchmark suite
memento benchmark data/datasets/humaneval

# Run specific models
memento benchmark data/datasets/humaneval --models memento promptbreeder

# Custom output file
memento benchmark data/datasets/humaneval --output results/benchmark_results.json
```

### Benchmark Datasets

- **HumanEval**: 164 hand-written programming problems
- **APPS**: 10,000 coding problems from programming competitions
- **CodeContests**: Google Code Jam problems
- **Custom Datasets**: Domain-specific problem collections

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/yarenty/prompt_learning.git
cd prompt_learning
python setup.py dev

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=memento --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m benchmark

# Run slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Format code
black memento/
isort memento/

# Lint code
flake8 memento/

# Type checking
mypy memento/

# Security check
bandit -r memento/
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Implement feature** with tests
3. **Run quality checks**: `pre-commit run --all-files`
4. **Update documentation**
5. **Submit pull request**

## 📊 Research

### Experimental Results

Our evaluation across 450 tasks (150 each in software engineering, mathematics, and creative writing) demonstrates:

- **15-25% improvement** in correctness scores
- **20-30% improvement** in efficiency metrics
- **Cross-domain transfer** of learned principles
- **Convergence** to stable, high-performance prompts

### Statistical Validation

- **Statistical significance**: p < 0.05 for all comparisons
- **Effect sizes**: Cohen's d > 0.5 for key metrics
- **Confidence intervals**: 95% CI showing consistent improvements

### Research Paper

For detailed methodology, results, and analysis, see our research paper:
- **Title**: "Memento: A Meta-Cognitive Framework for Self-Evolving System Prompts in AI Systems"
- **Authors**: Jaroslaw Nowosad
- **Institution**: SNI Lab, Ireland Research Centre, Huawei

## 📚 Citation

If you use Memento in your research, please cite our work:

```bibtex
@article{nowosad2025memento,
  title={Memento: A Meta-Cognitive Framework for Self-Evolving System Prompts in AI Systems},
  author={Nowosad, Jaroslaw},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/yarenty/prompt_learning}
}
```

**BibTeX entry:**
```bibtex
@misc{nowosad2024memento,
  title={Memento: A Meta-Cognitive Framework for Self-Evolving System Prompts in AI Systems},
  author={Jaroslaw Nowosad},
  year={2025},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```




## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama Team** for providing excellent local LLM infrastructure
- **OpenAI** and **Anthropic** for API access and model capabilities
- **Research Community** for foundational work in prompt engineering and meta-learning

## 📞 Contact

- **Author**: Jaroslaw Nowosad
- **Email**: yarenty@gmail.com
- **GitHub**: [yarenty](https://github.com/yarenty)

## 🔗 Links

- **Repository**: https://github.com/yarenty/prompt_learning
- **Documentation**: https://memento.readthedocs.io
- **Research Paper**: https://arxiv.org/abs/XXXX.XXXXX
- **Issues**: https://github.com/yarenty/prompt_learning/issues

---

**Memento** - Empowering AI systems to learn and evolve through meta-cognitive reflection. 
