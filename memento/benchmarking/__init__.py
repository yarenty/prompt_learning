"""
Memento Benchmarking Framework

This module provides comprehensive benchmarking capabilities for comparing
Memento against other prompt evolution approaches including:
- PromptBreeder (evolutionary optimization)
- Self-Evolving GPT (experience accumulation)
- Auto-Evolve (self-reasoning framework)

The framework includes:
- Baseline implementations of competing methods
- Standardized evaluation metrics
- Statistical analysis tools
- Visualization capabilities
- Integration with open-source datasets
"""

from .baselines import AutoEvolve, PromptBreeder, SelfEvolvingGPT
from .datasets import APPSDataset, CodeContestsDataset, HumanEvalDataset
from .evaluation import BenchmarkRunner, EvaluationMetrics, StatisticalAnalyzer
from .visualization import ComparisonPlotter, ResultsVisualizer

__all__ = [
    # Baseline implementations
    "PromptBreeder",
    "SelfEvolvingGPT",
    "AutoEvolve",
    # Evaluation framework
    "BenchmarkRunner",
    "EvaluationMetrics",
    "StatisticalAnalyzer",
    # Datasets
    "HumanEvalDataset",
    "APPSDataset",
    "CodeContestsDataset",
    # Visualization
    "ResultsVisualizer",
    "ComparisonPlotter",
]
