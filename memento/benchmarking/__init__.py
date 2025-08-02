"""
Professional Benchmarking Framework for Memento

This module provides comprehensive benchmarking capabilities using:
- Standard open-source datasets (HumanEval, MATH, BiGGen-Bench, etc.)
- Professional evaluation metrics and statistical analysis
- Comparative analysis against established baselines
- Reproducible and peer-reviewed evaluation protocols

Components:
- StandardBenchmarkRunner: Main benchmarking orchestrator
- Baseline implementations: PromptBreeder, Self-Evolving GPT, Auto-Evolve
- Evaluation metrics and statistical analysis tools
- Visualization and reporting utilities
"""

from .baselines.auto_evolve import AutoEvolve

# Baseline implementations for comparison
from .baselines.prompt_breeder import PromptBreeder
from .baselines.self_evolving_gpt import SelfEvolvingGPT

# Core benchmarking framework
from .evaluation.benchmark_runner import StandardBenchmarkRunner
from .evaluation.metrics import EvaluationMetrics
from .evaluation.statistical_analyzer import StatisticalAnalyzer
from .visualization.comparison_plotter import ComparisonPlotter

# Visualization and reporting
from .visualization.results_visualizer import ResultsVisualizer

__all__ = [
    # Main benchmarking framework (RECOMMENDED)
    "StandardBenchmarkRunner",
    "EvaluationMetrics",
    "StatisticalAnalyzer",
    # Baseline implementations
    "PromptBreeder",
    "SelfEvolvingGPT",
    "AutoEvolve",
    # Visualization tools
    "ResultsVisualizer",
    "ComparisonPlotter",
]
