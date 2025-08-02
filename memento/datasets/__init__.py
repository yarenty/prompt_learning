"""
Professional Dataset Collection for Memento Evaluation

This module provides access to both standard open-source datasets and custom datasets:

RECOMMENDED APPROACH - Standard Open-Source Datasets:
- HumanEval, BigCodeBench, APPS for software engineering
- MATH, GSM8K for mathematics
- WritingBench, BiGGen-Bench for creative writing

FALLBACK APPROACH - Custom Generated Datasets:
- Software Engineering: Algorithm implementation, data structures, design patterns
- Mathematics: Algebra, calculus, proofs, optimization, statistics
- Creative Writing: Story generation, essays, documentation, style adaptation

Using standard datasets ensures reproducibility, peer-reviewed quality, and professional credibility.
"""

from .creative_writing import CreativeWritingDataset
from .dataset_manager import DatasetManager
from .evaluation_suite import EvaluationSuite
from .mathematics import MathematicsDataset

# Priority 2: Custom generated datasets (FALLBACK)
from .software_engineering import SoftwareEngineeringDataset

# Priority 1: Standard open-source datasets (RECOMMENDED)
from .standard_datasets import StandardDatasetManager, StandardEvaluationRunner

__all__ = [
    # Standard datasets (RECOMMENDED)
    "StandardDatasetManager",
    "StandardEvaluationRunner",
    # Custom datasets (FALLBACK)
    "SoftwareEngineeringDataset",
    "MathematicsDataset",
    "CreativeWritingDataset",
    "DatasetManager",
    "EvaluationSuite",
]
