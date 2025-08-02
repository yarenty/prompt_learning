"""
Memento: A Meta-Cognitive Framework for Self-Evolving System Prompts in AI Systems

A novel framework that enables large language models (LLMs) to improve their
problem-solving capabilities through self-evolving system prompts autonomously.
"""

__version__ = "0.1.0"
__author__ = "Jaroslaw Nowosad"
__email__ = "jaroslaw.nowosad@huawei.com"
__description__ = "A Meta-Cognitive Framework for Self-Evolving System Prompts in AI Systems"

from .core.collector import FeedbackCollector

# Core imports
from .core.learner import PromptLearner
from .core.processor import PromptProcessor

# Exception imports
from .exceptions import (
    CollectionError,
    ConfigurationError,
    EvaluationError,
    EvolutionError,
    ExtractionError,
    MementoError,
    ProcessingError,
    ReflectionError,
    StorageError,
    UpdateError,
    ValidationError,
)

# Version info
__all__ = [
    "PromptLearner",
    "FeedbackCollector",
    "PromptProcessor",
    "MementoError",
    "ValidationError",
    "EvaluationError",
    "EvolutionError",
    "CollectionError",
    "ReflectionError",
    "ProcessingError",
    "ExtractionError",
    "UpdateError",
    "StorageError",
    "ConfigurationError",
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]
