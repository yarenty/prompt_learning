"""
Core components of the Memento framework.

This module contains the main classes and functionality for the meta-cognitive
learning system.
"""

from ..exceptions import (
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
from .base import BaseCollector, BaseLearner, BaseProcessor
from .collector import FeedbackCollector
from .learner import PromptLearner
from .processor import PromptProcessor

__all__ = [
    # Abstract base classes
    "BaseLearner",
    "BaseCollector",
    "BaseProcessor",
    # Concrete implementations
    "PromptLearner",
    "FeedbackCollector",
    "PromptProcessor",
    # Exception classes
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
]
