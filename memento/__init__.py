"""
Memento: Meta-Cognitive Framework for Self-Evolving System Prompts

A comprehensive framework for prompt evolution through self-learning and feedback integration.
Includes benchmarking capabilities against other prompt evolution methods.
"""

# Benchmarking framework
from . import benchmarking
from .config import EvaluationBackend, ModelConfig, ModelType
from .core import FeedbackCollector, PromptLearner, PromptProcessor
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

__version__ = "0.4.0"

__all__ = [
    # Core functionality
    "PromptLearner",
    "FeedbackCollector",
    "PromptProcessor",
    # Configuration
    "ModelConfig",
    "ModelType",
    "EvaluationBackend",
    # Exceptions
    "MementoError",
    "ValidationError",
    "EvaluationError",
    "EvolutionError",
    "CollectionError",
    "ProcessingError",
    "ExtractionError",
    "ReflectionError",
    "StorageError",
    "ConfigurationError",
    "UpdateError",
    # Benchmarking
    "benchmarking",
]
