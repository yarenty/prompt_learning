"""
Custom exception classes for the Memento framework.

This module defines all custom exceptions used throughout the framework
to avoid circular import issues.
"""


class MementoError(Exception):
    """Base exception for all Memento framework errors."""

    pass


class ValidationError(MementoError):
    """Raised when input validation fails."""

    pass


class EvaluationError(MementoError):
    """Raised when evaluation operations fail."""

    pass


class EvolutionError(MementoError):
    """Raised when prompt evolution fails."""

    pass


class CollectionError(MementoError):
    """Raised when feedback collection fails."""

    pass


class ReflectionError(MementoError):
    """Raised when reflection generation fails."""

    pass


class ProcessingError(MementoError):
    """Raised when feedback processing fails."""

    pass


class ExtractionError(MementoError):
    """Raised when insight extraction fails."""

    pass


class UpdateError(MementoError):
    """Raised when prompt update fails."""

    pass


class StorageError(MementoError):
    """Raised when storage operations fail."""

    pass


class ConfigurationError(MementoError):
    """Raised when configuration is invalid."""

    pass
