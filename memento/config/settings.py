"""
Main settings configuration for the Memento framework.

This module provides the main Settings class that combines all configuration
components and handles loading from environment variables and files.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field

from .models import (
    BenchmarkConfig,
    EvaluationConfig,
    LearningConfig,
    ModelConfig,
    StorageConfig,
)


class Settings(BaseSettings):
    """
    Main settings class for the Memento framework.

    This class combines all configuration components and provides
    a unified interface for accessing settings throughout the application.
    """

    # Environment and application settings
    environment: str = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Configuration components
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig, description="Storage configuration"
    )
    learning: LearningConfig = Field(
        default_factory=LearningConfig, description="Learning configuration"
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig, description="Benchmark configuration"
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False

        # Allow environment variable overrides
        fields = {
            "environment": {"env": "MEMENTO_ENV"},
            "debug": {"env": "MEMENTO_DEBUG"},
            "log_level": {"env": "MEMENTO_LOG_LEVEL"},
        }

    def __init__(self, **kwargs):
        """Initialize settings with optional overrides."""
        super().__init__(**kwargs)
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate settings after initialization."""
        # Ensure all required directories exist
        self.storage.base_path.mkdir(parents=True, exist_ok=True)
        self.storage.feedback_path.mkdir(parents=True, exist_ok=True)
        self.storage.evolution_path.mkdir(parents=True, exist_ok=True)
        self.storage.logs_path.mkdir(parents=True, exist_ok=True)
        self.storage.cache_path.mkdir(parents=True, exist_ok=True)

        # Validate model configuration
        if self.model.model_type.value == "openai" and not self.model.api_key:
            raise ValueError("OpenAI API key is required when using OpenAI models")

        if self.model.model_type.value == "anthropic" and not self.model.api_key:
            raise ValueError(
                "Anthropic API key is required when using Anthropic models"
            )

    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self.evaluation

    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        return self.storage

    def get_learning_config(self) -> LearningConfig:
        """Get learning configuration."""
        return self.learning

    def get_benchmark_config(self) -> BenchmarkConfig:
        """Get benchmark configuration."""
        return self.benchmark

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"

    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment.lower() == "testing"

    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "model": self.model.dict(),
            "evaluation": self.evaluation.dict(),
            "storage": {
                "base_path": str(self.storage.base_path),
                "feedback_path": str(self.storage.feedback_path),
                "evolution_path": str(self.storage.evolution_path),
                "logs_path": str(self.storage.logs_path),
                "cache_path": str(self.storage.cache_path),
            },
            "learning": self.learning.dict(),
            "benchmark": {
                **self.benchmark.dict(),
                "dataset_path": str(self.benchmark.dataset_path),
                "results_path": str(self.benchmark.results_path),
            },
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function provides a singleton pattern for settings,
    ensuring consistent configuration across the application.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Reload settings from environment and files.

    This function clears the cached settings and creates a new instance,
    useful for testing or when configuration changes need to be picked up.

    Returns:
        Settings: The new settings instance.
    """
    get_settings.cache_clear()
    return get_settings()
