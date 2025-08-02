"""
Configuration models for the Memento framework.

This module defines the data models used for configuration management.
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ModelType(str, Enum):
    """Supported model types."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class EvaluationBackend(str, Enum):
    """Supported evaluation backends."""

    LLM = "llm"
    HUMAN = "human"
    AUTOMATED = "automated"


class ModelConfig(BaseModel):
    """Configuration for LLM models."""

    model_type: ModelType = Field(default=ModelType.OLLAMA, description="Type of model to use")
    model_name: str = Field(default="codellama", description="Name of the specific model")
    api_key: Optional[str] = Field(default=None, description="API key for the model service")
    base_url: Optional[str] = Field(default=None, description="Base URL for the model service")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, ge=1, description="Maximum tokens to generate")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")

    @validator("model_name")
    def validate_model_name(cls, v):
        """Validate model name is not empty."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class EvaluationConfig(BaseModel):
    """Configuration for evaluation settings."""

    criteria: List[str] = Field(
        default=["correctness", "efficiency", "readability", "maintainability"],
        description="Evaluation criteria to use",
    )
    backend: EvaluationBackend = Field(default=EvaluationBackend.LLM, description="Evaluation backend to use")
    batch_size: int = Field(default=10, ge=1, description="Batch size for evaluations")
    cache_results: bool = Field(default=True, description="Whether to cache evaluation results")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    @validator("criteria")
    def validate_criteria(cls, v):
        """Validate evaluation criteria."""
        if not v:
            raise ValueError("At least one evaluation criterion must be specified")
        return [criterion.lower().strip() for criterion in v]


class StorageConfig(BaseModel):
    """Configuration for data storage."""

    base_path: Path = Field(default=Path("data"), description="Base path for data storage")
    feedback_path: Path = Field(default=Path("data/feedback"), description="Path for feedback data")
    evolution_path: Path = Field(default=Path("data/evolution"), description="Path for evolution data")
    logs_path: Path = Field(default=Path("logs"), description="Path for log files")
    cache_path: Path = Field(default=Path("data/cache"), description="Path for cache data")

    @validator(
        "base_path",
        "feedback_path",
        "evolution_path",
        "logs_path",
        "cache_path",
        pre=True,
    )
    def create_paths(cls, v):
        """Create directories if they don't exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


class LearningConfig(BaseModel):
    """Configuration for learning parameters."""

    max_iterations: int = Field(default=50, ge=1, description="Maximum learning iterations")
    convergence_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Threshold for convergence detection"
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for principle extraction",
    )
    principle_similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for principle similarity matching",
    )

    @validator("max_iterations")
    def validate_max_iterations(cls, v):
        """Validate maximum iterations."""
        if v < 1:
            raise ValueError("Maximum iterations must be at least 1")
        return v


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarking."""

    baseline_models: List[str] = Field(
        default=["promptbreeder", "self-evolving-gpt", "auto-evolve"],
        description="Baseline models to compare against",
    )
    dataset_path: Path = Field(default=Path("data/datasets"), description="Path to benchmark datasets")
    results_path: Path = Field(default=Path("results"), description="Path for benchmark results")
    statistical_significance_level: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Statistical significance level for tests",
    )

    @validator("dataset_path", "results_path", pre=True)
    def create_paths(cls, v):
        """Create directories if they don't exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
