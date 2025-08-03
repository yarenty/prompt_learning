"""Core metric type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class BaseMetric:
    """Base metric type."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric(BaseMetric):
    """Performance-specific metric."""

    unit: str
    component: str
    operation: str


@dataclass
class EvaluationMetric(BaseMetric):
    """Evaluation-specific metric."""

    criteria: List[str]
    scores: List[float]
    weight: Optional[Dict[str, float]] = None


@dataclass
class ResourceMetric(BaseMetric):
    """Resource usage metric."""

    resource_type: str  # 'memory', 'cpu', 'gpu', 'network'
    limit: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class QualityMetric(BaseMetric):
    """Quality assessment metric."""

    category: str  # 'code', 'prompt', 'solution'
    aspects: Dict[str, float]  # Different quality aspects and their scores
    requirements: Optional[List[str]] = None
