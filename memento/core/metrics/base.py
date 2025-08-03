"""Base metrics collection functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from scipy import stats

from .types import BaseMetric, EvaluationMetric, PerformanceMetric, QualityMetric, ResourceMetric

logger = logging.getLogger(__name__)


class BaseMetricsCollector:
    """Base class for all metrics collection."""

    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize metrics collector.

        Args:
            storage_path: Optional path for metrics storage
        """
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        self.metrics: List[BaseMetric] = []

    def add_metric(self, metric: BaseMetric) -> None:
        """Add a metric to the collection.

        Args:
            metric: Metric to add
        """
        self.metrics.append(metric)
        logger.debug(f"Added metric: {metric.name}={metric.value}")

    def get_metrics(
        self,
        metric_type: Optional[Type[BaseMetric]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **filters,
    ) -> List[BaseMetric]:
        """Get metrics with optional filtering.

        Args:
            metric_type: Filter by metric type
            start_time: Filter by start time
            end_time: Filter by end time
            **filters: Additional attribute filters

        Returns:
            List of filtered metrics
        """
        filtered = self.metrics

        if metric_type:
            filtered = [m for m in filtered if isinstance(m, metric_type)]

        if start_time:
            filtered = [m for m in filtered if m.timestamp >= start_time]

        if end_time:
            filtered = [m for m in filtered if m.timestamp <= end_time]

        # Apply additional filters
        for key, value in filters.items():
            filtered = [m for m in filtered if hasattr(m, key) and getattr(m, key) == value]

        return filtered

    def get_statistics(self, metric_type: Optional[Type[BaseMetric]] = None, **filters) -> Dict[str, Any]:
        """Calculate statistics for metrics.

        Args:
            metric_type: Filter by metric type
            **filters: Additional filters

        Returns:
            Dictionary with statistics
        """
        metrics = self.get_metrics(metric_type, **filters)
        values = [m.value for m in metrics]

        if not values:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}

        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    def calculate_confidence_interval(
        self, metric_type: Optional[Type[BaseMetric]] = None, confidence: float = 0.95, **filters
    ) -> Dict[str, float]:
        """Calculate confidence interval for metrics.

        Args:
            metric_type: Filter by metric type
            confidence: Confidence level (0-1)
            **filters: Additional filters

        Returns:
            Dictionary with interval statistics
        """
        metrics = self.get_metrics(metric_type, **filters)
        values = [m.value for m in metrics]

        if len(values) < 2:
            return {
                "mean": float(np.mean(values)) if values else None,
                "ci_lower": None,
                "ci_upper": None,
                "confidence": confidence,
            }

        mean = np.mean(values)
        ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=stats.sem(values))

        return {"mean": float(mean), "ci_lower": float(ci[0]), "ci_upper": float(ci[1]), "confidence": confidence}

    def calculate_effect_size(self, group_a: List[BaseMetric], group_b: List[BaseMetric]) -> Dict[str, float]:
        """Calculate effect size between two groups.

        Args:
            group_a: First group of metrics
            group_b: Second group of metrics

        Returns:
            Dictionary with effect size statistics
        """
        values_a = [m.value for m in group_a]
        values_b = [m.value for m in group_b]

        if len(values_a) < 2 or len(values_b) < 2:
            return {"cohens_d": None, "effect_size": None, "interpretation": None}

        # Calculate Cohen's d
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_pooled = np.sqrt(
            ((len(values_a) - 1) * np.var(values_a) + (len(values_b) - 1) * np.var(values_b))
            / (len(values_a) + len(values_b) - 2)
        )

        cohens_d = (mean_a - mean_b) / std_pooled

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {"cohens_d": float(cohens_d), "effect_size": float(abs(cohens_d)), "interpretation": interpretation}

    def save_metrics(self, filename: Optional[str] = None) -> Path:
        """Save metrics to file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if not self.storage_path:
            raise ValueError("Storage path not set")

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        file_path = self.storage_path / filename

        # Convert metrics to dict
        metrics_data = []
        for metric in self.metrics:
            metric_dict = {
                "type": metric.__class__.__name__,
                "timestamp": metric.timestamp.isoformat(),
                **{k: v for k, v in metric.__dict__.items() if k != "timestamp"},
            }
            metrics_data.append(metric_dict)

        with open(file_path, "w") as f:
            json.dump({"metrics": metrics_data, "exported_at": datetime.now().isoformat()}, f, indent=2)

        logger.info(f"Saved {len(self.metrics)} metrics to {file_path}")
        return file_path

    def load_metrics(self, file_path: Union[str, Path]) -> None:
        """Load metrics from file.

        Args:
            file_path: Path to metrics file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        metric_types = {
            "BaseMetric": BaseMetric,
            "PerformanceMetric": PerformanceMetric,
            "EvaluationMetric": EvaluationMetric,
            "ResourceMetric": ResourceMetric,
            "QualityMetric": QualityMetric,
        }

        self.metrics = []
        for metric_data in data["metrics"]:
            metric_type = metric_types[metric_data.pop("type")]
            metric_data["timestamp"] = datetime.fromisoformat(metric_data["timestamp"])
            self.metrics.append(metric_type(**metric_data))

        logger.info(f"Loaded {len(self.metrics)} metrics from {file_path}")

    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()
        logger.info("Cleared all metrics")
