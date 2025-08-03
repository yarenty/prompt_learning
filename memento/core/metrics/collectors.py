"""Specialized metric collectors."""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psutil
from scipy import stats

from .base import BaseMetricsCollector
from .types import EvaluationMetric, PerformanceMetric, QualityMetric, ResourceMetric


class PerformanceCollector(BaseMetricsCollector):
    """Performance metrics collection."""

    def add_timing(
        self, name: str, duration: float, component: str, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add timing metric.

        Args:
            name: Metric name
            duration: Time duration in seconds
            component: Component name
            operation: Operation name
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name, value=duration, unit="seconds", component=component, operation=operation, metadata=metadata or {}
        )
        self.add_metric(metric)

    def start_operation(
        self, name: str, component: str, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start timing an operation.

        Args:
            name: Operation name
            component: Component name
            operation: Operation type
            metadata: Additional metadata

        Returns:
            Operation ID
        """
        op_id = f"{name}_{time.time_ns()}"
        metadata = metadata or {}
        metadata["start_time"] = time.time()
        metadata["op_id"] = op_id

        # Add start marker
        self.add_metric(
            PerformanceMetric(
                name=f"{name}_start",
                value=0.0,
                unit="marker",
                component=component,
                operation=operation,
                metadata=metadata,
            )
        )

        return op_id

    def end_operation(
        self, op_id: str, component: str, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """End timing an operation.

        Args:
            op_id: Operation ID from start_operation
            component: Component name
            operation: Operation type
            metadata: Additional metadata

        Returns:
            Duration in seconds
        """
        # Find start metric
        start_metrics = [
            m for m in self.metrics if isinstance(m, PerformanceMetric) and m.metadata.get("op_id") == op_id
        ]

        if not start_metrics:
            raise ValueError(f"No start metric found for operation {op_id}")

        start_metric = start_metrics[-1]
        start_time = start_metric.metadata["start_time"]
        duration = time.time() - start_time

        # Add end metric
        self.add_timing(
            name=start_metric.name.replace("_start", ""),
            duration=duration,
            component=component,
            operation=operation,
            metadata={**(metadata or {}), "op_id": op_id},
        )

        return duration


class ResourceCollector(BaseMetricsCollector):
    """Resource usage metrics collection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process = psutil.Process()

    def collect_memory_usage(self, name: str = "memory_usage", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Collect current memory usage."""
        memory = self.process.memory_info()

        self.add_metric(
            ResourceMetric(
                name=name,
                value=memory.rss / 1024 / 1024,  # Convert to MB
                resource_type="memory",
                metadata={
                    "virtual_memory": memory.vms / 1024 / 1024,
                    "percent": self.process.memory_percent(),
                    **(metadata or {}),
                },
            )
        )

    def collect_cpu_usage(
        self, name: str = "cpu_usage", interval: float = 0.1, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Collect CPU usage."""
        cpu_percent = self.process.cpu_percent(interval=interval)

        self.add_metric(
            ResourceMetric(
                name=name,
                value=cpu_percent,
                resource_type="cpu",
                metadata={
                    "cores": psutil.cpu_count(),
                    "system_percent": psutil.cpu_percent(interval=0),
                    **(metadata or {}),
                },
            )
        )

    def start_resource_monitoring(
        self, interval: float = 1.0, collect_memory: bool = True, collect_cpu: bool = True
    ) -> None:
        """Start continuous resource monitoring."""
        raise NotImplementedError("Continuous monitoring not yet implemented")


class EvaluationCollector(BaseMetricsCollector):
    """Evaluation metrics collection."""

    def add_scores(
        self,
        name: str,
        criteria: List[str],
        scores: List[float],
        weights: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add evaluation scores.

        Args:
            name: Metric name
            criteria: Evaluation criteria
            scores: Scores for each criterion
            weights: Optional weights for criteria
            metadata: Additional metadata
        """
        if len(criteria) != len(scores):
            raise ValueError("Number of criteria must match number of scores")

        if weights and set(weights.keys()) != set(criteria):
            raise ValueError("Weights must match criteria exactly")

        # Calculate weighted average if weights provided
        if weights:
            weighted_scores = [scores[i] * weights[criteria[i]] for i in range(len(scores))]
            value = sum(weighted_scores) / sum(weights.values())
        else:
            value = sum(scores) / len(scores)

        self.add_metric(
            EvaluationMetric(
                name=name, value=value, criteria=criteria, scores=scores, weight=weights, metadata=metadata or {}
            )
        )

    def get_criteria_statistics(self, criterion: str, **filters) -> Dict[str, float]:
        """Get statistics for a specific criterion.

        Args:
            criterion: Criterion name
            **filters: Additional filters

        Returns:
            Statistics for the criterion
        """
        metrics = self.get_metrics(EvaluationMetric, **filters)

        # Extract scores for the criterion
        criterion_scores = []
        for metric in metrics:
            if criterion in metric.criteria:
                idx = metric.criteria.index(criterion)
                criterion_scores.append(metric.scores[idx])

        if not criterion_scores:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}

        return {
            "count": len(criterion_scores),
            "mean": float(np.mean(criterion_scores)),
            "std": float(np.std(criterion_scores)),
            "min": float(np.min(criterion_scores)),
            "max": float(np.max(criterion_scores)),
        }


class QualityCollector(BaseMetricsCollector):
    """Quality metrics collection."""

    def add_quality_score(
        self,
        name: str,
        category: str,
        aspects: Dict[str, float],
        requirements: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add quality score.

        Args:
            name: Metric name
            category: Quality category
            aspects: Scores for different aspects
            requirements: Optional requirements met
            metadata: Additional metadata
        """
        # Validate scores
        if not all(0 <= score <= 1 for score in aspects.values()):
            raise ValueError("All aspect scores must be between 0 and 1")

        # Calculate overall score
        value = sum(aspects.values()) / len(aspects)

        self.add_metric(
            QualityMetric(
                name=name,
                value=value,
                category=category,
                aspects=aspects,
                requirements=requirements,
                metadata=metadata or {},
            )
        )

    def get_aspect_trends(self, category: str, aspect: str, window: int = 10) -> Dict[str, List[float]]:
        """Get trends for a quality aspect.

        Args:
            category: Quality category
            aspect: Aspect name
            window: Window size for moving average

        Returns:
            Trend data
        """
        metrics = self.get_metrics(QualityMetric, category=category)

        # Extract aspect scores
        scores = []
        timestamps = []
        for metric in metrics:
            if aspect in metric.aspects:
                scores.append(metric.aspects[aspect])
                timestamps.append(metric.timestamp)

        if not scores:
            return {"scores": [], "timestamps": [], "moving_avg": []}

        # Calculate moving average
        moving_avg = []
        for i in range(len(scores)):
            start = max(0, i - window + 1)
            moving_avg.append(sum(scores[start : i + 1]) / (i - start + 1))

        return {"scores": scores, "timestamps": timestamps, "moving_avg": moving_avg}
