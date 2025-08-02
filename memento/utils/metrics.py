"""
Performance metrics collection for the Memento framework.

This module provides utilities for tracking, collecting, and analyzing
performance metrics across the framework components.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.logger import LoggerMixin


@dataclass
class PerformanceMetric:
    """Represents a single performance metric."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]
    component: str
    operation: str


@dataclass
class EvaluationMetric:
    """Represents evaluation performance metrics."""

    prompt_length: int
    problem_complexity: float
    evaluation_time: float
    criteria_count: int
    average_score: float
    max_score: float
    min_score: float
    score_variance: float
    timestamp: datetime


class MetricsCollector(LoggerMixin):
    """Collects and manages performance metrics."""

    def __init__(self, storage_path: Union[str, Path]):
        super().__init__()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metrics: List[PerformanceMetric] = []
        self.evaluation_metrics: List[EvaluationMetric] = []

    def add_metric(
        self,
        name: str,
        value: float,
        unit: str,
        component: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a performance metric.

        Args:
            name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            component: Component name
            operation: Operation name
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            metadata=metadata or {},
            component=component,
            operation=operation,
        )
        self.metrics.append(metric)
        self.logger.debug(f"Added metric: {name}={value}{unit} for {component}.{operation}")

    def add_evaluation_metric(
        self,
        prompt_length: int,
        problem_complexity: float,
        evaluation_time: float,
        criteria_count: int,
        scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add evaluation-specific metrics.

        Args:
            prompt_length: Length of the prompt
            problem_complexity: Complexity score of the problem
            evaluation_time: Time taken for evaluation
            criteria_count: Number of evaluation criteria
            scores: List of evaluation scores
            metadata: Additional metadata
        """
        if not scores:
            raise ValueError("Scores list cannot be empty")

        metric = EvaluationMetric(
            prompt_length=prompt_length,
            problem_complexity=problem_complexity,
            evaluation_time=evaluation_time,
            criteria_count=criteria_count,
            average_score=sum(scores) / len(scores),
            max_score=max(scores),
            min_score=min(scores),
            score_variance=self._calculate_variance(scores),
            timestamp=datetime.now(),
        )
        self.evaluation_metrics.append(metric)
        self.logger.debug(f"Added evaluation metric: avg_score={metric.average_score:.3f}")

    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores."""
        if len(scores) < 2:
            return 0.0

        mean = sum(scores) / len(scores)
        squared_diff_sum = sum((score - mean) ** 2 for score in scores)
        return squared_diff_sum / (len(scores) - 1)

    def get_metrics(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[PerformanceMetric]:
        """
        Get metrics with optional filtering.

        Args:
            component: Filter by component
            operation: Filter by operation
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Filtered list of metrics
        """
        filtered = self.metrics

        if component:
            filtered = [m for m in filtered if m.component == component]

        if operation:
            filtered = [m for m in filtered if m.operation == operation]

        if start_time:
            filtered = [m for m in filtered if m.timestamp >= start_time]

        if end_time:
            filtered = [m for m in filtered if m.timestamp <= end_time]

        return filtered

    def get_evaluation_metrics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[EvaluationMetric]:
        """
        Get evaluation metrics with optional time filtering.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Filtered list of evaluation metrics
        """
        filtered = self.evaluation_metrics

        if start_time:
            filtered = [m for m in filtered if m.timestamp >= start_time]

        if end_time:
            filtered = [m for m in filtered if m.timestamp <= end_time]

        return filtered

    def get_summary_stats(
        self, component: Optional[str] = None, operation: Optional[str] = None, metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for metrics.

        Args:
            component: Filter by component
            operation: Filter by operation
            metric_name: Filter by metric name

        Returns:
            Dictionary with summary statistics
        """
        metrics = self.get_metrics(component, operation)

        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]

        if not metrics:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "std": self._calculate_std(values),
        }

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        squared_diff_sum = sum((value - mean) ** 2 for value in values)
        variance = squared_diff_sum / (len(values) - 1)
        return variance**0.5

    def save_metrics(self, filename: Optional[str] = None) -> None:
        """
        Save metrics to file.

        Args:
            filename: Optional custom filename
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        file_path = self.storage_path / filename

        data = {
            "metrics": [asdict(m) for m in self.metrics],
            "evaluation_metrics": [asdict(m) for m in self.evaluation_metrics],
            "exported_at": datetime.now().isoformat(),
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Saved {len(self.metrics)} metrics to {file_path}")

    def load_metrics(self, filename: str) -> None:
        """
        Load metrics from file.

        Args:
            filename: Name of the file to load
        """
        file_path = self.storage_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert back to metric objects
        self.metrics = [PerformanceMetric(**m) for m in data.get("metrics", [])]
        self.evaluation_metrics = [EvaluationMetric(**m) for m in data.get("evaluation_metrics", [])]

        self.logger.info(f"Loaded {len(self.metrics)} metrics from {file_path}")

    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()
        self.evaluation_metrics.clear()
        self.logger.info("Cleared all metrics")


@contextmanager
def timing_context(
    collector: MetricsCollector, operation: str, component: str, metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for timing operations.

    Args:
        collector: Metrics collector instance
        operation: Operation name
        component: Component name
        metadata: Additional metadata
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        collector.add_metric(
            name="duration", value=duration, unit="seconds", component=component, operation=operation, metadata=metadata
        )


class PerformanceMonitor:
    """Monitors and tracks performance across the framework."""

    def __init__(self, storage_path: Union[str, Path]):
        self.collector = MetricsCollector(storage_path)
        self.active_timers: Dict[str, float] = {}

    def start_timer(self, timer_id: str) -> None:
        """Start a timer."""
        self.active_timers[timer_id] = time.time()

    def stop_timer(
        self, timer_id: str, component: str, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Stop a timer and record the metric.

        Args:
            timer_id: Timer identifier
            component: Component name
            operation: Operation name
            metadata: Additional metadata

        Returns:
            Duration in seconds
        """
        if timer_id not in self.active_timers:
            raise ValueError(f"Timer '{timer_id}' not found")

        start_time = self.active_timers.pop(timer_id)
        duration = time.time() - start_time

        self.collector.add_metric(
            name="duration", value=duration, unit="seconds", component=component, operation=operation, metadata=metadata
        )

        return duration

    def record_evaluation(
        self,
        prompt_length: int,
        problem_complexity: float,
        evaluation_time: float,
        criteria_count: int,
        scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record evaluation metrics."""
        self.collector.add_evaluation_metric(
            prompt_length=prompt_length,
            problem_complexity=problem_complexity,
            evaluation_time=evaluation_time,
            criteria_count=criteria_count,
            scores=scores,
            metadata=metadata,
        )

    def get_performance_report(
        self, component: Optional[str] = None, time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Generate a performance report.

        Args:
            component: Filter by component
            time_window: Time window for filtering

        Returns:
            Performance report dictionary
        """
        end_time = datetime.now()
        start_time = end_time - time_window if time_window else None

        metrics = self.collector.get_metrics(component=component, start_time=start_time, end_time=end_time)

        if not metrics:
            return {"message": "No metrics found for the specified criteria"}

        # Group by operation
        operations = {}
        for metric in metrics:
            op = metric.operation
            if op not in operations:
                operations[op] = []
            operations[op].append(metric)

        # Calculate statistics for each operation
        report = {
            "generated_at": datetime.now().isoformat(),
            "time_window": {"start": start_time.isoformat() if start_time else None, "end": end_time.isoformat()},
            "component": component,
            "total_metrics": len(metrics),
            "operations": {},
        }

        for op, op_metrics in operations.items():
            durations = [m.value for m in op_metrics if m.name == "duration"]
            if durations:
                report["operations"][op] = {
                    "count": len(durations),
                    "total_time": sum(durations),
                    "average_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations),
                }

        return report
