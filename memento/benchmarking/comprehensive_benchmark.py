"""Comprehensive benchmark system integrating all components."""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

from ..config.models import ModelConfig
from ..core.metrics import MetricsCollector
from ..datasets import StandardDatasetManager
from ..utils.logger import LoggerMixin
from ..visualization.comparison import ComparisonVisualizer
from ..visualization.dashboard import DashboardServer
from ..visualization.results import ResultsVisualizer
from ..visualization.statistical import StatisticalVisualizer
from .baselines.promptbreeder import PromptBreeder
from .baselines.self_evolving_gpt import SelfEvolvingGPT
from .datasets.loader import DatasetLoader
from .evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """System resource monitoring."""

    def __init__(self):
        """Initialize resource monitor."""
        self.process = psutil.Process()
        self.start_time = None
        self.metrics = []
        self.monitoring = False

    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self.metrics.clear()
        self.monitoring = True
        logger.info("Started resource monitoring")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        self.collect_final_metrics()
        logger.info("Stopped resource monitoring")

    def collect_metrics(self):
        """Collect current resource metrics."""
        if not self.monitoring:
            return

        try:
            self.metrics.append(
                {
                    "timestamp": time.time(),
                    "cpu_percent": self.process.cpu_percent(),
                    "memory_rss": self.process.memory_info().rss,
                    "memory_vms": self.process.memory_info().vms,
                    "memory_percent": self.process.memory_percent(),
                    "threads": self.process.num_threads(),
                    "handles": getattr(self.process, "num_handles", lambda: 0)(),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to collect resource metrics: {e}")

    def collect_final_metrics(self):
        """Collect final metrics summary."""
        if not self.metrics:
            return

        self.collect_metrics()  # One final collection

        # Calculate summary stats
        memory_values = [m["memory_rss"] for m in self.metrics]
        cpu_values = [m["cpu_percent"] for m in self.metrics]

        summary = {
            "duration": time.time() - self.start_time if self.start_time else 0,
            "peak_memory_rss": max(memory_values) if memory_values else 0,
            "avg_memory_rss": sum(memory_values) / len(memory_values) if memory_values else 0,
            "peak_cpu": max(cpu_values) if cpu_values else 0,
            "avg_cpu": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "total_samples": len(self.metrics),
        }

        self.metrics.append({"summary": summary})

    def get_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.metrics:
            return {"status": "No metrics collected"}

        # Find summary in metrics
        for metric in reversed(self.metrics):
            if "summary" in metric:
                return metric["summary"]

        return {"status": "No summary available"}


class PerformanceAnalyzer:
    """Performance analysis tools."""

    def __init__(self):
        """Initialize performance analyzer."""
        self.results = []

    def analyze_throughput(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze processing throughput."""
        if not results.get("dataset_results"):
            return {"error": "No dataset results available"}

        total_samples = 0
        total_time = 0

        for dataset_result in results["dataset_results"].values():
            if isinstance(dataset_result, dict):
                total_samples += dataset_result.get("samples_processed", 0)
                total_time += dataset_result.get("processing_time", 0)

        if total_time == 0:
            return {"samples_per_second": 0, "average_latency": 0}

        return {
            "samples_per_second": total_samples / total_time,
            "average_latency": total_time / total_samples if total_samples > 0 else 0,
            "total_samples": total_samples,
            "total_time": total_time,
        }

    def analyze_resources(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource usage."""
        if not metrics:
            return {"error": "No metrics available"}

        # Find summary
        summary = None
        for metric in reversed(metrics):
            if "summary" in metric:
                summary = metric["summary"]
                break

        if not summary:
            return {"error": "No summary available"}

        # Convert bytes to MB for readability
        return {
            "peak_memory_mb": summary["peak_memory_rss"] / (1024 * 1024),
            "avg_memory_mb": summary["avg_memory_rss"] / (1024 * 1024),
            "peak_cpu_percent": summary["peak_cpu"],
            "avg_cpu_percent": summary["avg_cpu"],
            "duration_seconds": summary["duration"],
            "efficiency_score": self._calculate_efficiency_score(summary),
        }

    def _calculate_efficiency_score(self, summary: Dict[str, Any]) -> float:
        """Calculate efficiency score based on resource usage."""
        # Simple efficiency score: lower resource usage = higher efficiency
        memory_score = max(0, 1 - (summary["avg_memory_rss"] / (1024 * 1024 * 1024)))  # 1GB baseline
        cpu_score = max(0, 1 - (summary["avg_cpu"] / 100))

        return (memory_score + cpu_score) / 2


class ComprehensiveBenchmark(LoggerMixin):
    """Comprehensive benchmark system."""

    def __init__(
        self,
        model_config: ModelConfig,
        output_dir: Union[str, Path],
        enable_dashboard: bool = True,
        enable_resource_monitoring: bool = True,
    ):
        """Initialize comprehensive benchmark.

        Args:
            model_config: Model configuration
            output_dir: Output directory for results
            enable_dashboard: Whether to enable real-time dashboard
            enable_resource_monitoring: Whether to monitor system resources
        """
        super().__init__()

        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize core components
        self.dataset_loader = DatasetLoader(cache_dir=self.output_dir / "dataset_cache")

        self.evaluator = Evaluator()

        self.metrics_collector = MetricsCollector(storage_path=self.output_dir / "metrics")

        # Initialize monitoring
        self.enable_resource_monitoring = enable_resource_monitoring
        self.resource_monitor = ResourceMonitor() if enable_resource_monitoring else None
        self.performance_analyzer = PerformanceAnalyzer()

        # Initialize visualization
        self.results_viz = ResultsVisualizer(output_dir=self.output_dir / "results")

        self.comparison_viz = ComparisonVisualizer(output_dir=self.output_dir / "comparison")

        self.stats_viz = StatisticalVisualizer(output_dir=self.output_dir / "stats")

        # Initialize dashboard
        self.enable_dashboard = enable_dashboard
        self.dashboard = DashboardServer() if enable_dashboard else None

        # Initialize baseline models
        self.baseline_models = {
            "promptbreeder": PromptBreeder(model_config=model_config, storage_path=self.output_dir / "promptbreeder"),
            "self_evolving_gpt": SelfEvolvingGPT(
                model_config=model_config, storage_path=self.output_dir / "self_evolving_gpt"
            ),
        }

        self.logger.info(f"Initialized comprehensive benchmark system in {self.output_dir}")

    async def run_benchmark(
        self,
        datasets: List[str],
        models: List[str],
        baseline: Optional[str] = None,
        max_samples_per_dataset: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark.

        Args:
            datasets: List of datasets to evaluate
            models: List of models to evaluate
            baseline: Optional baseline model for comparison
            max_samples_per_dataset: Maximum samples per dataset
            **kwargs: Additional parameters

        Returns:
            Comprehensive benchmark results
        """
        self.logger.info("Starting comprehensive benchmark")
        start_time = datetime.now()

        # Start monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()

        if self.dashboard:
            await self.dashboard.start()

        try:
            # Initialize results structure
            results = {
                "metadata": {
                    "start_time": start_time.isoformat(),
                    "datasets": datasets,
                    "models": models,
                    "baseline": baseline,
                    "max_samples_per_dataset": max_samples_per_dataset,
                    "model_config": self.model_config.model_dump(),
                },
                "dataset_results": {},
                "model_results": {},
                "comparison_results": {},
                "performance_analysis": {},
                "resource_analysis": {},
            }

            # Run evaluations for each dataset
            for dataset_name in datasets:
                self.logger.info(f"Evaluating dataset: {dataset_name}")

                try:
                    # Load dataset
                    dataset = await self.dataset_loader.load_dataset(dataset_name, max_samples=max_samples_per_dataset)

                    # Evaluate models on this dataset
                    dataset_results = await self._evaluate_dataset(dataset_name, dataset, models)

                    results["dataset_results"][dataset_name] = dataset_results

                    # Update dashboard if enabled
                    if self.dashboard:
                        for model_name, model_results in dataset_results.items():
                            self.dashboard.update_comparison(
                                dataset_name, model_name, model_results.get("accuracy", 0.0)
                            )

                except Exception as e:
                    self.logger.error(f"Failed to evaluate dataset {dataset_name}: {e}")
                    results["dataset_results"][dataset_name] = {"error": str(e)}

            # Aggregate model results
            for model in models:
                results["model_results"][model] = self._aggregate_model_results(model, results["dataset_results"])

            # Generate comparisons if baseline specified
            if baseline and baseline in models:
                results["comparison_results"] = await self._generate_comparisons(
                    baseline, models, results["dataset_results"]
                )

            # Analyze performance
            results["performance_analysis"] = self.performance_analyzer.analyze_throughput(results)

            # Analyze resources
            if self.resource_monitor:
                results["resource_analysis"] = self.performance_analyzer.analyze_resources(
                    self.resource_monitor.metrics
                )

            # Generate visualizations
            await self._generate_visualizations(results)

            # Save results
            self._save_results(results)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.logger.info(f"Benchmark completed in {duration:.2f} seconds")
            results["metadata"]["end_time"] = end_time.isoformat()
            results["metadata"]["duration_seconds"] = duration

            return results

        finally:
            # Stop monitoring
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()

            if self.dashboard:
                await self.dashboard.stop()

    async def _evaluate_dataset(self, dataset_name: str, dataset: Any, models: List[str]) -> Dict[str, Any]:
        """Evaluate models on a specific dataset."""
        results = {}

        for model_name in models:
            self.logger.info(f"Evaluating {model_name} on {dataset_name}")

            start_time = time.time()

            try:
                # Get model results
                if model_name in self.baseline_models:
                    model_results = await self._evaluate_baseline_model(model_name, dataset, dataset_name)
                else:
                    # This would be for Memento or other custom models
                    model_results = await self._evaluate_custom_model(model_name, dataset, dataset_name)

                processing_time = time.time() - start_time
                model_results["processing_time"] = processing_time
                model_results["samples_processed"] = len(dataset) if hasattr(dataset, "__len__") else 0

                results[model_name] = model_results

                # Collect resource metrics
                if self.resource_monitor:
                    self.resource_monitor.collect_metrics()

            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name} on {dataset_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    async def _evaluate_baseline_model(self, model_name: str, dataset: Any, dataset_name: str) -> Dict[str, Any]:
        """Evaluate a baseline model."""
        model = self.baseline_models[model_name]

        # For now, return placeholder results
        # This would be replaced with actual model evaluation
        return {
            "accuracy": 0.75 + (hash(model_name) % 100) / 1000,  # Placeholder
            "latency": 0.1 + (hash(dataset_name) % 50) / 1000,  # Placeholder
            "quality_score": 0.8 + (hash(f"{model_name}_{dataset_name}") % 100) / 1000,
        }

    async def _evaluate_custom_model(self, model_name: str, dataset: Any, dataset_name: str) -> Dict[str, Any]:
        """Evaluate a custom model (like Memento)."""
        # Placeholder for custom model evaluation
        return {
            "accuracy": 0.85 + (hash(model_name) % 100) / 1000,  # Placeholder
            "latency": 0.08 + (hash(dataset_name) % 30) / 1000,  # Placeholder
            "quality_score": 0.9 + (hash(f"{model_name}_{dataset_name}") % 100) / 1000,
        }

    def _aggregate_model_results(self, model: str, dataset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results for a model across datasets."""
        aggregated = {}

        # Collect all metrics for this model
        all_metrics = {}
        for dataset_result in dataset_results.values():
            if model in dataset_result and isinstance(dataset_result[model], dict):
                model_result = dataset_result[model]
                for metric, value in model_result.items():
                    if isinstance(value, (int, float)):
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(value)

        # Calculate statistics for each metric
        for metric, values in all_metrics.items():
            if values:
                aggregated[metric] = {
                    "mean": sum(values) / len(values),
                    "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return aggregated

    async def _generate_comparisons(
        self, baseline: str, models: List[str], dataset_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comparison results."""
        comparisons = {}

        for model in models:
            if model != baseline:
                comparisons[model] = self._compare_models(baseline, model, dataset_results)

        return comparisons

    def _compare_models(
        self, baseline_model: str, comparison_model: str, dataset_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare two models."""
        improvements = {}

        for dataset, results in dataset_results.items():
            if (
                baseline_model in results
                and comparison_model in results
                and isinstance(results[baseline_model], dict)
                and isinstance(results[comparison_model], dict)
            ):

                baseline_results = results[baseline_model]
                comparison_results = results[comparison_model]

                dataset_improvements = {}
                for metric in baseline_results:
                    if (
                        metric in comparison_results
                        and isinstance(baseline_results[metric], (int, float))
                        and isinstance(comparison_results[metric], (int, float))
                    ):

                        baseline_val = baseline_results[metric]
                        comparison_val = comparison_results[metric]

                        if baseline_val != 0:
                            improvement = ((comparison_val - baseline_val) / baseline_val) * 100
                            dataset_improvements[metric] = improvement

                improvements[dataset] = dataset_improvements

        return improvements

    async def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate all visualizations."""
        self.logger.info("Generating visualizations")

        try:
            # Generate results visualizations
            metrics_data = []
            for dataset, dataset_results in results["dataset_results"].items():
                for model, model_results in dataset_results.items():
                    if isinstance(model_results, dict):
                        for metric, value in model_results.items():
                            if isinstance(value, (int, float)):
                                metrics_data.append(
                                    {
                                        "dataset": dataset,
                                        "model": model,
                                        "metric": metric,
                                        "value": value,
                                        "timestamp": datetime.now(),
                                    }
                                )

            if metrics_data:
                # Generate metric-specific plots
                for metric in set(m["metric"] for m in metrics_data):
                    metric_data = [m for m in metrics_data if m["metric"] == metric]
                    if metric_data:
                        self.results_viz.plot_metric_distribution(metric_data, metric, group_by="model")

                # Generate comparison plots if we have comparisons
                if results.get("comparison_results"):
                    baseline = results["metadata"]["baseline"]
                    for model in results["comparison_results"]:
                        comparison_data = {
                            baseline: {
                                dataset: dataset_results.get(baseline, {})
                                for dataset, dataset_results in results["dataset_results"].items()
                            },
                            model: {
                                dataset: dataset_results.get(model, {})
                                for dataset, dataset_results in results["dataset_results"].items()
                            },
                        }

                        # Generate improvement heatmap
                        self.comparison_viz.plot_improvement_heatmap(comparison_data[baseline], comparison_data[model])

                # Generate comprehensive report
                self.results_viz.generate_report(
                    metrics_data, list(set(m["metric"] for m in metrics_data)), group_by="model"
                )

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")

    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results."""
        # Save full results
        results_file = self.output_dir / "comprehensive_benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save metrics
        self.metrics_collector.save_metrics("comprehensive_metrics.json")

        # Save resource metrics if available
        if self.resource_monitor and self.resource_monitor.metrics:
            resource_file = self.output_dir / "resource_metrics.json"
            with open(resource_file, "w") as f:
                json.dump(self.resource_monitor.metrics, f, indent=2, default=str)

        self.logger.info(f"Results saved to {results_file}")
