"""Comprehensive benchmark runner with dashboard and resource monitoring."""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..config.models import ModelConfig
from ..utils.metrics import MetricsCollector
from ..visualization.comparison import ComparisonVisualizer
from ..visualization.dashboard import DashboardServer
from ..visualization.results import ResultsVisualizer
from ..visualization.statistical import StatisticalVisualizer
# Import the base ComprehensiveBenchmark from main
from .main import ComprehensiveBenchmark as BaseComprehensiveBenchmark

logger = logging.getLogger(__name__)


class EnhancedComprehensiveBenchmark:
    """Enhanced comprehensive benchmark runner with dashboard and resource monitoring."""

    def __init__(
        self,
        model_config: ModelConfig,
        output_dir: Union[str, Path],
        enable_dashboard: bool = True,
        enable_resource_monitoring: bool = True,
        metrics: Optional[List[str]] = None,
    ):
        """Initialize enhanced benchmark runner.

        Args:
            model_config: Model configuration
            output_dir: Directory for outputs
            enable_dashboard: Whether to enable real-time dashboard
            enable_resource_monitoring: Whether to monitor system resources
            metrics: Optional list of metrics to track
        """
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_dashboard = enable_dashboard
        self.enable_resource_monitoring = enable_resource_monitoring

        # Initialize base benchmark
        self.base_benchmark = BaseComprehensiveBenchmark(
            output_dir=self.output_dir,
            metrics=metrics or ["accuracy", "latency", "memory_usage", "quality_score"],
            monitor_resources=enable_resource_monitoring,
        )

        # Initialize dashboard if enabled
        self.dashboard = None
        if enable_dashboard:
            self.dashboard = DashboardServer(
                host="localhost", port=8050, update_interval=1.0
            )

        # Performance tracking
        self.start_time = None
        self.performance_metrics = {}

    async def run_benchmark(
        self,
        datasets: List[str],
        models: List[str],
        baseline: Optional[str] = None,
        max_samples_per_dataset: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark with enhanced features.

        Args:
            datasets: List of datasets to evaluate
            models: List of models to evaluate
            baseline: Optional baseline model for comparison
            max_samples_per_dataset: Maximum samples per dataset
            **kwargs: Additional parameters

        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting enhanced comprehensive benchmark")
        self.start_time = time.time()

        # Start dashboard if enabled
        dashboard_task = None
        if self.dashboard:
            dashboard_task = asyncio.create_task(self.dashboard.start())

        try:
            # Run base benchmark with dashboard updates
            results = await self._run_benchmark_with_dashboard_updates(
                datasets=datasets,
                models=models,
                baseline=baseline,
                max_samples_per_dataset=max_samples_per_dataset,
                **kwargs,
            )

            # Add enhanced metrics
            enhanced_results = await self._add_enhanced_metrics(results)

            # Add performance analysis
            enhanced_results["performance_analysis"] = self._analyze_performance()

            # Add resource analysis if monitoring enabled
            if self.enable_resource_monitoring:
                enhanced_results["resource_analysis"] = self._analyze_resources()

            # Save enhanced results
            self._save_enhanced_results(enhanced_results)

            logger.info("Enhanced benchmark completed successfully")
            return enhanced_results

        finally:
            # Stop dashboard if running
            if dashboard_task:
                dashboard_task.cancel()
                try:
                    await dashboard_task
                except asyncio.CancelledError:
                    pass

    async def _add_enhanced_metrics(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Add enhanced metrics to base results."""
        enhanced_results = base_results.copy()

        # Add throughput metrics
        total_samples = 0
        for dataset_name, dataset_results in base_results.get("dataset_results", {}).items():
            for model_name, model_results in dataset_results.items():
                if isinstance(model_results, dict):
                    responses = model_results.get("responses", [])
                    if isinstance(responses, list):
                        total_samples += len(responses)

        total_time = time.time() - self.start_time
        enhanced_results["throughput"] = {
            "samples_per_second": total_samples / total_time if total_time > 0 else 0,
            "total_samples": total_samples,
            "total_time": total_time,
        }

        # Add model efficiency metrics
        model_efficiency = {}
        for model_name, results in base_results.get("model_results", {}).items():
            if isinstance(results, dict):
                # Calculate efficiency based on accuracy vs latency
                accuracy_data = results.get("accuracy", {})
                latency_data = results.get("latency", {})
                
                if isinstance(accuracy_data, dict):
                    accuracy = accuracy_data.get("mean", 0)
                else:
                    accuracy = accuracy_data if isinstance(accuracy_data, (int, float)) else 0
                
                if isinstance(latency_data, dict):
                    latency = latency_data.get("mean", 0)
                else:
                    latency = latency_data if isinstance(latency_data, (int, float)) else 0
                
                efficiency = accuracy / latency if latency > 0 else 0
                model_efficiency[model_name] = {
                    "efficiency_score": efficiency,
                    "accuracy_latency_ratio": efficiency,
                }

        enhanced_results["model_efficiency"] = model_efficiency

        return enhanced_results

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze benchmark performance."""
        if not self.start_time:
            return {}

        total_time = time.time() - self.start_time
        
        return {
            "total_execution_time": total_time,
            "samples_per_second": self.performance_metrics.get("total_samples", 0) / total_time if total_time > 0 else 0,
            "average_latency": self.performance_metrics.get("average_latency", 0),
            "peak_throughput": self.performance_metrics.get("peak_throughput", 0),
        }

    def _analyze_resources(self) -> Dict[str, Any]:
        """Analyze resource usage."""
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            return {
                "peak_memory_mb": memory_info.rss / 1024 / 1024,
                "avg_cpu_percent": cpu_percent,
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "cpu_count": psutil.cpu_count(),
                "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024,
            }
        except ImportError:
            logger.warning("psutil not available, skipping resource analysis")
            return {}

    def _save_enhanced_results(self, results: Dict[str, Any]) -> None:
        """Save enhanced results."""
        import json
        
        # Save comprehensive results
        results_path = self.output_dir / "comprehensive_benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save performance summary
        performance_path = self.output_dir / "performance_summary.json"
        performance_summary = {
            "metadata": results.get("metadata", {}),
            "performance_analysis": results.get("performance_analysis", {}),
            "resource_analysis": results.get("resource_analysis", {}),
            "throughput": results.get("throughput", {}),
        }
        
        with open(performance_path, "w") as f:
            json.dump(performance_summary, f, indent=2, default=str)

        logger.info(f"Enhanced results saved to {self.output_dir}")

    async def _run_benchmark_with_dashboard_updates(
        self,
        datasets: List[str],
        models: List[str],
        baseline: Optional[str] = None,
        max_samples_per_dataset: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run benchmark with real-time dashboard updates."""
        
        # Initialize progress tracking
        total_datasets = len(datasets)
        total_models = len(models)
        completed_datasets = 0
        completed_models = 0
        
        # Update dashboard with initial state
        if self.dashboard:
            self.dashboard.update_metric("total_datasets", total_datasets)
            self.dashboard.update_metric("total_models", total_models)
            self.dashboard.update_metric("completed_datasets", 0)
            self.dashboard.update_metric("completed_models", 0)
            self.dashboard.update_metric("progress_percentage", 0.0)
        
        # Run the base benchmark
        results = await self.base_benchmark.run_benchmark(
            datasets=datasets,
            models=models,
            baseline=baseline,
            max_samples_per_dataset=max_samples_per_dataset,
            **kwargs,
        )
        
        # Update dashboard with results as they come in
        if self.dashboard and "dataset_results" in results:
            for dataset_name, dataset_results in results["dataset_results"].items():
                completed_datasets += 1
                
                if self.dashboard:
                    self.dashboard.update_metric("completed_datasets", completed_datasets)
                    self.dashboard.update_metric("progress_percentage", (completed_datasets / total_datasets) * 100)
                
                for model_name, model_results in dataset_results.items():
                    if isinstance(model_results, dict):
                        # Update model-specific metrics
                        for metric in ["accuracy", "latency", "quality_score"]:
                            if metric in model_results:
                                value = model_results[metric]
                                if isinstance(value, dict) and "mean" in value:
                                    value = value["mean"]
                                
                                if self.dashboard:
                                    self.dashboard.update_metric(f"{model_name}_{metric}", value)
                                    self.dashboard.update_comparison(metric, model_name, value)
                        
                        # Update resource metrics if available
                        if "resources" in model_results:
                            resources = model_results["resources"]
                            if isinstance(resources, dict):
                                if "peak_memory_mb" in resources and self.dashboard:
                                    self.dashboard.update_resource("memory_mb", resources["peak_memory_mb"])
                                if "avg_cpu_percent" in resources and self.dashboard:
                                    self.dashboard.update_resource("cpu_percent", resources["avg_cpu_percent"])
        
        return results


# Alias for backward compatibility
ComprehensiveBenchmark = EnhancedComprehensiveBenchmark
