"""Main benchmark runner."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.metrics import MetricsCollector
from ..visualization.comparison import ComparisonVisualizer
from ..visualization.results import ResultsVisualizer
from ..visualization.statistical import StatisticalVisualizer

logger = logging.getLogger(__name__)


class ComprehensiveBenchmark:
    """Main benchmark runner."""

    def __init__(
        self, output_dir: Union[str, Path], metrics: Optional[List[str]] = None, monitor_resources: bool = True
    ):
        """Initialize benchmark runner.

        Args:
            output_dir: Directory for outputs
            metrics: Optional list of metrics to track
            monitor_resources: Whether to monitor system resources
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize collectors and visualizers
        self.metrics_collector = MetricsCollector(storage_path=self.output_dir / "metrics")

        self.results_viz = ResultsVisualizer(output_dir=self.output_dir / "results")

        self.comparison_viz = ComparisonVisualizer(output_dir=self.output_dir / "comparison")

        self.stats_viz = StatisticalVisualizer(output_dir=self.output_dir / "stats")

        # Configure metrics
        self.metrics = metrics or ["accuracy", "latency", "memory_usage", "quality_score"]

        self.monitor_resources = monitor_resources

    async def run_benchmark(
        self, datasets: List[str], models: List[str], baseline: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark.

        Args:
            datasets: List of datasets to evaluate
            models: List of models to evaluate
            baseline: Optional baseline model for comparison
            **kwargs: Additional parameters

        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting comprehensive benchmark")
        start_time = datetime.now()

        # Initialize results
        results = {
            "metadata": {
                "start_time": start_time.isoformat(),
                "datasets": datasets,
                "models": models,
                "baseline": baseline,
                "metrics": self.metrics,
            },
            "dataset_results": {},
            "model_results": {},
            "comparison_results": {},
            "statistical_results": {},
        }

        try:
            # Run evaluations
            for dataset in datasets:
                logger.info(f"Evaluating dataset: {dataset}")
                dataset_results = await self._evaluate_dataset(dataset, models, **kwargs)
                results["dataset_results"][dataset] = dataset_results

            # Generate model-specific results
            for model in models:
                model_results = self._aggregate_model_results(model, results["dataset_results"])
                results["model_results"][model] = model_results

            # Generate comparisons if baseline specified
            if baseline and baseline in models:
                logger.info("Generating comparisons")
                comparison_results = await self._generate_comparisons(baseline, models, results["dataset_results"])
                results["comparison_results"] = comparison_results

            # Generate visualizations
            logger.info("Generating visualizations")
            await self._generate_visualizations(results)

            # Save results
            self._save_results(results)

            logger.info("Benchmark completed successfully")
            return results

        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            raise

    async def _evaluate_dataset(self, dataset: str, models: List[str], **kwargs) -> Dict[str, Any]:
        """Evaluate models on a dataset.

        Args:
            dataset: Dataset name
            models: List of models
            **kwargs: Additional parameters

        Returns:
            Dictionary with evaluation results
        """
        results = {}

        for model in models:
            logger.info(f"Evaluating {model} on {dataset}")

            # Track resources if enabled
            if self.monitor_resources:
                self.metrics_collector.start_resource_monitoring()

            # Run evaluation
            model_results = await self._evaluate_model(model, dataset, **kwargs)

            # Stop resource monitoring
            if self.monitor_resources:
                resource_metrics = self.metrics_collector.stop_resource_monitoring()
                model_results["resources"] = resource_metrics

            results[model] = model_results

        return results

    async def _evaluate_model(self, model: str, dataset: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a model on a dataset.

        Args:
            model: Model name
            dataset: Dataset name
            **kwargs: Additional parameters

        Returns:
            Dictionary with evaluation results
        """
        # TODO: Implement actual model evaluation
        # This is a placeholder that returns random metrics
        import numpy as np

        return {metric: float(np.random.normal(0.8, 0.1)) for metric in self.metrics}

    def _aggregate_model_results(self, model: str, dataset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results for a model across datasets.

        Args:
            model: Model name
            dataset_results: Results for each dataset

        Returns:
            Dictionary with aggregated results
        """
        aggregated = {metric: [] for metric in self.metrics}

        # Collect all values for each metric
        for dataset_result in dataset_results.values():
            model_result = dataset_result.get(model, {})
            for metric in self.metrics:
                if metric in model_result:
                    aggregated[metric].append(model_result[metric])

        # Calculate statistics
        statistics = {}
        for metric, values in aggregated.items():
            if values:
                statistics[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }

        return statistics

    async def _generate_comparisons(
        self, baseline: str, models: List[str], dataset_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comparison results.

        Args:
            baseline: Baseline model
            models: List of models
            dataset_results: Results for each dataset

        Returns:
            Dictionary with comparison results
        """
        comparisons = {}

        for model in models:
            if model != baseline:
                # Collect results for comparison
                baseline_results = {}
                model_results = {}

                for dataset, results in dataset_results.items():
                    if baseline in results and model in results:
                        baseline_results[dataset] = results[baseline]
                        model_results[dataset] = results[model]

                # Generate comparison metrics
                comparisons[model] = {
                    "effect_sizes": self._calculate_effect_sizes(baseline_results, model_results),
                    "improvements": self._calculate_improvements(baseline_results, model_results),
                }

        return comparisons

    def _calculate_effect_sizes(
        self, baseline_results: Dict[str, Dict[str, float]], model_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate effect sizes between baseline and model.

        Args:
            baseline_results: Baseline model results
            model_results: Comparison model results

        Returns:
            Dictionary mapping metrics to effect sizes
        """
        effect_sizes = {}

        for metric in self.metrics:
            baseline_values = []
            model_values = []

            for dataset in baseline_results:
                if dataset in model_results:
                    if metric in baseline_results[dataset]:
                        baseline_values.append(baseline_results[dataset][metric])
                    if metric in model_results[dataset]:
                        model_values.append(model_results[dataset][metric])

            if baseline_values and model_values:
                # Calculate Cohen's d
                d = (np.mean(model_values) - np.mean(baseline_values)) / np.sqrt(
                    (
                        (len(model_values) - 1) * np.var(model_values)
                        + (len(baseline_values) - 1) * np.var(baseline_values)
                    )
                    / (len(model_values) + len(baseline_values) - 2)
                )
                effect_sizes[metric] = float(d)

        return effect_sizes

    def _calculate_improvements(
        self, baseline_results: Dict[str, Dict[str, float]], model_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate improvements over baseline.

        Args:
            baseline_results: Baseline model results
            model_results: Comparison model results

        Returns:
            Dictionary mapping datasets and metrics to improvements
        """
        improvements = {}

        for dataset in baseline_results:
            if dataset in model_results:
                improvements[dataset] = {}

                for metric in self.metrics:
                    if metric in baseline_results[dataset] and metric in model_results[dataset]:
                        baseline = baseline_results[dataset][metric]
                        model = model_results[dataset][metric]

                        if baseline != 0:
                            improvement = ((model - baseline) / baseline) * 100
                            improvements[dataset][metric] = float(improvement)

        return improvements

    async def _generate_visualizations(self, results: Dict[str, Any]) -> None:
        """Generate all visualizations.

        Args:
            results: Benchmark results
        """
        # Generate results visualizations
        for metric in self.metrics:
            self.results_viz.plot_metric_history(results["dataset_results"], metric)

            self.results_viz.plot_metric_distribution(results["dataset_results"], metric)

        # Generate comparison visualizations
        if results["comparison_results"]:
            for metric in self.metrics:
                self.comparison_viz.plot_method_comparison(
                    results["dataset_results"], metric, baseline=results["metadata"]["baseline"]
                )

            for model in results["comparison_results"]:
                self.comparison_viz.plot_improvement_heatmap(
                    results["dataset_results"][results["metadata"]["baseline"]], results["dataset_results"][model]
                )

        # Generate statistical visualizations
        if results["comparison_results"]:
            for metric in self.metrics:
                data = {
                    model: [
                        results["dataset_results"][dataset][model][metric]
                        for dataset in results["dataset_results"]
                        if model in results["dataset_results"][dataset]
                        and metric in results["dataset_results"][dataset][model]
                    ]
                    for model in results["metadata"]["models"]
                }

                self.stats_viz.plot_confidence_intervals(data)

                if results["metadata"]["baseline"]:
                    baseline = results["metadata"]["baseline"]
                    baseline_data = {
                        dataset: results["dataset_results"][dataset][baseline]
                        for dataset in results["dataset_results"]
                        if baseline in results["dataset_results"][dataset]
                    }

                    for model in results["metadata"]["models"]:
                        if model != baseline:
                            model_data = {
                                dataset: results["dataset_results"][dataset][model]
                                for dataset in results["dataset_results"]
                                if model in results["dataset_results"][dataset]
                            }

                            self.stats_viz.plot_effect_sizes(baseline_data, model_data)

        # Generate summary report
        self.results_viz.generate_report(results["dataset_results"], self.metrics)

        if results["comparison_results"]:
            self.comparison_viz.create_comparison_report(
                results["dataset_results"], results["metadata"]["baseline"], self.metrics
            )

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results.

        Args:
            results: Results to save
        """
        import json

        # Save full results
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save metrics
        self.metrics_collector.save_metrics("benchmark_metrics.json")

        logger.info(f"Results saved to {self.output_dir}")
