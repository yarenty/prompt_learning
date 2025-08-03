"""Tests for comprehensive benchmark system."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memento.benchmarking.comprehensive_benchmark import (
    ComprehensiveBenchmark,
    PerformanceAnalyzer,
    ResourceMonitor,
)
from memento.config.models import ModelConfig, ModelType


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(model_type=ModelType.OLLAMA, model_name="test_model")


@pytest.fixture
def mock_dataset():
    """Create mock dataset."""
    return [
        {"id": 1, "description": "Test problem 1", "expected": "answer1"},
        {"id": 2, "description": "Test problem 2", "expected": "answer2"},
    ]


class TestResourceMonitor:
    """Test resource monitoring functionality."""

    def test_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor()
        assert monitor.process is not None
        assert monitor.start_time is None
        assert monitor.metrics == []
        assert not monitor.monitoring

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = ResourceMonitor()

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring
        assert monitor.start_time is not None
        assert monitor.metrics == []

        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring

    @patch("psutil.Process")
    def test_collect_metrics(self, mock_process_class):
        """Test metrics collection."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 50.0
        mock_process.memory_info.return_value = MagicMock(rss=1024 * 1024, vms=2048 * 1024)
        mock_process.memory_percent.return_value = 25.0
        mock_process.num_threads.return_value = 4
        mock_process.num_handles.return_value = 100
        mock_process_class.return_value = mock_process

        monitor = ResourceMonitor()
        monitor.start_monitoring()
        monitor.collect_metrics()

        assert len(monitor.metrics) == 1
        metric = monitor.metrics[0]
        assert metric["cpu_percent"] == 50.0
        assert metric["memory_rss"] == 1024 * 1024
        assert metric["memory_percent"] == 25.0
        assert metric["threads"] == 4

    def test_get_summary(self):
        """Test getting resource usage summary."""
        monitor = ResourceMonitor()

        # Test with no metrics
        summary = monitor.get_summary()
        assert "status" in summary

        # Test with mock metrics
        monitor.metrics = [{"summary": {"duration": 10.0, "peak_memory_rss": 1024 * 1024}}]
        summary = monitor.get_summary()
        assert summary["duration"] == 10.0
        assert summary["peak_memory_rss"] == 1024 * 1024


class TestPerformanceAnalyzer:
    """Test performance analysis functionality."""

    def test_initialization(self):
        """Test performance analyzer initialization."""
        analyzer = PerformanceAnalyzer()
        assert analyzer.results == []

    def test_analyze_throughput(self):
        """Test throughput analysis."""
        analyzer = PerformanceAnalyzer()

        # Test with empty results
        results = {"dataset_results": {}}
        throughput = analyzer.analyze_throughput(results)
        assert "error" in throughput

        # Test with valid results
        results = {
            "dataset_results": {
                "dataset1": {"samples_processed": 100, "processing_time": 10.0},
                "dataset2": {"samples_processed": 50, "processing_time": 5.0},
            }
        }
        throughput = analyzer.analyze_throughput(results)
        assert throughput["samples_per_second"] == 10.0  # (100+50)/(10+5)
        assert throughput["average_latency"] == 0.1  # (10+5)/(100+50)
        assert throughput["total_samples"] == 150
        assert throughput["total_time"] == 15.0

    def test_analyze_resources(self):
        """Test resource analysis."""
        analyzer = PerformanceAnalyzer()

        # Test with empty metrics
        resources = analyzer.analyze_resources([])
        assert "error" in resources

        # Test with valid metrics
        metrics = [
            {
                "summary": {
                    "peak_memory_rss": 1024 * 1024 * 100,  # 100MB
                    "avg_memory_rss": 1024 * 1024 * 50,  # 50MB
                    "peak_cpu": 80.0,
                    "avg_cpu": 40.0,
                    "duration": 30.0,
                }
            }
        ]
        resources = analyzer.analyze_resources(metrics)
        assert resources["peak_memory_mb"] == 100.0
        assert resources["avg_memory_mb"] == 50.0
        assert resources["peak_cpu_percent"] == 80.0
        assert resources["avg_cpu_percent"] == 40.0
        assert resources["duration_seconds"] == 30.0
        assert "efficiency_score" in resources


class TestComprehensiveBenchmark:
    """Test comprehensive benchmark functionality."""

    def test_initialization(self, model_config, temp_output_dir):
        """Test benchmark initialization."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=False,
        )

        assert benchmark.model_config == model_config
        assert benchmark.output_dir == temp_output_dir
        assert benchmark.dataset_loader is not None
        assert benchmark.evaluator is not None
        assert benchmark.metrics_collector is not None
        assert benchmark.results_viz is not None
        assert benchmark.comparison_viz is not None
        assert benchmark.stats_viz is not None
        assert benchmark.resource_monitor is None  # Disabled
        assert benchmark.dashboard is None  # Disabled

    def test_initialization_with_monitoring(self, model_config, temp_output_dir):
        """Test benchmark initialization with monitoring enabled."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=True,
            enable_resource_monitoring=True,
        )

        assert benchmark.resource_monitor is not None
        assert benchmark.dashboard is not None

    @pytest.mark.asyncio
    async def test_evaluate_baseline_model(self, model_config, temp_output_dir, mock_dataset):
        """Test baseline model evaluation."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=False,
        )

        # Mock baseline model
        with patch.object(benchmark.baseline_models["promptbreeder"], "evolve") as mock_evolve:
            mock_evolve.return_value = AsyncMock()

            result = await benchmark._evaluate_baseline_model("promptbreeder", mock_dataset, "test_dataset")

            assert isinstance(result, dict)
            assert "accuracy" in result
            assert "latency" in result
            assert "quality_score" in result
            assert 0 <= result["accuracy"] <= 1
            assert result["latency"] > 0
            assert 0 <= result["quality_score"] <= 1

    @pytest.mark.asyncio
    async def test_evaluate_custom_model(self, model_config, temp_output_dir, mock_dataset):
        """Test custom model evaluation."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=False,
        )

        result = await benchmark._evaluate_custom_model("memento", mock_dataset, "test_dataset")

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "latency" in result
        assert "quality_score" in result
        assert 0 <= result["accuracy"] <= 1
        assert result["latency"] > 0
        assert 0 <= result["quality_score"] <= 1

    def test_aggregate_model_results(self, model_config, temp_output_dir):
        """Test model results aggregation."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=False,
        )

        dataset_results = {
            "dataset1": {"model1": {"accuracy": 0.8, "latency": 0.1}, "model2": {"accuracy": 0.7, "latency": 0.2}},
            "dataset2": {"model1": {"accuracy": 0.9, "latency": 0.05}, "model2": {"accuracy": 0.75, "latency": 0.15}},
        }

        aggregated = benchmark._aggregate_model_results("model1", dataset_results)

        assert "accuracy" in aggregated
        assert "latency" in aggregated
        assert aggregated["accuracy"]["mean"] == 0.85  # (0.8 + 0.9) / 2
        assert aggregated["latency"]["mean"] == 0.075  # (0.1 + 0.05) / 2
        assert aggregated["accuracy"]["count"] == 2
        assert aggregated["latency"]["count"] == 2

    def test_compare_models(self, model_config, temp_output_dir):
        """Test model comparison."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=False,
        )

        dataset_results = {
            "dataset1": {"baseline": {"accuracy": 0.7, "latency": 0.2}, "improved": {"accuracy": 0.8, "latency": 0.15}}
        }

        comparison = benchmark._compare_models("baseline", "improved", dataset_results)

        assert "dataset1" in comparison
        assert "accuracy" in comparison["dataset1"]
        assert "latency" in comparison["dataset1"]

        # Check improvement percentages
        acc_improvement = comparison["dataset1"]["accuracy"]
        lat_improvement = comparison["dataset1"]["latency"]

        assert abs(acc_improvement - 14.29) < 0.1  # (0.8-0.7)/0.7 * 100 ≈ 14.29%
        assert abs(lat_improvement - (-25.0)) < 0.1  # (0.15-0.2)/0.2 * 100 = -25%

    @pytest.mark.asyncio
    async def test_run_benchmark_minimal(self, model_config, temp_output_dir):
        """Test minimal benchmark run."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=False,
        )

        # Mock dataset loader
        with patch.object(benchmark.dataset_loader, "load_dataset") as mock_load:
            mock_load.return_value = mock_dataset = [{"id": 1, "description": "test", "expected": "answer"}]

            # Mock evaluation methods
            with patch.object(benchmark, "_evaluate_baseline_model") as mock_eval_baseline:
                with patch.object(benchmark, "_evaluate_custom_model") as mock_eval_custom:
                    mock_eval_baseline.return_value = {"accuracy": 0.8, "latency": 0.1}
                    mock_eval_custom.return_value = {"accuracy": 0.9, "latency": 0.08}

                    # Mock visualization methods
                    with patch.object(benchmark, "_generate_visualizations") as mock_viz:
                        mock_viz.return_value = None

                        results = await benchmark.run_benchmark(
                            datasets=["test_dataset"],
                            models=["memento", "promptbreeder"],
                            baseline="promptbreeder",
                            max_samples_per_dataset=5,
                        )

                        # Verify results structure
                        assert "metadata" in results
                        assert "dataset_results" in results
                        assert "model_results" in results
                        assert "comparison_results" in results
                        assert "performance_analysis" in results

                        # Verify metadata
                        metadata = results["metadata"]
                        assert metadata["datasets"] == ["test_dataset"]
                        assert metadata["models"] == ["memento", "promptbreeder"]
                        assert metadata["baseline"] == "promptbreeder"
                        assert metadata["max_samples_per_dataset"] == 5

                        # Verify dataset results
                        assert "test_dataset" in results["dataset_results"]
                        dataset_result = results["dataset_results"]["test_dataset"]
                        assert "memento" in dataset_result
                        assert "promptbreeder" in dataset_result

                        # Verify model results aggregation
                        assert "memento" in results["model_results"]
                        assert "promptbreeder" in results["model_results"]

                        # Verify comparison results
                        assert "memento" in results["comparison_results"]

    def test_save_results(self, model_config, temp_output_dir):
        """Test results saving."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=False,
        )

        test_results = {"metadata": {"test": "data"}, "dataset_results": {"dataset1": {"model1": {"accuracy": 0.8}}}}

        benchmark._save_results(test_results)

        # Check if results file was created
        results_file = temp_output_dir / "comprehensive_benchmark_results.json"
        assert results_file.exists()

        # Verify content
        with open(results_file) as f:
            saved_results = json.load(f)

        assert saved_results["metadata"]["test"] == "data"
        assert saved_results["dataset_results"]["dataset1"]["model1"]["accuracy"] == 0.8


class TestIntegration:
    """Integration tests for the comprehensive benchmark system."""

    @pytest.mark.asyncio
    async def test_full_benchmark_flow(self, model_config, temp_output_dir):
        """Test complete benchmark flow end-to-end."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=True,
        )

        # Create minimal mock implementations
        mock_dataset = [{"id": 1, "description": "test problem", "expected": "answer"}]

        with patch.object(benchmark.dataset_loader, "load_dataset") as mock_load:
            mock_load.return_value = mock_dataset

            with patch.object(benchmark, "_evaluate_baseline_model") as mock_eval_baseline:
                with patch.object(benchmark, "_evaluate_custom_model") as mock_eval_custom:
                    with patch.object(benchmark, "_generate_visualizations") as mock_viz:
                        mock_eval_baseline.return_value = {"accuracy": 0.75, "latency": 0.12, "quality_score": 0.8}
                        mock_eval_custom.return_value = {"accuracy": 0.85, "latency": 0.10, "quality_score": 0.9}
                        mock_viz.return_value = None

                        results = await benchmark.run_benchmark(
                            datasets=["humaneval"], models=["memento", "promptbreeder"], baseline="promptbreeder"
                        )

                        # Verify complete results structure
                        assert all(
                            key in results
                            for key in [
                                "metadata",
                                "dataset_results",
                                "model_results",
                                "comparison_results",
                                "performance_analysis",
                                "resource_analysis",
                            ]
                        )

                        # Verify improvement calculation
                        memento_comparison = results["comparison_results"]["memento"]
                        assert "humaneval" in memento_comparison

                        # Verify files were created
                        assert (temp_output_dir / "comprehensive_benchmark_results.json").exists()

                        # Verify resource monitoring worked
                        if benchmark.resource_monitor:
                            assert len(benchmark.resource_monitor.metrics) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, model_config, temp_output_dir):
        """Test error handling in benchmark system."""
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=temp_output_dir,
            enable_dashboard=False,
            enable_resource_monitoring=False,
        )

        # Mock dataset loader to raise exception
        with patch.object(benchmark.dataset_loader, "load_dataset") as mock_load:
            mock_load.side_effect = Exception("Dataset loading failed")

            with patch.object(benchmark, "_generate_visualizations") as mock_viz:
                mock_viz.return_value = None

                results = await benchmark.run_benchmark(datasets=["failing_dataset"], models=["memento"], baseline=None)

                # Verify error was captured
                assert "failing_dataset" in results["dataset_results"]
                assert "error" in results["dataset_results"]["failing_dataset"]
                assert "Dataset loading failed" in results["dataset_results"]["failing_dataset"]["error"]
