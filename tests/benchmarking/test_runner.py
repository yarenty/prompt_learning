"""Tests for benchmark runner."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from memento.benchmarking.main import ComprehensiveBenchmark


@pytest.fixture
def benchmark():
    """Create temporary benchmark runner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ComprehensiveBenchmark(output_dir=tmpdir)


@pytest.fixture
def sample_results():
    """Generate sample benchmark results."""
    np.random.seed(42)

    return {
        "dataset_results": {
            "dataset1": {
                "model1": {"accuracy": 0.85, "latency": 0.15, "memory_usage": 512.0, "quality_score": 0.92},
                "model2": {"accuracy": 0.82, "latency": 0.18, "memory_usage": 498.0, "quality_score": 0.89},
            },
            "dataset2": {
                "model1": {"accuracy": 0.88, "latency": 0.12, "memory_usage": 525.0, "quality_score": 0.94},
                "model2": {"accuracy": 0.84, "latency": 0.16, "memory_usage": 508.0, "quality_score": 0.91},
            },
        },
        "metadata": {
            "datasets": ["dataset1", "dataset2"],
            "models": ["model1", "model2"],
            "baseline": "model1",
            "metrics": ["accuracy", "latency", "memory_usage", "quality_score"],
        },
    }


@pytest.mark.asyncio
async def test_run_benchmark(benchmark):
    """Test running benchmark."""
    # Run benchmark
    results = await benchmark.run_benchmark(
        datasets=["dataset1", "dataset2"], models=["model1", "model2"], baseline="model1"
    )

    # Check results structure
    assert "metadata" in results
    assert "dataset_results" in results
    assert "model_results" in results
    assert "comparison_results" in results

    # Check metadata
    assert results["metadata"]["datasets"] == ["dataset1", "dataset2"]
    assert results["metadata"]["models"] == ["model1", "model2"]
    assert results["metadata"]["baseline"] == "model1"

    # Check output files
    assert (benchmark.output_dir / "benchmark_results.json").exists()
    assert (benchmark.output_dir / "metrics" / "benchmark_metrics.json").exists()


@pytest.mark.asyncio
async def test_evaluate_dataset(benchmark):
    """Test dataset evaluation."""
    results = await benchmark._evaluate_dataset("dataset1", ["model1", "model2"])

    # Check results for each model
    assert "model1" in results
    assert "model2" in results

    # Check metrics
    for model_results in results.values():
        for metric in benchmark.metrics:
            assert metric in model_results
            assert isinstance(model_results[metric], float)
            assert 0 <= model_results[metric] <= 1


@pytest.mark.asyncio
async def test_evaluate_model(benchmark):
    """Test model evaluation."""
    results = await benchmark._evaluate_model("model1", "dataset1")

    # Check metrics
    for metric in benchmark.metrics:
        assert metric in results
        assert isinstance(results[metric], float)
        assert 0 <= results[metric] <= 1


def test_aggregate_model_results(benchmark, sample_results):
    """Test model results aggregation."""
    results = benchmark._aggregate_model_results("model1", sample_results["dataset_results"])

    # Check statistics for each metric
    for metric in benchmark.metrics:
        assert metric in results
        stats = results[metric]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats


@pytest.mark.asyncio
async def test_generate_comparisons(benchmark, sample_results):
    """Test comparison generation."""
    comparisons = await benchmark._generate_comparisons(
        "model1", ["model1", "model2"], sample_results["dataset_results"]
    )

    # Check comparison results
    assert "model2" in comparisons
    comparison = comparisons["model2"]
    assert "effect_sizes" in comparison
    assert "improvements" in comparison

    # Check effect sizes
    effect_sizes = comparison["effect_sizes"]
    for metric in benchmark.metrics:
        assert metric in effect_sizes
        assert isinstance(effect_sizes[metric], float)

    # Check improvements
    improvements = comparison["improvements"]
    for dataset in sample_results["dataset_results"]:
        assert dataset in improvements
        for metric in benchmark.metrics:
            assert metric in improvements[dataset]
            assert isinstance(improvements[dataset][metric], float)


def test_calculate_effect_sizes(benchmark):
    """Test effect size calculation."""
    baseline_results = {
        "dataset1": {"accuracy": 0.85, "latency": 0.15},
        "dataset2": {"accuracy": 0.88, "latency": 0.12},
    }

    model_results = {"dataset1": {"accuracy": 0.82, "latency": 0.18}, "dataset2": {"accuracy": 0.84, "latency": 0.16}}

    effect_sizes = benchmark._calculate_effect_sizes(baseline_results, model_results)

    # Check effect sizes
    assert "accuracy" in effect_sizes
    assert "latency" in effect_sizes
    assert isinstance(effect_sizes["accuracy"], float)
    assert isinstance(effect_sizes["latency"], float)


def test_calculate_improvements(benchmark):
    """Test improvement calculation."""
    baseline_results = {
        "dataset1": {"accuracy": 0.85, "latency": 0.15},
        "dataset2": {"accuracy": 0.88, "latency": 0.12},
    }

    model_results = {"dataset1": {"accuracy": 0.89, "latency": 0.14}, "dataset2": {"accuracy": 0.92, "latency": 0.11}}

    improvements = benchmark._calculate_improvements(baseline_results, model_results)

    # Check improvements
    for dataset in ["dataset1", "dataset2"]:
        assert dataset in improvements
        for metric in ["accuracy", "latency"]:
            assert metric in improvements[dataset]
            assert isinstance(improvements[dataset][metric], float)


@pytest.mark.asyncio
async def test_generate_visualizations(benchmark, sample_results):
    """Test visualization generation."""
    # Generate visualizations
    await benchmark._generate_visualizations(sample_results)

    # Check visualization files
    viz_dir = benchmark.output_dir
    assert any(viz_dir.glob("results/*.html"))
    assert any(viz_dir.glob("comparison/*.html"))
    assert any(viz_dir.glob("stats/*.html"))


def test_save_results(benchmark, sample_results):
    """Test results saving."""
    # Save results
    benchmark._save_results(sample_results)

    # Check saved files
    results_file = benchmark.output_dir / "benchmark_results.json"
    assert results_file.exists()

    metrics_file = benchmark.output_dir / "metrics" / "benchmark_metrics.json"
    assert metrics_file.exists()


def test_error_handling(benchmark):
    """Test error handling."""
    # Test with invalid dataset
    with pytest.raises(ValueError):
        benchmark.run_benchmark(datasets=["invalid_dataset"], models=["model1"])

    # Test with invalid model
    with pytest.raises(ValueError):
        benchmark.run_benchmark(datasets=["dataset1"], models=["invalid_model"])

    # Test with invalid baseline
    with pytest.raises(ValueError):
        benchmark.run_benchmark(datasets=["dataset1"], models=["model1"], baseline="invalid_model")

    # Test with empty datasets
    with pytest.raises(ValueError):
        benchmark.run_benchmark(datasets=[], models=["model1"])

    # Test with empty models
    with pytest.raises(ValueError):
        benchmark.run_benchmark(datasets=["dataset1"], models=[])
