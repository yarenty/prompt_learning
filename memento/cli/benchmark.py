"""CLI interface for benchmarking."""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

import click

from ..benchmarking.main import ComprehensiveBenchmark


@click.group()
def benchmark():
    """Benchmark command group."""
    pass


@benchmark.command()
@click.option("--datasets", "-d", multiple=True, help="Datasets to benchmark")
@click.option("--models", "-m", multiple=True, help="Models to evaluate")
@click.option("--output-dir", "-o", default="benchmark_results", help="Output directory")
@click.option("--cache-dir", "-c", default=None, help="Cache directory")
@click.option("--max-samples", "-n", type=int, default=None, help="Maximum samples per dataset")
def run(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    output_dir: str = "benchmark_results",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Run benchmarks.

    Example:
        memento benchmark run -d humaneval gsm8k -m memento promptbreeder
    """
    # Initialize benchmark
    benchmark = ComprehensiveBenchmark(output_dir=output_dir, cache_dir=cache_dir)

    # Convert max samples
    max_samples_dict = (
        {dataset: max_samples for dataset in (datasets or benchmark.datasets.keys())} if max_samples else None
    )

    # Run benchmarks
    results = asyncio.run(benchmark.run_benchmarks(datasets=datasets, models=models, max_samples=max_samples_dict))

    # Print summary
    click.echo(f"\n✨ Benchmark complete!")
    click.echo(f"📊 Report: {results['report']}")
    click.echo(f"💾 Results: {results['results_file']}")


@benchmark.command()
@click.argument("model")
@click.argument("dataset")
@click.option("--max-samples", "-n", type=int, default=None, help="Maximum samples")
@click.option("--output-dir", "-o", default="benchmark_results", help="Output directory")
def evaluate(model: str, dataset: str, max_samples: Optional[int] = None, output_dir: str = "benchmark_results"):
    """Evaluate single model on dataset.

    Example:
        memento benchmark evaluate memento humaneval -n 100
    """
    # Initialize benchmark
    benchmark = ComprehensiveBenchmark(output_dir=output_dir)

    # Run evaluation
    results = asyncio.run(benchmark.evaluate_model(model=model, dataset=dataset, max_samples=max_samples))

    # Print summary
    click.echo(f"\n✨ Evaluation complete!")
    click.echo(f"📊 Report: {results['report']}")
    click.echo(f"💾 Results: {results['results_file']}")


@benchmark.command()
def list_datasets():
    """List available datasets."""
    benchmark = ComprehensiveBenchmark()

    click.echo("\n📚 Available Datasets:")
    for name, config in benchmark.datasets.items():
        click.echo(f"\n{name}:")
        click.echo(f"  Max samples: {config['max_samples']}")
        if "split" in config:
            click.echo(f"  Split: {config['split']}")


@benchmark.command()
def list_models():
    """List available models."""
    benchmark = ComprehensiveBenchmark()

    click.echo("\n🤖 Available Models:")
    for name in benchmark.models:
        click.echo(f"- {name}")


if __name__ == "__main__":
    benchmark()
