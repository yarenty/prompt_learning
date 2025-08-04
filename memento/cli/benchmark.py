"""CLI interface for benchmarking."""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

import click

from ..benchmarking.comprehensive_benchmark import ComprehensiveBenchmark
from ..config.models import ModelConfig, ModelType


@click.group()
def benchmark():
    """Benchmark command group."""
    pass


@benchmark.command()
@click.option("--datasets", "-d", multiple=True, default=["humaneval"], help="Datasets to benchmark")
@click.option("--models", "-m", multiple=True, default=["memento"], help="Models to evaluate")
@click.option("--output-dir", "-o", default="benchmark_results", help="Output directory")
@click.option("--max-samples", "-n", type=int, default=None, help="Maximum samples per dataset")
@click.option("--model-type", default="ollama", help="Model type")
@click.option("--model-name", default="llama3.2", help="Model name")
@click.option("--baseline", default="promptbreeder", help="Baseline model for comparison")
@click.option("--no-dashboard", is_flag=True, help="Disable dashboard")
@click.option("--no-monitoring", is_flag=True, help="Disable resource monitoring")
def run(
    datasets: List[str],
    models: List[str],
    output_dir: str = "benchmark_results",
    max_samples: Optional[int] = None,
    model_type: str = "ollama",
    model_name: str = "llama3.2",
    baseline: str = "promptbreeder",
    no_dashboard: bool = False,
    no_monitoring: bool = False,
):
    """Run comprehensive benchmarks.

    Example:
        memento benchmark run -d humaneval gsm8k -m memento promptbreeder
    """
    # Create model config
    model_config = ModelConfig(
        model_type=ModelType(model_type),
        model_name=model_name,
        temperature=0.7,
        max_tokens=2048,
    )

    # Initialize benchmark
    benchmark = ComprehensiveBenchmark(
        model_config=model_config,
        output_dir=output_dir,
        enable_dashboard=not no_dashboard,
        enable_resource_monitoring=not no_monitoring,
    )

    # Run benchmarks
    results = asyncio.run(
        benchmark.run_benchmark(
            datasets=list(datasets),
            models=list(models),
            baseline=baseline,
            max_samples_per_dataset=max_samples,
        )
    )

    # Print summary
    click.echo(f"\n✨ Benchmark complete!")
    click.echo(f"📊 Results saved to: {output_dir}")
    
    # Print performance summary
    if "performance_analysis" in results:
        perf = results["performance_analysis"]
        if "samples_per_second" in perf:
            click.echo(f"⚡ Throughput: {perf['samples_per_second']:.2f} samples/sec")
    
    # Print model results
    if "model_results" in results:
        click.echo(f"\n🤖 Model Performance:")
        for model, metrics in results["model_results"].items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                acc = metrics["accuracy"]
                if isinstance(acc, dict) and "mean" in acc:
                    click.echo(f"  {model}: {acc['mean']:.3f} ± {acc.get('std', 0):.3f}")


@benchmark.command()
@click.argument("model")
@click.argument("dataset")
@click.option("--max-samples", "-n", type=int, default=None, help="Maximum samples")
@click.option("--output-dir", "-o", default="benchmark_results", help="Output directory")
@click.option("--model-type", default="ollama", help="Model type")
@click.option("--model-name", default="llama3.2", help="Model name")
def evaluate(
    model: str,
    dataset: str,
    max_samples: Optional[int] = None,
    output_dir: str = "benchmark_results",
    model_type: str = "ollama",
    model_name: str = "llama3.2",
):
    """Evaluate single model on dataset.

    Example:
        memento benchmark evaluate memento humaneval -n 100
    """
    # Create model config
    model_config = ModelConfig(
        model_type=ModelType(model_type),
        model_name=model_name,
        temperature=0.7,
        max_tokens=2048,
    )

    # Initialize benchmark
    benchmark = ComprehensiveBenchmark(
        model_config=model_config,
        output_dir=output_dir,
        enable_dashboard=False,
        enable_resource_monitoring=False,
    )

    # Run evaluation
    results = asyncio.run(
        benchmark.run_benchmark(
            datasets=[dataset],
            models=[model],
            max_samples_per_dataset=max_samples,
        )
    )

    # Print summary
    click.echo(f"\n✨ Evaluation complete!")
    click.echo(f"📊 Results saved to: {output_dir}")


@benchmark.command()
def list_datasets():
    """List available datasets."""
    from ..benchmarking.datasets.loader import DatasetLoader
    
    loader = DatasetLoader()
    available_datasets = list(loader.datasets.keys())

    click.echo("\n📚 Available Datasets:")
    for name in available_datasets:
        click.echo(f"- {name}")


@benchmark.command()
def list_models():
    """List available models."""
    available_models = ["memento", "promptbreeder", "self_evolving_gpt", "auto_evolve"]

    click.echo("\n🤖 Available Models:")
    for name in available_models:
        click.echo(f"- {name}")


if __name__ == "__main__":
    benchmark()
