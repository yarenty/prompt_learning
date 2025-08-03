"""Enhanced CLI interface for comprehensive benchmarking."""

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import click

from ..benchmarking.comprehensive_benchmark import ComprehensiveBenchmark
from ..config.models import ModelConfig, ModelType


@click.group()
def benchmark():
    """Comprehensive benchmarking commands."""
    pass


@benchmark.command()
@click.option(
    "--datasets", multiple=True, default=["humaneval", "gsm8k", "writingbench"], help="Datasets to benchmark on"
)
@click.option(
    "--models", multiple=True, default=["memento", "promptbreeder", "self_evolving_gpt"], help="Models to evaluate"
)
@click.option("--baseline", default="promptbreeder", help="Baseline model for comparison")
@click.option("--output-dir", default="comprehensive_benchmark_results", help="Output directory for results")
@click.option("--max-samples", type=int, help="Maximum samples per dataset")
@click.option(
    "--model-type", type=click.Choice(["ollama", "openai", "anthropic"]), default="ollama", help="Model type to use"
)
@click.option("--model-name", default="llama2", help="Model name")
@click.option("--dashboard/--no-dashboard", default=True, help="Enable real-time dashboard")
@click.option("--monitoring/--no-monitoring", default=True, help="Enable resource monitoring")
@click.option("--config-file", type=click.Path(exists=True), help="Configuration file (YAML/JSON)")
def run(
    datasets: List[str],
    models: List[str],
    baseline: str,
    output_dir: str,
    max_samples: Optional[int],
    model_type: str,
    model_name: str,
    dashboard: bool,
    monitoring: bool,
    config_file: Optional[str],
):
    """Run comprehensive benchmark."""
    click.echo("🚀 Starting Comprehensive Benchmark")
    click.echo("=" * 50)

    # Load configuration
    if config_file:
        config = _load_config(config_file)
        datasets = config.get("datasets", datasets)
        models = config.get("models", models)
        baseline = config.get("baseline", baseline)
        max_samples = config.get("max_samples", max_samples)
        dashboard = config.get("dashboard", dashboard)
        monitoring = config.get("monitoring", monitoring)

        # Model config from file
        model_config_dict = config.get("model_config", {})
        model_config = ModelConfig(
            model_type=ModelType(model_config_dict.get("model_type", model_type)),
            model_name=model_config_dict.get("model_name", model_name),
            **{k: v for k, v in model_config_dict.items() if k not in ["model_type", "model_name"]},
        )
    else:
        # Create model config from CLI args
        model_config = ModelConfig(model_type=ModelType(model_type), model_name=model_name)

    click.echo(f"📊 Datasets: {', '.join(datasets)}")
    click.echo(f"🤖 Models: {', '.join(models)}")
    click.echo(f"📈 Baseline: {baseline}")
    click.echo(f"💾 Output: {output_dir}")
    click.echo(f"🎯 Max samples: {max_samples or 'All'}")
    click.echo(f"📱 Dashboard: {'Enabled' if dashboard else 'Disabled'}")
    click.echo(f"📊 Monitoring: {'Enabled' if monitoring else 'Disabled'}")
    click.echo()

    # Run benchmark
    asyncio.run(
        _run_benchmark(
            model_config=model_config,
            datasets=list(datasets),
            models=list(models),
            baseline=baseline,
            output_dir=output_dir,
            max_samples=max_samples,
            dashboard=dashboard,
            monitoring=monitoring,
        )
    )


@benchmark.command()
@click.option("--results-dir", required=True, help="Directory containing benchmark results")
@click.option(
    "--output-format",
    type=click.Choice(["html", "pdf", "json", "csv"]),
    default="html",
    help="Output format for report",
)
@click.option("--include-plots/--no-plots", default=True, help="Include visualization plots")
def report(results_dir: str, output_format: str, include_plots: bool):
    """Generate comprehensive report from benchmark results."""
    click.echo("📊 Generating Comprehensive Report")
    click.echo("=" * 40)

    results_path = Path(results_dir)
    if not results_path.exists():
        click.echo(f"❌ Results directory not found: {results_dir}")
        return

    # Look for results file
    results_file = results_path / "comprehensive_benchmark_results.json"
    if not results_file.exists():
        click.echo(f"❌ Results file not found: {results_file}")
        return

    click.echo(f"📁 Results: {results_file}")
    click.echo(f"📄 Format: {output_format}")
    click.echo(f"📈 Plots: {'Included' if include_plots else 'Excluded'}")

    # Load and process results
    with open(results_file) as f:
        results = json.load(f)

    _generate_report(results, output_format, include_plots, results_path)
    click.echo("✅ Report generated successfully!")


@benchmark.command()
@click.option("--datasets", multiple=True, help="Datasets to validate")
@click.option("--cache-dir", help="Cache directory for datasets")
@click.option("--max-samples", type=int, help="Maximum samples to validate per dataset")
def validate_datasets(datasets: Optional[List[str]], cache_dir: Optional[str], max_samples: Optional[int]):
    """Validate dataset availability and integrity."""
    click.echo("🔍 Validating Datasets")
    click.echo("=" * 30)

    from ..benchmarking.datasets.loader import DatasetLoader

    loader = DatasetLoader(cache_dir=cache_dir)
    available_datasets = list(loader.datasets.keys())

    if not datasets:
        datasets = available_datasets

    click.echo(f"📋 Available datasets: {', '.join(available_datasets)}")
    click.echo(f"🎯 Validating: {', '.join(datasets)}")
    click.echo()

    asyncio.run(_validate_datasets(loader, datasets, max_samples))


@benchmark.command()
@click.option("--host", default="localhost", help="Dashboard host")
@click.option("--port", default=8050, type=int, help="Dashboard port")
@click.option("--update-interval", default=1.0, type=float, help="Update interval in seconds")
def dashboard_only(host: str, port: int, update_interval: float):
    """Run dashboard server only (for monitoring external benchmarks)."""
    click.echo("📱 Starting Dashboard Server")
    click.echo("=" * 35)

    from ..visualization.dashboard import DashboardServer

    dashboard = DashboardServer(host=host, port=port, update_interval=update_interval)

    click.echo(f"🌐 Host: {host}")
    click.echo(f"🔌 Port: {port}")
    click.echo(f"⏱️  Update interval: {update_interval}s")
    click.echo(f"🔗 URL: http://{host}:{port}")
    click.echo()
    click.echo("Press Ctrl+C to stop...")

    try:
        asyncio.run(dashboard.start())
    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped")


@benchmark.command()
@click.option(
    "--template", type=click.Choice(["basic", "advanced", "research"]), default="basic", help="Configuration template"
)
@click.option("--output", default="benchmark_config.yaml", help="Output configuration file")
def init_config(template: str, output: str):
    """Initialize benchmark configuration file."""
    click.echo(f"⚙️  Creating {template} configuration template")

    config = _create_config_template(template)

    output_path = Path(output)
    with open(output_path, "w") as f:
        if output_path.suffix.lower() == ".json":
            json.dump(config, f, indent=2)
        else:
            import yaml

            yaml.dump(config, f, default_flow_style=False)

    click.echo(f"✅ Configuration saved to: {output}")
    click.echo(f"📝 Edit the file and use --config-file flag to use it")


async def _run_benchmark(
    model_config: ModelConfig,
    datasets: List[str],
    models: List[str],
    baseline: str,
    output_dir: str,
    max_samples: Optional[int],
    dashboard: bool,
    monitoring: bool,
):
    """Run the comprehensive benchmark."""
    benchmark = ComprehensiveBenchmark(
        model_config=model_config,
        output_dir=output_dir,
        enable_dashboard=dashboard,
        enable_resource_monitoring=monitoring,
    )

    try:
        results = await benchmark.run_benchmark(
            datasets=datasets, models=models, baseline=baseline, max_samples_per_dataset=max_samples
        )

        click.echo("\n✅ Benchmark completed successfully!")
        click.echo(f"📊 Results saved to: {output_dir}")

        # Print summary
        _print_results_summary(results)

    except Exception as e:
        click.echo(f"\n❌ Benchmark failed: {e}")
        raise


async def _validate_datasets(loader, datasets: List[str], max_samples: Optional[int]):
    """Validate datasets."""
    for dataset_name in datasets:
        click.echo(f"🔍 Validating {dataset_name}...")

        try:
            dataset = await loader.load_dataset(dataset_name, max_samples=max_samples or 5)

            if hasattr(dataset, "__len__"):
                size = len(dataset)
                click.echo(f"  ✅ Loaded {size} samples")
            else:
                click.echo(f"  ✅ Dataset loaded successfully")

            # Basic validation
            if hasattr(dataset, "__iter__"):
                first_item = next(iter(dataset), None)
                if first_item:
                    click.echo(
                        f"  📋 Sample keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Non-dict item'}"
                    )

        except Exception as e:
            click.echo(f"  ❌ Failed: {e}")


def _load_config(config_file: str) -> dict:
    """Load configuration from file."""
    config_path = Path(config_file)

    with open(config_path) as f:
        if config_path.suffix.lower() == ".json":
            return json.load(f)
        else:
            import yaml

            return yaml.safe_load(f)


def _create_config_template(template: str) -> dict:
    """Create configuration template."""
    base_config = {
        "datasets": ["humaneval", "gsm8k", "writingbench"],
        "models": ["memento", "promptbreeder", "self_evolving_gpt"],
        "baseline": "promptbreeder",
        "max_samples": None,
        "dashboard": True,
        "monitoring": True,
        "model_config": {"model_type": "ollama", "model_name": "llama2", "temperature": 0.7, "max_tokens": 1000},
    }

    if template == "advanced":
        base_config.update(
            {
                "datasets": [
                    "humaneval",
                    "apps",
                    "livecodebench",
                    "gsm8k",
                    "math_hard",
                    "mmlu_math",
                    "writingbench",
                    "creativity",
                ],
                "models": ["memento", "promptbreeder", "self_evolving_gpt", "auto_evolve"],
                "max_samples": 100,
                "evaluation_config": {
                    "programming": {"timeout": 10, "enable_quality_metrics": True, "enable_runtime_metrics": True},
                    "mathematics": {"enable_reasoning_analysis": True, "enable_complexity_analysis": True},
                    "writing": {"enable_coherence_analysis": True, "enable_style_analysis": True},
                },
            }
        )
    elif template == "research":
        base_config.update(
            {
                "datasets": [
                    "humaneval",
                    "bigcodebench",
                    "apps",
                    "livecodebench",
                    "gsm8k",
                    "math_hard",
                    "mmlu_math",
                    "writingbench",
                    "creativity",
                    "biggen_bench",
                ],
                "models": ["memento", "promptbreeder", "self_evolving_gpt", "auto_evolve"],
                "baseline": "promptbreeder",
                "max_samples": None,  # Use full datasets
                "statistical_analysis": {
                    "confidence_level": 0.95,
                    "enable_effect_size": True,
                    "enable_power_analysis": True,
                    "bootstrap_samples": 1000,
                },
                "visualization": {
                    "enable_interactive_plots": True,
                    "export_formats": ["html", "pdf", "png"],
                    "publication_ready": True,
                },
            }
        )

    return base_config


def _generate_report(results: dict, output_format: str, include_plots: bool, output_dir: Path):
    """Generate report from results."""
    # This would integrate with the visualization system
    # For now, just save a summary

    summary = {
        "benchmark_info": results.get("metadata", {}),
        "performance_summary": results.get("performance_analysis", {}),
        "resource_summary": results.get("resource_analysis", {}),
        "model_comparison": results.get("comparison_results", {}),
    }

    report_file = output_dir / f"benchmark_report.{output_format}"

    if output_format == "json":
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)
    elif output_format == "html":
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Benchmark Report</title></head>
        <body>
        <h1>Comprehensive Benchmark Report</h1>
        <pre>{json.dumps(summary, indent=2)}</pre>
        </body>
        </html>
        """
        with open(report_file, "w") as f:
            f.write(html_content)

    click.echo(f"📄 Report saved: {report_file}")


def _print_results_summary(results: dict):
    """Print results summary."""
    click.echo("\n📊 Results Summary:")
    click.echo("-" * 30)

    # Performance analysis
    perf = results.get("performance_analysis", {})
    if perf and "samples_per_second" in perf:
        click.echo(f"⚡ Throughput: {perf['samples_per_second']:.2f} samples/sec")
        click.echo(f"⏱️  Avg Latency: {perf['average_latency']:.3f} sec")

    # Resource analysis
    resources = results.get("resource_analysis", {})
    if resources and "peak_memory_mb" in resources:
        click.echo(f"💾 Peak Memory: {resources['peak_memory_mb']:.1f} MB")
        click.echo(f"🔋 Avg CPU: {resources['avg_cpu_percent']:.1f}%")

    # Model results
    model_results = results.get("model_results", {})
    if model_results:
        click.echo("\n🤖 Model Performance:")
        for model, metrics in model_results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                acc = metrics["accuracy"]
                if isinstance(acc, dict) and "mean" in acc:
                    click.echo(f"  {model}: {acc['mean']:.3f} ± {acc.get('std', 0):.3f}")

    # Comparisons
    comparisons = results.get("comparison_results", {})
    if comparisons:
        baseline = results.get("metadata", {}).get("baseline", "baseline")
        click.echo(f"\n📈 Improvements over {baseline}:")
        for model, improvements in comparisons.items():
            if isinstance(improvements, dict):
                # Average improvement across datasets
                all_improvements = []
                for dataset_improvements in improvements.values():
                    if isinstance(dataset_improvements, dict):
                        all_improvements.extend(
                            [v for v in dataset_improvements.values() if isinstance(v, (int, float))]
                        )

                if all_improvements:
                    avg_improvement = sum(all_improvements) / len(all_improvements)
                    click.echo(f"  {model}: {avg_improvement:+.1f}%")


if __name__ == "__main__":
    benchmark()
