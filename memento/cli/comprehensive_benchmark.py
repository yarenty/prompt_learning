"""Enhanced CLI interface for comprehensive benchmarking."""

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import typer

from ..benchmarking.comprehensive_benchmark import ComprehensiveBenchmark
from ..config.models import ModelConfig, ModelType

benchmark = typer.Typer(name="benchmark", help="Comprehensive benchmarking commands")


@benchmark.command()
def run(
    datasets: List[str] = typer.Option(
        ["humaneval", "gsm8k", "writingbench"], "--datasets", help="Datasets to benchmark on"
    ),
    models: List[str] = typer.Option(
        ["memento", "promptbreeder", "self_evolving_gpt"], "--models", help="Models to evaluate"
    ),
    baseline: str = typer.Option("promptbreeder", "--baseline", help="Baseline model for comparison"),
    output_dir: str = typer.Option(
        "comprehensive_benchmark_results", "--output-dir", help="Output directory for results"
    ),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Maximum samples per dataset"),
    model_type: str = typer.Option("ollama", "--model-type", help="Model type to use"),
    model_name: str = typer.Option("llama2", "--model-name", help="Model name"),
    dashboard: bool = typer.Option(True, "--dashboard/--no-dashboard", help="Enable real-time dashboard"),
    monitoring: bool = typer.Option(True, "--monitoring/--no-monitoring", help="Enable resource monitoring"),
    config_file: Optional[str] = typer.Option(None, "--config-file", help="Configuration file (YAML/JSON)"),
):
    """Run comprehensive benchmark."""
    typer.echo("🚀 Starting Comprehensive Benchmark")
    typer.echo("=" * 50)

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

    typer.echo(f"📊 Datasets: {', '.join(datasets)}")
    typer.echo(f"🤖 Models: {', '.join(models)}")
    typer.echo(f"📈 Baseline: {baseline}")
    typer.echo(f"💾 Output: {output_dir}")
    typer.echo(f"🎯 Max samples: {max_samples or 'All'}")
    typer.echo(f"📱 Dashboard: {'Enabled' if dashboard else 'Disabled'}")
    typer.echo(f"📊 Monitoring: {'Enabled' if monitoring else 'Disabled'}")
    typer.echo()

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
def report(
    results_dir: str = typer.Option(..., "--results-dir", help="Directory containing benchmark results"),
    output_format: str = typer.Option("html", "--output-format", help="Output format for report"),
    include_plots: bool = typer.Option(True, "--include-plots/--no-plots", help="Include visualization plots"),
):
    """Generate comprehensive report from benchmark results."""
    typer.echo("📊 Generating Comprehensive Report")
    typer.echo("=" * 40)

    results_path = Path(results_dir)
    if not results_path.exists():
        typer.echo(f"❌ Results directory not found: {results_dir}")
        return

    # Look for results file
    results_file = results_path / "comprehensive_benchmark_results.json"
    if not results_file.exists():
        typer.echo(f"❌ Results file not found: {results_file}")
        return

    typer.echo(f"📁 Results: {results_file}")
    typer.echo(f"📄 Format: {output_format}")
    typer.echo(f"📈 Plots: {'Included' if include_plots else 'Excluded'}")

    # Load and process results
    with open(results_file) as f:
        results = json.load(f)

    _generate_report(results, output_format, include_plots, results_path)
    typer.echo("✅ Report generated successfully!")


@benchmark.command()
def validate_datasets(
    datasets: Optional[List[str]] = typer.Option(None, "--datasets", help="Datasets to validate"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", help="Cache directory for datasets"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Maximum samples to validate per dataset"),
):
    """Validate dataset availability and integrity."""
    typer.echo("🔍 Validating Datasets")
    typer.echo("=" * 30)

    from ..benchmarking.datasets.loader import DatasetLoader

    loader = DatasetLoader(cache_dir=cache_dir)
    available_datasets = list(loader.datasets.keys())

    if not datasets:
        datasets = available_datasets

    typer.echo(f"📋 Available datasets: {', '.join(available_datasets)}")
    typer.echo(f"🎯 Validating: {', '.join(datasets)}")
    typer.echo()

    asyncio.run(_validate_datasets(loader, datasets, max_samples))


@benchmark.command()
def dashboard_only(
    host: str = typer.Option("localhost", "--host", help="Dashboard host"),
    port: int = typer.Option(8050, "--port", help="Dashboard port"),
    update_interval: float = typer.Option(1.0, "--update-interval", help="Update interval in seconds"),
):
    """Run dashboard server only (for monitoring external benchmarks)."""
    typer.echo("📱 Starting Dashboard Server")
    typer.echo("=" * 35)

    from ..visualization.dashboard import DashboardServer

    dashboard = DashboardServer(host=host, port=port, update_interval=update_interval)

    typer.echo(f"🌐 Host: {host}")
    typer.echo(f"🔌 Port: {port}")
    typer.echo(f"⏱️  Update interval: {update_interval}s")
    typer.echo(f"🔗 URL: http://{host}:{port}")
    typer.echo()
    typer.echo("Press Ctrl+C to stop...")

    try:
        asyncio.run(dashboard.start())
    except KeyboardInterrupt:
        typer.echo("\n👋 Dashboard stopped")


@benchmark.command()
def init_config(
    template: str = typer.Option("basic", "--template", help="Configuration template"),
    output: str = typer.Option("benchmark_config.yaml", "--output", help="Output configuration file"),
):
    """Initialize benchmark configuration file."""
    typer.echo(f"⚙️  Creating {template} configuration template")

    config = _create_config_template(template)

    output_path = Path(output)
    with open(output_path, "w") as f:
        if output_path.suffix.lower() == ".json":
            json.dump(config, f, indent=2)
        else:
            import yaml

            yaml.dump(config, f, default_flow_style=False)

    typer.echo(f"✅ Configuration saved to: {output}")
    typer.echo("📝 Edit the file and use --config-file flag to use it")


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

        typer.echo("\n✅ Benchmark completed successfully!")
        typer.echo(f"📊 Results saved to: {output_dir}")

        # Print summary
        _print_results_summary(results)

    except Exception as e:
        typer.echo(f"\n❌ Benchmark failed: {e}")
        raise


async def _validate_datasets(loader, datasets: List[str], max_samples: Optional[int]):
    """Validate datasets."""
    for dataset_name in datasets:
        typer.echo(f"🔍 Validating {dataset_name}...")

        try:
            dataset = await loader.load_dataset(dataset_name, max_samples=max_samples or 5)

            if hasattr(dataset, "__len__"):
                size = len(dataset)
                typer.echo(f"  ✅ Loaded {size} samples")
            else:
                typer.echo("  ✅ Dataset loaded successfully")

            # Basic validation
            if hasattr(dataset, "__iter__"):
                first_item = next(iter(dataset), None)
                if first_item:
                    typer.echo(
                        f"  📋 Sample keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Non-dict item'}"
                    )

        except Exception as e:
            typer.echo(f"  ❌ Failed: {e}")


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

    typer.echo(f"📄 Report saved: {report_file}")


def _print_results_summary(results: dict):
    """Print results summary."""
    typer.echo("\n📊 Results Summary:")
    typer.echo("-" * 30)

    # Performance analysis
    perf = results.get("performance_analysis", {})
    if perf and "samples_per_second" in perf:
        typer.echo(f"⚡ Throughput: {perf['samples_per_second']:.2f} samples/sec")
        typer.echo(f"⏱️  Avg Latency: {perf['average_latency']:.3f} sec")

    # Resource analysis
    resources = results.get("resource_analysis", {})
    if resources and "peak_memory_mb" in resources:
        typer.echo(f"💾 Peak Memory: {resources['peak_memory_mb']:.1f} MB")
        typer.echo(f"🔋 Avg CPU: {resources['avg_cpu_percent']:.1f}%")

    # Model results
    model_results = results.get("model_results", {})
    if model_results:
        typer.echo("\n🤖 Model Performance:")
        for model, metrics in model_results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                acc = metrics["accuracy"]
                if isinstance(acc, dict) and "mean" in acc:
                    typer.echo(f"  {model}: {acc['mean']:.3f} ± {acc.get('std', 0):.3f}")

    # Comparisons
    comparisons = results.get("comparison_results", {})
    if comparisons:
        baseline = results.get("metadata", {}).get("baseline", "baseline")
        typer.echo(f"\n📈 Improvements over {baseline}:")
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
                    typer.echo(f"  {model}: {avg_improvement:+.1f}%")


if __name__ == "__main__":
    benchmark()
