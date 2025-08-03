"""
Command-line interface for the Memento framework.

This module provides the main CLI entry point for running experiments,
managing the framework, and accessing various features.
"""

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..config import get_settings
from ..core.collector import FeedbackCollector
from ..core.learner import PromptLearner
from ..core.processor import PromptProcessor
from ..utils.logger import get_logger, setup_logger
from .comprehensive_benchmark import benchmark

# Create Typer app
app = typer.Typer(
    name="memento",
    help=("A Meta-Cognitive Framework for Self-Evolving System Prompts in AI Systems"),
    add_completion=False,
)

# Rich console for better output
console = Console()


@app.command()
def version():
    """Show version information."""
    from .. import __author__, __email__, __version__

    table = Table(title="Memento Framework")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Author", __author__)
    table.add_row("Email", __email__)

    console.print(table)


@app.command()
def init(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model to use"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Initialize the Memento framework."""
    try:
        # Setup logging
        logger = setup_logger("memento", level=log_level)
        logger.info("Initializing Memento framework")

        # Get settings
        settings = get_settings()

        # Create necessary directories
        for path in [
            settings.storage.base_path,
            settings.storage.feedback_path,
            settings.storage.evolution_path,
            settings.storage.logs_path,
            settings.storage.cache_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")

        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing components...", total=3)

            # Initialize learner
            _ = PromptLearner(model=model)
            progress.update(task, advance=1)

            # Initialize collector
            _ = FeedbackCollector(model=model)
            progress.update(task, advance=1)

            # Initialize processor
            _ = PromptProcessor(model=model)
            progress.update(task, advance=1)

        console.print("✅ Memento framework initialized successfully!", style="green")
        logger.info("Framework initialization completed")

    except Exception as e:
        console.print(f"❌ Error initializing framework: {e}", style="red")
        logger.error(f"Initialization failed: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    problem_file: Path = typer.Argument(..., help="Path to problem file"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Model to use"),
    iterations: int = typer.Option(10, "--iterations", "-i", help="Number of iterations"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Run a learning experiment."""
    try:
        logger = get_logger("memento")
        logger.info(f"Starting experiment with {iterations} iterations")

        # Load problem
        if not problem_file.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_file}")

        with open(problem_file, "r") as f:
            _ = f.read()

        # Initialize components
        _ = PromptLearner(model=model)
        _ = FeedbackCollector(model=model)
        _ = PromptProcessor(model=model)

        # Run experiment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running experiment...", total=iterations)

            for i in range(iterations):
                progress.update(task, description=f"Iteration {i+1}/{iterations}")

                # TODO: Implement actual experiment logic
                # This is a placeholder for the learning cycle

                progress.update(task, advance=1)

        console.print("✅ Experiment completed successfully!", style="green")
        logger.info("Experiment completed")

    except Exception as e:
        console.print(f"❌ Error running experiment: {e}", style="red")
        logger.error(f"Experiment failed: {e}")
        raise typer.Exit(1)


@app.command()
def legacy_benchmark(
    dataset_path: Path = typer.Argument(..., help="Path to benchmark dataset"),
    models: list[str] = typer.Option(
        ["memento", "promptbreeder", "self-evolving-gpt", "auto-evolve"],
        "--models",
        "-m",
        help="Models to benchmark",
    ),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Run legacy benchmark comparison."""
    try:
        logger = get_logger("memento")
        logger.info(f"Starting legacy benchmark with models: {models}")

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # TODO: Implement benchmark logic
        console.print("🔄 Legacy benchmark functionality coming soon...", style="yellow")
        logger.info("Legacy benchmark completed")

    except Exception as e:
        console.print(f"❌ Error running legacy benchmark: {e}", style="red")
        logger.error(f"Legacy benchmark failed: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """Show framework status."""
    try:
        settings = get_settings()

        table = Table(title="Memento Framework Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Path", style="blue")

        # Check directories
        for name, path in [
            ("Base Data", settings.storage.base_path),
            ("Feedback", settings.storage.feedback_path),
            ("Evolution", settings.storage.evolution_path),
            ("Logs", settings.storage.logs_path),
            ("Cache", settings.storage.cache_path),
        ]:
            status = "✅ Exists" if path.exists() else "❌ Missing"
            table.add_row(name, status, str(path))

        # Check model configuration
        model_status = "✅ Configured" if settings.model.model_name else "❌ Not configured"
        table.add_row("Model", model_status, settings.model.model_name)

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error checking status: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def clean(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clean framework data and cache."""
    try:
        settings = get_settings()

        if not confirm:
            console.print("⚠️  This will delete all framework data and cache files.")
            if not typer.confirm("Are you sure you want to continue?"):
                console.print("Operation cancelled.")
                return

        # Clean directories
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Cleaning framework data...", total=4)

            # Clean feedback data
            if settings.storage.feedback_path.exists():
                for file in settings.storage.feedback_path.glob("*"):
                    file.unlink()
            progress.update(task, advance=1)

            # Clean evolution data
            if settings.storage.evolution_path.exists():
                for file in settings.storage.evolution_path.glob("*"):
                    file.unlink()
            progress.update(task, advance=1)

            # Clean cache
            if settings.storage.cache_path.exists():
                for file in settings.storage.cache_path.glob("*"):
                    file.unlink()
            progress.update(task, advance=1)

            # Clean logs (keep recent ones)
            if settings.storage.logs_path.exists():
                for file in settings.storage.logs_path.glob("*.log"):
                    if file.stat().st_mtime < (time.time() - 7 * 24 * 3600):  # 7 days
                        file.unlink()
            progress.update(task, advance=1)

        console.print("✅ Framework data cleaned successfully!", style="green")

    except Exception as e:
        console.print(f"❌ Error cleaning data: {e}", style="red")
        raise typer.Exit(1)


# Add benchmark subcommand group
app.add_typer(benchmark, name="benchmark", help="Comprehensive benchmarking commands")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
