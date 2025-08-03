"""Visualization tools for benchmarking results."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader


class BenchmarkVisualizer:
    """Visualizer for benchmark results."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir or "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load HTML templates
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def plot_performance_comparison(self, results: Dict, metrics: Optional[List[str]] = None) -> None:
        """Plot performance comparison across models.

        Args:
            results: Benchmark results
            metrics: Metrics to plot
        """
        # Extract metrics
        metrics = metrics or ["correctness", "efficiency", "quality"]

        # Prepare data
        data = []
        for dataset_model, scores in results.items():
            dataset, model = dataset_model.split("_")
            for metric in metrics:
                if metric in scores:
                    data.append({"Dataset": dataset, "Model": model, "Metric": metric, "Score": scores[metric]})

        df = pd.DataFrame(data)

        # Create grouped bar plot
        fig = px.bar(
            df,
            x="Dataset",
            y="Score",
            color="Model",
            barmode="group",
            facet_col="Metric",
            title="Model Performance Comparison",
        )

        # Save plot
        fig.write_html(Path(self.output_dir) / "performance_comparison.html")

    def plot_learning_curves(self, results: Dict, metric: str = "correctness") -> None:
        """Plot learning curves.

        Args:
            results: Benchmark results with history
            metric: Metric to plot
        """
        # Prepare data
        data = []
        for dataset_model, scores in results.items():
            if "history" in scores:
                dataset, model = dataset_model.split("_")
                for step, score in enumerate(scores["history"][metric]):
                    data.append({"Dataset": dataset, "Model": model, "Step": step, "Score": score})

        df = pd.DataFrame(data)

        # Create line plot
        fig = px.line(df, x="Step", y="Score", color="Model", facet_col="Dataset", title=f"Learning Curves - {metric}")

        # Save plot
        fig.write_html(Path(self.output_dir) / "learning_curves.html")

    def plot_error_analysis(self, results: Dict) -> None:
        """Plot error analysis.

        Args:
            results: Benchmark results with error info
        """
        # Prepare data
        data = []
        for dataset_model, scores in results.items():
            if "errors" in scores:
                dataset, model = dataset_model.split("_")
                for error_type, count in scores["errors"].items():
                    data.append({"Dataset": dataset, "Model": model, "Error": error_type, "Count": count})

        df = pd.DataFrame(data)

        # Create stacked bar plot
        fig = px.bar(df, x="Dataset", y="Count", color="Error", facet_col="Model", title="Error Analysis")

        # Save plot
        fig.write_html(Path(self.output_dir) / "error_analysis.html")

    def generate_report(self, results: Dict, config: Optional[Dict] = None) -> str:
        """Generate HTML report.

        Args:
            results: Benchmark results
            config: Benchmark configuration

        Returns:
            Path to generated report
        """
        # Load template
        template = self.env.get_template("report.html")

        # Prepare data
        report_data = {
            "results": results,
            "config": config or {},
            "summary": self._generate_summary(results),
            "plots": {
                "performance": "performance_comparison.html",
                "learning": "learning_curves.html",
                "errors": "error_analysis.html",
            },
        }

        # Generate report
        report_path = Path(self.output_dir) / "report.html"
        with open(report_path, "w") as f:
            f.write(template.render(**report_data))

        return str(report_path)

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate results summary."""
        summary = {
            "datasets": len(set(k.split("_")[0] for k in results)),
            "models": len(set(k.split("_")[1] for k in results)),
            "best_model": {},
            "metrics": {},
        }

        # Calculate metrics
        for dataset_model, scores in results.items():
            dataset, model = dataset_model.split("_")

            for metric, score in scores.items():
                if isinstance(score, (int, float)):
                    if metric not in summary["metrics"]:
                        summary["metrics"][metric] = {"min": score, "max": score, "avg": [score]}
                    else:
                        summary["metrics"][metric]["min"] = min(summary["metrics"][metric]["min"], score)
                        summary["metrics"][metric]["max"] = max(summary["metrics"][metric]["max"], score)
                        summary["metrics"][metric]["avg"].append(score)

        # Calculate averages
        for metric in summary["metrics"]:
            scores = summary["metrics"][metric]["avg"]
            summary["metrics"][metric]["avg"] = np.mean(scores)

            # Find best model
            best_score = summary["metrics"][metric]["max"]
            for dataset_model, scores in results.items():
                if scores.get(metric) == best_score:
                    dataset, model = dataset_model.split("_")
                    summary["best_model"][metric] = {"model": model, "dataset": dataset, "score": best_score}

        return summary
