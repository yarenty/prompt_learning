"""Comparison visualization functionality."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .base import BaseVisualizer


class ComparisonVisualizer(BaseVisualizer):
    """Comparison visualization for benchmarking."""

    def plot_method_comparison(
        self,
        results: Dict[str, List[float]],
        metric: str,
        baseline: Optional[str] = None,
        confidence: float = 0.95,
        **kwargs,
    ) -> Path:
        """Plot method comparison boxplots.

        Args:
            results: Dictionary mapping method names to lists of scores
            metric: Name of the metric being compared
            baseline: Optional baseline method for comparison
            confidence: Confidence level for statistical tests
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        # Create figure
        fig = self.create_plotly_figure()

        # Prepare data for plotting
        data = []
        for method, scores in results.items():
            for score in scores:
                data.append({"Method": method, "Score": score})

        df = pd.DataFrame(data)

        # Create box plots
        fig.add_trace(go.Box(x=df["Method"], y=df["Score"], boxpoints="all", jitter=0.3, pointpos=-1.8, name="Scores"))

        # Add statistical annotations
        if baseline and baseline in results:
            baseline_scores = results[baseline]
            y_max = df["Score"].max()

            for method, scores in results.items():
                if method != baseline:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(baseline_scores, scores)

                    # Calculate effect size (Cohen's d)
                    d = (np.mean(scores) - np.mean(baseline_scores)) / np.sqrt(
                        ((len(scores) - 1) * np.var(scores) + (len(baseline_scores) - 1) * np.var(baseline_scores))
                        / (len(scores) + len(baseline_scores) - 2)
                    )

                    # Add annotation
                    annotation = f"p={p_value:.3f}\nd={d:.2f}"
                    if p_value < 0.001:
                        annotation = "***\n" + annotation
                    elif p_value < 0.01:
                        annotation = "**\n" + annotation
                    elif p_value < 0.05:
                        annotation = "*\n" + annotation

                    fig.add_annotation(x=method, y=y_max * 1.1, text=annotation, showarrow=False, font=dict(size=10))

        # Update layout
        fig.update_layout(
            title=f"Method Comparison: {metric}", xaxis_title="Method", yaxis_title=metric, showlegend=False, **kwargs
        )

        # Save plot
        paths = self.save_plotly_figure(fig, f"method_comparison_{metric}")
        return paths["html"]

    def plot_improvement_heatmap(
        self, baseline_results: Dict[str, Dict[str, float]], improved_results: Dict[str, Dict[str, float]], **kwargs
    ) -> Path:
        """Plot improvement heatmap.

        Args:
            baseline_results: Baseline results {dataset: {metric: score}}
            improved_results: Improved results {dataset: {metric: score}}
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        # Calculate improvements
        improvements = {}
        for dataset in baseline_results:
            if dataset in improved_results:
                improvements[dataset] = {}
                for metric in baseline_results[dataset]:
                    if metric in improved_results[dataset]:
                        baseline = baseline_results[dataset][metric]
                        improved = improved_results[dataset][metric]

                        if baseline != 0:
                            improvement = ((improved - baseline) / baseline) * 100
                        else:
                            improvement = np.inf if improved > 0 else 0

                        improvements[dataset][metric] = improvement

        # Convert to DataFrame
        df = pd.DataFrame(improvements).T

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale="RdYlGn",
                text=np.round(df.values, 1),
                texttemplate="%{text}%",
                textfont={"size": 10},
                colorbar=dict(title="Improvement (%)", titleside="right"),
            )
        )

        # Update layout
        fig.update_layout(
            title="Performance Improvement Heatmap", xaxis_title="Metric", yaxis_title="Dataset", **kwargs
        )

        # Save plot
        paths = self.save_plotly_figure(fig, "improvement_heatmap")
        return paths["html"]

    def plot_statistical_tests(
        self, method_a: List[float], method_b: List[float], name_a: str = "Method A", name_b: str = "Method B", **kwargs
    ) -> Path:
        """Plot statistical test results.

        Args:
            method_a: Scores from first method
            method_b: Scores from second method
            name_a: Name of first method
            name_b: Name of second method
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        # Perform statistical tests
        t_stat, p_value = stats.ttest_ind(method_a, method_b)

        # Calculate effect size
        d = (np.mean(method_b) - np.mean(method_a)) / np.sqrt(
            ((len(method_b) - 1) * np.var(method_b) + (len(method_a) - 1) * np.var(method_a))
            / (len(method_b) + len(method_a) - 2)
        )

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=["Score Distribution", "Effect Size", "P-Value", "Statistical Power"]
        )

        # Distribution plot
        fig.add_trace(go.Violin(y=method_a, name=name_a, box_visible=True, meanline_visible=True), row=1, col=1)

        fig.add_trace(go.Violin(y=method_b, name=name_b, box_visible=True, meanline_visible=True), row=1, col=1)

        # Effect size plot
        effect_sizes = [-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8]
        labels = ["Large-", "Medium-", "Small-", "No Effect", "Small+", "Medium+", "Large+"]

        fig.add_trace(
            go.Scatter(
                x=effect_sizes,
                y=[0] * len(effect_sizes),
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=10, color="gray"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[d],
                y=[0],
                mode="markers",
                marker=dict(size=15, color=self.colors["primary"]),
                name="Observed",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # P-value plot
        thresholds = [0.001, 0.01, 0.05, 0.1]
        colors = ["darkred", "red", "orange", "yellow"]

        for i, (threshold, color) in enumerate(zip(thresholds, colors)):
            fig.add_shape(
                type="rect",
                x0=0,
                x1=threshold,
                y0=0,
                y1=1,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0,
                row=2,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=[p_value],
                y=[0.5],
                mode="markers",
                marker=dict(size=15, color=self.colors["primary"]),
                name="P-Value",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Power analysis
        sample_sizes = np.linspace(10, 200, 50)
        power = []

        for n in sample_sizes:
            # Calculate power for current sample size
            power.append(stats.power.TTestIndPower().power(effect_size=abs(d), nobs=n, alpha=0.05))

        fig.add_trace(
            go.Scatter(
                x=sample_sizes, y=power, mode="lines", line=dict(color=self.colors["primary"]), name="Power Curve"
            ),
            row=2,
            col=2,
        )

        # Add current sample size point
        current_power = stats.power.TTestIndPower().power(
            effect_size=abs(d), nobs=min(len(method_a), len(method_b)), alpha=0.05
        )

        fig.add_trace(
            go.Scatter(
                x=[min(len(method_a), len(method_b))],
                y=[current_power],
                mode="markers",
                marker=dict(size=15, color=self.colors["accent"]),
                name="Current",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(title="Statistical Analysis", showlegend=True, height=800, **kwargs)

        # Update axes
        fig.update_xaxes(title_text="Effect Size", row=1, col=2)
        fig.update_xaxes(title_text="P-Value", row=2, col=1)
        fig.update_xaxes(title_text="Sample Size", row=2, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=2)

        # Save plot
        paths = self.save_plotly_figure(fig, "statistical_analysis")
        return paths["html"]

    def create_comparison_report(
        self, results: Dict[str, Dict[str, List[float]]], baseline: str, metrics: List[str], **kwargs
    ) -> Path:
        """Generate comprehensive comparison report.

        Args:
            results: Results {dataset: {method: scores}}
            baseline: Baseline method name
            metrics: List of metrics to include
            **kwargs: Additional parameters

        Returns:
            Path to saved report
        """
        plot_paths = {}

        # Method comparison plots
        for metric in metrics:
            metric_results = {method: scores[metric] for method, scores in results.items() if metric in scores}

            plot_paths[f"comparison_{metric}"] = self.plot_method_comparison(metric_results, metric, baseline=baseline)

        # Improvement heatmap
        baseline_results = {dataset: results[dataset][baseline] for dataset in results if baseline in results[dataset]}

        for method in results:
            if method != baseline:
                improved_results = {
                    dataset: results[dataset][method] for dataset in results if method in results[dataset]
                }

                plot_paths[f"heatmap_{method}"] = self.plot_improvement_heatmap(baseline_results, improved_results)

        # Statistical tests
        for method in results:
            if method != baseline:
                for metric in metrics:
                    if metric in results[baseline] and metric in results[method]:
                        plot_paths[f"stats_{method}_{metric}"] = self.plot_statistical_tests(
                            results[baseline][metric], results[method][metric], name_a=baseline, name_b=method
                        )

        # Generate report
        report_path = self.render_template(
            "comparison_report.html",
            "comparison_report.html",
            results=results,
            baseline=baseline,
            metrics=metrics,
            plot_paths=plot_paths,
            **kwargs,
        )

        return report_path
