"""Results visualization functionality."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import BaseVisualizer


class ResultsVisualizer(BaseVisualizer):
    """Results-specific visualization."""

    def plot_metric_history(
        self,
        metrics: List[Dict[str, Any]],
        metric_name: str,
        group_by: Optional[str] = None,
        rolling_window: Optional[int] = None,
        **kwargs,
    ) -> Path:
        """Plot metric history.

        Args:
            metrics: List of metrics
            metric_name: Name of metric to plot
            group_by: Optional grouping field
            rolling_window: Window for moving average
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        # Prepare data
        df = pd.DataFrame(metrics)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if rolling_window:
            df[f"{metric_name}_ma"] = df[metric_name].rolling(rolling_window).mean()

        # Create figure
        fig = self.create_plotly_figure()

        if group_by:
            for group in df[group_by].unique():
                group_data = df[df[group_by] == group]

                # Raw values
                fig.add_trace(
                    go.Scatter(
                        x=group_data["timestamp"],
                        y=group_data[metric_name],
                        name=f"{group} (raw)",
                        mode="markers",
                        marker=dict(size=6),
                        opacity=0.5,
                    )
                )

                if rolling_window:
                    # Moving average
                    fig.add_trace(
                        go.Scatter(
                            x=group_data["timestamp"],
                            y=group_data[f"{metric_name}_ma"],
                            name=f"{group} (MA-{rolling_window})",
                            mode="lines",
                            line=dict(width=2),
                        )
                    )
        else:
            # Raw values
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[metric_name],
                    name="Raw values",
                    mode="markers",
                    marker=dict(size=6),
                    opacity=0.5,
                )
            )

            if rolling_window:
                # Moving average
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df[f"{metric_name}_ma"],
                        name=f"Moving Avg ({rolling_window})",
                        mode="lines",
                        line=dict(width=2),
                    )
                )

        # Update layout
        fig.update_layout(title=f"{metric_name} History", xaxis_title="Time", yaxis_title=metric_name, **kwargs)

        # Save plot
        paths = self.save_plotly_figure(fig, f"metric_history_{metric_name}")
        return paths["html"]

    def plot_metric_distribution(
        self, metrics: List[Dict[str, Any]], metric_name: str, group_by: Optional[str] = None, **kwargs
    ) -> Path:
        """Plot metric distribution.

        Args:
            metrics: List of metrics
            metric_name: Name of metric to plot
            group_by: Optional grouping field
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        df = pd.DataFrame(metrics)

        # Create figure
        fig = self.create_plotly_figure()

        if group_by:
            # Create violin plots for each group
            for group in df[group_by].unique():
                group_data = df[df[group_by] == group][metric_name]

                fig.add_trace(go.Violin(y=group_data, name=group, box_visible=True, meanline_visible=True))
        else:
            # Single violin plot
            fig.add_trace(go.Violin(y=df[metric_name], box_visible=True, meanline_visible=True))

        # Update layout
        fig.update_layout(title=f"{metric_name} Distribution", yaxis_title=metric_name, **kwargs)

        # Save plot
        paths = self.save_plotly_figure(fig, f"metric_distribution_{metric_name}")
        return paths["html"]

    def plot_metric_correlation(
        self, metrics: List[Dict[str, Any]], metric_x: str, metric_y: str, group_by: Optional[str] = None, **kwargs
    ) -> Path:
        """Plot correlation between metrics.

        Args:
            metrics: List of metrics
            metric_x: First metric name
            metric_y: Second metric name
            group_by: Optional grouping field
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        df = pd.DataFrame(metrics)

        # Create figure
        fig = self.create_plotly_figure()

        if group_by:
            for group in df[group_by].unique():
                group_data = df[df[group_by] == group]

                fig.add_trace(
                    go.Scatter(
                        x=group_data[metric_x],
                        y=group_data[metric_y],
                        name=group,
                        mode="markers",
                        marker=dict(size=8),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[metric_x],
                    y=df[metric_y],
                    mode="markers",
                    marker=dict(size=8),
                )
            )

        # Add trend line
        if not group_by:
            z = np.polyfit(df[metric_x], df[metric_y], 1)
            p = np.poly1d(z)

            x_range = np.linspace(df[metric_x].min(), df[metric_x].max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    name="Trend",
                    mode="lines",
                    line=dict(dash="dash"),
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Correlation: {metric_x} vs {metric_y}", xaxis_title=metric_x, yaxis_title=metric_y, **kwargs
        )

        # Save plot
        paths = self.save_plotly_figure(fig, f"metric_correlation_{metric_x}_{metric_y}")
        return paths["html"]

    def create_summary_dashboard(
        self, metrics: List[Dict[str, Any]], metric_names: List[str], group_by: Optional[str] = None, **kwargs
    ) -> Path:
        """Create summary dashboard with multiple plots.

        Args:
            metrics: List of metrics
            metric_names: Names of metrics to include
            group_by: Optional grouping field
            **kwargs: Additional parameters

        Returns:
            Path to saved dashboard
        """
        df = pd.DataFrame(metrics)

        # Create subplot grid
        n_metrics = len(metric_names)
        n_rows = (n_metrics + 1) // 2  # 2 columns

        fig = make_subplots(
            rows=n_rows, cols=2, subplot_titles=[f"{name} Summary" for name in metric_names], vertical_spacing=0.1
        )

        # Add plots for each metric
        for i, metric in enumerate(metric_names):
            row = i // 2 + 1
            col = i % 2 + 1

            if group_by:
                for group in df[group_by].unique():
                    group_data = df[df[group_by] == group]

                    # Time series
                    fig.add_trace(
                        go.Scatter(
                            x=group_data["timestamp"],
                            y=group_data[metric],
                            name=f"{group} - {metric}",
                            mode="lines+markers",
                            showlegend=True if i == 0 else False,
                        ),
                        row=row,
                        col=col,
                    )
            else:
                fig.add_trace(
                    go.Scatter(x=df["timestamp"], y=df[metric], name=metric, mode="lines+markers"), row=row, col=col
                )

        # Update layout
        fig.update_layout(height=300 * n_rows, title="Metrics Summary Dashboard", showlegend=True, **kwargs)

        # Save dashboard
        paths = self.save_plotly_figure(fig, "metrics_dashboard", formats=["html"])
        return paths["html"]

    def generate_report(
        self, metrics: List[Dict[str, Any]], metric_names: List[str], group_by: Optional[str] = None, **kwargs
    ) -> Path:
        """Generate comprehensive HTML report.

        Args:
            metrics: List of metrics
            metric_names: Names of metrics to include
            group_by: Optional grouping field
            **kwargs: Additional parameters

        Returns:
            Path to saved report
        """
        # Generate all plots
        plot_paths = {}

        # History plots
        for metric in metric_names:
            plot_paths[f"{metric}_history"] = self.plot_metric_history(metrics, metric, group_by=group_by)

            plot_paths[f"{metric}_distribution"] = self.plot_metric_distribution(metrics, metric, group_by=group_by)

        # Correlation matrix
        for i, metric_x in enumerate(metric_names):
            for metric_y in metric_names[i + 1 :]:
                plot_paths[f"correlation_{metric_x}_{metric_y}"] = self.plot_metric_correlation(
                    metrics, metric_x, metric_y, group_by=group_by
                )

        # Dashboard
        plot_paths["dashboard"] = self.create_summary_dashboard(metrics, metric_names, group_by=group_by)

        # Calculate statistics
        stats = {}
        df = pd.DataFrame(metrics)

        for metric in metric_names:
            if group_by:
                group_stats = {}
                for group in df[group_by].unique():
                    group_data = df[df[group_by] == group][metric]
                    group_stats[group] = {
                        "mean": group_data.mean(),
                        "std": group_data.std(),
                        "min": group_data.min(),
                        "max": group_data.max(),
                        "median": group_data.median(),
                    }
                stats[metric] = group_stats
            else:
                stats[metric] = {
                    "mean": df[metric].mean(),
                    "std": df[metric].std(),
                    "min": df[metric].min(),
                    "max": df[metric].max(),
                    "median": df[metric].median(),
                }

        # Render report template
        report_path = self.render_template(
            "results_report.html",
            "results_report.html",
            metrics=metrics,
            metric_names=metric_names,
            group_by=group_by,
            plot_paths=plot_paths,
            stats=stats,
            **kwargs,
        )

        return report_path
