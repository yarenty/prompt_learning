"""Statistical visualization functionality."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm

from .base import BaseVisualizer


class StatisticalVisualizer(BaseVisualizer):
    """Statistical analysis visualization."""

    def plot_confidence_intervals(
        self,
        data: Dict[str, List[float]],
        confidence_levels: List[float] = [0.90, 0.95, 0.99],
        n_bootstrap: int = 10000,
        **kwargs,
    ) -> Path:
        """Plot confidence intervals at multiple levels.

        Args:
            data: Dictionary mapping names to lists of values
            confidence_levels: List of confidence levels to plot
            n_bootstrap: Number of bootstrap samples
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        # Create figure
        fig = self.create_plotly_figure()

        # Calculate intervals for each dataset
        for name, values in data.items():
            # Bootstrap sampling
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))

            # Calculate intervals for each confidence level
            intervals = []
            for level in confidence_levels:
                lower = np.percentile(bootstrap_means, (1 - level) * 100 / 2)
                upper = np.percentile(bootstrap_means, (1 + level) * 100 / 2)
                intervals.append((lower, upper))

            # Plot mean and intervals
            mean = np.mean(values)

            # Add mean point
            fig.add_trace(
                go.Scatter(
                    x=[mean],
                    y=[name],
                    mode="markers",
                    marker=dict(size=10, color=self.colors["primary"]),
                    name=f"{name} (mean)",
                    showlegend=False,
                )
            )

            # Add intervals
            for (lower, upper), level in zip(intervals, confidence_levels):
                fig.add_trace(
                    go.Scatter(
                        x=[lower, upper],
                        y=[name, name],
                        mode="lines",
                        line=dict(
                            width=4, color=self.colors["primary"], opacity=0.2 + 0.2 * confidence_levels.index(level)
                        ),
                        name=f"{level*100:.0f}% CI",
                        showlegend=True,
                    )
                )

        # Update layout
        fig.update_layout(title="Confidence Intervals Analysis", xaxis_title="Value", yaxis_title="Dataset", **kwargs)

        # Save plot
        paths = self.save_plotly_figure(fig, "confidence_intervals")
        return paths["html"]

    def plot_effect_sizes(
        self, control_data: Dict[str, List[float]], treatment_data: Dict[str, List[float]], **kwargs
    ) -> Path:
        """Plot effect size analysis with interpretations.

        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Effect Size Comparison",
                "Effect Size Interpretation",
                "Sample Size Analysis",
                "Power Analysis",
            ],
        )

        # Calculate effect sizes
        effect_sizes = {}
        for name in control_data:
            if name in treatment_data:
                control = control_data[name]
                treatment = treatment_data[name]

                # Calculate Cohen's d
                d = (np.mean(treatment) - np.mean(control)) / np.sqrt(
                    ((len(treatment) - 1) * np.var(treatment) + (len(control) - 1) * np.var(control))
                    / (len(treatment) + len(control) - 2)
                )
                effect_sizes[name] = d

        # Effect size comparison
        names = list(effect_sizes.keys())
        sizes = list(effect_sizes.values())

        fig.add_trace(go.Bar(x=names, y=sizes, marker_color=self.colors["primary"]), row=1, col=1)

        # Effect size interpretation
        thresholds = [-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8]
        labels = ["Large-", "Medium-", "Small-", "No Effect", "Small+", "Medium+", "Large+"]

        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=[0] * len(thresholds),
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=10, color="gray"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        for name, size in effect_sizes.items():
            fig.add_trace(go.Scatter(x=[size], y=[0], mode="markers", marker=dict(size=15), name=name), row=1, col=2)

        # Sample size analysis
        min_n = 10
        max_n = 200
        ns = np.linspace(min_n, max_n, 50)

        for name, d in effect_sizes.items():
            powers = []
            for n in ns:
                power = stats.power.TTestIndPower().power(effect_size=abs(d), nobs=n, alpha=0.05)
                powers.append(power)

            fig.add_trace(go.Scatter(x=ns, y=powers, name=name, mode="lines"), row=2, col=1)

        # Add power threshold
        fig.add_shape(
            type="line", x0=min_n, x1=max_n, y0=0.8, y1=0.8, line=dict(color="red", dash="dash"), row=2, col=1
        )

        # Power analysis
        effect_range = np.linspace(0.1, 1.0, 50)
        sample_sizes = [20, 50, 100, 200]

        for n in sample_sizes:
            powers = []
            for d in effect_range:
                power = stats.power.TTestIndPower().power(effect_size=d, nobs=n, alpha=0.05)
                powers.append(power)

            fig.add_trace(go.Scatter(x=effect_range, y=powers, name=f"n={n}", mode="lines"), row=2, col=2)

        # Update layout
        fig.update_xaxes(title_text="Effect Size", row=1, col=1)
        fig.update_xaxes(title_text="Effect Size", row=1, col=2)
        fig.update_xaxes(title_text="Sample Size", row=2, col=1)
        fig.update_xaxes(title_text="Effect Size", row=2, col=2)

        fig.update_yaxes(title_text="Cohen's d", row=1, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=2)

        fig.update_layout(height=800, title="Effect Size Analysis", showlegend=True, **kwargs)

        # Save plot
        paths = self.save_plotly_figure(fig, "effect_size_analysis")
        return paths["html"]

    def plot_power_analysis(
        self, effect_size: float, alpha: float = 0.05, power_target: float = 0.80, **kwargs
    ) -> Path:
        """Plot power analysis and sample size estimation.

        Args:
            effect_size: Expected effect size
            alpha: Significance level
            power_target: Target power level
            **kwargs: Additional plot parameters

        Returns:
            Path to saved plot
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Power vs Sample Size",
                "Required Sample Size",
                "Power vs Effect Size",
                "Operating Characteristics",
            ],
        )

        # Power vs Sample Size
        ns = np.linspace(10, 200, 50)
        powers = []

        for n in ns:
            power = stats.power.TTestIndPower().power(effect_size=effect_size, nobs=n, alpha=alpha)
            powers.append(power)

        fig.add_trace(
            go.Scatter(x=ns, y=powers, mode="lines", line=dict(color=self.colors["primary"]), name="Power Curve"),
            row=1,
            col=1,
        )

        # Add power target line
        fig.add_shape(
            type="line",
            x0=10,
            x1=200,
            y0=power_target,
            y1=power_target,
            line=dict(color="red", dash="dash"),
            row=1,
            col=1,
        )

        # Required Sample Size
        required_n = stats.power.TTestIndPower().solve_power(effect_size=effect_size, power=power_target, alpha=alpha)

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=required_n,
                title="Required Sample Size",
                number=dict(font=dict(size=50), valueformat=".0f"),
            ),
            row=1,
            col=2,
        )

        # Power vs Effect Size
        effect_range = np.linspace(0.1, 1.0, 50)
        sample_sizes = [20, 50, 100, 200]

        for n in sample_sizes:
            powers = []
            for d in effect_range:
                power = stats.power.TTestIndPower().power(effect_size=d, nobs=n, alpha=alpha)
                powers.append(power)

            fig.add_trace(go.Scatter(x=effect_range, y=powers, name=f"n={n}", mode="lines"), row=2, col=1)

        # Operating Characteristics
        alphas = [0.01, 0.05, 0.10]
        powers = []

        for a in alphas:
            power = stats.power.TTestIndPower().power(effect_size=effect_size, nobs=required_n, alpha=a)
            powers.append(power)

        fig.add_trace(
            go.Bar(x=[f"α={a}" for a in alphas], y=powers, marker_color=self.colors["primary"], name="Power"),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="Sample Size", row=1, col=1)
        fig.update_xaxes(title_text="Effect Size", row=2, col=1)
        fig.update_xaxes(title_text="Significance Level", row=2, col=2)

        fig.update_yaxes(title_text="Power", row=1, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=2)

        fig.update_layout(height=800, title="Power Analysis", showlegend=True, **kwargs)

        # Save plot
        paths = self.save_plotly_figure(fig, "power_analysis")
        return paths["html"]
