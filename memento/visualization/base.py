"""Base visualization functionality."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class BaseVisualizer:
    """Base visualization functionality."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        theme: str = "professional",
        template_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize visualizer.

        Args:
            output_dir: Directory for saving visualizations
            theme: Visual theme to use
            template_dir: Directory containing templates
        """
        self.output_dir = Path(output_dir) if output_dir else Path("visualization_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        self.data_dir = self.output_dir / "data"

        for directory in [self.plots_dir, self.reports_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Set up template environment
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            self.template_dir = Path(__file__).parent / "templates"

        self.env = Environment(loader=FileSystemLoader(self.template_dir), autoescape=True)

        # Set up theme
        self.set_theme(theme)

    def set_theme(self, theme: str) -> None:
        """Set visualization theme.

        Args:
            theme: Theme name
        """
        if theme == "professional":
            plt.style.use("seaborn-v0_8-whitegrid")
            self.colors = {
                "primary": "#2E8B57",  # Sea Green
                "secondary": "#4169E1",  # Royal Blue
                "accent": "#DC143C",  # Crimson
                "neutral": "#696969",  # Dim Gray
                "highlight": "#FFD700",  # Gold
            }
            self.figure_size = (12, 8)
            self.dpi = 300
            self.font_size = 11

            plt.rcParams.update(
                {
                    "figure.figsize": self.figure_size,
                    "figure.dpi": self.dpi,
                    "font.size": self.font_size,
                    "axes.labelsize": 12,
                    "axes.titlesize": 14,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                }
            )

        elif theme == "dark":
            plt.style.use("dark_background")
            self.colors = {
                "primary": "#00FF7F",  # Spring Green
                "secondary": "#4169E1",  # Royal Blue
                "accent": "#FF4500",  # Orange Red
                "neutral": "#A9A9A9",  # Dark Gray
                "highlight": "#FFD700",  # Gold
            }
            # Dark theme settings...

        else:
            raise ValueError(f"Unknown theme: {theme}")

    def create_figure(
        self, width: Optional[int] = None, height: Optional[int] = None, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create new figure with current theme.

        Args:
            width: Figure width
            height: Figure height
            **kwargs: Additional figure parameters

        Returns:
            Figure and axes objects
        """
        size = (width or self.figure_size[0], height or self.figure_size[1])
        return plt.subplots(figsize=size, **kwargs)

    def save_figure(self, fig: plt.Figure, name: str, formats: List[str] = ["png", "pdf"], **kwargs) -> Dict[str, Path]:
        """Save figure in multiple formats.

        Args:
            fig: Figure to save
            name: Base name for files
            formats: Output formats
            **kwargs: Additional save parameters

        Returns:
            Dictionary mapping formats to file paths
        """
        paths = {}
        for fmt in formats:
            path = self.plots_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight", **kwargs)
            paths[fmt] = path

        plt.close(fig)
        return paths

    def create_plotly_figure(self, **kwargs) -> go.Figure:
        """Create Plotly figure with current theme.

        Args:
            **kwargs: Figure parameters

        Returns:
            Plotly figure
        """
        fig = go.Figure(**kwargs)

        # Apply theme
        fig.update_layout(
            font=dict(size=self.font_size),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=40, b=40, l=40, r=40),
            showlegend=True,
        )

        return fig

    def save_plotly_figure(
        self, fig: go.Figure, name: str, formats: List[str] = ["html", "json"], **kwargs
    ) -> Dict[str, Path]:
        """Save Plotly figure.

        Args:
            fig: Figure to save
            name: Base name for files
            formats: Output formats
            **kwargs: Additional save parameters

        Returns:
            Dictionary mapping formats to file paths
        """
        paths = {}
        for fmt in formats:
            path = self.plots_dir / f"{name}.{fmt}"

            if fmt == "html":
                fig.write_html(path, **kwargs)
            elif fmt == "json":
                fig.write_json(path, **kwargs)
            else:
                fig.write_image(path, **kwargs)

            paths[fmt] = path

        return paths

    def render_template(self, template_name: str, output_name: str, **context) -> Path:
        """Render template to file.

        Args:
            template_name: Name of template file
            output_name: Name for output file
            **context: Template context

        Returns:
            Path to output file
        """
        template = self.env.get_template(template_name)
        output = template.render(**context)

        output_path = self.reports_dir / output_name
        with open(output_path, "w") as f:
            f.write(output)

        return output_path

    def save_data(self, data: Any, name: str, format: str = "json") -> Path:
        """Save data to file.

        Args:
            data: Data to save
            name: File name
            format: Output format

        Returns:
            Path to saved file
        """
        path = self.data_dir / f"{name}.{format}"

        if format == "json":
            pd.DataFrame(data).to_json(path)
        elif format == "csv":
            pd.DataFrame(data).to_csv(path, index=False)
        elif format == "parquet":
            pd.DataFrame(data).to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return path
