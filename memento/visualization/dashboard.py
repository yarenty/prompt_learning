"""Real-time monitoring dashboard."""

from datetime import datetime
from typing import Dict, List

import dash
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots


class DashboardServer:
    """Real-time monitoring dashboard."""

    def __init__(
        self, host: str = "localhost", port: int = 8050, update_interval: float = 1.0, history_size: int = 100
    ):
        """Initialize dashboard server.

        Args:
            host: Server host
            port: Server port
            update_interval: Update interval in seconds
            history_size: Number of data points to keep
        """
        self.app = dash.Dash(__name__)
        self.host = host
        self.port = port
        self.update_interval = update_interval
        self.history_size = history_size

        # Initialize data storage
        self.metrics: Dict[str, List[float]] = {}
        self.timestamps: List[datetime] = []
        self.resources: Dict[str, List[float]] = {}
        self.comparisons: Dict[str, Dict[str, List[float]]] = {}

        # Setup callbacks
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div(
            [
                # Header
                html.H1("Memento Benchmark Monitor", style={"textAlign": "center"}),
                # Metrics Section
                html.Div(
                    [
                        html.H2("Performance Metrics"),
                        dcc.Graph(id="metrics-live"),
                        dcc.Checklist(
                            id="metric-selector", options=[], value=[], inline=True  # Will be updated dynamically
                        ),
                    ]
                ),
                # Resources Section
                html.Div(
                    [
                        html.H2("Resource Usage"),
                        dcc.Graph(id="resources-live"),
                        dcc.Checklist(
                            id="resource-selector",
                            options=[
                                {"label": "Memory", "value": "memory"},
                                {"label": "CPU", "value": "cpu"},
                                {"label": "GPU", "value": "gpu"},
                            ],
                            value=["memory", "cpu"],
                            inline=True,
                        ),
                    ]
                ),
                # Comparison Section
                html.Div(
                    [
                        html.H2("Model Comparison"),
                        dcc.Graph(id="comparison-live"),
                        dcc.Dropdown(
                            id="comparison-metric",
                            options=[],  # Will be updated dynamically
                            placeholder="Select metric",
                        ),
                    ]
                ),
                # Auto-refresh
                dcc.Interval(
                    id="interval-component",
                    interval=self.update_interval * 1000,  # Convert to milliseconds
                    n_intervals=0,
                ),
            ]
        )

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [
                dash.Output("metrics-live", "figure"),
                dash.Output("metric-selector", "options"),
                dash.Output("metric-selector", "value"),
            ],
            [dash.Input("interval-component", "n_intervals"), dash.Input("metric-selector", "value")],
        )
        def update_metrics(n_intervals, selected_metrics):
            # Update available metrics
            available_metrics = list(self.metrics.keys())
            options = [{"label": m, "value": m} for m in available_metrics]

            # Use all metrics if none selected
            if not selected_metrics:
                selected_metrics = available_metrics

            # Create figure
            fig = go.Figure()

            for metric in selected_metrics:
                if metric in self.metrics:
                    fig.add_trace(
                        go.Scatter(x=self.timestamps, y=self.metrics[metric], name=metric, mode="lines+markers")
                    )

            fig.update_layout(
                title="Performance Metrics Over Time", xaxis_title="Time", yaxis_title="Value", height=400
            )

            return fig, options, selected_metrics

        @self.app.callback(
            dash.Output("resources-live", "figure"),
            [dash.Input("interval-component", "n_intervals"), dash.Input("resource-selector", "value")],
        )
        def update_resources(n_intervals, selected_resources):
            fig = make_subplots(rows=len(selected_resources), cols=1)

            for i, resource in enumerate(selected_resources, 1):
                if resource in self.resources:
                    fig.add_trace(
                        go.Scatter(
                            x=self.timestamps, y=self.resources[resource], name=resource.upper(), fill="tozeroy"
                        ),
                        row=i,
                        col=1,
                    )

            fig.update_layout(title="Resource Usage", height=300 * len(selected_resources), showlegend=True)

            return fig

        @self.app.callback(
            [dash.Output("comparison-live", "figure"), dash.Output("comparison-metric", "options")],
            [dash.Input("interval-component", "n_intervals"), dash.Input("comparison-metric", "value")],
        )
        def update_comparison(n_intervals, selected_metric):
            # Update available metrics
            available_metrics = list(self.metrics.keys())
            options = [{"label": m, "value": m} for m in available_metrics]

            # Create figure
            fig = go.Figure()

            if selected_metric and selected_metric in self.comparisons:
                comparison = self.comparisons[selected_metric]

                for model, values in comparison.items():
                    fig.add_trace(go.Box(y=values, name=model, boxpoints="all", jitter=0.3, pointpos=-1.8))

            fig.update_layout(
                title=f"Model Comparison: {selected_metric}" if selected_metric else "Model Comparison",
                yaxis_title="Value",
                height=400,
                showlegend=True,
            )

            return fig, options

    async def start(self):
        """Start dashboard server."""
        self.app.run(host=self.host, port=self.port, debug=False)

    async def stop(self):
        """Stop dashboard server."""
        # Dash doesn't provide a clean way to stop the server
        # This is a workaround
        func = self.app.server.shutdown
        if func is None:
            func = self.app.server.stop
        func()

    def update_metric(self, name: str, value: float):
        """Update metric value.

        Args:
            name: Metric name
            value: New value
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(value)

        # Trim history
        if len(self.metrics[name]) > self.history_size:
            self.metrics[name].pop(0)

        # Update timestamp
        self.timestamps.append(datetime.now())
        if len(self.timestamps) > self.history_size:
            self.timestamps.pop(0)

    def update_resource(self, name: str, value: float):
        """Update resource usage.

        Args:
            name: Resource name
            value: New value
        """
        if name not in self.resources:
            self.resources[name] = []

        self.resources[name].append(value)

        # Trim history
        if len(self.resources[name]) > self.history_size:
            self.resources[name].pop(0)

    def update_comparison(self, metric: str, model: str, value: float):
        """Update comparison data.

        Args:
            metric: Metric name
            model: Model name
            value: New value
        """
        if metric not in self.comparisons:
            self.comparisons[metric] = {}

        if model not in self.comparisons[metric]:
            self.comparisons[metric][model] = []

        self.comparisons[metric][model].append(value)

        # Trim history
        if len(self.comparisons[metric][model]) > self.history_size:
            self.comparisons[metric][model].pop(0)

    def clear_data(self):
        """Clear all stored data."""
        self.metrics.clear()
        self.timestamps.clear()
        self.resources.clear()
        self.comparisons.clear()
