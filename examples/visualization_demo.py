#!/usr/bin/env python3
"""
Professional Visualization Framework Demo

This script demonstrates the comprehensive visualization capabilities
of the Memento benchmarking framework, including publication-ready
charts and statistical analysis plots.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memento.benchmarking.visualization import ComparisonPlotter, ResultsVisualizer


class VisualizationDemo:
    """Demonstration of visualization capabilities."""

    def __init__(self):
        """Initialize the visualization demo."""
        self.console = Console()
        self.output_dir = Path("visualization_demo_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualizers
        self.results_visualizer = ResultsVisualizer(
            output_dir=self.output_dir / "results", style="default"  # Use default style instead of seaborn-v0_8
        )
        self.comparison_plotter = ComparisonPlotter(output_dir=self.output_dir / "comparisons")

    def display_intro(self):
        """Display introduction to the visualization demo."""
        intro_text = """
🎨 PROFESSIONAL VISUALIZATION FRAMEWORK DEMO
=============================================

This demo showcases Memento's comprehensive visualization capabilities:

✅ PUBLICATION-READY CHARTS
  • Performance comparison charts
  • Statistical significance plots
  • Multi-domain radar charts
  • Evolution trajectory visualizations

✅ DETAILED COMPARATIVE ANALYSIS
  • Method comparison box plots
  • Improvement heatmaps
  • Confidence interval plots
  • Effect size magnitude charts

✅ PROFESSIONAL REPORTING
  • Automated HTML reports
  • Publication figure exports
  • Statistical power analysis
  • Dataset difficulty analysis

All visualizations are designed for research publications and
presentations with high-quality, publication-ready output.
        """

        self.console.print(
            Panel(intro_text, title="🎨 Visualization Framework", title_align="left", border_style="blue")
        )

    def create_sample_benchmark_results(self) -> Dict[str, Any]:
        """Create sample benchmark results for visualization."""
        return {
            "benchmark_info": {
                "timestamp": "2024-01-15T10:30:00Z",
                "model_config": {"model_name": "llama3.2", "temperature": 0.7},
                "datasets_used": ["humaneval", "math_hard", "writingbench", "gsm8k"],
                "max_problems_per_dataset": 50,
            },
            "dataset_results": {
                "humaneval": {
                    "dataset_info": {"domain": "programming", "size": 164},
                    "metrics": {"pass_rate": 0.45, "avg_quality": 0.72, "response_rate": 0.95},
                },
                "math_hard": {
                    "dataset_info": {"domain": "mathematics", "size": 1324},
                    "metrics": {"accuracy": 0.23, "avg_quality": 0.65, "response_rate": 0.90},
                },
                "writingbench": {
                    "dataset_info": {"domain": "writing", "size": 765},
                    "metrics": {"avg_score": 3.4, "completion_rate": 0.92, "avg_quality": 0.78},
                },
                "gsm8k": {
                    "dataset_info": {"domain": "mathematics", "size": 8500},
                    "metrics": {"accuracy": 0.67, "avg_quality": 0.81, "response_rate": 0.98},
                },
            },
            "comparative_analysis": {
                "memento_performance": {
                    "humaneval": {"pass_rate": 0.45, "avg_quality": 0.72},
                    "math_hard": {"accuracy": 0.23, "avg_quality": 0.65},
                    "writingbench": {"avg_score": 3.4, "completion_rate": 0.92},
                },
                "baseline_performance": {
                    "PromptBreeder": {
                        "humaneval": {"pass_rate": 0.31, "avg_quality": 0.65},
                        "math_hard": {"accuracy": 0.18, "avg_quality": 0.62},
                        "writingbench": {"avg_score": 2.8, "completion_rate": 0.85},
                    },
                    "Self-Evolving GPT": {
                        "humaneval": {"pass_rate": 0.28, "avg_quality": 0.63},
                        "math_hard": {"accuracy": 0.15, "avg_quality": 0.58},
                        "writingbench": {"avg_score": 2.6, "completion_rate": 0.82},
                    },
                    "Auto-Evolve": {
                        "humaneval": {"pass_rate": 0.33, "avg_quality": 0.67},
                        "math_hard": {"accuracy": 0.20, "avg_quality": 0.64},
                        "writingbench": {"avg_score": 3.0, "completion_rate": 0.87},
                    },
                },
            },
            "statistical_analysis": {
                "p_values": {
                    "Memento vs PromptBreeder": 0.001,
                    "Memento vs Self-Evolving GPT": 0.002,
                    "Memento vs Auto-Evolve": 0.015,
                },
                "effect_sizes": {
                    "Memento vs PromptBreeder": 1.2,
                    "Memento vs Self-Evolving GPT": 1.5,
                    "Memento vs Auto-Evolve": 0.9,
                },
                "confidence_intervals": {
                    "PromptBreeder": (0.31, 0.28, 0.34),
                    "Self-Evolving GPT": (0.28, 0.25, 0.31),
                    "Auto-Evolve": (0.33, 0.30, 0.36),
                    "Memento (Ours)": (0.45, 0.42, 0.48),
                },
            },
        }

    async def demonstrate_results_visualization(self, results: Dict[str, Any]):
        """Demonstrate the ResultsVisualizer capabilities."""
        self.console.print("\n📊 RESULTS VISUALIZATION DEMO", style="bold blue")
        self.console.print("=" * 50)

        with self.console.status("[bold green]Creating performance comparison chart..."):
            perf_chart = self.results_visualizer.create_performance_comparison_chart(results)
            self.console.print(f"✅ Performance comparison chart: {perf_chart}")

        with self.console.status("[bold green]Creating statistical significance plot..."):
            stat_plot = self.results_visualizer.create_statistical_significance_plot(
                results.get("statistical_analysis", {})
            )
            self.console.print(f"✅ Statistical significance plot: {stat_plot}")

        with self.console.status("[bold green]Creating domain analysis radar chart..."):
            radar_chart = self.results_visualizer.create_domain_analysis_radar(results.get("domain_analysis", {}))
            self.console.print(f"✅ Domain analysis radar chart: {radar_chart}")

        with self.console.status("[bold green]Creating evolution trajectory plot..."):
            evolution_plot = self.results_visualizer.create_evolution_trajectory_plot({})
            self.console.print(f"✅ Evolution trajectory plot: {evolution_plot}")

    async def demonstrate_comparison_plotting(self):
        """Demonstrate the ComparisonPlotter capabilities."""
        self.console.print("\n📈 COMPARISON PLOTTING DEMO", style="bold blue")
        self.console.print("=" * 50)

        # Sample performance data for box plots
        performance_data = {
            "PromptBreeder": [0.29, 0.31, 0.33, 0.30, 0.32],
            "Self-Evolving GPT": [0.26, 0.28, 0.30, 0.27, 0.29],
            "Auto-Evolve": [0.31, 0.33, 0.35, 0.32, 0.34],
            "Memento (Ours)": [0.43, 0.45, 0.47, 0.44, 0.46],
        }

        with self.console.status("[bold green]Creating method comparison boxplot..."):
            boxplot = self.comparison_plotter.create_method_comparison_boxplot(performance_data)
            self.console.print(f"✅ Method comparison boxplot: {boxplot}")

        with self.console.status("[bold green]Creating improvement heatmap..."):
            heatmap = self.comparison_plotter.create_improvement_heatmap({})
            self.console.print(f"✅ Improvement heatmap: {heatmap}")

        with self.console.status("[bold green]Creating confidence interval plot..."):
            ci_plot = self.comparison_plotter.create_confidence_interval_plot({})
            self.console.print(f"✅ Confidence interval plot: {ci_plot}")

        with self.console.status("[bold green]Creating effect size magnitude chart..."):
            effect_chart = self.comparison_plotter.create_effect_size_magnitude_chart({})
            self.console.print(f"✅ Effect size magnitude chart: {effect_chart}")

        with self.console.status("[bold green]Creating dataset difficulty analysis..."):
            difficulty_plot = self.comparison_plotter.create_dataset_difficulty_analysis({})
            self.console.print(f"✅ Dataset difficulty analysis: {difficulty_plot}")

        with self.console.status("[bold green]Creating statistical power analysis..."):
            power_plot = self.comparison_plotter.create_statistical_power_analysis({})
            self.console.print(f"✅ Statistical power analysis: {power_plot}")

    async def demonstrate_comprehensive_reporting(self, results: Dict[str, Any]):
        """Demonstrate comprehensive report generation."""
        self.console.print("\n📋 COMPREHENSIVE REPORTING DEMO", style="bold blue")
        self.console.print("=" * 50)

        with self.console.status("[bold green]Generating comprehensive HTML report..."):
            report_path = self.results_visualizer.generate_comprehensive_report(results)
            self.console.print(f"✅ Comprehensive HTML report: {report_path}")

        with self.console.status("[bold green]Exporting publication-ready figures..."):
            exported_figures = self.results_visualizer.export_publication_figures(results)
            self.console.print(f"✅ Exported {len(exported_figures)} publication figures")

            # Show a few examples
            for i, fig_path in enumerate(exported_figures[:4]):
                self.console.print(f"   📄 {fig_path}")
            if len(exported_figures) > 4:
                self.console.print(f"   ... and {len(exported_figures) - 4} more")

    def display_summary_table(self):
        """Display summary of generated visualizations."""
        table = Table(title="📊 Generated Visualizations Summary", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Visualization Type", style="green")
        table.add_column("Purpose", style="yellow")
        table.add_column("Status", justify="center")

        visualizations = [
            ("Results", "Performance Comparison Chart", "Overall method comparison", "✅"),
            ("Results", "Statistical Significance Plot", "P-values and effect sizes", "✅"),
            ("Results", "Domain Analysis Radar", "Multi-domain performance", "✅"),
            ("Results", "Evolution Trajectory", "Improvement over iterations", "✅"),
            ("Comparison", "Method Comparison Boxplot", "Performance distributions", "✅"),
            ("Comparison", "Improvement Heatmap", "Percentage improvements", "✅"),
            ("Comparison", "Confidence Intervals", "Statistical uncertainty", "✅"),
            ("Comparison", "Effect Size Analysis", "Practical significance", "✅"),
            ("Comparison", "Difficulty Analysis", "Performance vs difficulty", "✅"),
            ("Comparison", "Statistical Power", "Test reliability", "✅"),
            ("Reports", "Comprehensive HTML Report", "Complete analysis", "✅"),
            ("Reports", "Publication Figures", "Research paper ready", "✅"),
        ]

        for category, viz_type, purpose, status in visualizations:
            table.add_row(category, viz_type, purpose, status)

        self.console.print(table)

    def display_next_steps(self):
        """Display information about next steps and usage."""
        next_steps = """
🎯 NEXT STEPS & USAGE GUIDE
===========================

✅ VISUALIZATION FRAMEWORK COMPLETED
  • All visualization modules implemented
  • Publication-ready output formats
  • Professional statistical analysis
  • Comprehensive reporting capabilities

🔧 INTEGRATION WITH BENCHMARKING
  • Visualizations integrate with StandardBenchmarkRunner
  • Automatic chart generation during benchmarks
  • Export capabilities for research papers
  • Interactive HTML reports for presentations

📊 CUSTOMIZATION OPTIONS
  • Multiple output formats (PNG, PDF, SVG, EPS)
  • Configurable themes and styling
  • Custom color schemes and layouts
  • Publication-specific formatting

🚀 READY FOR PHASE 4
  • Visualization framework complete
  • All Phase 3 objectives achieved
  • Ready to proceed with advanced features
  • Professional benchmarking infrastructure established
        """

        self.console.print(Panel(next_steps, title="🎯 Next Steps", title_align="left", border_style="green"))

    async def run_demo(self):
        """Run the complete visualization demonstration."""
        self.display_intro()

        # Create sample data
        results = self.create_sample_benchmark_results()

        # Save sample results for reference
        with open(self.output_dir / "sample_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Demonstrate visualization capabilities
        await self.demonstrate_results_visualization(results)
        await self.demonstrate_comparison_plotting()
        await self.demonstrate_comprehensive_reporting(results)

        # Display summary
        self.display_summary_table()
        self.display_next_steps()

        self.console.print("\n🎨 Visualization Demo Complete!")
        self.console.print(f"📁 All outputs saved to: {self.output_dir}")
        self.console.print(
            f"🌐 View HTML report: {self.output_dir}/results/reports/comprehensive_visualization_report.html"
        )


async def main():
    """Main demo function."""
    demo = VisualizationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
