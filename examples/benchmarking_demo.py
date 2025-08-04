#!/usr/bin/env python3
"""
Memento Benchmarking Demo

This demo showcases the comprehensive benchmarking framework that compares
Memento against other prompt evolution methods:
- PromptBreeder (evolutionary optimization)
- Self-Evolving GPT (experience accumulation)
- Auto-Evolve (self-reasoning framework)

Features demonstrated:
- Complete benchmarking infrastructure
- Statistical significance testing
- Performance comparison across methods
- Standardized evaluation metrics
"""

import asyncio
from pathlib import Path

from memento.benchmarking import BenchmarkRunner
from memento.benchmarking.evaluation import BenchmarkConfig
from memento.config import ModelConfig, ModelType


class BenchmarkingDemo:
    """Comprehensive benchmarking demonstration."""

    def __init__(self, base_path: str = "./benchmark_data"):
        """Initialize demo with storage paths."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Model configuration
        self.model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="llama3.2", temperature=0.7)

        # Benchmark configuration
        self.benchmark_config = BenchmarkConfig(
            num_runs=3,  # Reduced for demo
            max_iterations=5,  # Reduced for demo
            timeout_minutes=30,
            save_intermediate=True,
            parallel_runs=False,
        )

        # Initialize benchmark runner
        self.benchmark_runner = BenchmarkRunner(
            model_config=self.model_config, storage_path=self.base_path, config=self.benchmark_config
        )

        print("🏆 Memento Benchmarking Demo Initialized")
        print(f"📂 Storage path: {self.base_path}")
        print(f"🤖 Model: {self.model_config.model_name}")
        print(f"🔄 Runs per method: {self.benchmark_config.num_runs}")
        print(f"⚡ Max iterations: {self.benchmark_config.max_iterations}")

    def create_sample_datasets(self) -> dict:
        """Create sample datasets for benchmarking."""

        # Programming problems dataset
        programming_problems = [
            {
                "id": "p1",
                "description": "Write a function to reverse a string without using built-in reverse functions",
                "solution": "def reverse_string(s): return s[::-1]",
                "difficulty": "easy",
            },
            {
                "id": "p2",
                "description": "Implement a function to check if a string is a palindrome",
                "solution": "def is_palindrome(s): return s.lower() == s.lower()[::-1]",
                "difficulty": "easy",
            },
            {
                "id": "p3",
                "description": "Write a function to find the maximum element in a list",
                "solution": "def find_max(lst): return max(lst) if lst else None",
                "difficulty": "easy",
            },
            {
                "id": "p4",
                "description": "Implement binary search algorithm",
                "solution": "def binary_search(arr, target): # implementation here",
                "difficulty": "medium",
            },
            {
                "id": "p5",
                "description": "Write a function to calculate factorial recursively",
                "solution": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "difficulty": "medium",
            },
        ]

        # Math problems dataset
        math_problems = [
            {
                "id": "m1",
                "description": "Calculate the area of a circle given radius",
                "solution": "def circle_area(r): return 3.14159 * r * r",
                "difficulty": "easy",
            },
            {
                "id": "m2",
                "description": "Find the greatest common divisor of two numbers",
                "solution": "def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
                "difficulty": "medium",
            },
            {
                "id": "m3",
                "description": "Calculate compound interest",
                "solution": "def compound_interest(p, r, t): return p * (1 + r/100) ** t",
                "difficulty": "medium",
            },
        ]

        return {"programming": programming_problems, "mathematics": math_problems}

    async def demonstrate_single_method_comparison(self):
        """Demonstrate comparison between Memento and one baseline method."""
        print("\n" + "=" * 70)
        print("🔬 Single Method Comparison Demo")
        print("=" * 70)

        initial_prompt = "You are a helpful programming assistant. Provide clear, efficient solutions."
        problem_sets = {"demo": self.create_sample_datasets()["programming"][:3]}  # Small subset
        methods = ["memento", "promptbreeder"]  # Just two methods for quick demo

        print(f"📝 Initial Prompt: {initial_prompt}")
        print(f"📊 Methods: {', '.join(methods)}")
        print(f"🎯 Problems: {len(problem_sets['demo'])}")

        try:
            results = await self.benchmark_runner.run_comprehensive_benchmark(
                initial_prompt=initial_prompt, problem_sets=problem_sets, methods=methods
            )

            # Display results
            self._display_comparison_results(results)

        except Exception as e:
            print(f"❌ Comparison failed: {e}")

    async def demonstrate_full_benchmark(self):
        """Demonstrate full benchmarking across all methods and datasets."""
        print("\n" + "=" * 70)
        print("🏆 Full Benchmarking Demo")
        print("=" * 70)

        initial_prompt = "You are an expert problem solver. Provide accurate, well-reasoned solutions."
        problem_sets = self.create_sample_datasets()

        print(f"📝 Initial Prompt: {initial_prompt}")
        print(f"📚 Datasets: {list(problem_sets.keys())}")
        print("🔬 Methods: All (Memento, PromptBreeder, Self-Evolving GPT, Auto-Evolve)")

        # Note: This would take a long time with real LLM calls
        print("\n⚠️  Note: Full benchmark would take significant time with real LLM calls.")
        print("For demo purposes, we'll simulate a quick version...")

        try:
            # Run with reduced problem sets for demo
            demo_problem_sets = {name: problems[:2] for name, problems in problem_sets.items()}

            results = await self.benchmark_runner.run_comprehensive_benchmark(
                initial_prompt=initial_prompt,
                problem_sets=demo_problem_sets,
                methods=["memento", "promptbreeder"],  # Reduced for demo
            )

            # Display comprehensive results
            self._display_full_results(results)

        except Exception as e:
            print(f"❌ Full benchmark failed: {e}")

    def _display_comparison_results(self, results: dict):
        """Display comparison results in a readable format."""
        print("\n📊 Comparison Results:")
        print("-" * 50)

        if "statistical_analysis" in results:
            analysis = results["statistical_analysis"]

            # Method rankings
            if "method_rankings" in analysis:
                print("\n🏆 Method Rankings:")
                for ranking in analysis["method_rankings"]:
                    print(
                        f"  {ranking['rank']}. {ranking['method']}: "
                        f"{ranking['mean_performance']:.3f} ± {ranking['std_performance']:.3f} "
                        f"(Success: {ranking['success_rate']:.1%})"
                    )

            # Statistical significance
            if "pairwise_comparisons" in analysis:
                print("\n📈 Statistical Comparisons:")
                for comparison, data in analysis["pairwise_comparisons"].items():
                    significance = "✅ Significant" if data["significant"] else "❌ Not significant"
                    print(f"  {comparison}: {significance} (p={data['p_value']:.3f})")
                    print(f"    Winner: {data['winner']}, Effect size: {data['effect_size']:.3f}")

            # Summary
            if "summary" in analysis:
                summary = analysis["summary"]
                print("\n🎯 Summary:")
                print(f"  Best Method: {summary['best_method']}")
                print(f"  Best Performance: {summary['best_performance']:.3f}")
                print(f"  Total Runs: {summary['total_runs']}")

    def _display_full_results(self, results: dict):
        """Display full benchmark results."""
        print("\n📈 Full Benchmark Results:")
        print("=" * 60)

        # Dataset-specific results
        if "dataset_results" in results:
            for dataset_name, dataset_data in results["dataset_results"].items():
                print(f"\n📚 Dataset: {dataset_name}")
                print("-" * 30)

                for method, method_results in dataset_data.items():
                    performances = [r.final_performance for r in method_results]
                    avg_perf = sum(performances) / len(performances) if performances else 0
                    print(f"  {method}: {avg_perf:.3f} avg performance ({len(method_results)} runs)")

        # Overall analysis
        self._display_comparison_results(results)

        # Execution info
        if "execution_time" in results:
            print(f"\n⏱️  Total Execution Time: {results['execution_time']:.1f} seconds")

    async def demonstrate_method_analysis(self):
        """Demonstrate individual method analysis capabilities."""
        print("\n" + "=" * 70)
        print("🔍 Individual Method Analysis Demo")
        print("=" * 70)

        print("This demo would show detailed analysis of each method:")
        print("📊 PromptBreeder:")
        print("  - Population evolution over generations")
        print("  - Mutation and crossover effectiveness")
        print("  - Convergence patterns")

        print("\n🧠 Self-Evolving GPT:")
        print("  - Experience accumulation patterns")
        print("  - Learning rate adaptation")
        print("  - Memory utilization efficiency")

        print("\n🤔 Auto-Evolve:")
        print("  - Reasoning chain quality")
        print("  - Error detection accuracy")
        print("  - Confidence estimation reliability")

        print("\n🚀 Memento:")
        print("  - Feedback integration effectiveness")
        print("  - Principle extraction quality")
        print("  - Adaptive learning performance")

    async def run_complete_demo(self):
        """Run the complete benchmarking demonstration."""
        print("🎭 Starting Memento Benchmarking Demo - Phase 4")
        print("=" * 80)

        try:
            # Individual method analysis
            await self.demonstrate_method_analysis()

            # Single method comparison (quick)
            await self.demonstrate_single_method_comparison()

            # Note about full benchmark
            print("\n" + "=" * 70)
            print("📋 Full Benchmark Capabilities")
            print("=" * 70)
            print("The framework supports:")
            print("✅ All 4 methods: Memento, PromptBreeder, Self-Evolving GPT, Auto-Evolve")
            print("✅ Multiple datasets: Programming, Mathematics, Creative Writing")
            print("✅ Statistical analysis: T-tests, effect sizes, confidence intervals")
            print("✅ Performance metrics: Accuracy, improvement, convergence speed")
            print("✅ Visualization: Charts, comparisons, evolution tracking")

            # Summary
            print("\n" + "=" * 80)
            print("🎊 Phase 4 Benchmarking Framework Completed!")
            print("=" * 80)
            print("✅ Baseline implementations: PromptBreeder, Self-Evolving GPT, Auto-Evolve")
            print("✅ Comprehensive evaluation infrastructure")
            print("✅ Statistical significance testing")
            print("✅ Standardized comparison metrics")
            print("✅ Professional benchmarking capabilities")
            print(f"📂 Results saved to: {self.base_path}")
            print("\n🚀 Ready to demonstrate Memento's superiority through rigorous benchmarking!")

        except Exception as e:
            print(f"\n💥 Demo failed with error: {e}")
            raise


async def main():
    """Main demo function."""
    try:
        demo = BenchmarkingDemo()
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
