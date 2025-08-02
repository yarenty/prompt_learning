"""
BenchmarkRunner - Orchestrates comprehensive benchmarking experiments.

This module provides the main benchmarking infrastructure for comparing
Memento against baseline methods (PromptBreeder, Self-Evolving GPT, Auto-Evolve).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...config import ModelConfig
from ...core import PromptLearner
from ..baselines import AutoEvolve, PromptBreeder, SelfEvolvingGPT
from .metrics import EvaluationMetrics
from .statistical_analyzer import StatisticalAnalyzer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""

    num_runs: int = 5  # Number of independent runs for statistical significance
    max_iterations: int = 10  # Maximum iterations per method
    timeout_minutes: int = 60  # Timeout per method run
    save_intermediate: bool = True  # Save intermediate results
    parallel_runs: bool = False  # Run methods in parallel (if resources allow)


@dataclass
class MethodResult:
    """Results from a single method run."""

    method_name: str
    run_id: int
    initial_prompt: str
    final_prompt: str
    performance_history: List[float]
    final_performance: float
    improvement: float
    execution_time: float
    iterations_completed: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """
    Orchestrates comprehensive benchmarking experiments.

    This class manages the execution of multiple prompt evolution methods
    across multiple datasets and provides standardized comparison metrics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        storage_path: Path,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize BenchmarkRunner.

        Args:
            model_config: Model configuration for all methods
            storage_path: Path to store benchmark results
            config: Benchmark configuration
        """
        self.model_config = model_config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.config = config or BenchmarkConfig()

        # Initialize components
        self.metrics = EvaluationMetrics()
        self.statistical_analyzer = StatisticalAnalyzer()

        # Results storage
        self.results: Dict[str, List[MethodResult]] = {}
        self.benchmark_metadata: Dict[str, Any] = {}

    async def run_comprehensive_benchmark(
        self,
        initial_prompt: str,
        problem_sets: Dict[str, List[Dict[str, Any]]],
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all methods and datasets.

        Args:
            initial_prompt: Starting prompt for all methods
            problem_sets: Dictionary mapping dataset names to problem lists
            methods: List of method names to benchmark (default: all)

        Returns:
            Comprehensive benchmark results with statistical analysis
        """
        if methods is None:
            methods = ["memento", "promptbreeder", "self_evolving_gpt", "auto_evolve"]

        print("🏆 Starting Comprehensive Benchmark")
        print(f"📊 Methods: {', '.join(methods)}")
        print(f"📚 Datasets: {', '.join(problem_sets.keys())}")
        print(f"🔄 Runs per method: {self.config.num_runs}")

        benchmark_start_time = time.time()

        # Store benchmark metadata
        self.benchmark_metadata = {
            "start_time": benchmark_start_time,
            "methods": methods,
            "datasets": list(problem_sets.keys()),
            "config": asdict(self.config),
            "model_config": {
                "model_name": self.model_config.model_name,
                "model_type": self.model_config.model_type.value,
                "temperature": self.model_config.temperature,
            },
        }

        # Run benchmarks for each dataset
        dataset_results = {}
        for dataset_name, problems in problem_sets.items():
            print(f"\n📚 Benchmarking on dataset: {dataset_name}")

            dataset_results[dataset_name] = await self._benchmark_on_dataset(
                initial_prompt, problems, methods, dataset_name
            )

        # Aggregate results across datasets
        aggregated_results = self._aggregate_results(dataset_results)

        # Perform statistical analysis
        statistical_results = self._perform_statistical_analysis(aggregated_results)

        # Generate final report
        final_results = {
            "benchmark_metadata": self.benchmark_metadata,
            "dataset_results": dataset_results,
            "aggregated_results": aggregated_results,
            "statistical_analysis": statistical_results,
            "execution_time": time.time() - benchmark_start_time,
        }

        # Save results
        await self._save_benchmark_results(final_results)

        print(f"\n🎉 Benchmark completed in {final_results['execution_time']:.1f}s")

        return final_results

    async def _benchmark_on_dataset(
        self,
        initial_prompt: str,
        problems: List[Dict[str, Any]],
        methods: List[str],
        dataset_name: str,
    ) -> Dict[str, List[MethodResult]]:
        """Benchmark all methods on a single dataset."""
        dataset_results = {}

        for method_name in methods:
            print(f"  🔬 Testing {method_name}...")

            method_results = []
            for run_id in range(self.config.num_runs):
                print(f"    Run {run_id + 1}/{self.config.num_runs}")

                try:
                    result = await self._run_single_method(method_name, initial_prompt, problems, run_id)
                    method_results.append(result)

                    if self.config.save_intermediate:
                        await self._save_intermediate_result(result, dataset_name)

                except Exception as e:
                    print(f"    ❌ Run {run_id + 1} failed: {e}")
                    # Create failure result
                    failure_result = MethodResult(
                        method_name=method_name,
                        run_id=run_id,
                        initial_prompt=initial_prompt,
                        final_prompt=initial_prompt,
                        performance_history=[0.0],
                        final_performance=0.0,
                        improvement=0.0,
                        execution_time=0.0,
                        iterations_completed=0,
                        metadata={"error": str(e)},
                    )
                    method_results.append(failure_result)

            dataset_results[method_name] = method_results

        return dataset_results

    async def _run_single_method(
        self,
        method_name: str,
        initial_prompt: str,
        problems: List[Dict[str, Any]],
        run_id: int,
    ) -> MethodResult:
        """Run a single method on a problem set."""
        start_time = time.time()

        # Create evaluation function
        evaluation_function = self._create_evaluation_function(problems)

        if method_name == "memento":
            result = await self._run_memento(initial_prompt, evaluation_function, problems, run_id)
        elif method_name == "promptbreeder":
            result = await self._run_promptbreeder(initial_prompt, evaluation_function, problems, run_id)
        elif method_name == "self_evolving_gpt":
            result = await self._run_self_evolving_gpt(initial_prompt, evaluation_function, problems, run_id)
        elif method_name == "auto_evolve":
            result = await self._run_auto_evolve(initial_prompt, evaluation_function, problems, run_id)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        execution_time = time.time() - start_time

        # Calculate improvement
        initial_performance = result.performance_history[0] if result.performance_history else 0.0
        improvement = result.final_performance - initial_performance

        # Update result with timing and improvement
        result.execution_time = execution_time
        result.improvement = improvement

        return result

    def _create_evaluation_function(self, problems: List[Dict[str, Any]]):
        """Create evaluation function for the given problem set."""

        async def evaluate_prompt(prompt: str, problem_subset: Optional[List] = None) -> float:
            """Evaluate prompt performance on problems."""
            test_problems = problem_subset or problems[: min(10, len(problems))]  # Use subset for efficiency

            total_score = 0.0
            for problem in test_problems:
                try:
                    score = await self.metrics.evaluate_prompt_on_problem(prompt, problem, self.model_config)
                    total_score += score
                except Exception:
                    # Assign zero score for failed evaluations
                    pass

            return total_score / len(test_problems) if test_problems else 0.0

        return evaluate_prompt

    async def _run_memento(
        self,
        initial_prompt: str,
        evaluation_function: callable,
        problems: List[Dict[str, Any]],
        run_id: int,
    ) -> MethodResult:
        """Run Memento method."""
        # Use PromptLearner for Memento
        learner = PromptLearner(
            model_config=self.model_config,
            storage_path=self.storage_path / f"memento_run_{run_id}",
            enable_metrics=True,
        )

        performance_history = []
        current_prompt = initial_prompt

        # Simulate evolution iterations
        for iteration in range(self.config.max_iterations):
            # Evaluate current prompt
            performance = await evaluation_function(current_prompt)
            performance_history.append(performance)

            if iteration < self.config.max_iterations - 1:
                # Evolve prompt using Memento's approach
                # Use a sample problem for evolution
                sample_problem = {
                    "description": problems[iteration % len(problems)]["description"],
                    "solution": problems[iteration % len(problems)].get("solution", ""),
                }

                try:
                    evolution_result = await learner.evaluate_prompt_performance(
                        prompt=current_prompt,
                        problem=sample_problem,
                        evaluation_criteria=["correctness", "efficiency", "readability"],
                    )

                    # Evolve based on lessons learned
                    evolved_prompt = await learner.evolve_prompt(
                        current_prompt=current_prompt, lessons=evolution_result["lessons"]
                    )

                    current_prompt = evolved_prompt

                except Exception as e:
                    print(f"    Memento evolution error: {e}")
                    break

        return MethodResult(
            method_name="memento",
            run_id=run_id,
            initial_prompt=initial_prompt,
            final_prompt=current_prompt,
            performance_history=performance_history,
            final_performance=performance_history[-1] if performance_history else 0.0,
            improvement=0.0,  # Will be calculated later
            execution_time=0.0,  # Will be set later
            iterations_completed=len(performance_history),
            metadata={"method": "memento", "learner_type": "PromptLearner"},
        )

    async def _run_promptbreeder(
        self,
        initial_prompt: str,
        evaluation_function: callable,
        problems: List[Dict[str, Any]],
        run_id: int,
    ) -> MethodResult:
        """Run PromptBreeder method."""
        promptbreeder = PromptBreeder(
            model_config=self.model_config,
            population_size=10,  # Smaller for efficiency
            max_generations=self.config.max_iterations,
            storage_path=self.storage_path / f"promptbreeder_run_{run_id}",
        )

        try:
            best_individual = await promptbreeder.evolve(
                initial_prompt=initial_prompt, evaluation_function=evaluation_function, problem_set=problems
            )

            performance_history = [gen["best_fitness"] for gen in promptbreeder.evolution_history]

            return MethodResult(
                method_name="promptbreeder",
                run_id=run_id,
                initial_prompt=initial_prompt,
                final_prompt=best_individual.prompt,
                performance_history=performance_history,
                final_performance=best_individual.fitness,
                improvement=0.0,  # Will be calculated later
                execution_time=0.0,  # Will be set later
                iterations_completed=len(performance_history),
                metadata=promptbreeder.get_evolution_summary(),
            )

        except Exception as e:
            return MethodResult(
                method_name="promptbreeder",
                run_id=run_id,
                initial_prompt=initial_prompt,
                final_prompt=initial_prompt,
                performance_history=[0.0],
                final_performance=0.0,
                improvement=0.0,
                execution_time=0.0,
                iterations_completed=0,
                metadata={"error": str(e)},
            )

    async def _run_self_evolving_gpt(
        self,
        initial_prompt: str,
        evaluation_function: callable,
        problems: List[Dict[str, Any]],
        run_id: int,
    ) -> MethodResult:
        """Run Self-Evolving GPT method."""
        self_evolving = SelfEvolvingGPT(
            model_config=self.model_config, storage_path=self.storage_path / f"self_evolving_run_{run_id}"
        )

        try:
            final_prompt = await self_evolving.evolve(
                initial_prompt=initial_prompt,
                evaluation_function=evaluation_function,
                problem_set=problems,
                num_iterations=self.config.max_iterations,
            )

            return MethodResult(
                method_name="self_evolving_gpt",
                run_id=run_id,
                initial_prompt=initial_prompt,
                final_prompt=final_prompt,
                performance_history=self_evolving.performance_history,
                final_performance=self_evolving.performance_history[-1] if self_evolving.performance_history else 0.0,
                improvement=0.0,  # Will be calculated later
                execution_time=0.0,  # Will be set later
                iterations_completed=len(self_evolving.performance_history),
                metadata=self_evolving.get_evolution_summary(),
            )

        except Exception as e:
            return MethodResult(
                method_name="self_evolving_gpt",
                run_id=run_id,
                initial_prompt=initial_prompt,
                final_prompt=initial_prompt,
                performance_history=[0.0],
                final_performance=0.0,
                improvement=0.0,
                execution_time=0.0,
                iterations_completed=0,
                metadata={"error": str(e)},
            )

    async def _run_auto_evolve(
        self,
        initial_prompt: str,
        evaluation_function: callable,
        problems: List[Dict[str, Any]],
        run_id: int,
    ) -> MethodResult:
        """Run Auto-Evolve method."""
        auto_evolve = AutoEvolve(
            model_config=self.model_config, storage_path=self.storage_path / f"auto_evolve_run_{run_id}"
        )

        try:
            final_prompt = await auto_evolve.evolve(
                initial_prompt=initial_prompt,
                evaluation_function=evaluation_function,
                problem_set=problems,
                num_iterations=self.config.max_iterations,
            )

            # Extract performance history from improvement history
            performance_history = []
            for improvement in auto_evolve.improvement_history:
                # Use confidence as proxy for performance improvement
                performance_history.append(improvement.get("confidence", 0.5))

            if not performance_history:
                performance_history = [0.5]  # Default

            return MethodResult(
                method_name="auto_evolve",
                run_id=run_id,
                initial_prompt=initial_prompt,
                final_prompt=final_prompt,
                performance_history=performance_history,
                final_performance=performance_history[-1],
                improvement=0.0,  # Will be calculated later
                execution_time=0.0,  # Will be set later
                iterations_completed=len(performance_history),
                metadata=auto_evolve.get_evolution_summary(),
            )

        except Exception as e:
            return MethodResult(
                method_name="auto_evolve",
                run_id=run_id,
                initial_prompt=initial_prompt,
                final_prompt=initial_prompt,
                performance_history=[0.0],
                final_performance=0.0,
                improvement=0.0,
                execution_time=0.0,
                iterations_completed=0,
                metadata={"error": str(e)},
            )

    def _aggregate_results(self, dataset_results: Dict[str, Dict[str, List[MethodResult]]]) -> Dict[str, Any]:
        """Aggregate results across datasets."""
        aggregated = {}

        # Get all methods
        all_methods = set()
        for dataset_results_dict in dataset_results.values():
            all_methods.update(dataset_results_dict.keys())

        for method in all_methods:
            method_data = {
                "all_runs": [],
                "final_performances": [],
                "improvements": [],
                "execution_times": [],
                "iterations_completed": [],
            }

            # Collect data across all datasets
            for dataset_name, dataset_dict in dataset_results.items():
                if method in dataset_dict:
                    for result in dataset_dict[method]:
                        method_data["all_runs"].append(result)
                        method_data["final_performances"].append(result.final_performance)
                        method_data["improvements"].append(result.improvement)
                        method_data["execution_times"].append(result.execution_time)
                        method_data["iterations_completed"].append(result.iterations_completed)

            # Calculate statistics
            if method_data["final_performances"]:
                aggregated[method] = {
                    "mean_performance": np.mean(method_data["final_performances"]),
                    "std_performance": np.std(method_data["final_performances"]),
                    "mean_improvement": np.mean(method_data["improvements"]),
                    "std_improvement": np.std(method_data["improvements"]),
                    "mean_execution_time": np.mean(method_data["execution_times"]),
                    "mean_iterations": np.mean(method_data["iterations_completed"]),
                    "success_rate": sum(1 for p in method_data["final_performances"] if p > 0.1)
                    / len(method_data["final_performances"]),
                    "total_runs": len(method_data["final_performances"]),
                    "raw_data": method_data,
                }

        return aggregated

    def _perform_statistical_analysis(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on aggregated results."""
        return self.statistical_analyzer.analyze_method_comparison(aggregated_results)

    async def _save_intermediate_result(self, result: MethodResult, dataset_name: str) -> None:
        """Save intermediate result."""
        intermediate_dir = self.storage_path / "intermediate" / dataset_name
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{result.method_name}_run_{result.run_id}.json"
        filepath = intermediate_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    async def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Save final benchmark results."""
        timestamp = int(time.time())
        results_file = self.storage_path / f"benchmark_results_{timestamp}.json"

        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"📊 Benchmark results saved to: {results_file}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
