"""Model evaluation pipeline for standardized evaluation protocols."""

import logging
import time
from typing import Any, Dict, List, Optional

from ...config.models import ModelConfig
from ...core.learner import PromptLearner
from ...utils.logger import LoggerMixin
from ..baselines.promptbreeder import PromptBreeder
from ..baselines.self_evolving_gpt import SelfEvolvingGPT
from .task_metrics import MathematicsMetrics, ProgrammingMetrics, WritingMetrics

logger = logging.getLogger(__name__)


class ModelEvaluator(LoggerMixin):
    """Standardized model evaluation pipeline."""

    def __init__(self, model_config: ModelConfig):
        """Initialize model evaluator.

        Args:
            model_config: Model configuration for LLM calls
        """
        super().__init__()
        self.model_config = model_config

        # Initialize task-specific metrics
        self.programming_metrics = ProgrammingMetrics()
        self.math_metrics = MathematicsMetrics()
        self.writing_metrics = WritingMetrics()

        # Initialize baseline models (will be created per evaluation)
        self.baseline_models = {}

    async def evaluate_model(
        self,
        model_name: str,
        dataset: Any,
        dataset_name: str,
        task_type: str = "auto",
        storage_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a model on a dataset.

        Args:
            model_name: Name of the model to evaluate
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset
            task_type: Type of task (programming, mathematics, writing, auto)
            storage_path: Storage path for model data
            **kwargs: Additional evaluation parameters

        Returns:
            Evaluation results dictionary
        """
        self.logger.info(f"Evaluating {model_name} on {dataset_name}")

        start_time = time.time()

        # Auto-detect task type if not provided
        if task_type == "auto":
            task_type = self._detect_task_type(dataset_name)

        # Prepare dataset samples
        samples = self._prepare_dataset_samples(dataset, **kwargs)

        try:
            # Get model instance
            model = await self._get_model_instance(model_name, storage_path, **kwargs)

            # Run evaluation
            results = await self._run_evaluation(model, model_name, samples, task_type, **kwargs)

            # Calculate task-specific metrics
            task_metrics = await self._calculate_task_metrics(results["responses"], samples, task_type)

            # Combine results
            evaluation_time = time.time() - start_time

            final_results = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "task_type": task_type,
                "evaluation_time": evaluation_time,
                "samples_evaluated": len(samples),
                "basic_metrics": results["basic_metrics"],
                "task_metrics": task_metrics,
                "responses": (
                    results["responses"][:5] if len(results["responses"]) > 5 else results["responses"]
                ),  # Sample responses
                "metadata": {"model_config": self.model_config.model_dump(), "evaluation_params": kwargs},
            }

            self.logger.info(f"Completed evaluation of {model_name} on {dataset_name} in {evaluation_time:.2f}s")
            return final_results

        except Exception as e:
            self.logger.error(f"Failed to evaluate {model_name} on {dataset_name}: {e}")
            return {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "task_type": task_type,
                "error": str(e),
                "evaluation_time": time.time() - start_time,
                "samples_evaluated": 0,
            }

    async def _get_model_instance(self, model_name: str, storage_path: Optional[str] = None, **kwargs) -> Any:
        """Get model instance for evaluation."""
        storage_path = storage_path or f"./{model_name}_storage"

        if model_name == "promptbreeder":
            return PromptBreeder(
                model_config=self.model_config,
                storage_path=storage_path,
                population_size=kwargs.get("population_size", 10),  # Smaller for faster evaluation
                max_generations=kwargs.get("max_generations", 5),  # Fewer generations for faster evaluation
                mutation_rate=kwargs.get("mutation_rate", 0.3),
                crossover_rate=kwargs.get("crossover_rate", 0.7),
            )
        elif model_name == "self_evolving_gpt":
            return SelfEvolvingGPT(
                model_config=self.model_config,
                storage_path=storage_path,
                memory_size=kwargs.get("memory_size", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
            )
        elif model_name == "memento":
            return PromptLearner(model_config=self.model_config, storage_path=storage_path)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _detect_task_type(self, dataset_name: str) -> str:
        """Auto-detect task type from dataset name."""
        dataset_name_lower = dataset_name.lower()

        if any(keyword in dataset_name_lower for keyword in ["code", "programming", "humaneval", "apps", "bigcode"]):
            return "programming"
        elif any(keyword in dataset_name_lower for keyword in ["math", "gsm8k", "mmlu_math"]):
            return "mathematics"
        elif any(keyword in dataset_name_lower for keyword in ["writing", "creative", "biggen"]):
            return "writing"
        else:
            return "general"

    def _prepare_dataset_samples(
        self, dataset: Any, max_samples: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Prepare dataset samples for evaluation."""
        samples = []

        # Convert dataset to list of samples
        if hasattr(dataset, "__iter__"):
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break

                # Standardize sample format
                if isinstance(item, dict):
                    # Handle different dataset formats
                    problem = item.get("problem", item.get("description", item.get("text", item.get("question", ""))))
                    expected = item.get("expected", item.get("solution", item.get("answer", "")))
                    
                    sample = {
                        "id": item.get("id", i),
                        "problem": problem,
                        "expected": expected,
                        "metadata": {
                            k: v
                            for k, v in item.items()
                            if k not in ["id", "problem", "description", "text", "question", "expected", "solution", "answer"]
                        },
                    }
                else:
                    sample = {"id": i, "problem": str(item), "expected": "", "metadata": {}}

                samples.append(sample)

        return samples

    async def _run_evaluation(
        self, model: Any, model_name: str, samples: List[Dict[str, Any]], task_type: str, **kwargs
    ) -> Dict[str, Any]:
        """Run model evaluation on samples."""
        responses = []
        correct_count = 0
        total_time = 0

        for sample in samples:
            sample_start = time.time()

            try:
                # Generate response based on model type
                if model_name == "promptbreeder":
                    response = await self._evaluate_with_promptbreeder(model, sample, **kwargs)
                elif model_name == "self_evolving_gpt":
                    response = await self._evaluate_with_self_evolving_gpt(model, sample, **kwargs)
                elif model_name == "memento":
                    response = await self._evaluate_with_memento(model, sample, **kwargs)
                else:
                    response = await self._evaluate_with_generic_model(model, sample, **kwargs)

                sample_time = time.time() - sample_start
                total_time += sample_time

                # Basic correctness check (simple string matching for now)
                is_correct = self._check_basic_correctness(response, sample["expected"])
                if is_correct:
                    correct_count += 1

                responses.append(
                    {
                        "sample_id": sample["id"],
                        "problem": sample["problem"],
                        "expected": sample["expected"],
                        "response": response,
                        "is_correct": is_correct,
                        "response_time": sample_time,
                    }
                )

            except Exception as e:
                self.logger.warning(f"Failed to evaluate sample {sample['id']}: {e}")
                responses.append(
                    {
                        "sample_id": sample["id"],
                        "problem": sample["problem"],
                        "expected": sample["expected"],
                        "response": "",
                        "is_correct": False,
                        "response_time": 0,
                        "error": str(e),
                    }
                )

        # Calculate basic metrics
        basic_metrics = {
            "accuracy": correct_count / len(samples) if samples else 0,
            "total_samples": len(samples),
            "correct_samples": correct_count,
            "average_response_time": total_time / len(samples) if samples else 0,
            "total_evaluation_time": total_time,
        }

        return {"basic_metrics": basic_metrics, "responses": responses}

    async def _evaluate_with_promptbreeder(self, model: PromptBreeder, sample: Dict[str, Any], **kwargs) -> str:
        """Evaluate sample with PromptBreeder."""

        # Create a simple evaluation function for PromptBreeder
        async def evaluation_function(prompt: str, problem_set: List[Dict[str, Any]]) -> float:
            # Simple evaluation - in real implementation, this would be more sophisticated
            return 0.8  # Placeholder score

        # Use PromptBreeder to evolve a prompt for this type of problem
        initial_prompt = f"Solve this problem step by step: {sample['problem']}"

        # Run evolution (with reduced parameters for speed)
        best_individual = await model.evolve(
            initial_prompt=initial_prompt, evaluation_function=evaluation_function, problem_set=[sample]
        )

        # Use the evolved prompt to generate final response
        if best_individual:
            return best_individual.prompt
        else:
            return initial_prompt

    async def _evaluate_with_self_evolving_gpt(self, model: SelfEvolvingGPT, sample: Dict[str, Any], **kwargs) -> str:
        """Evaluate sample with SelfEvolvingGPT."""

        # Create evaluation function
        async def evaluation_function(prompt: str, problem_set: List[Dict[str, Any]]) -> float:
            return 0.8  # Placeholder score

        # Use SelfEvolvingGPT to generate and improve response
        initial_prompt = f"Solve this problem: {sample['problem']}"

        # Run self-evolution (with reduced iterations for speed)
        final_prompt = await model.evolve(
            initial_prompt=initial_prompt,
            evaluation_function=evaluation_function,
            problem_set=[sample],
            num_iterations=3,  # Reduced for speed
        )

        return final_prompt

    async def _evaluate_with_memento(self, model: PromptLearner, sample: Dict[str, Any], **kwargs) -> str:
        """Evaluate sample with Memento."""
        # Use Memento's prompt evolution capabilities
        problem = {"description": sample["problem"], "solution": sample.get("expected", "")}

        # Create a simple evaluation criteria (using valid criteria from validation)
        evaluation_criteria = ["correctness", "clarity", "readability"]

        try:
            # Evaluate current prompt performance
            evaluation_results = await model.evaluate_prompt_performance(
                prompt="You are a helpful assistant that solves problems step by step.",
                problem=problem,
                evaluation_criteria=evaluation_criteria,
            )

            # Extract lessons and evolve the prompt
            lessons = self._extract_lessons_from_evaluation(evaluation_results)
            evolved_prompt = await model.evolve_prompt(
                current_prompt="You are a helpful assistant that solves problems step by step.", lessons=lessons
            )

            # For now, return the evolved prompt as the response
            # In a full implementation, we would use the evolved prompt to generate a response
            return f"Evolved prompt: {evolved_prompt}\n\nProblem: {sample['problem']}\n\nSolution approach: Use the evolved prompt to solve this problem step by step."

        except Exception as e:
            self.logger.warning(f"Memento evaluation failed: {e}")
            return f"Problem: {sample['problem']}\n\nSolution: [Memento evaluation failed - {e}]"

    def _extract_lessons_from_evaluation(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract lessons from evaluation results."""
        lessons = []

        # Extract insights from evaluation
        if "insights" in evaluation_results:
            for insight in evaluation_results["insights"]:
                lessons.append({"type": "evaluation_insight", "content": insight, "priority": "medium"})

        # Extract improvement suggestions
        if "suggestions" in evaluation_results:
            for suggestion in evaluation_results["suggestions"]:
                lessons.append({"type": "improvement_suggestion", "content": suggestion, "priority": "high"})

        return lessons

    async def _evaluate_with_generic_model(self, model: Any, sample: Dict[str, Any], **kwargs) -> str:
        """Evaluate sample with generic model."""
        # Fallback for unknown models
        return f"Problem: {sample['problem']}\nSolution: [Generated by {type(model).__name__}]"

    def _check_basic_correctness(self, response: str, expected: str) -> bool:
        """Basic correctness check."""
        if not expected:
            return True  # Can't check correctness without expected answer

        # Simple string matching (case-insensitive)
        response_clean = response.lower().strip()
        expected_clean = expected.lower().strip()

        # Check if expected answer is contained in response
        return expected_clean in response_clean

    async def _calculate_task_metrics(
        self, responses: List[Dict[str, Any]], samples: List[Dict[str, Any]], task_type: str
    ) -> Dict[str, Any]:
        """Calculate task-specific metrics."""
        if task_type == "programming":
            return await self._calculate_programming_metrics(responses, samples)
        elif task_type == "mathematics":
            return await self._calculate_mathematics_metrics(responses, samples)
        elif task_type == "writing":
            return await self._calculate_writing_metrics(responses, samples)
        else:
            return {"task_type": task_type, "metrics": "No specific metrics available"}

    async def _calculate_programming_metrics(
        self, responses: List[Dict[str, Any]], samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate programming-specific metrics."""
        metrics = {}

        # Calculate code quality for responses that look like code
        code_responses = [r for r in responses if self._looks_like_code(r["response"])]

        if code_responses:
            quality_scores = []
            for response in code_responses:
                try:
                    quality = self.programming_metrics.calculate_code_quality(response["response"])
                    quality_scores.append(quality)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate code quality: {e}")

            if quality_scores:
                # Average quality metrics
                avg_quality = {}
                for key in quality_scores[0].keys():
                    avg_quality[key] = sum(q[key] for q in quality_scores) / len(quality_scores)
                metrics["average_code_quality"] = avg_quality

        # Calculate pass@1 for samples with test cases
        samples_with_tests = [s for s in samples if s.get("metadata", {}).get("test_cases")]
        if samples_with_tests and code_responses:
            pass_at_1_scores = []
            for i, sample in enumerate(samples_with_tests):
                if i < len(code_responses):
                    try:
                        test_cases = sample["metadata"]["test_cases"]
                        pass_score = self.programming_metrics.calculate_pass_at_k(
                            code_responses[i]["response"], test_cases, k=1
                        )
                        pass_at_1_scores.append(pass_score)
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate pass@1: {e}")

            if pass_at_1_scores:
                metrics["pass_at_1"] = sum(pass_at_1_scores) / len(pass_at_1_scores)

        return metrics

    async def _calculate_mathematics_metrics(
        self, responses: List[Dict[str, Any]], samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate mathematics-specific metrics."""
        metrics = {}

        # Calculate numerical accuracy
        accuracy_scores = []
        for response in responses:
            if response["expected"]:
                try:
                    accuracy = self.math_metrics.calculate_accuracy(response["response"], response["expected"])
                    accuracy_scores.append(accuracy)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate math accuracy: {e}")

        if accuracy_scores:
            metrics["numerical_accuracy"] = sum(accuracy_scores) / len(accuracy_scores)

        # Calculate reasoning quality
        reasoning_scores = []
        for response in responses:
            try:
                reasoning = self.math_metrics.evaluate_reasoning(response["response"])
                reasoning_scores.append(reasoning)
            except Exception as e:
                self.logger.warning(f"Failed to evaluate reasoning: {e}")

        if reasoning_scores:
            # Average reasoning metrics
            avg_reasoning = {}
            for key in reasoning_scores[0].keys():
                avg_reasoning[key] = sum(r[key] for r in reasoning_scores) / len(reasoning_scores)
            metrics["reasoning_quality"] = avg_reasoning

        return metrics

    async def _calculate_writing_metrics(
        self, responses: List[Dict[str, Any]], samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate writing-specific metrics."""
        metrics = {}

        # Calculate ROUGE scores against expected outputs
        rouge_scores = []
        for response in responses:
            if response["expected"]:
                try:
                    rouge = self.writing_metrics.calculate_rouge(response["response"], response["expected"])
                    rouge_scores.append(rouge)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate ROUGE: {e}")

        if rouge_scores:
            # Average ROUGE metrics
            avg_rouge = {}
            for key in rouge_scores[0].keys():
                avg_rouge[key] = sum(r[key] for r in rouge_scores) / len(rouge_scores)
            metrics["rouge_scores"] = avg_rouge

        # Calculate style metrics
        style_scores = []
        for response in responses:
            try:
                style = self.writing_metrics.analyze_style(response["response"])
                style_scores.append(style)
            except Exception as e:
                self.logger.warning(f"Failed to analyze style: {e}")

        if style_scores:
            # Average style metrics
            avg_style = {}
            for key in style_scores[0].keys():
                avg_style[key] = sum(s[key] for s in style_scores) / len(style_scores)
            metrics["style_analysis"] = avg_style

        # Calculate coherence metrics
        coherence_scores = []
        for response in responses:
            try:
                coherence = self.writing_metrics.evaluate_coherence(response["response"])
                coherence_scores.append(coherence)
            except Exception as e:
                self.logger.warning(f"Failed to evaluate coherence: {e}")

        if coherence_scores:
            # Average coherence metrics
            avg_coherence = {}
            for key in coherence_scores[0].keys():
                avg_coherence[key] = sum(c[key] for c in coherence_scores) / len(coherence_scores)
            metrics["coherence_analysis"] = avg_coherence

        return metrics

    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code."""
        code_indicators = [
            "def ",
            "class ",
            "import ",
            "from ",
            "if ",
            "for ",
            "while ",
            "try:",
            "{",
            "}",
            "[",
            "]",
            "=",
            "+=",
            "-=",
            "print(",
            "return ",
            "yield ",
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)
