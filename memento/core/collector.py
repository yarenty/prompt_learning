"""
Enhanced feedback collection system with async support, caching, and multiple evaluation backends.

This module provides a professional-grade FeedbackCollector that implements the BaseCollector
interface with comprehensive error handling, validation, performance metrics, and caching.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ollama

from ..config import EvaluationBackend, ModelConfig
from ..exceptions import CollectionError, EvaluationError, ReflectionError, ValidationError
from ..utils.metrics import PerformanceMonitor
from ..utils.validation import (
    validate_evaluation_criteria,
    validate_feedback_data,
    validate_json_response,
)
from .base import BaseCollector


class FeedbackCollector(BaseCollector):
    """
    Enhanced feedback collector with async support, caching, and multiple evaluation backends.

    Features:
    - Async LLM operations for better performance
    - Multiple evaluation backends (LLM, human, automated)
    - Caching system for evaluation results
    - Batch processing capabilities
    - Comprehensive validation and error handling
    - Performance metrics collection
    """

    def __init__(
        self,
        model_config: ModelConfig,
        storage_path: Union[str, Path],
        enable_metrics: bool = True,
        cache_evaluations: bool = True,
        evaluation_backend: EvaluationBackend = EvaluationBackend.LLM,
        **kwargs: Any,
    ):
        """
        Initialize the enhanced feedback collector.

        Args:
            model_config: Configuration for the LLM model
            storage_path: Path to store feedback data
            enable_metrics: Whether to collect performance metrics
            cache_evaluations: Whether to cache evaluation results
            evaluation_backend: Backend to use for evaluations
            **kwargs: Additional configuration options
        """
        super().__init__(model_config, storage_path, **kwargs)

        self.cache_evaluations = cache_evaluations
        self.evaluation_backend = evaluation_backend
        self._evaluation_cache: Dict[str, Dict[str, float]] = {}
        self._reflection_cache: Dict[str, str] = {}

        # Initialize performance monitoring
        self.monitor: Optional[PerformanceMonitor] = None
        if enable_metrics:
            self.monitor = PerformanceMonitor(storage_path=self.storage_path / "metrics")

        # Create cache directory
        self.cache_path = self.storage_path / "cache"
        self.cache_path.mkdir(exist_ok=True)

        self.logger.info(
            f"FeedbackCollector initialized with backend={evaluation_backend.value}, "
            f"caching={'enabled' if cache_evaluations else 'disabled'}, "
            f"metrics={'enabled' if enable_metrics else 'disabled'}"
        )

    async def collect_solution_feedback(
        self, problem: str, solution: str, evaluation_criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Collect comprehensive feedback for a solution with async support.

        Args:
            problem: The original problem statement
            solution: The proposed solution
            evaluation_criteria: List of criteria to evaluate the solution against

        Returns:
            Dict containing comprehensive feedback data

        Raises:
            CollectionError: If feedback collection fails
            ValidationError: If inputs are invalid
        """
        timer_id = None
        if self.monitor:
            timer_id = f"feedback_collection_{id(self)}"
            self.monitor.start_timer(timer_id)

        try:
            # Validate inputs
            if not problem or not problem.strip():
                raise ValidationError("Problem cannot be empty")
            if not solution or not solution.strip():
                raise ValidationError("Solution cannot be empty")
            validate_evaluation_criteria(evaluation_criteria)

            self.logger.info(f"Collecting feedback for solution with {len(evaluation_criteria)} criteria")

            # Collect evaluation and reflection concurrently
            evaluation_task = self.evaluate_solution(solution, evaluation_criteria)
            reflection_task = self.generate_reflection(problem, solution)

            evaluation, reflection = await asyncio.gather(evaluation_task, reflection_task)

            # Create comprehensive feedback data
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "problem": problem,
                "solution": solution,
                "evaluation_criteria": evaluation_criteria,
                "evaluation": evaluation,
                "reflection": reflection,
                "backend": self.evaluation_backend.value,
                "model_config": {
                    "model_name": self.model_config.model_name,
                    "model_type": self.model_config.model_type.value,
                    "temperature": self.model_config.temperature,
                },
                "metrics": {},
            }

            # Add performance metrics if available
            if self.monitor and timer_id:
                duration = self.monitor.stop_timer(
                    timer_id,
                    "collector",
                    "collect_solution_feedback",
                    metadata={"criteria_count": len(evaluation_criteria)},
                )
                feedback["metrics"] = {"duration": duration}

            # Validate feedback data
            validate_feedback_data(feedback)

            # Store feedback
            await self._store_feedback(feedback)

            self.logger.info("Feedback collection completed successfully")
            return feedback

        except Exception as e:
            if self.monitor and timer_id:
                self.monitor.stop_timer(timer_id, "collector", "collect_solution_feedback", metadata={"error": str(e)})

            if isinstance(e, (ValidationError, EvaluationError, ReflectionError)):
                raise

            raise CollectionError(f"Failed to collect solution feedback: {e}") from e

    async def evaluate_solution(self, solution: str, criteria: List[str]) -> Dict[str, float]:
        """
        Evaluate solution quality against given criteria with caching support.

        Args:
            solution: The solution to evaluate
            criteria: List of evaluation criteria

        Returns:
            Dictionary mapping criteria to scores (0.0 to 1.0)

        Raises:
            EvaluationError: If evaluation fails
        """
        # Check cache first
        cache_key = self._get_cache_key(solution, criteria)
        if self.cache_evaluations and cache_key in self._evaluation_cache:
            self.logger.debug("Using cached evaluation result")
            return self._evaluation_cache[cache_key]

        try:
            if self.evaluation_backend == EvaluationBackend.LLM:
                result = await self._evaluate_with_llm(solution, criteria)
            elif self.evaluation_backend == EvaluationBackend.AUTOMATED:
                result = await self._evaluate_automated(solution, criteria)
            else:
                raise EvaluationError(f"Unsupported evaluation backend: {self.evaluation_backend}")

            # Cache the result
            if self.cache_evaluations:
                self._evaluation_cache[cache_key] = result

            return result

        except Exception as e:
            if isinstance(e, EvaluationError):
                raise
            raise EvaluationError(f"Failed to evaluate solution: {e}") from e

    async def generate_reflection(self, problem: str, solution: str) -> str:
        """
        Generate reflection on the problem-solving process with caching support.

        Args:
            problem: The original problem
            solution: The proposed solution

        Returns:
            Generated reflection text

        Raises:
            ReflectionError: If reflection generation fails
        """
        # Check cache first
        cache_key = self._get_cache_key(problem + solution, ["reflection"])
        if self.cache_evaluations and cache_key in self._reflection_cache:
            self.logger.debug("Using cached reflection result")
            return self._reflection_cache[cache_key]

        try:
            reflection_prompt = f"""
            Analyze the following problem-solving process and provide insights:

            Problem:
            {problem}

            Solution:
            {solution}

            Please provide a reflection that covers:
            1. The approach taken to solve the problem
            2. Strengths and weaknesses of the solution
            3. Alternative approaches that could be considered
            4. Lessons learned from this solution

            Keep the reflection concise but insightful.
            """

            start_time = time.time()
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=reflection_prompt,
                options={
                    "temperature": self.model_config.temperature,
                    "num_predict": 500,
                },
            )
            duration = time.time() - start_time

            reflection = response["response"].strip()

            if not reflection:
                raise ReflectionError("Generated reflection is empty")

            # Cache the result
            if self.cache_evaluations:
                self._reflection_cache[cache_key] = reflection

            self.logger.debug(f"Generated reflection in {duration:.2f}s")
            return reflection

        except Exception as e:
            if isinstance(e, ReflectionError):
                raise
            raise ReflectionError(f"Failed to generate reflection: {e}") from e

    async def _evaluate_with_llm(self, solution: str, criteria: List[str]) -> Dict[str, float]:
        """Evaluate solution using LLM backend."""
        criteria_str = ", ".join(criteria)
        evaluation_prompt = f"""
        Evaluate the following solution against these criteria: {criteria_str}
        For each criterion, provide a score between 0.0 and 1.0.
        
        Solution:
        {solution}
        
        Respond with a JSON object where keys are criteria and values are numeric scores.
        Example: {{"correctness": 0.85, "efficiency": 0.70, "readability": 0.90}}
        """

        start_time = time.time()
        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=evaluation_prompt,
            options={
                "temperature": 0.1,  # Lower temperature for more consistent scoring
                "num_predict": 200,
            },
        )
        duration = time.time() - start_time

        try:
            # Extract and validate JSON response
            response_text = response["response"].strip()
            evaluation_data = validate_json_response(response_text)

            # Ensure all criteria are present and scores are valid
            result = {}
            for criterion in criteria:
                if criterion not in evaluation_data:
                    self.logger.warning(f"Missing evaluation for criterion: {criterion}")
                    result[criterion] = 0.5  # Default neutral score
                else:
                    score = float(evaluation_data[criterion])
                    result[criterion] = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            self.logger.debug(f"LLM evaluation completed in {duration:.2f}s")
            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise EvaluationError(f"Invalid LLM evaluation response: {e}") from e

    async def _evaluate_automated(self, solution: str, criteria: List[str]) -> Dict[str, float]:
        """Evaluate solution using automated heuristics."""
        result = {}

        for criterion in criteria:
            if criterion.lower() in ["readability", "clarity"]:
                # Simple readability heuristic based on length and complexity
                lines = solution.split("\n")
                avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
                score = max(0.0, min(1.0, 1.0 - (avg_line_length - 50) / 100))
                result[criterion] = score

            elif criterion.lower() in ["efficiency", "performance"]:
                # Simple efficiency heuristic based on algorithmic patterns
                score = 0.7  # Default moderate score
                if "for" in solution.lower() and "for" in solution.lower():
                    score -= 0.2  # Nested loops penalty
                if "sort" in solution.lower():
                    score += 0.1  # Sorting bonus
                result[criterion] = max(0.0, min(1.0, score))

            else:
                # Default neutral score for unknown criteria
                result[criterion] = 0.6

        await asyncio.sleep(0.1)  # Simulate processing time
        return result

    async def _store_feedback(self, feedback: Dict[str, Any]) -> None:
        """Store feedback data to file system."""
        timestamp = feedback["timestamp"].replace(":", "-").replace(".", "-")
        filename = f"feedback_{timestamp}.json"
        filepath = self.storage_path / filename

        try:
            await asyncio.to_thread(filepath.write_text, json.dumps(feedback, indent=2, ensure_ascii=False))
            self.logger.debug(f"Feedback stored to {filepath}")
        except Exception as e:
            raise CollectionError(f"Failed to store feedback: {e}") from e

    def _get_cache_key(self, content: str, criteria: List[str]) -> str:
        """Generate cache key for content and criteria."""
        import hashlib

        combined = content + "|".join(sorted(criteria))
        return hashlib.md5(combined.encode()).hexdigest()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance metrics report."""
        if not self.monitor:
            return {"message": "Performance monitoring is disabled"}
        return self.monitor.get_performance_report()

    def clear_cache(self) -> None:
        """Clear evaluation and reflection caches."""
        self._evaluation_cache.clear()
        self._reflection_cache.clear()
        self.logger.info("Caches cleared")

    async def batch_collect_feedback(
        self, problems_solutions: List[Dict[str, Any]], evaluation_criteria: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Collect feedback for multiple problem-solution pairs in batch.

        Args:
            problems_solutions: List of dicts with 'problem' and 'solution' keys
            evaluation_criteria: Common evaluation criteria for all solutions

        Returns:
            List of feedback dictionaries

        Raises:
            CollectionError: If batch collection fails
        """
        try:
            tasks = []
            for item in problems_solutions:
                task = self.collect_solution_feedback(item["problem"], item["solution"], evaluation_criteria)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            feedback_list = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to collect feedback for item {i}: {result}")
                    continue
                feedback_list.append(result)

            self.logger.info(
                f"Batch feedback collection completed: {len(feedback_list)}/{len(problems_solutions)} successful"
            )
            return feedback_list

        except Exception as e:
            raise CollectionError(f"Batch feedback collection failed: {e}") from e
