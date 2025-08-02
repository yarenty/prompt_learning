"""
Enhanced core module for system prompt evolution through self-learning.

This module provides the PromptLearner class with async support, comprehensive
error handling, configuration validation, and performance metrics collection.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import ollama

from ..config import ModelConfig
from ..exceptions import EvaluationError, EvolutionError, StorageError, ValidationError
from ..utils.metrics import PerformanceMonitor
from ..utils.validation import (
    validate_evaluation_criteria,
    validate_json_response,
    validate_lesson,
    validate_problem,
    validate_prompt,
)
from .base import BaseLearner


class PromptLearner(BaseLearner):
    """
    Enhanced prompt learning system with async support and comprehensive error handling.

    This class implements the BaseLearner interface and provides:
    - Async/await support for all operations
    - Comprehensive error handling with custom exceptions
    - Input validation for all parameters
    - Performance metrics collection
    - Configuration validation
    """

    def __init__(
        self, model_config: ModelConfig, storage_path: Union[str, Path], enable_metrics: bool = True, **kwargs: Any
    ):
        """
        Initialize the PromptLearner.

        Args:
            model_config: Model configuration
            storage_path: Path for storing evolution data
            enable_metrics: Whether to enable performance metrics collection
            **kwargs: Additional configuration options

        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__(model_config, storage_path, **kwargs)

        # Initialize performance monitoring
        self.enable_metrics = enable_metrics
        if self.enable_metrics:
            metrics_path = self.storage_path / "metrics"
            self.monitor = PerformanceMonitor(metrics_path)
        else:
            self.monitor = None

        # Validate model configuration
        self._validate_model_config()

        self.logger.info(f"Initialized PromptLearner with model: {self.model_config.model_name}")

    def _validate_model_config(self) -> None:
        """Validate the model configuration."""
        if not self.model_config.model_name:
            raise ValidationError("Model name is required")

        if self.model_config.model_type.value not in ["ollama", "openai", "anthropic", "local"]:
            raise ValidationError(f"Unsupported model type: {self.model_config.model_type.value}")

    async def evaluate_prompt_performance(
        self, prompt: str, problem: Dict[str, str], evaluation_criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate how well a system prompt performs on a given problem.

        Args:
            prompt: The system prompt to evaluate
            problem: Dictionary containing problem description and solution
            evaluation_criteria: List of criteria to evaluate against

        Returns:
            Dictionary containing evaluation results and insights

        Raises:
            ValidationError: If inputs are invalid
            EvaluationError: If evaluation fails
        """
        # Input validation
        try:
            validated_prompt = validate_prompt(prompt)
            validated_problem = validate_problem(problem)
            validated_criteria = validate_evaluation_criteria(evaluation_criteria)
        except ValidationError as e:
            self.logger.error(f"Validation error: {e}")
            raise

        # Start performance monitoring
        timer_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        if self.monitor:
            self.monitor.start_timer(timer_id)

        try:
            self.logger.info(f"Starting prompt evaluation for problem: {validated_problem['description'][:50]}...")

            # Create evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(validated_prompt, validated_problem, validated_criteria)

            # Generate evaluation using LLM
            evaluation_response = await self._generate_evaluation(evaluation_prompt)

            # Parse and validate evaluation
            evaluation = self._parse_evaluation_response(evaluation_response, validated_criteria)

            # Extract lessons learned
            lessons = self._extract_lessons(evaluation)

            # Record performance metrics
            if self.monitor:
                evaluation_time = self.monitor.stop_timer(
                    timer_id,
                    "learner",
                    "evaluate_prompt_performance",
                    metadata={"prompt_length": len(validated_prompt)},
                )

                scores = [v.get("score", 0.0) if isinstance(v, dict) else v for v in evaluation.values()]
                self.monitor.record_evaluation(
                    prompt_length=len(validated_prompt),
                    problem_complexity=self._calculate_problem_complexity(validated_problem),
                    evaluation_time=evaluation_time,
                    criteria_count=len(validated_criteria),
                    scores=scores,
                )

            result = {
                "timestamp": datetime.now().isoformat(),
                "prompt": validated_prompt,
                "problem": validated_problem,
                "evaluation": evaluation,
                "lessons": lessons,
            }

            self.logger.info(f"Completed prompt evaluation with {len(lessons)} lessons extracted")
            return result

        except Exception as e:
            if self.monitor:
                self.monitor.stop_timer(timer_id, "learner", "evaluate_prompt_performance")

            self.logger.error(f"Evaluation failed: {e}")
            raise EvaluationError(f"Prompt evaluation failed: {e}") from e

    async def evolve_prompt(self, current_prompt: str, lessons: List[Dict[str, Any]]) -> str:
        """
        Evolve a system prompt by incorporating lessons learned.

        Args:
            current_prompt: The current system prompt
            lessons: List of lessons learned from evaluations

        Returns:
            Updated system prompt

        Raises:
            ValidationError: If inputs are invalid
            EvolutionError: If prompt evolution fails
        """
        # Input validation
        try:
            validated_prompt = validate_prompt(current_prompt)
            validated_lessons = [validate_lesson(lesson) for lesson in lessons]
        except ValidationError as e:
            self.logger.error(f"Validation error: {e}")
            raise

        # Start performance monitoring
        timer_id = f"evolve_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        if self.monitor:
            self.monitor.start_timer(timer_id)

        try:
            self.logger.info(f"Starting prompt evolution with {len(validated_lessons)} lessons")

            # Create evolution prompt
            evolution_prompt = self._create_evolution_prompt(validated_prompt, validated_lessons)

            # Generate evolved prompt using LLM
            evolved_prompt = await self._generate_evolved_prompt(evolution_prompt)

            # Validate evolved prompt
            validated_evolved_prompt = validate_prompt(evolved_prompt)

            # Record performance metrics
            if self.monitor:
                evolution_time = self.monitor.stop_timer(
                    timer_id, "learner", "evolve_prompt", metadata={"lessons_count": len(validated_lessons)}
                )

            self.logger.info(f"Completed prompt evolution in {evolution_time if self.monitor else 'unknown'} seconds")
            return validated_evolved_prompt

        except Exception as e:
            if self.monitor:
                self.monitor.stop_timer(timer_id, "learner", "evolve_prompt")

            self.logger.error(f"Prompt evolution failed: {e}")
            raise EvolutionError(f"Prompt evolution failed: {e}") from e

    async def save_evolution_step(
        self, prompt_type: str, current_prompt: str, updated_prompt: str, evaluation_results: List[Dict[str, Any]]
    ) -> None:
        """
        Save a step in the prompt evolution process.

        Args:
            prompt_type: Type/category of the prompt
            current_prompt: The prompt before evolution
            updated_prompt: The prompt after evolution
            evaluation_results: Results from the evaluation

        Raises:
            ValidationError: If inputs are invalid
            StorageError: If saving fails
        """
        # Input validation
        if not isinstance(prompt_type, str) or not prompt_type.strip():
            raise ValidationError("Prompt type must be a non-empty string")

        try:
            validated_current = validate_prompt(current_prompt)
            validated_updated = validate_prompt(updated_prompt)
        except ValidationError as e:
            self.logger.error(f"Validation error: {e}")
            raise

        # Start performance monitoring
        timer_id = f"save_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        if self.monitor:
            self.monitor.start_timer(timer_id)

        try:
            step = {
                "timestamp": datetime.now().isoformat(),
                "prompt_type": prompt_type.strip(),
                "current_prompt": validated_current,
                "updated_prompt": validated_updated,
                "evaluation_results": evaluation_results,
            }

            # Add to in-memory history
            self.evolution_history.append(step)

            # Save to file
            await self._save_evolution_file(step, prompt_type)

            # Record performance metrics
            if self.monitor:
                self.monitor.stop_timer(
                    timer_id, "learner", "save_evolution_step", metadata={"prompt_type": prompt_type}
                )

            self.logger.info(f"Saved evolution step for {prompt_type}")

        except Exception as e:
            if self.monitor:
                self.monitor.stop_timer(timer_id, "learner", "save_evolution_step")

            self.logger.error(f"Failed to save evolution step: {e}")
            raise StorageError(f"Failed to save evolution step: {e}") from e

    def _create_evaluation_prompt(self, prompt: str, problem: Dict[str, str], criteria: List[str]) -> str:
        """Create the evaluation prompt."""
        criteria_str = ", ".join(criteria)
        return f"""
        You are evaluating a system prompt's performance on a coding problem.
        
        System Prompt:
        {prompt}
        
        Problem:
        {problem['description']}
        
        Solution:
        {problem['solution']}
        
        Evaluate the solution based on these criteria: {criteria_str}
        For each criterion, provide:
        1. A score between 0.0 and 1.0
        2. A brief explanation of the score
        3. A lesson learned that could improve the system prompt
        
        Provide your evaluation in JSON format with the following structure:
        {{
            "criterion_name": {{
                "score": 0.85,
                "explanation": "Brief explanation",
                "lesson": "Actionable lesson learned"
            }}
        }}
        """

    def _create_evolution_prompt(self, current_prompt: str, lessons: List[Dict[str, Any]]) -> str:
        """Create the evolution prompt."""
        lessons_text = "\n".join([f"- {lesson['criterion']}: {lesson['lesson']}" for lesson in lessons])

        return f"""
        You are evolving a system prompt by incorporating lessons learned.
        
        Current System Prompt:
        {current_prompt}
        
        Lessons Learned:
        {lessons_text}
        
        Create an improved version of the system prompt that incorporates these lessons.
        The new prompt should:
        1. Maintain the original personality and approach
        2. Integrate the lessons naturally
        3. Be more effective at solving similar problems
        4. Be clear and concise
        
        Provide the updated system prompt.
        """

    async def _generate_evaluation(self, prompt: str) -> str:
        """Generate evaluation using the LLM."""
        try:
            if self.model_config.model_type.value == "ollama":
                response = await asyncio.to_thread(
                    ollama.generate, model=self.model_config.model_name, prompt=prompt, format="json"
                )
                return response.response
            else:
                # TODO: Implement other model types
                raise EvaluationError(f"Model type {self.model_config.model_type.value} not yet implemented")
        except Exception as e:
            raise EvaluationError(f"Failed to generate evaluation: {e}") from e

    async def _generate_evolved_prompt(self, prompt: str) -> str:
        """Generate evolved prompt using the LLM."""
        try:
            if self.model_config.model_type.value == "ollama":
                response = await asyncio.to_thread(ollama.generate, model=self.model_config.model_name, prompt=prompt)
                return response.response.strip()
            else:
                # TODO: Implement other model types
                raise EvolutionError(f"Model type {self.model_config.model_type.value} not yet implemented")
        except Exception as e:
            raise EvolutionError(f"Failed to generate evolved prompt: {e}") from e

    def _parse_evaluation_response(self, response: str, criteria: List[str]) -> Dict[str, Any]:
        """Parse and validate the evaluation response."""
        try:
            evaluation = validate_json_response(response)
        except ValidationError as e:
            raise EvaluationError(f"Invalid evaluation response: {e}") from e

        # Ensure all criteria are present
        for criterion in criteria:
            if criterion not in evaluation:
                evaluation[criterion] = {"score": 0.0, "explanation": "", "lesson": ""}

        return evaluation

    def _extract_lessons(self, evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract lessons learned from evaluation results."""
        lessons = []
        for criterion, data in evaluation.items():
            if isinstance(data, dict) and "lesson" in data:
                lesson = {
                    "criterion": criterion,
                    "score": data.get("score", 0.0),
                    "explanation": data.get("explanation", ""),
                    "lesson": data["lesson"],
                }
                lessons.append(lesson)
        return lessons

    def _calculate_problem_complexity(self, problem: Dict[str, str]) -> float:
        """Calculate a simple complexity score for the problem."""
        # Simple heuristic based on description and solution length
        desc_length = len(problem["description"])
        sol_length = len(problem["solution"])

        # Normalize to 0-1 range
        complexity = min(1.0, (desc_length + sol_length) / 1000.0)
        return complexity

    async def _save_evolution_file(self, step: Dict[str, Any], prompt_type: str) -> None:
        """Save evolution step to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evolution_{prompt_type}_{timestamp}.json"
        file_path = self.storage_path / filename

        try:
            await asyncio.to_thread(file_path.write_text, json.dumps(step, indent=2))
        except Exception as e:
            raise StorageError(f"Failed to write evolution file: {e}") from e

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for this learner."""
        if not self.monitor:
            return {"message": "Performance monitoring is disabled"}

        return self.monitor.get_performance_report(component="learner")
