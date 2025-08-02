"""
Evaluation metrics for prompt evolution benchmarking.
"""

import asyncio
from typing import Any, Dict

import ollama

from ...config import ModelConfig


class EvaluationMetrics:
    """Standardized evaluation metrics for prompt comparison."""

    async def evaluate_prompt_on_problem(
        self, prompt: str, problem: Dict[str, Any], model_config: ModelConfig
    ) -> float:
        """Evaluate a prompt on a single problem."""
        try:
            # Generate solution using the prompt
            solution = await self._generate_solution(prompt, problem["description"], model_config)

            # Evaluate solution quality
            score = await self._evaluate_solution_quality(solution, problem, model_config)

            return score
        except Exception:
            return 0.0

    async def _generate_solution(self, prompt: str, problem: str, model_config: ModelConfig) -> str:
        """Generate solution using the prompt."""
        full_prompt = f"{prompt}\n\nProblem: {problem}\n\nSolution:"

        response = await asyncio.to_thread(
            ollama.generate,
            model=model_config.model_name,
            prompt=full_prompt,
            options={"temperature": 0.3, "num_predict": 200},
        )

        return response["response"].strip()

    async def _evaluate_solution_quality(
        self, solution: str, problem: Dict[str, Any], model_config: ModelConfig
    ) -> float:
        """Evaluate solution quality."""
        evaluation_prompt = f"""
        Rate this solution on a scale of 0.0 to 1.0:
        
        Problem: {problem["description"]}
        Solution: {solution}
        
        Return only a number between 0.0 and 1.0.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=model_config.model_name,
                prompt=evaluation_prompt,
                options={"temperature": 0.1, "num_predict": 10},
            )

            score = float(response["response"].strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5
