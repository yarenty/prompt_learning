"""
Abstract base classes for core Memento framework components.

This module defines the interfaces that all core components must implement,
enabling extensibility and consistent behavior across different implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import ModelConfig
from ..utils.logger import LoggerMixin


class BaseLearner(ABC, LoggerMixin):
    """Abstract base class for prompt learning strategies."""

    def __init__(self, model_config: ModelConfig, storage_path: Union[str, Path], **kwargs: Any):
        super().__init__()
        self.model_config = model_config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.evolution_history: List[Dict[str, Any]] = []

    @abstractmethod
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
            EvaluationError: If evaluation fails
            ValidationError: If inputs are invalid
        """
        pass

    @abstractmethod
    async def evolve_prompt(self, current_prompt: str, lessons: List[Dict[str, Any]]) -> str:
        """
        Evolve a system prompt by incorporating lessons learned.

        Args:
            current_prompt: The current system prompt
            lessons: List of lessons learned from evaluations

        Returns:
            Updated system prompt

        Raises:
            EvolutionError: If prompt evolution fails
            ValidationError: If inputs are invalid
        """
        pass

    @abstractmethod
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
            StorageError: If saving fails
        """
        pass

    def get_evolution_history(self, prompt_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the evolution history for a specific prompt type or all types.

        Args:
            prompt_type: Optional filter for specific prompt type

        Returns:
            List of evolution history entries
        """
        if prompt_type:
            return [step for step in self.evolution_history if step.get("prompt_type") == prompt_type]
        return self.evolution_history.copy()


class BaseCollector(ABC, LoggerMixin):
    """Abstract base class for feedback collection strategies."""

    def __init__(self, model_config: ModelConfig, storage_path: Union[str, Path], **kwargs: Any):
        super().__init__()
        self.model_config = model_config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def collect_solution_feedback(
        self, problem: str, solution: str, evaluation_criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Collect feedback for a solution including quality metrics and reflections.

        Args:
            problem: The original problem statement
            solution: The proposed solution
            evaluation_criteria: List of criteria to evaluate the solution against

        Returns:
            Dict containing feedback data

        Raises:
            CollectionError: If feedback collection fails
            ValidationError: If inputs are invalid
        """
        pass

    @abstractmethod
    async def evaluate_solution(self, solution: str, criteria: List[str]) -> Dict[str, float]:
        """
        Evaluate solution quality against given criteria.

        Args:
            solution: The solution to evaluate
            criteria: List of evaluation criteria

        Returns:
            Dictionary mapping criteria to scores

        Raises:
            EvaluationError: If evaluation fails
        """
        pass

    @abstractmethod
    async def generate_reflection(self, problem: str, solution: str) -> str:
        """
        Generate reflection on the problem-solving process.

        Args:
            problem: The original problem
            solution: The proposed solution

        Returns:
            Generated reflection text

        Raises:
            ReflectionError: If reflection generation fails
        """
        pass


class BaseProcessor(ABC, LoggerMixin):
    """Abstract base class for feedback processing strategies."""

    def __init__(
        self, model_config: ModelConfig, feedback_path: Union[str, Path], prompt_path: Union[str, Path], **kwargs: Any
    ):
        super().__init__()
        self.model_config = model_config
        self.feedback_path = Path(feedback_path)
        self.prompt_path = Path(prompt_path)
        self.prompt_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def process_feedback(self) -> List[Dict[str, Any]]:
        """
        Process all feedback files and extract insights.

        Returns:
            List of extracted insights

        Raises:
            ProcessingError: If processing fails
        """
        pass

    @abstractmethod
    async def extract_insights(self, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and cluster similar insights from feedback.

        Args:
            feedback: List of feedback data

        Returns:
            List of extracted insights

        Raises:
            ExtractionError: If insight extraction fails
        """
        pass

    @abstractmethod
    async def update_system_prompt(self, insights: List[Dict[str, Any]]) -> str:
        """
        Update the system prompt with new insights.

        Args:
            insights: List of insights to incorporate

        Returns:
            Updated system prompt

        Raises:
            UpdateError: If prompt update fails
        """
        pass
