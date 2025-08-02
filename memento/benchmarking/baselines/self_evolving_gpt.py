"""
Self-Evolving GPT Implementation

Based on experience accumulation and memory management approaches.
This implementation focuses on learning from past interactions and
gradually improving performance through experience replay.

Key features:
- Experience memory system
- Adaptive learning rates
- Experience replay mechanism
- Performance-based experience weighting
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ollama

from ...config import ModelConfig


@dataclass
class Experience:
    """Represents a learning experience."""

    problem: str
    prompt_used: str
    solution_generated: str
    feedback_score: float
    timestamp: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def success(self) -> bool:
        """Whether this experience was successful."""
        return self.feedback_score > 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create from dictionary."""
        return cls(**data)


class ExperienceMemory:
    """Memory system for storing and retrieving experiences."""

    def __init__(self, max_size: int = 1000, decay_factor: float = 0.95):
        """
        Initialize experience memory.

        Args:
            max_size: Maximum number of experiences to store
            decay_factor: Factor for decaying old experience importance
        """
        self.max_size = max_size
        self.decay_factor = decay_factor
        self.experiences: deque = deque(maxlen=max_size)
        self.success_rate_history: List[float] = []

    def add_experience(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        self.experiences.append(experience)

        # Update success rate history
        recent_experiences = list(self.experiences)[-50:]  # Last 50 experiences
        if recent_experiences:
            success_rate = sum(exp.success for exp in recent_experiences) / len(recent_experiences)
            self.success_rate_history.append(success_rate)

    def get_relevant_experiences(self, problem: str, top_k: int = 10) -> List[Experience]:
        """
        Retrieve most relevant experiences for a given problem.

        Args:
            problem: Current problem to solve
            top_k: Number of top experiences to return

        Returns:
            List of most relevant experiences
        """
        if not self.experiences:
            return []

        # Simple relevance scoring based on problem similarity
        scored_experiences = []
        for exp in self.experiences:
            relevance_score = self._calculate_relevance(problem, exp)
            scored_experiences.append((relevance_score, exp))

        # Sort by relevance and return top-k
        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored_experiences[:top_k]]

    def _calculate_relevance(self, problem: str, experience: Experience) -> float:
        """Calculate relevance score between problem and experience."""
        # Simple word overlap scoring
        problem_words = set(problem.lower().split())
        exp_words = set(experience.problem.lower().split())

        if not problem_words or not exp_words:
            return 0.0

        overlap = len(problem_words.intersection(exp_words))
        union = len(problem_words.union(exp_words))

        jaccard_similarity = overlap / union if union > 0 else 0.0

        # Weight by success and recency
        success_weight = 1.2 if experience.success else 0.8
        recency_weight = self._calculate_recency_weight(experience.timestamp)

        return jaccard_similarity * success_weight * recency_weight

    def _calculate_recency_weight(self, timestamp: float) -> float:
        """Calculate weight based on experience recency."""
        current_time = time.time()
        age_hours = (current_time - timestamp) / 3600

        # Exponential decay with half-life of 24 hours
        return np.exp(-age_hours / 24)

    def get_success_rate(self) -> float:
        """Get current success rate."""
        if not self.experiences:
            return 0.0

        recent_experiences = list(self.experiences)[-100:]  # Last 100 experiences
        return sum(exp.success for exp in recent_experiences) / len(recent_experiences)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.experiences:
            return {"total_experiences": 0, "success_rate": 0.0}

        total_experiences = len(self.experiences)
        success_rate = self.get_success_rate()

        # Calculate improvement trend
        if len(self.success_rate_history) >= 2:
            recent_trend = np.mean(self.success_rate_history[-10:]) - np.mean(self.success_rate_history[-20:-10])
        else:
            recent_trend = 0.0

        return {
            "total_experiences": total_experiences,
            "success_rate": success_rate,
            "improvement_trend": recent_trend,
            "memory_utilization": total_experiences / self.max_size,
        }


class SelfEvolvingGPT:
    """
    Self-Evolving GPT implementation with experience-based learning.

    This class implements a system that:
    - Accumulates experiences from problem-solving attempts
    - Uses experience replay to improve future performance
    - Adapts learning rates based on success patterns
    - Maintains a memory of successful strategies
    """

    def __init__(
        self,
        model_config: ModelConfig,
        memory_size: int = 1000,
        learning_rate: float = 0.1,
        adaptation_rate: float = 0.05,
        experience_replay_ratio: float = 0.3,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize Self-Evolving GPT.

        Args:
            model_config: Model configuration for LLM calls
            memory_size: Maximum size of experience memory
            learning_rate: Base learning rate for prompt updates
            adaptation_rate: Rate of learning rate adaptation
            experience_replay_ratio: Fraction of training using experience replay
            storage_path: Path to store experiences and models
        """
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.base_learning_rate = learning_rate
        self.adaptation_rate = adaptation_rate
        self.experience_replay_ratio = experience_replay_ratio

        self.storage_path = storage_path or Path("./self_evolving_gpt_data")
        self.storage_path.mkdir(exist_ok=True)

        # Initialize memory system
        self.memory = ExperienceMemory(max_size=memory_size)

        # Evolution state
        self.current_prompt = ""
        self.evolution_history: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []

        # Load existing experiences if available
        self._load_experiences()

    async def evolve(
        self,
        initial_prompt: str,
        evaluation_function: callable,
        problem_set: List[Dict[str, Any]],
        num_iterations: int = 20,
    ) -> str:
        """
        Run the self-evolution process.

        Args:
            initial_prompt: Starting prompt
            evaluation_function: Function to evaluate prompt performance
            problem_set: Set of problems for training
            num_iterations: Number of evolution iterations

        Returns:
            Final evolved prompt
        """
        self.current_prompt = initial_prompt

        for iteration in range(num_iterations):
            print(f"Self-Evolution Iteration {iteration + 1}/{num_iterations}")

            # Evaluate current prompt
            performance = await evaluation_function(self.current_prompt, problem_set)
            self.performance_history.append(performance)

            # Process problems and collect experiences
            await self._collect_experiences(problem_set)

            # Update prompt based on experiences
            if iteration < num_iterations - 1:  # Don't update on last iteration
                await self._update_prompt_from_experiences()

            # Adapt learning rate
            self._adapt_learning_rate()

            # Log progress
            self._log_iteration(iteration, performance)

        # Save final state
        await self._save_state()

        return self.current_prompt

    async def _collect_experiences(self, problem_set: List[Dict[str, Any]]) -> None:
        """Collect experiences by solving problems with current prompt."""
        for problem_data in problem_set:
            try:
                # Generate solution using current prompt
                solution = await self._generate_solution(self.current_prompt, problem_data["description"])

                # Evaluate the solution
                feedback_score = await self._evaluate_solution(solution, problem_data)

                # Create experience
                experience = Experience(
                    problem=problem_data["description"],
                    prompt_used=self.current_prompt,
                    solution_generated=solution,
                    feedback_score=feedback_score,
                    timestamp=time.time(),
                    metadata={"problem_id": problem_data.get("id", "unknown")},
                )

                # Add to memory
                self.memory.add_experience(experience)

            except Exception as e:
                print(f"Error processing problem: {e}")
                continue

    async def _generate_solution(self, prompt: str, problem: str) -> str:
        """Generate solution using the current prompt."""
        full_prompt = f"{prompt}\n\nProblem: {problem}\n\nSolution:"

        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=full_prompt,
            options={"temperature": 0.3, "num_predict": 300},
        )

        return response["response"].strip()

    async def _evaluate_solution(self, solution: str, problem_data: Dict[str, Any]) -> float:
        """Evaluate a solution and return feedback score."""
        # Simple evaluation based on solution quality
        evaluation_prompt = f"""
        Evaluate this solution to the given problem on a scale of 0.0 to 1.0:

        Problem: {problem_data["description"]}
        Solution: {solution}

        Consider:
        - Correctness of the approach
        - Completeness of the solution
        - Code quality (if applicable)
        - Clarity of explanation

        Return only a single number between 0.0 and 1.0.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=evaluation_prompt,
                options={"temperature": 0.1, "num_predict": 50},
            )

            score_text = response["response"].strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except (ValueError, Exception):
            return 0.5  # Default neutral score

    async def _update_prompt_from_experiences(self) -> None:
        """Update the current prompt based on accumulated experiences."""
        # Get recent successful experiences
        successful_experiences = [exp for exp in list(self.memory.experiences)[-50:] if exp.success]

        if not successful_experiences:
            return

        # Analyze patterns in successful experiences
        patterns = await self._analyze_success_patterns(successful_experiences)

        # Generate prompt improvements
        improved_prompt = await self._generate_improved_prompt(patterns)

        if improved_prompt and improved_prompt != self.current_prompt:
            self.current_prompt = improved_prompt

    async def _analyze_success_patterns(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Analyze patterns in successful experiences."""
        analysis_prompt = f"""
        Analyze these successful problem-solving experiences and identify key patterns:

        Experiences:
        {self._format_experiences_for_analysis(experiences[:10])}

        Identify:
        1. Common approaches that led to success
        2. Effective problem-solving strategies
        3. Patterns in language or structure
        4. Key elements that should be preserved

        Return a JSON object with your analysis.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=analysis_prompt,
                options={"temperature": 0.3, "num_predict": 400},
            )

            # Try to parse JSON response
            analysis_text = response["response"].strip()
            if analysis_text.startswith("{"):
                return json.loads(analysis_text)
            else:
                return {"analysis": analysis_text}

        except Exception as e:
            return {"error": str(e), "fallback_analysis": "Pattern analysis failed"}

    def _format_experiences_for_analysis(self, experiences: List[Experience]) -> str:
        """Format experiences for analysis prompt."""
        formatted = []
        for i, exp in enumerate(experiences, 1):
            formatted.append(
                f"""
            Experience {i}:
            Problem: {exp.problem[:100]}...
            Solution: {exp.solution_generated[:150]}...
            Score: {exp.feedback_score:.2f}
            """
            )
        return "\n".join(formatted)

    async def _generate_improved_prompt(self, patterns: Dict[str, Any]) -> str:
        """Generate an improved prompt based on success patterns."""
        improvement_prompt = f"""
        Current system prompt:
        {self.current_prompt}

        Success patterns identified:
        {json.dumps(patterns, indent=2)}

        Generate an improved version of the system prompt that:
        1. Incorporates the successful patterns
        2. Maintains the core functionality
        3. Addresses any identified weaknesses
        4. Remains clear and actionable

        Return only the improved prompt without explanation.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=improvement_prompt,
                options={"temperature": 0.4, "num_predict": 300},
            )

            return response["response"].strip()

        except Exception:
            return self.current_prompt  # Return unchanged if improvement fails

    def _adapt_learning_rate(self) -> None:
        """Adapt learning rate based on recent performance."""
        if len(self.performance_history) < 2:
            return

        # Calculate recent improvement
        recent_performance = np.mean(self.performance_history[-5:])
        older_performance = (
            np.mean(self.performance_history[-10:-5]) if len(self.performance_history) >= 10 else recent_performance
        )

        improvement = recent_performance - older_performance

        if improvement > 0:
            # Increase learning rate if improving
            self.learning_rate = min(self.learning_rate * (1 + self.adaptation_rate), self.base_learning_rate * 2)
        else:
            # Decrease learning rate if not improving
            self.learning_rate = max(self.learning_rate * (1 - self.adaptation_rate), self.base_learning_rate * 0.1)

    def _log_iteration(self, iteration: int, performance: float) -> None:
        """Log iteration results."""
        memory_stats = self.memory.get_memory_stats()

        iteration_data = {
            "iteration": iteration,
            "performance": performance,
            "learning_rate": self.learning_rate,
            "memory_stats": memory_stats,
            "prompt_length": len(self.current_prompt),
            "timestamp": time.time(),
        }

        self.evolution_history.append(iteration_data)

        print(f"  Performance: {performance:.3f}")
        print(f"  Success Rate: {memory_stats['success_rate']:.3f}")
        print(f"  Learning Rate: {self.learning_rate:.4f}")
        print(f"  Experiences: {memory_stats['total_experiences']}")

    async def _save_state(self) -> None:
        """Save current state and experiences."""
        # Save experiences
        experiences_data = [exp.to_dict() for exp in self.memory.experiences]
        experiences_file = self.storage_path / "experiences.json"
        with open(experiences_file, "w") as f:
            json.dump(experiences_data, f, indent=2)

        # Save evolution results
        results = {
            "final_prompt": self.current_prompt,
            "evolution_history": self.evolution_history,
            "performance_history": self.performance_history,
            "memory_stats": self.memory.get_memory_stats(),
            "parameters": {
                "learning_rate": self.learning_rate,
                "base_learning_rate": self.base_learning_rate,
                "adaptation_rate": self.adaptation_rate,
                "experience_replay_ratio": self.experience_replay_ratio,
            },
        }

        results_file = self.storage_path / f"self_evolving_results_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"State saved to {self.storage_path}")

    def _load_experiences(self) -> None:
        """Load existing experiences if available."""
        experiences_file = self.storage_path / "experiences.json"
        if experiences_file.exists():
            try:
                with open(experiences_file, "r") as f:
                    experiences_data = json.load(f)

                for exp_data in experiences_data:
                    experience = Experience.from_dict(exp_data)
                    self.memory.add_experience(experience)

                print(f"Loaded {len(experiences_data)} existing experiences")

            except Exception as e:
                print(f"Failed to load experiences: {e}")

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of the evolution process."""
        if not self.evolution_history:
            return {"status": "No evolution history available"}

        initial_performance = self.performance_history[0] if self.performance_history else 0
        final_performance = self.performance_history[-1] if self.performance_history else 0
        improvement = final_performance - initial_performance

        return {
            "total_iterations": len(self.evolution_history),
            "initial_performance": initial_performance,
            "final_performance": final_performance,
            "improvement": improvement,
            "improvement_percentage": (improvement / initial_performance * 100) if initial_performance > 0 else 0,
            "final_prompt": self.current_prompt,
            "memory_stats": self.memory.get_memory_stats(),
            "learning_rate": self.learning_rate,
        }
