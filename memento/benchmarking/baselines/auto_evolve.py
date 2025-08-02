"""
Auto-Evolve Implementation

Based on self-reasoning and error correction approaches.
This implementation focuses on metacognitive reasoning about
prompt effectiveness and systematic error correction.

Key features:
- Self-reasoning framework
- Error detection and correction
- Reasoning chain validation
- Confidence estimation
- Iterative self-improvement
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama

from ...config import ModelConfig


@dataclass
class ReasoningStep:
    """Represents a step in the reasoning chain."""

    step_number: int
    description: str
    reasoning: str
    confidence: float
    evidence: List[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorAnalysis:
    """Analysis of errors and potential corrections."""

    error_type: str
    description: str
    root_cause: str
    suggested_fix: str
    confidence: float
    examples: List[str] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoEvolve:
    """
    Auto-Evolve implementation with self-reasoning framework.

    This class implements a system that:
    - Uses metacognitive reasoning to analyze prompt effectiveness
    - Detects and corrects systematic errors
    - Validates reasoning chains for consistency
    - Estimates confidence in improvements
    - Iteratively self-improves through reasoning
    """

    def __init__(
        self,
        model_config: ModelConfig,
        confidence_threshold: float = 0.7,
        max_reasoning_steps: int = 5,
        error_detection_sensitivity: float = 0.3,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize Auto-Evolve.

        Args:
            model_config: Model configuration for LLM calls
            confidence_threshold: Minimum confidence for accepting changes
            max_reasoning_steps: Maximum steps in reasoning chain
            error_detection_sensitivity: Sensitivity for error detection
            storage_path: Path to store reasoning logs and results
        """
        self.model_config = model_config
        self.confidence_threshold = confidence_threshold
        self.max_reasoning_steps = max_reasoning_steps
        self.error_detection_sensitivity = error_detection_sensitivity

        self.storage_path = storage_path or Path("./auto_evolve_data")
        self.storage_path.mkdir(exist_ok=True)

        # Evolution state
        self.current_prompt = ""
        self.reasoning_history: List[List[ReasoningStep]] = []
        self.error_history: List[ErrorAnalysis] = []
        self.improvement_history: List[Dict[str, Any]] = []
        self.confidence_history: List[float] = []

    async def evolve(
        self,
        initial_prompt: str,
        evaluation_function: callable,
        problem_set: List[Dict[str, Any]],
        num_iterations: int = 10,
    ) -> str:
        """
        Run the auto-evolution process.

        Args:
            initial_prompt: Starting prompt
            evaluation_function: Function to evaluate prompt performance
            problem_set: Set of problems for evaluation
            num_iterations: Number of evolution iterations

        Returns:
            Final evolved prompt
        """
        self.current_prompt = initial_prompt

        for iteration in range(num_iterations):
            print(f"Auto-Evolve Iteration {iteration + 1}/{num_iterations}")

            # Evaluate current prompt
            performance = await evaluation_function(self.current_prompt, problem_set)

            # Generate reasoning chain about current performance
            reasoning_chain = await self._generate_reasoning_chain(self.current_prompt, problem_set, performance)
            self.reasoning_history.append(reasoning_chain)

            # Detect errors and areas for improvement
            error_analysis = await self._detect_errors(self.current_prompt, problem_set, reasoning_chain)
            if error_analysis:
                self.error_history.extend(error_analysis)

            # Generate improvements based on reasoning and error analysis
            if iteration < num_iterations - 1:  # Don't improve on last iteration
                improved_prompt, confidence = await self._generate_improvement(reasoning_chain, error_analysis)

                if confidence >= self.confidence_threshold:
                    self.current_prompt = improved_prompt
                    self.confidence_history.append(confidence)
                    print(f"  Prompt improved with confidence: {confidence:.3f}")
                else:
                    print(f"  Improvement rejected (confidence: {confidence:.3f})")

            # Log iteration
            self._log_iteration(iteration, performance, reasoning_chain)

        # Save final results
        await self._save_results()

        return self.current_prompt

    async def _generate_reasoning_chain(
        self,
        prompt: str,
        problem_set: List[Dict[str, Any]],
        performance: float,
    ) -> List[ReasoningStep]:
        """Generate a reasoning chain about prompt effectiveness."""
        reasoning_prompt = f"""
        Analyze the effectiveness of this system prompt based on its performance:

        System Prompt: {prompt}
        Performance Score: {performance:.3f}
        Sample Problems: {self._format_problems_sample(problem_set[:3])}

        Generate a step-by-step reasoning chain (max {self.max_reasoning_steps} steps) that analyzes:
        1. What aspects of the prompt work well
        2. What aspects might be causing issues
        3. How the prompt aligns with the problem requirements
        4. Potential areas for improvement
        5. Overall assessment of prompt effectiveness

        For each step, provide:
        - Step description
        - Detailed reasoning
        - Confidence level (0.0-1.0)
        - Supporting evidence

        Format as JSON array of steps.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=reasoning_prompt,
                options={"temperature": 0.3, "num_predict": 600},
            )

            reasoning_text = response["response"].strip()

            # Try to parse JSON response
            if reasoning_text.startswith("["):
                steps_data = json.loads(reasoning_text)
                reasoning_steps = []

                for i, step_data in enumerate(steps_data[: self.max_reasoning_steps]):
                    step = ReasoningStep(
                        step_number=i + 1,
                        description=step_data.get("description", f"Step {i+1}"),
                        reasoning=step_data.get("reasoning", ""),
                        confidence=float(step_data.get("confidence", 0.5)),
                        evidence=step_data.get("evidence", []),
                    )
                    reasoning_steps.append(step)

                return reasoning_steps
            else:
                # Fallback: create single reasoning step
                return [
                    ReasoningStep(
                        step_number=1, description="Analysis", reasoning=reasoning_text, confidence=0.5, evidence=[]
                    )
                ]

        except Exception as e:
            # Fallback reasoning step
            return [
                ReasoningStep(
                    step_number=1,
                    description="Fallback Analysis",
                    reasoning=f"Analysis failed: {str(e)}. Performance: {performance:.3f}",
                    confidence=0.3,
                    evidence=[],
                )
            ]

    def _format_problems_sample(self, problems: List[Dict[str, Any]]) -> str:
        """Format a sample of problems for analysis."""
        formatted = []
        for i, problem in enumerate(problems, 1):
            formatted.append(f"Problem {i}: {problem.get('description', 'N/A')[:100]}...")
        return "\n".join(formatted)

    async def _detect_errors(
        self,
        prompt: str,
        problem_set: List[Dict[str, Any]],
        reasoning_chain: List[ReasoningStep],
    ) -> List[ErrorAnalysis]:
        """Detect systematic errors and issues."""
        # Test prompt on sample problems to identify failure patterns
        test_results = await self._test_prompt_on_samples(prompt, problem_set[:5])

        error_detection_prompt = f"""
        Analyze the following for systematic errors and issues:

        System Prompt: {prompt}
        
        Reasoning Chain:
        {self._format_reasoning_chain(reasoning_chain)}
        
        Test Results:
        {self._format_test_results(test_results)}

        Identify systematic errors, patterns of failure, or areas for improvement.
        For each error identified, provide:
        - Error type (e.g., "ambiguity", "missing_instruction", "over_complexity")
        - Description of the error
        - Root cause analysis
        - Suggested fix
        - Confidence in the analysis (0.0-1.0)
        - Examples demonstrating the error

        Return as JSON array of error analyses.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=error_detection_prompt,
                options={"temperature": 0.2, "num_predict": 500},
            )

            error_text = response["response"].strip()

            if error_text.startswith("["):
                errors_data = json.loads(error_text)
                error_analyses = []

                for error_data in errors_data:
                    error = ErrorAnalysis(
                        error_type=error_data.get("error_type", "unknown"),
                        description=error_data.get("description", ""),
                        root_cause=error_data.get("root_cause", ""),
                        suggested_fix=error_data.get("suggested_fix", ""),
                        confidence=float(error_data.get("confidence", 0.5)),
                        examples=error_data.get("examples", []),
                    )
                    error_analyses.append(error)

                return error_analyses
            else:
                return []

        except Exception as e:
            print(f"Error detection failed: {e}")
            return []

    async def _test_prompt_on_samples(self, prompt: str, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test prompt on sample problems to gather performance data."""
        results = []

        for sample in samples:
            try:
                # Generate solution
                solution = await self._generate_solution(prompt, sample["description"])

                # Simple quality assessment
                quality = await self._assess_solution_quality(solution, sample)

                results.append(
                    {
                        "problem": sample["description"][:100],
                        "solution": solution[:200],
                        "quality": quality,
                        "success": quality > 0.5,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "problem": sample["description"][:100],
                        "solution": f"Error: {str(e)}",
                        "quality": 0.0,
                        "success": False,
                    }
                )

        return results

    async def _generate_solution(self, prompt: str, problem: str) -> str:
        """Generate solution using the prompt."""
        full_prompt = f"{prompt}\n\nProblem: {problem}\n\nSolution:"

        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=full_prompt,
            options={"temperature": 0.3, "num_predict": 200},
        )

        return response["response"].strip()

    async def _assess_solution_quality(self, solution: str, problem_data: Dict[str, Any]) -> float:
        """Assess the quality of a generated solution."""
        assessment_prompt = f"""
        Rate the quality of this solution on a scale of 0.0 to 1.0:

        Problem: {problem_data["description"]}
        Solution: {solution}

        Consider correctness, completeness, and clarity.
        Return only a number between 0.0 and 1.0.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=assessment_prompt,
                options={"temperature": 0.1, "num_predict": 10},
            )

            score = float(response["response"].strip())
            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5  # Default neutral score

    def _format_reasoning_chain(self, chain: List[ReasoningStep]) -> str:
        """Format reasoning chain for prompts."""
        formatted = []
        for step in chain:
            formatted.append(
                f"""
            Step {step.step_number}: {step.description}
            Reasoning: {step.reasoning}
            Confidence: {step.confidence:.2f}
            """
            )
        return "\n".join(formatted)

    def _format_test_results(self, results: List[Dict[str, Any]]) -> str:
        """Format test results for analysis."""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"""
            Test {i}:
            Problem: {result['problem']}
            Solution: {result['solution']}
            Quality: {result['quality']:.2f}
            Success: {result['success']}
            """
            )
        return "\n".join(formatted)

    async def _generate_improvement(
        self,
        reasoning_chain: List[ReasoningStep],
        error_analyses: List[ErrorAnalysis],
    ) -> Tuple[str, float]:
        """Generate an improved prompt based on reasoning and error analysis."""
        improvement_prompt = f"""
        Current system prompt:
        {self.current_prompt}

        Reasoning analysis:
        {self._format_reasoning_chain(reasoning_chain)}

        Identified errors:
        {self._format_error_analyses(error_analyses)}

        Generate an improved version of the system prompt that:
        1. Addresses the identified errors and issues
        2. Incorporates insights from the reasoning analysis
        3. Maintains clarity and effectiveness
        4. Builds on the strengths identified

        Also provide a confidence score (0.0-1.0) for this improvement.

        Format as JSON:
        {{
            "improved_prompt": "...",
            "confidence": 0.X,
            "reasoning": "explanation of changes"
        }}
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=improvement_prompt,
                options={"temperature": 0.4, "num_predict": 400},
            )

            response_text = response["response"].strip()

            if response_text.startswith("{"):
                improvement_data = json.loads(response_text)
                improved_prompt = improvement_data.get("improved_prompt", self.current_prompt)
                confidence = float(improvement_data.get("confidence", 0.5))

                # Log the improvement reasoning
                self.improvement_history.append(
                    {
                        "iteration": len(self.improvement_history),
                        "original_prompt": self.current_prompt,
                        "improved_prompt": improved_prompt,
                        "confidence": confidence,
                        "reasoning": improvement_data.get("reasoning", ""),
                        "timestamp": time.time(),
                    }
                )

                return improved_prompt, confidence
            else:
                return self.current_prompt, 0.0

        except Exception as e:
            print(f"Improvement generation failed: {e}")
            return self.current_prompt, 0.0

    def _format_error_analyses(self, analyses: List[ErrorAnalysis]) -> str:
        """Format error analyses for prompts."""
        if not analyses:
            return "No systematic errors detected."

        formatted = []
        for i, analysis in enumerate(analyses, 1):
            formatted.append(
                f"""
            Error {i}: {analysis.error_type}
            Description: {analysis.description}
            Root Cause: {analysis.root_cause}
            Suggested Fix: {analysis.suggested_fix}
            Confidence: {analysis.confidence:.2f}
            """
            )
        return "\n".join(formatted)

    def _log_iteration(
        self,
        iteration: int,
        performance: float,
        reasoning_chain: List[ReasoningStep],
    ) -> None:
        """Log iteration results."""
        avg_reasoning_confidence = np.mean([step.confidence for step in reasoning_chain])

        print(f"  Performance: {performance:.3f}")
        print(f"  Reasoning Steps: {len(reasoning_chain)}")
        print(f"  Avg Reasoning Confidence: {avg_reasoning_confidence:.3f}")
        high_confidence_errors = [e for e in self.error_history if e.confidence > self.error_detection_sensitivity]
        print(f"  Errors Detected: {len(high_confidence_errors)}")

    async def _save_results(self) -> None:
        """Save evolution results and reasoning logs."""
        results = {
            "final_prompt": self.current_prompt,
            "reasoning_history": [[step.to_dict() for step in chain] for chain in self.reasoning_history],
            "error_history": [error.to_dict() for error in self.error_history],
            "improvement_history": self.improvement_history,
            "confidence_history": self.confidence_history,
            "parameters": {
                "confidence_threshold": self.confidence_threshold,
                "max_reasoning_steps": self.max_reasoning_steps,
                "error_detection_sensitivity": self.error_detection_sensitivity,
            },
        }

        results_file = self.storage_path / f"auto_evolve_results_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_file}")

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of the evolution process."""
        if not self.improvement_history:
            return {"status": "No improvements made"}

        total_errors = len(self.error_history)
        high_confidence_errors = len([e for e in self.error_history if e.confidence > self.confidence_threshold])

        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.0

        return {
            "total_iterations": len(self.reasoning_history),
            "improvements_made": len(self.improvement_history),
            "average_improvement_confidence": avg_confidence,
            "total_errors_detected": total_errors,
            "high_confidence_errors": high_confidence_errors,
            "final_prompt": self.current_prompt,
            "reasoning_steps_total": sum(len(chain) for chain in self.reasoning_history),
            "error_types": list(set(error.error_type for error in self.error_history)),
        }
