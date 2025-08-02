"""
Evaluation Suite

Comprehensive evaluation system for assessing AI performance on dataset problems.
Supports both automated and human evaluation methods.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama

from ..config import ModelConfig
from .dataset_manager import DatasetManager, Problem


@dataclass
class EvaluationResult:
    """Result of evaluating a solution to a problem."""

    problem_id: str
    domain: str
    solution: str
    scores: Dict[str, float]  # criterion -> score
    overall_score: float
    feedback: str
    evaluation_method: str  # "automated", "human", "hybrid"
    evaluator_info: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationSession:
    """A complete evaluation session across multiple problems."""

    session_id: str
    method_name: str  # "memento", "promptbreeder", etc.
    problems_evaluated: List[str]  # problem IDs
    results: List[EvaluationResult]
    session_statistics: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AutomatedEvaluator:
    """Automated evaluation using LLM-based assessment."""

    def __init__(self, model_config: ModelConfig):
        """Initialize automated evaluator."""
        self.model_config = model_config

    async def evaluate_solution(self, problem: Problem, solution: str) -> EvaluationResult:
        """Evaluate a solution using automated LLM assessment."""
        evaluation_prompt = self._create_evaluation_prompt(problem, solution)

        try:
            response = await ollama.AsyncClient().chat(
                model=self.model_config.model_name,
                messages=[{"role": "user", "content": evaluation_prompt}],
                options={"temperature": 0.1, "num_predict": 500},
            )

            evaluation_text = response["message"]["content"].strip()
            scores, overall_score, feedback = self._parse_evaluation_response(evaluation_text, problem)

            return EvaluationResult(
                problem_id=problem.id,
                domain=self._get_problem_domain(problem),
                solution=solution,
                scores=scores,
                overall_score=overall_score,
                feedback=feedback,
                evaluation_method="automated",
                evaluator_info={
                    "model": self.model_config.model_name,
                    "temperature": 0.1,
                    "evaluation_prompt_length": len(evaluation_prompt),
                },
                timestamp=time.time(),
            )

        except Exception as e:
            return EvaluationResult(
                problem_id=problem.id,
                domain=self._get_problem_domain(problem),
                solution=solution,
                scores={},
                overall_score=0.0,
                feedback=f"Evaluation failed: {str(e)}",
                evaluation_method="automated",
                evaluator_info={"error": str(e)},
                timestamp=time.time(),
            )

    def _create_evaluation_prompt(self, problem: Problem, solution: str) -> str:
        """Create evaluation prompt based on problem type."""
        domain = self._get_problem_domain(problem)

        if domain == "software_engineering":
            return self._create_software_evaluation_prompt(problem, solution)
        elif domain == "mathematics":
            return self._create_math_evaluation_prompt(problem, solution)
        elif domain == "creative_writing":
            return self._create_writing_evaluation_prompt(problem, solution)
        else:
            return self._create_generic_evaluation_prompt(problem, solution)

    def _create_software_evaluation_prompt(self, problem, solution: str) -> str:
        """Create evaluation prompt for software engineering problems."""
        return f"""
You are an expert software engineer evaluating a coding solution.

PROBLEM:
Title: {problem.title}
Description: {problem.description}
Category: {problem.category}
Difficulty: {problem.difficulty}

EVALUATION CRITERIA:
{', '.join(problem.evaluation_criteria)}

SOLUTION TO EVALUATE:
{solution}

Please evaluate this solution on a scale of 0-10 for each criterion and provide an overall score.
Format your response as JSON:
{{
    "scores": {{"criterion1": score, "criterion2": score, ...}},
    "overall_score": overall_score,
    "feedback": "detailed feedback explaining the scores"
}}
"""

    def _create_math_evaluation_prompt(self, problem, solution: str) -> str:
        """Create evaluation prompt for mathematics problems."""
        return f"""
You are an expert mathematician evaluating a mathematical solution.

PROBLEM:
Title: {problem.title}
Statement: {problem.statement}
Domain: {problem.domain}
Difficulty: {problem.difficulty}
Expected Answer: {problem.expected_answer}

SOLUTION TO EVALUATE:
{solution}

Please evaluate this solution on a scale of 0-10 considering:
- Mathematical correctness
- Solution approach
- Clarity of explanation
- Completeness

Format your response as JSON:
{{
    "scores": {{"correctness": score, "approach": score, "clarity": score, "completeness": score}},
    "overall_score": overall_score,
    "feedback": "detailed mathematical feedback"
}}
"""

    def _create_writing_evaluation_prompt(self, problem, solution: str) -> str:
        """Create evaluation prompt for creative writing problems."""
        return f"""
You are an expert writing instructor evaluating a creative writing piece.

PROBLEM:
Title: {problem.title}
Prompt: {problem.prompt}
Category: {problem.category}
Genre: {problem.genre}
Target Audience: {problem.target_audience}
Word Count Range: {problem.word_count_range[0]}-{problem.word_count_range[1]}

EVALUATION CRITERIA:
{', '.join(problem.evaluation_criteria)}

WRITING TO EVALUATE:
{solution}

Please evaluate this writing on a scale of 0-10 for each criterion.
Format your response as JSON:
{{
    "scores": {{"criterion1": score, "criterion2": score, ...}},
    "overall_score": overall_score,
    "feedback": "detailed writing feedback"
}}
"""

    def _create_generic_evaluation_prompt(self, problem, solution: str) -> str:
        """Create generic evaluation prompt."""
        return f"""
You are an expert evaluator assessing a solution to a problem.

PROBLEM:
{getattr(problem, 'title', 'N/A')}
{getattr(problem, 'description', getattr(problem, 'statement', getattr(problem, 'prompt', 'N/A')))}

SOLUTION TO EVALUATE:
{solution}

Please evaluate this solution on a scale of 0-10 considering quality, correctness, and completeness.
Format your response as JSON:
{{
    "scores": {{"quality": score, "correctness": score, "completeness": score}},
    "overall_score": overall_score,
    "feedback": "detailed feedback"
}}
"""

    def _parse_evaluation_response(self, response: str, problem: Problem) -> tuple[Dict[str, float], float, str]:
        """Parse LLM evaluation response."""
        try:
            # Try to extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                eval_data = json.loads(json_str)

                scores = eval_data.get("scores", {})
                overall_score = eval_data.get("overall_score", 0.0)
                feedback = eval_data.get("feedback", "No feedback provided")

                return scores, float(overall_score), feedback

        except Exception as e:
            print(f"Error parsing evaluation response: {e}")

        # Fallback: estimate score from response text
        score_keywords = ["excellent", "good", "fair", "poor", "terrible"]
        score_values = [9.0, 7.0, 5.0, 3.0, 1.0]

        response_lower = response.lower()
        for keyword, value in zip(score_keywords, score_values):
            if keyword in response_lower:
                return {"overall": value}, value, response

        return {"overall": 5.0}, 5.0, response

    def _get_problem_domain(self, problem: Problem) -> str:
        """Determine problem domain."""
        if hasattr(problem, "category"):
            if problem.category in ["algorithm", "data_structure", "design_pattern", "architecture", "testing"]:
                return "software_engineering"
            elif problem.category in ["story", "essay", "documentation", "problem_solving", "style_adaptation"]:
                return "creative_writing"
        elif hasattr(problem, "domain"):
            if problem.domain in ["algebra", "calculus", "proof", "optimization", "statistics"]:
                return "mathematics"

        return "unknown"


class HumanEvaluator:
    """Human evaluation interface."""

    def __init__(self, evaluator_id: str):
        """Initialize human evaluator."""
        self.evaluator_id = evaluator_id

    def evaluate_solution(self, problem: Problem, solution: str) -> EvaluationResult:
        """Collect human evaluation (interactive)."""
        print(f"\n{'='*60}")
        print(f"HUMAN EVALUATION - Problem ID: {problem.id}")
        print(f"{'='*60}")

        # Display problem
        print("\nPROBLEM:")
        print(f"Title: {getattr(problem, 'title', 'N/A')}")
        if hasattr(problem, "description"):
            print(f"Description: {problem.description}")
        elif hasattr(problem, "statement"):
            print(f"Statement: {problem.statement}")
        elif hasattr(problem, "prompt"):
            print(f"Prompt: {problem.prompt}")

        # Display solution
        print("\nSOLUTION:")
        print(solution)

        # Collect evaluation
        print("\nEVALUATION:")

        # Get evaluation criteria
        criteria = getattr(problem, "evaluation_criteria", ["quality", "correctness", "completeness"])
        scores = {}

        for criterion in criteria:
            while True:
                try:
                    score = float(input(f"Rate {criterion} (0-10): "))
                    if 0 <= score <= 10:
                        scores[criterion] = score
                        break
                    else:
                        print("Please enter a score between 0 and 10")
                except ValueError:
                    print("Please enter a valid number")

        overall_score = sum(scores.values()) / len(scores)
        feedback = input("Feedback (optional): ").strip()

        return EvaluationResult(
            problem_id=problem.id,
            domain=self._get_problem_domain(problem),
            solution=solution,
            scores=scores,
            overall_score=overall_score,
            feedback=feedback or "No feedback provided",
            evaluation_method="human",
            evaluator_info={"evaluator_id": self.evaluator_id},
            timestamp=time.time(),
        )

    def _get_problem_domain(self, problem: Problem) -> str:
        """Determine problem domain."""
        if hasattr(problem, "category"):
            if problem.category in ["algorithm", "data_structure", "design_pattern", "architecture", "testing"]:
                return "software_engineering"
            elif problem.category in ["story", "essay", "documentation", "problem_solving", "style_adaptation"]:
                return "creative_writing"
        elif hasattr(problem, "domain"):
            if problem.domain in ["algebra", "calculus", "proof", "optimization", "statistics"]:
                return "mathematics"

        return "unknown"


class EvaluationSuite:
    """Comprehensive evaluation suite managing all evaluation methods."""

    def __init__(self, dataset_manager: DatasetManager, model_config: ModelConfig, storage_path: Optional[Path] = None):
        """Initialize evaluation suite."""
        self.dataset_manager = dataset_manager
        self.model_config = model_config
        self.storage_path = storage_path or Path("evaluation_results")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize evaluators
        self.automated_evaluator = AutomatedEvaluator(model_config)
        self.human_evaluator = None  # Created when needed

        # Track evaluation sessions
        self.sessions: List[EvaluationSession] = []

    async def evaluate_method_performance(
        self, method_name: str, problem_ids: List[str], solutions: List[str], evaluation_method: str = "automated"
    ) -> EvaluationSession:
        """Evaluate a method's performance on a set of problems."""
        session_id = f"{method_name}_{int(time.time())}"
        session = EvaluationSession(
            session_id=session_id,
            method_name=method_name,
            problems_evaluated=problem_ids,
            results=[],
            session_statistics={},
            start_time=time.time(),
        )

        # Evaluate each problem-solution pair
        for problem_id, solution in zip(problem_ids, solutions):
            problem = self.dataset_manager.get_problem_by_id(problem_id)
            if not problem:
                print(f"Warning: Problem {problem_id} not found")
                continue

            if evaluation_method == "automated":
                result = await self.automated_evaluator.evaluate_solution(problem, solution)
            elif evaluation_method == "human":
                if not self.human_evaluator:
                    self.human_evaluator = HumanEvaluator("default_evaluator")
                result = self.human_evaluator.evaluate_solution(problem, solution)
            else:
                raise ValueError(f"Unknown evaluation method: {evaluation_method}")

            session.results.append(result)

        # Calculate session statistics
        session.end_time = time.time()
        session.session_statistics = self._calculate_session_statistics(session)

        # Save session
        self._save_session(session)
        self.sessions.append(session)

        return session

    def _calculate_session_statistics(self, session: EvaluationSession) -> Dict[str, Any]:
        """Calculate statistics for an evaluation session."""
        if not session.results:
            return {}

        scores = [result.overall_score for result in session.results]
        domain_scores = {}

        for result in session.results:
            if result.domain not in domain_scores:
                domain_scores[result.domain] = []
            domain_scores[result.domain].append(result.overall_score)

        stats = {
            "total_problems": len(session.results),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "domain_performance": {
                domain: {
                    "mean": sum(domain_scores[domain]) / len(domain_scores[domain]),
                    "count": len(domain_scores[domain]),
                }
                for domain in domain_scores
            },
            "evaluation_duration": session.end_time - session.start_time if session.end_time else 0,
        }

        return stats

    def _save_session(self, session: EvaluationSession) -> None:
        """Save evaluation session to file."""
        session_file = self.storage_path / f"{session.session_id}.json"

        try:
            with open(session_file, "w") as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving session: {e}")

    def load_session(self, session_id: str) -> Optional[EvaluationSession]:
        """Load evaluation session from file."""
        session_file = self.storage_path / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, "r") as f:
                session_data = json.load(f)

            # Convert results back to EvaluationResult objects
            session_data["results"] = [
                EvaluationResult.from_dict(result_data) for result_data in session_data["results"]
            ]

            return EvaluationSession(**session_data)

        except Exception as e:
            print(f"Error loading session: {e}")
            return None

    def compare_methods(self, session_ids: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple evaluation sessions."""
        sessions = []
        for session_id in session_ids:
            session = self.load_session(session_id)
            if session:
                sessions.append(session)

        if not sessions:
            return {"error": "No valid sessions found"}

        comparison = {
            "methods": [session.method_name for session in sessions],
            "performance_comparison": {},
            "statistical_significance": {},
            "summary": {},
        }

        # Compare mean scores
        for session in sessions:
            method_name = session.method_name
            comparison["performance_comparison"][method_name] = {
                "mean_score": session.session_statistics.get("mean_score", 0),
                "total_problems": session.session_statistics.get("total_problems", 0),
                "domain_performance": session.session_statistics.get("domain_performance", {}),
            }

        # Determine best performing method
        best_method = max(comparison["performance_comparison"].items(), key=lambda x: x[1]["mean_score"])

        comparison["summary"] = {
            "best_method": best_method[0],
            "best_score": best_method[1]["mean_score"],
            "methods_compared": len(sessions),
            "total_evaluations": sum(session.session_statistics.get("total_problems", 0) for session in sessions),
        }

        return comparison

    def generate_evaluation_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        session = self.load_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        report = {
            "session_info": {
                "session_id": session.session_id,
                "method_name": session.method_name,
                "evaluation_date": time.ctime(session.start_time),
                "duration": session.session_statistics.get("evaluation_duration", 0),
            },
            "performance_summary": session.session_statistics,
            "detailed_results": [result.to_dict() for result in session.results],
            "insights": self._generate_insights(session),
        }

        return report

    def _generate_insights(self, session: EvaluationSession) -> List[str]:
        """Generate insights from evaluation session."""
        insights = []
        stats = session.session_statistics

        if stats.get("mean_score", 0) > 8.0:
            insights.append("Excellent overall performance across problems")
        elif stats.get("mean_score", 0) > 6.0:
            insights.append("Good performance with room for improvement")
        else:
            insights.append("Performance needs significant improvement")

        # Domain-specific insights
        domain_perf = stats.get("domain_performance", {})
        if domain_perf:
            best_domain = max(domain_perf.items(), key=lambda x: x[1]["mean"])
            worst_domain = min(domain_perf.items(), key=lambda x: x[1]["mean"])

            insights.append(f"Strongest performance in {best_domain[0]} domain")
            insights.append(f"Weakest performance in {worst_domain[0]} domain")

        return insights
