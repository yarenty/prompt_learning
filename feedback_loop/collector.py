"""
Feedback collection system for gathering and evaluating solution quality and reflections.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import ollama
from pathlib import Path

class FeedbackCollector:
    def __init__(self, storage_path: str = "data/feedback", model: str = "codellama"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.model = model
        
    def collect_solution_feedback(
        self,
        problem: str,
        solution: str,
        evaluation_criteria: List[str]
    ) -> Dict:
        """
        Collect feedback for a solution including quality metrics and reflections.
        
        Args:
            problem: The original problem statement
            solution: The proposed solution
            evaluation_criteria: List of criteria to evaluate the solution against
            
        Returns:
            Dict containing feedback data
        """
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "problem": problem,
            "solution": solution,
            "evaluation": self._evaluate_solution(solution, evaluation_criteria),
            "reflection": self._generate_reflection(problem, solution)
        }
        
        self._store_feedback(feedback)
        return feedback
    
    def _evaluate_solution(
        self,
        solution: str,
        criteria: List[str]
    ) -> Dict[str, float]:
        """Evaluate solution quality against given criteria."""
        evaluation_prompt = f"""
        Evaluate the following code solution against these criteria: {', '.join(criteria)}
        For each criterion, provide a score between 0.0 and 1.0.
        
        Solution:
        {solution}
        
        Provide your evaluation in JSON format with scores for each criterion.
        """
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=evaluation_prompt,
                format="json"
            )
            evaluation = json.loads(response.response)
            return {criterion: float(evaluation.get(criterion, 0.0)) for criterion in criteria}
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {criterion: 0.0 for criterion in criteria}
    
    def _generate_reflection(
        self,
        problem: str,
        solution: str
    ) -> str:
        """Generate reflection on the problem-solving process."""
        reflection_prompt = f"""
        Analyze the following problem and solution:
        
        Problem: {problem}
        
        Solution: {solution}
        
        What general principles or strategies did you learn that could help solve similar problems in the future?
        Focus on extracting reusable patterns and approaches.
        Provide a concise, actionable insight that could be applied to similar coding problems.
        """
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=reflection_prompt
            )
            return response.response.strip()
        except Exception as e:
            print(f"Error in reflection generation: {e}")
            return "Failed to generate reflection"
    
    def _store_feedback(self, feedback: Dict) -> None:
        """Store feedback data to disk."""
        timestamp = feedback["timestamp"].replace(":", "-")
        file_path = self.storage_path / f"feedback_{timestamp}.json"
        
        with open(file_path, "w") as f:
            json.dump(feedback, f, indent=2) 