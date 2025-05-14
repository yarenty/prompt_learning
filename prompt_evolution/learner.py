"""
Core module for system prompt evolution through self-learning.
"""
from typing import Dict, List, Optional
import json
from pathlib import Path
import ollama
from datetime import datetime

class PromptLearner:
    def __init__(
        self,
        model: str = "codellama",
        storage_path: str = "data/prompt_evolution",
        initial_prompts: Optional[Dict[str, str]] = None
    ):
        self.model = model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.initial_prompts = initial_prompts or {}
        self.evolution_history = []
        
    def evaluate_prompt_performance(
        self,
        prompt: str,
        problem: Dict[str, str],
        evaluation_criteria: List[str]
    ) -> Dict:
        """
        Evaluate how well a system prompt performs on a given problem.
        
        Args:
            prompt: The system prompt to evaluate
            problem: Dictionary containing problem description and solution
            evaluation_criteria: List of criteria to evaluate against
            
        Returns:
            Dictionary containing evaluation results and insights
        """
        evaluation_prompt = f"""
        You are evaluating a system prompt's performance on a coding problem.
        
        System Prompt:
        {prompt}
        
        Problem:
        {problem['description']}
        
        Solution:
        {problem['solution']}
        
        Evaluate the solution based on these criteria: {', '.join(evaluation_criteria)}
        For each criterion, provide:
        1. A score between 0.0 and 1.0
        2. A brief explanation of the score
        3. A lesson learned that could improve the system prompt
        
        Provide your evaluation in JSON format.
        """
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=evaluation_prompt,
                format="json"
            )
            evaluation = json.loads(response.response)
            
            # Extract lessons learned
            lessons = self._extract_lessons(evaluation)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "problem": problem,
                "evaluation": evaluation,
                "lessons": lessons
            }
        except Exception as e:
            print(f"Error in prompt evaluation: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "problem": problem,
                "error": str(e)
            }
    
    def evolve_prompt(
        self,
        current_prompt: str,
        lessons: List[Dict]
    ) -> str:
        """
        Evolve a system prompt by incorporating lessons learned.
        
        Args:
            current_prompt: The current system prompt
            lessons: List of lessons learned from evaluations
            
        Returns:
            Updated system prompt
        """
        evolution_prompt = f"""
        You are evolving a system prompt by incorporating lessons learned.
        
        Current System Prompt:
        {current_prompt}
        
        Lessons Learned:
        {json.dumps(lessons, indent=2)}
        
        Create an improved version of the system prompt that incorporates these lessons.
        The new prompt should:
        1. Maintain the original personality and approach
        2. Integrate the lessons naturally
        3. Be more effective at solving similar problems
        4. Be clear and concise
        
        Provide the updated system prompt.
        """
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=evolution_prompt
            )
            return response.response.strip()
        except Exception as e:
            print(f"Error in prompt evolution: {e}")
            return current_prompt
    
    def _extract_lessons(self, evaluation: Dict) -> List[Dict]:
        """Extract lessons learned from evaluation results."""
        lessons = []
        for criterion, data in evaluation.items():
            if isinstance(data, dict) and "lesson" in data:
                lessons.append({
                    "criterion": criterion,
                    "score": data.get("score", 0.0),
                    "explanation": data.get("explanation", ""),
                    "lesson": data["lesson"]
                })
        return lessons
    
    def save_evolution_step(
        self,
        prompt_type: str,
        current_prompt: str,
        updated_prompt: str,
        evaluation_results: List[Dict]
    ) -> None:
        """Save a step in the prompt evolution process."""
        step = {
            "timestamp": datetime.now().isoformat(),
            "prompt_type": prompt_type,
            "current_prompt": current_prompt,
            "updated_prompt": updated_prompt,
            "evaluation_results": evaluation_results
        }
        
        self.evolution_history.append(step)
        
        # Save to file
        file_path = self.storage_path / f"evolution_{prompt_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file_path, "w") as f:
            json.dump(step, f, indent=2)
    
    def get_evolution_history(self, prompt_type: Optional[str] = None) -> List[Dict]:
        """Get the evolution history for a specific prompt type or all types."""
        if prompt_type:
            return [step for step in self.evolution_history if step["prompt_type"] == prompt_type]
        return self.evolution_history 