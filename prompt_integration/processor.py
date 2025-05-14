"""
System for processing feedback and integrating insights into the system prompt.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import ollama

class PromptProcessor:
    def __init__(
        self,
        feedback_path: str = "data/feedback",
        prompt_path: str = "data/prompts",
        model: str = "codellama"
    ):
        self.feedback_path = Path(feedback_path)
        self.prompt_path = Path(prompt_path)
        self.prompt_path.mkdir(parents=True, exist_ok=True)
        self.model = model
        
    def process_feedback(self) -> List[Dict]:
        """Process all feedback files and extract insights."""
        feedback_files = list(self.feedback_path.glob("feedback_*.json"))
        all_feedback = []
        
        for file_path in feedback_files:
            with open(file_path) as f:
                feedback = json.load(f)
                all_feedback.append(feedback)
        
        return self._extract_insights(all_feedback)
    
    def _extract_insights(self, feedback: List[Dict]) -> List[Dict]:
        """Extract and cluster similar insights from feedback."""
        reflections = [f["reflection"] for f in feedback]
        
        # Convert reflections to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(reflections)
        
        # Cluster similar reflections
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
        
        # Group reflections by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(reflections[idx])
        
        # Extract key insights from each cluster
        insights = []
        for cluster_id, cluster_reflections in clusters.items():
            if cluster_id == -1:  # Skip noise points
                continue
                
            insight = self._synthesize_cluster_insight(cluster_reflections)
            insights.append({
                "cluster_id": cluster_id,
                "insight": insight,
                "support_count": len(cluster_reflections)
            })
        
        return insights
    
    def _synthesize_cluster_insight(self, reflections: List[str]) -> str:
        """Synthesize a single insight from a cluster of similar reflections."""
        synthesis_prompt = f"""
        Analyze these similar reflections and synthesize them into a single, clear insight:
        
        Reflections:
        {chr(10).join(f'- {r}' for r in reflections)}
        
        Provide a concise, actionable insight that captures the common pattern or principle.
        Focus on making it specific and applicable to future coding problems.
        """
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=synthesis_prompt
            )
            return response.response.strip()
        except Exception as e:
            print(f"Error in insight synthesis: {e}")
            return "Failed to synthesize insight"
    
    def update_system_prompt(self, insights: List[Dict]) -> str:
        """Update the system prompt with new insights."""
        current_prompt = self._load_current_prompt()
        
        # Format new insights
        new_sections = []
        for insight in insights:
            if insight["support_count"] >= 2:  # Only include well-supported insights
                new_sections.append(
                    f"# Insight from {insight['support_count']} similar cases:\n"
                    f"{insight['insight']}\n"
                )
        
        # Combine with current prompt
        updated_prompt = current_prompt + "\n\n" + "\n".join(new_sections)
        
        # Save updated prompt
        self._save_prompt(updated_prompt)
        
        return updated_prompt
    
    def _load_current_prompt(self) -> str:
        """Load the current system prompt."""
        prompt_file = self.prompt_path / "current_prompt.txt"
        if prompt_file.exists():
            return prompt_file.read_text()
        return "# Initial System Prompt\n\n"
    
    def _save_prompt(self, prompt: str) -> None:
        """Save the updated system prompt."""
        # Save current version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_file = self.prompt_path / f"prompt_{timestamp}.txt"
        version_file.write_text(prompt)
        
        # Update current version
        current_file = self.prompt_path / "current_prompt.txt"
        current_file.write_text(prompt) 