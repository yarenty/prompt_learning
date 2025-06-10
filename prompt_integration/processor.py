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
        if not feedback:
            return []
            
        # Extract reflections and their evaluations
        reflections = []
        for f in feedback:
            reflection = f["reflection"]
            evaluation = f.get("evaluation", {})
            # Add evaluation scores to reflection for better context
            if evaluation:
                reflection += f"\nEvaluation scores: {json.dumps(evaluation)}"
            reflections.append(reflection)
        
        # Convert reflections to TF-IDF vectors with better parameters
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)  # Include both unigrams and bigrams
        )
        X = vectorizer.fit_transform(reflections)
        
        # Use more lenient clustering parameters
        clustering = DBSCAN(
            eps=0.5,  # Increased similarity threshold
            min_samples=1,  # Allow single samples to form clusters
            metric='cosine'  # Use cosine similarity
        ).fit(X)
        
        # Group reflections by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(reflections[idx])
        
        # Extract key insights from each cluster
        insights = []
        for cluster_id, cluster_reflections in clusters.items():
            if cluster_id == -1:  # Handle noise points as individual insights
                for reflection in cluster_reflections:
                    insight = self._synthesize_single_insight(reflection)
                    insights.append({
                        "cluster_id": f"single_{len(insights)}",
                        "insight": insight,
                        "support_count": 1
                    })
            else:
                insight = self._synthesize_cluster_insight(cluster_reflections)
                insights.append({
                    "cluster_id": cluster_id,
                    "insight": insight,
                    "support_count": len(cluster_reflections)
                })
        
        return insights
    
    def _synthesize_single_insight(self, reflection: str) -> str:
        """Synthesize an insight from a single reflection."""
        synthesis_prompt = f"""
        Analyze this reflection and extract a key insight:
        
        Reflection:
        {reflection}
        
        Provide a concise, actionable insight that captures the main learning point.
        Focus on making it specific and applicable to future coding problems.
        """
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=synthesis_prompt
            )
            return response.response.strip()
        except Exception as e:
            print(f"Error in single insight synthesis: {e}")
            return "Failed to synthesize insight"
    
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
        
        # Format insights for synthesis
        insights_text = "\n\n".join([
            f"Insight {i+1} (supported by {insight['support_count']} cases):\n{insight['insight']}"
            for i, insight in enumerate(insights)
        ])
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        You are tasked with creating a concise, well-structured system prompt for an AI code generation system.
        
        CURRENT SYSTEM PROMPT:
        {current_prompt}
        
        NEW INSIGHTS TO INCORPORATE:
        {insights_text}
        
        Please create an updated system prompt that:
        1. Maintains a clear, professional tone
        2. Organizes insights into logical sections (Core Principles, Solution Approach, Key Techniques, Code Quality Standards)
        3. Is concise and actionable
        4. Integrates new insights naturally with existing content
        5. Focuses on practical, reusable patterns and best practices
        
        Format the prompt with clear section headers and bullet points where appropriate.
        The final prompt should be comprehensive yet concise, focusing on the most important and widely applicable insights.
        """
        
        try:
            # Generate updated prompt using Ollama
            response = ollama.generate(
                model=self.model,
                prompt=synthesis_prompt
            )
            updated_prompt = response.response.strip()
            
            # Save updated prompt
            self._save_prompt(updated_prompt)
            
            return updated_prompt
        except Exception as e:
            print(f"Error in prompt synthesis: {e}")
            return current_prompt
    
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