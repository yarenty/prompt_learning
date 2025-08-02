"""
Enhanced prompt processing system with async support, principle extraction, and conflict resolution.

This module provides a professional-grade PromptProcessor that implements the BaseProcessor
interface with comprehensive error handling, validation, principle versioning, and confidence scoring.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ollama
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import ModelConfig
from ..exceptions import ExtractionError, ProcessingError, UpdateError
from ..utils.metrics import PerformanceMonitor
from ..utils.validation import validate_json_response
from .base import BaseProcessor


class PrincipleVersion:
    """Represents a versioned principle with confidence and metadata."""

    def __init__(
        self,
        principle: str,
        confidence: float,
        version: int = 1,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.principle = principle
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        self.version = version
        self.created_at = created_at or datetime.now()
        self.metadata = metadata or {}
        self.superseded = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "principle": self.principle,
            "confidence": self.confidence,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "superseded": self.superseded,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrincipleVersion":
        """Create from dictionary representation."""
        created_at = datetime.fromisoformat(data["created_at"])
        principle = cls(
            principle=data["principle"],
            confidence=data["confidence"],
            version=data["version"],
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )
        principle.superseded = data.get("superseded", False)
        return principle


class PromptProcessor(BaseProcessor):
    """
    Enhanced prompt processor with principle extraction, versioning, and conflict resolution.

    Features:
    - Async feedback processing for better performance
    - Advanced principle extraction with confidence scoring
    - Principle versioning system for tracking evolution
    - Conflict resolution for contradictory insights
    - Performance metrics collection
    - Comprehensive validation and error handling
    """

    def __init__(
        self,
        model_config: ModelConfig,
        feedback_path: Union[str, Path],
        prompt_path: Union[str, Path],
        enable_metrics: bool = True,
        min_confidence_threshold: float = 0.6,
        max_principles: int = 50,
        **kwargs: Any,
    ):
        """
        Initialize the enhanced prompt processor.

        Args:
            model_config: Configuration for the LLM model
            feedback_path: Path to feedback data
            prompt_path: Path to store processed prompts
            enable_metrics: Whether to collect performance metrics
            min_confidence_threshold: Minimum confidence for principle acceptance
            max_principles: Maximum number of principles to maintain
            **kwargs: Additional configuration options
        """
        super().__init__(model_config, feedback_path, prompt_path, **kwargs)

        self.min_confidence_threshold = max(0.0, min(1.0, min_confidence_threshold))
        self.max_principles = max(1, max_principles)

        # Initialize performance monitoring
        self.monitor: Optional[PerformanceMonitor] = None
        if enable_metrics:
            self.monitor = PerformanceMonitor(storage_path=self.prompt_path / "metrics")

        # Principle versioning system
        self.principles: Dict[str, List[PrincipleVersion]] = {}
        self.principle_clusters: Dict[str, List[str]] = {}

        # Create versioning directory
        self.versions_path = self.prompt_path / "versions"
        self.versions_path.mkdir(exist_ok=True)

        self.logger.info(
            f"PromptProcessor initialized with confidence_threshold={min_confidence_threshold}, "
            f"max_principles={max_principles}, metrics={'enabled' if enable_metrics else 'disabled'}"
        )

    async def process_feedback(self) -> List[Dict[str, Any]]:
        """
        Process all feedback files and extract insights with async support.

        Returns:
            List of extracted insights with confidence scores

        Raises:
            ProcessingError: If processing fails
        """
        timer_id = None
        if self.monitor:
            timer_id = f"process_feedback_{id(self)}"
            self.monitor.start_timer(timer_id)

        try:
            self.logger.info("Starting feedback processing")

            # Load all feedback files
            feedback_files = list(self.feedback_path.glob("feedback_*.json"))
            if not feedback_files:
                self.logger.warning("No feedback files found")
                return []

            self.logger.info(f"Found {len(feedback_files)} feedback files")

            # Load feedback data concurrently
            feedback_data = await self._load_feedback_files(feedback_files)

            # Extract insights from feedback
            insights = await self.extract_insights(feedback_data)

            # Update performance metrics
            if self.monitor and timer_id:
                self.monitor.stop_timer(
                    timer_id,
                    "processor",
                    "process_feedback",
                    metadata={"feedback_files": len(feedback_files), "insights": len(insights)},
                )

            self.logger.info(f"Processed {len(feedback_files)} files, extracted {len(insights)} insights")
            return insights

        except Exception as e:
            if self.monitor and timer_id:
                self.monitor.stop_timer(timer_id, "processor", "process_feedback", metadata={"error": str(e)})

            if isinstance(e, (ProcessingError, ExtractionError)):
                raise

            raise ProcessingError(f"Failed to process feedback: {e}") from e

    async def extract_insights(self, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and cluster similar insights from feedback with advanced algorithms.

        Args:
            feedback: List of feedback data

        Returns:
            List of extracted insights with confidence scores and clustering

        Raises:
            ExtractionError: If insight extraction fails
        """
        if not feedback:
            return []

        try:
            self.logger.info(f"Extracting insights from {len(feedback)} feedback items")

            # Extract reflections and evaluations
            reflections = []
            evaluations = []

            for f in feedback:
                if "reflection" in f and f["reflection"]:
                    reflections.append(f["reflection"])
                if "evaluation" in f and isinstance(f["evaluation"], dict):
                    evaluations.append(f["evaluation"])

            # Extract principles from reflections
            principles = await self._extract_principles_from_reflections(reflections)

            # Extract patterns from evaluations
            patterns = await self._extract_patterns_from_evaluations(evaluations)

            # Combine and cluster insights
            all_insights = principles + patterns
            clustered_insights = await self._cluster_insights(all_insights)

            # Score confidence for each insight
            scored_insights = await self._score_insight_confidence(clustered_insights)

            # Filter by confidence threshold
            filtered_insights = [
                insight
                for insight in scored_insights
                if insight.get("confidence", 0.0) >= self.min_confidence_threshold
            ]

            self.logger.info(
                f"Extracted {len(all_insights)} raw insights, "
                f"clustered into {len(clustered_insights)}, "
                f"filtered to {len(filtered_insights)} high-confidence insights"
            )

            return filtered_insights

        except Exception as e:
            raise ExtractionError(f"Failed to extract insights: {e}") from e

    async def update_system_prompt(self, insights: List[Dict[str, Any]]) -> str:
        """
        Update the system prompt with new insights using conflict resolution.

        Args:
            insights: List of insights to incorporate

        Returns:
            Updated system prompt

        Raises:
            UpdateError: If prompt update fails
        """
        try:
            self.logger.info(f"Updating system prompt with {len(insights)} insights")

            # Load current system prompt
            current_prompt = await self._load_current_prompt()

            # Resolve conflicts between insights
            resolved_insights = await self._resolve_insight_conflicts(insights)

            # Version and store principles
            await self._version_principles(resolved_insights)

            # Generate updated prompt
            updated_prompt = await self._generate_updated_prompt(current_prompt, resolved_insights)

            # Validate and store the new prompt
            await self._store_updated_prompt(updated_prompt)

            self.logger.info("System prompt updated successfully")
            return updated_prompt

        except Exception as e:
            raise UpdateError(f"Failed to update system prompt: {e}") from e

    async def _load_feedback_files(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Load feedback files concurrently."""

        async def load_file(file_path: Path) -> Optional[Dict[str, Any]]:
            try:
                content = await asyncio.to_thread(file_path.read_text)
                return json.loads(content)
            except Exception as e:
                self.logger.error(f"Failed to load feedback file {file_path}: {e}")
                return None

        tasks = [load_file(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        feedback_data = []
        for result in results:
            if isinstance(result, dict):
                feedback_data.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Error loading feedback: {result}")

        return feedback_data

    async def _extract_principles_from_reflections(self, reflections: List[str]) -> List[Dict[str, Any]]:
        """Extract principles from reflection texts using LLM."""
        if not reflections:
            return []

        # Combine reflections for batch processing
        combined_reflections = "\n\n---\n\n".join(reflections)

        extraction_prompt = f"""
        Extract general programming principles and best practices from the following reflections.
        Focus on actionable insights that can improve code quality and problem-solving approaches.
        
        Reflections:
        {combined_reflections}
        
        Provide your response as a JSON array of objects, each with:
        - "principle": A clear, actionable principle
        - "category": The category (e.g., "performance", "readability", "architecture")
        - "examples": Brief examples of when to apply this principle
        
        Example format:
        [
            {{
                "principle": "Use iterative approaches instead of recursion for large datasets to avoid stack overflow",
                "category": "performance",
                "examples": "fibonacci, factorial, tree traversal with large depths"
            }}
        ]
        """

        try:
            start_time = time.time()
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=extraction_prompt,
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent extraction
                    "num_predict": 1000,
                },
            )
            duration = time.time() - start_time

            response_text = response["response"].strip()
            principles_data = validate_json_response(response_text)

            if not isinstance(principles_data, list):
                raise ExtractionError("Expected list of principles from LLM")

            principles = []
            for item in principles_data:
                if isinstance(item, dict) and "principle" in item:
                    principles.append(
                        {
                            "type": "principle",
                            "content": item["principle"],
                            "category": item.get("category", "general"),
                            "examples": item.get("examples", ""),
                            "source": "reflection",
                            "extraction_time": duration,
                        }
                    )

            self.logger.debug(f"Extracted {len(principles)} principles in {duration:.2f}s")
            return principles

        except Exception as e:
            raise ExtractionError(f"Failed to extract principles: {e}") from e

    async def _extract_patterns_from_evaluations(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns from evaluation scores."""
        if not evaluations:
            return []

        patterns = []

        # Analyze score distributions
        criteria_scores = {}
        for eval_data in evaluations:
            for criterion, score in eval_data.items():
                if isinstance(score, (int, float)):
                    if criterion not in criteria_scores:
                        criteria_scores[criterion] = []
                    criteria_scores[criterion].append(score)

        # Generate insights from score patterns
        for criterion, scores in criteria_scores.items():
            if len(scores) >= 3:  # Need minimum data for meaningful patterns
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)

                if avg_score < 0.6:
                    patterns.append(
                        {
                            "type": "pattern",
                            "content": f"Solutions consistently score low on {criterion} (avg: {avg_score:.2f})",
                            "category": "weakness",
                            "criterion": criterion,
                            "avg_score": avg_score,
                            "source": "evaluation",
                        }
                    )
                elif avg_score > 0.8:
                    patterns.append(
                        {
                            "type": "pattern",
                            "content": f"Solutions consistently excel in {criterion} (avg: {avg_score:.2f})",
                            "category": "strength",
                            "criterion": criterion,
                            "avg_score": avg_score,
                            "source": "evaluation",
                        }
                    )

                if max_score - min_score > 0.5:
                    patterns.append(
                        {
                            "type": "pattern",
                            "content": f"High variability in {criterion} scores suggests inconsistent approach",
                            "category": "consistency",
                            "criterion": criterion,
                            "variability": max_score - min_score,
                            "source": "evaluation",
                        }
                    )

        self.logger.debug(f"Extracted {len(patterns)} patterns from evaluations")
        return patterns

    async def _cluster_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar insights using TF-IDF and DBSCAN."""
        if len(insights) < 2:
            return insights

        try:
            # Extract text content for clustering
            texts = [insight["content"] for insight in insights]

            # Vectorize using TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=100, stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.8
            )

            vectors = vectorizer.fit_transform(texts)

            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
            cluster_labels = clustering.fit_predict(vectors.toarray())

            # Group insights by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(insights[i])

            # Merge similar insights within clusters
            clustered_insights = []
            for label, cluster_insights in clusters.items():
                if label == -1:  # Noise points (unclustered)
                    clustered_insights.extend(cluster_insights)
                else:
                    # Merge insights in the cluster
                    merged_insight = await self._merge_cluster_insights(cluster_insights)
                    clustered_insights.append(merged_insight)

            self.logger.debug(f"Clustered {len(insights)} insights into {len(clustered_insights)} groups")
            return clustered_insights

        except Exception as e:
            self.logger.warning(f"Clustering failed, returning original insights: {e}")
            return insights

    async def _merge_cluster_insights(self, cluster_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar insights within a cluster."""
        if len(cluster_insights) == 1:
            return cluster_insights[0]

        # Use the most comprehensive insight as base
        base_insight = max(cluster_insights, key=lambda x: len(x["content"]))

        # Combine categories and sources
        categories = set()
        sources = set()

        for insight in cluster_insights:
            categories.add(insight.get("category", "general"))
            sources.add(insight.get("source", "unknown"))

        merged = base_insight.copy()
        merged["categories"] = list(categories)
        merged["sources"] = list(sources)
        merged["cluster_size"] = len(cluster_insights)
        merged["merged_from"] = [insight["content"] for insight in cluster_insights if insight != base_insight]

        return merged

    async def _score_insight_confidence(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score confidence for each insight based on various factors."""
        scored_insights = []

        for insight in insights:
            confidence = 0.5  # Base confidence

            # Factor 1: Source reliability
            if insight.get("source") == "evaluation":
                confidence += 0.2  # Evaluation-based insights are more reliable

            # Factor 2: Cluster size (if clustered)
            cluster_size = insight.get("cluster_size", 1)
            if cluster_size > 1:
                confidence += min(0.3, cluster_size * 0.1)

            # Factor 3: Content length and specificity
            content_length = len(insight["content"])
            if content_length > 50:  # More detailed insights
                confidence += 0.1
            if content_length > 100:
                confidence += 0.1

            # Factor 4: Category-specific adjustments
            category = insight.get("category", "general")
            if category in ["performance", "security", "correctness"]:
                confidence += 0.1  # Critical categories get bonus

            # Factor 5: Numerical evidence (for patterns)
            if "avg_score" in insight:
                score = insight["avg_score"]
                if score < 0.3 or score > 0.9:  # Extreme scores are more significant
                    confidence += 0.1

            # Clamp confidence to [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            insight["confidence"] = confidence
            scored_insights.append(insight)

        # Sort by confidence (highest first)
        scored_insights.sort(key=lambda x: x["confidence"], reverse=True)

        return scored_insights

    async def _resolve_insight_conflicts(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between contradictory insights."""
        # Group insights by category
        category_groups = {}
        for insight in insights:
            category = insight.get("category", "general")
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(insight)

        resolved_insights = []

        for category, group_insights in category_groups.items():
            if len(group_insights) <= 1:
                resolved_insights.extend(group_insights)
                continue

            # Detect conflicts within category
            conflicts = await self._detect_conflicts(group_insights)

            if not conflicts:
                resolved_insights.extend(group_insights)
            else:
                # Resolve conflicts by confidence score
                resolved = await self._resolve_conflicts_by_confidence(conflicts)
                resolved_insights.extend(resolved)

        return resolved_insights

    async def _detect_conflicts(self, insights: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Detect conflicting insights using LLM analysis."""
        if len(insights) < 2:
            return []

        conflicts = []

        # Simple heuristic: check for opposing keywords
        opposing_pairs = [
            ("fast", "slow"),
            ("efficient", "inefficient"),
            ("simple", "complex"),
            ("readable", "unreadable"),
            ("good", "bad"),
            ("better", "worse"),
        ]

        for i in range(len(insights)):
            for j in range(i + 1, len(insights)):
                insight1, insight2 = insights[i], insights[j]
                content1, content2 = insight1["content"].lower(), insight2["content"].lower()

                for word1, word2 in opposing_pairs:
                    if (word1 in content1 and word2 in content2) or (word2 in content1 and word1 in content2):
                        conflicts.append((insight1, insight2))
                        break

        return conflicts

    async def _resolve_conflicts_by_confidence(
        self, conflicts: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Resolve conflicts by selecting higher confidence insights."""
        resolved = []
        processed = set()

        for insight1, insight2 in conflicts:
            id1, id2 = id(insight1), id(insight2)

            if id1 in processed or id2 in processed:
                continue

            # Choose higher confidence insight
            if insight1.get("confidence", 0) >= insight2.get("confidence", 0):
                resolved.append(insight1)
                processed.add(id1)
                self.logger.debug(f"Resolved conflict: kept '{insight1['content'][:50]}...'")
            else:
                resolved.append(insight2)
                processed.add(id2)
                self.logger.debug(f"Resolved conflict: kept '{insight2['content'][:50]}...'")

        return resolved

    async def _version_principles(self, insights: List[Dict[str, Any]]) -> None:
        """Version and store principles for tracking evolution."""
        for insight in insights:
            if insight.get("type") == "principle":
                principle_text = insight["content"]
                confidence = insight.get("confidence", 0.5)
                category = insight.get("category", "general")

                # Create versioned principle
                principle = PrincipleVersion(
                    principle=principle_text,
                    confidence=confidence,
                    metadata={
                        "category": category,
                        "source": insight.get("source", "unknown"),
                        "examples": insight.get("examples", ""),
                    },
                )

                # Store in versioning system
                if category not in self.principles:
                    self.principles[category] = []

                # Check if this supersedes existing principles
                await self._check_principle_supersession(category, principle)

                self.principles[category].append(principle)

        # Save versioning state
        await self._save_principle_versions()

    async def _check_principle_supersession(self, category: str, new_principle: PrincipleVersion) -> None:
        """Check if new principle supersedes existing ones."""
        if category not in self.principles:
            return

        for existing in self.principles[category]:
            if existing.superseded:
                continue

            # Simple similarity check (could be enhanced with semantic similarity)
            similarity = await self._calculate_principle_similarity(existing.principle, new_principle.principle)

            if similarity > 0.8:  # High similarity threshold
                if new_principle.confidence > existing.confidence:
                    existing.superseded = True
                    new_principle.version = existing.version + 1
                    self.logger.info(f"Principle superseded: v{existing.version} -> v{new_principle.version}")

    async def _calculate_principle_similarity(self, principle1: str, principle2: str) -> float:
        """Calculate similarity between two principles (simplified implementation)."""
        # Simple word overlap similarity (could be enhanced with embeddings)
        words1 = set(principle1.lower().split())
        words2 = set(principle2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def _load_current_prompt(self) -> str:
        """Load the current system prompt."""
        prompt_file = self.prompt_path / "system_prompt.txt"

        if prompt_file.exists():
            return await asyncio.to_thread(prompt_file.read_text)
        else:
            # Default system prompt
            return """You are a helpful coding assistant. Write clean, efficient, and well-documented code.
Follow best practices and explain your reasoning when solving problems."""

    async def _generate_updated_prompt(self, current_prompt: str, insights: List[Dict[str, Any]]) -> str:
        """Generate updated system prompt incorporating insights."""
        # Extract principles for prompt integration
        principles = [insight["content"] for insight in insights if insight.get("type") == "principle"]
        patterns = [insight["content"] for insight in insights if insight.get("type") == "pattern"]

        update_prompt = f"""
        Update the following system prompt by incorporating the new insights and principles.
        Maintain the original tone and structure while seamlessly integrating new guidance.
        
        Current System Prompt:
        {current_prompt}
        
        New Principles to Integrate:
        {chr(10).join(f"- {p}" for p in principles)}
        
        Patterns to Address:
        {chr(10).join(f"- {p}" for p in patterns)}
        
        Generate an improved system prompt that incorporates these insights naturally.
        Keep it concise but comprehensive.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=update_prompt,
                options={
                    "temperature": 0.4,
                    "num_predict": 800,
                },
            )

            updated_prompt = response["response"].strip()

            if len(updated_prompt) < 50:  # Sanity check
                raise UpdateError("Generated prompt is too short")

            return updated_prompt

        except Exception as e:
            raise UpdateError(f"Failed to generate updated prompt: {e}") from e

    async def _store_updated_prompt(self, prompt: str) -> None:
        """Store the updated prompt with versioning."""
        # Store current version
        prompt_file = self.prompt_path / "system_prompt.txt"
        await asyncio.to_thread(prompt_file.write_text, prompt)

        # Store versioned copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_file = self.versions_path / f"system_prompt_{timestamp}.txt"
        await asyncio.to_thread(version_file.write_text, prompt)

        self.logger.info(f"Updated prompt stored: {prompt_file}, versioned: {version_file}")

    async def _save_principle_versions(self) -> None:
        """Save principle versioning state."""
        version_data = {}

        for category, principles in self.principles.items():
            version_data[category] = [p.to_dict() for p in principles]

        version_file = self.versions_path / "principle_versions.json"
        await asyncio.to_thread(version_file.write_text, json.dumps(version_data, indent=2, ensure_ascii=False))

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance metrics report."""
        if not self.monitor:
            return {"message": "Performance monitoring is disabled"}
        return self.monitor.get_performance_report()

    def get_principle_summary(self) -> Dict[str, Any]:
        """Get summary of current principles by category."""
        summary = {
            "total_principles": sum(len(principles) for principles in self.principles.values()),
            "categories": {},
            "active_principles": 0,
        }

        for category, principles in self.principles.items():
            active = [p for p in principles if not p.superseded]
            summary["categories"][category] = {
                "total": len(principles),
                "active": len(active),
                "avg_confidence": sum(p.confidence for p in active) / len(active) if active else 0.0,
            }
            summary["active_principles"] += len(active)

        return summary
