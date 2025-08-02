"""
Tests for the enhanced PromptProcessor with async support, principle extraction, and versioning.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from memento.config import ModelConfig, ModelType
from memento.core.processor import PrincipleVersion, PromptProcessor
from memento.exceptions import ExtractionError


class TestPrincipleVersion:
    """Test cases for PrincipleVersion class."""

    def test_principle_version_creation(self):
        """Test PrincipleVersion creation and validation."""
        principle = PrincipleVersion(
            principle="Use iterative approaches for large datasets", confidence=0.85, version=1
        )

        assert principle.principle == "Use iterative approaches for large datasets"
        assert principle.confidence == 0.85
        assert principle.version == 1
        assert not principle.superseded
        assert principle.created_at is not None

    def test_principle_version_confidence_clamping(self):
        """Test confidence score clamping to [0, 1] range."""
        # Test upper bound
        principle1 = PrincipleVersion("test", confidence=1.5)
        assert principle1.confidence == 1.0

        # Test lower bound
        principle2 = PrincipleVersion("test", confidence=-0.5)
        assert principle2.confidence == 0.0

        # Test normal range
        principle3 = PrincipleVersion("test", confidence=0.7)
        assert principle3.confidence == 0.7

    def test_principle_version_serialization(self):
        """Test to_dict and from_dict methods."""
        original = PrincipleVersion(
            principle="Test principle", confidence=0.8, version=2, metadata={"category": "performance"}
        )

        # Test serialization
        data = original.to_dict()
        assert data["principle"] == "Test principle"
        assert data["confidence"] == 0.8
        assert data["version"] == 2
        assert data["metadata"]["category"] == "performance"

        # Test deserialization
        restored = PrincipleVersion.from_dict(data)
        assert restored.principle == original.principle
        assert restored.confidence == original.confidence
        assert restored.version == original.version
        assert restored.metadata == original.metadata


class TestPromptProcessor:
    """Test cases for enhanced PromptProcessor."""

    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.3)

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            feedback_path = temp_path / "feedback"
            prompt_path = temp_path / "prompts"
            feedback_path.mkdir()
            prompt_path.mkdir()
            yield feedback_path, prompt_path

    @pytest.fixture
    def processor(self, model_config, temp_paths):
        """Create PromptProcessor instance."""
        feedback_path, prompt_path = temp_paths
        return PromptProcessor(
            model_config=model_config,
            feedback_path=feedback_path,
            prompt_path=prompt_path,
            enable_metrics=True,
            min_confidence_threshold=0.6,
            max_principles=20,
        )

    @pytest.fixture
    def sample_feedback_data(self):
        """Sample feedback data for testing."""
        return [
            {
                "timestamp": "2024-01-01T10:00:00",
                "problem": "Calculate fibonacci numbers",
                "solution": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
                "evaluation": {"correctness": 0.9, "efficiency": 0.3, "readability": 0.8},
                "reflection": "The recursive solution is correct but inefficient for large numbers due to exponential time complexity. Consider using dynamic programming or iterative approach.",
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "problem": "Sort an array",
                "solution": "def sort_array(arr): return sorted(arr)",
                "evaluation": {"correctness": 1.0, "efficiency": 0.8, "readability": 0.9},
                "reflection": "Using built-in sorted() is efficient and readable. Good choice for general sorting needs.",
            },
        ]

    def test_initialization(self, model_config, temp_paths):
        """Test PromptProcessor initialization."""
        feedback_path, prompt_path = temp_paths
        processor = PromptProcessor(
            model_config=model_config,
            feedback_path=feedback_path,
            prompt_path=prompt_path,
            enable_metrics=False,
            min_confidence_threshold=0.7,
            max_principles=30,
        )

        assert processor.model_config == model_config
        assert processor.feedback_path == feedback_path
        assert processor.prompt_path == prompt_path
        assert processor.min_confidence_threshold == 0.7
        assert processor.max_principles == 30
        assert processor.monitor is None
        assert processor.versions_path.exists()

    def test_initialization_with_metrics(self, model_config, temp_paths):
        """Test PromptProcessor initialization with metrics enabled."""
        feedback_path, prompt_path = temp_paths
        processor = PromptProcessor(
            model_config=model_config, feedback_path=feedback_path, prompt_path=prompt_path, enable_metrics=True
        )

        assert processor.monitor is not None
        assert processor.min_confidence_threshold == 0.6  # default
        assert processor.max_principles == 50  # default

    @pytest.mark.asyncio
    async def test_process_feedback_no_files(self, processor):
        """Test processing feedback when no files exist."""
        result = await processor.process_feedback()
        assert result == []

    @pytest.mark.asyncio
    async def test_process_feedback_with_files(self, processor, temp_paths, sample_feedback_data):
        """Test processing feedback with actual files."""
        feedback_path, _ = temp_paths

        # Create sample feedback files
        for i, feedback in enumerate(sample_feedback_data):
            file_path = feedback_path / f"feedback_{i}.json"
            file_path.write_text(json.dumps(feedback, indent=2))

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {
                "response": json.dumps(
                    [
                        {
                            "principle": "Use iterative approaches instead of recursion for large datasets",
                            "category": "performance",
                            "examples": "fibonacci, factorial calculations",
                        }
                    ]
                )
            }

            result = await processor.process_feedback()

            assert isinstance(result, list)
            assert len(result) > 0
            # Should have insights from both principles and patterns

    @pytest.mark.asyncio
    async def test_extract_insights_empty_feedback(self, processor):
        """Test insight extraction with empty feedback."""
        result = await processor.extract_insights([])
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_insights_with_reflections(self, processor, sample_feedback_data):
        """Test insight extraction from reflections."""
        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {
                "response": json.dumps(
                    [
                        {
                            "principle": "Avoid recursive solutions for problems with overlapping subproblems",
                            "category": "performance",
                            "examples": "fibonacci, dynamic programming problems",
                        },
                        {
                            "principle": "Use built-in functions when available for better performance",
                            "category": "efficiency",
                            "examples": "sorting, searching operations",
                        },
                    ]
                )
            }

            insights = await processor.extract_insights(sample_feedback_data)

            assert isinstance(insights, list)
            assert len(insights) > 0

            # Check that insights have required fields
            for insight in insights:
                assert "type" in insight
                assert "content" in insight
                assert "confidence" in insight
                assert 0.0 <= insight["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_extract_insights_with_patterns(self, processor, sample_feedback_data):
        """Test pattern extraction from evaluation data."""
        insights = await processor.extract_insights(sample_feedback_data)

        # Should extract patterns from evaluation scores
        pattern_insights = [i for i in insights if i.get("type") == "pattern"]
        assert len(pattern_insights) > 0

        # Check for efficiency pattern (low average in first example)
        efficiency_patterns = [p for p in pattern_insights if "efficiency" in p.get("criterion", "")]
        assert len(efficiency_patterns) > 0

    @pytest.mark.asyncio
    async def test_extract_insights_confidence_filtering(self, processor, sample_feedback_data):
        """Test that low-confidence insights are filtered out."""
        processor.min_confidence_threshold = 0.9  # Very high threshold

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {
                "response": json.dumps(
                    [{"principle": "Test principle", "category": "general", "examples": "test examples"}]
                )
            }

            insights = await processor.extract_insights(sample_feedback_data)

            # Should filter out insights below threshold
            for insight in insights:
                assert insight["confidence"] >= 0.9

    @pytest.mark.asyncio
    async def test_extract_insights_invalid_json(self, processor, sample_feedback_data):
        """Test handling of invalid JSON from LLM."""
        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": "invalid json"}

            with pytest.raises(ExtractionError, match="Failed to extract principles"):
                await processor.extract_insights(sample_feedback_data)

    @pytest.mark.asyncio
    async def test_update_system_prompt_basic(self, processor, temp_paths):
        """Test basic system prompt update."""
        _, prompt_path = temp_paths

        # Create initial prompt
        initial_prompt = "You are a helpful coding assistant."
        prompt_file = prompt_path / "system_prompt.txt"
        prompt_file.write_text(initial_prompt)

        insights = [
            {
                "type": "principle",
                "content": "Always validate input parameters",
                "confidence": 0.8,
                "category": "validation",
            }
        ]

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {
                "response": "You are a helpful coding assistant. Always validate input parameters before processing."
            }

            updated_prompt = await processor.update_system_prompt(insights)

            assert len(updated_prompt) > len(initial_prompt)
            assert "validate" in updated_prompt.lower()

    @pytest.mark.asyncio
    async def test_update_system_prompt_with_versioning(self, processor, temp_paths):
        """Test system prompt update creates versions."""
        _, prompt_path = temp_paths

        insights = [
            {
                "type": "principle",
                "content": "Use descriptive variable names",
                "confidence": 0.9,
                "category": "readability",
            }
        ]

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": "Updated system prompt with new principles."}

            await processor.update_system_prompt(insights)

            # Check that versioned file was created
            version_files = list(processor.versions_path.glob("system_prompt_*.txt"))
            assert len(version_files) >= 1

    @pytest.mark.asyncio
    async def test_principle_versioning(self, processor):
        """Test principle versioning system."""
        insights = [
            {
                "type": "principle",
                "content": "Use caching for expensive operations",
                "confidence": 0.8,
                "category": "performance",
            },
            {
                "type": "principle",
                "content": "Always cache expensive computations",  # Similar principle
                "confidence": 0.9,
                "category": "performance",
            },
        ]

        await processor._version_principles(insights)

        assert "performance" in processor.principles
        assert len(processor.principles["performance"]) == 2

        # Check that versioning file was created
        version_file = processor.versions_path / "principle_versions.json"
        assert version_file.exists()

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, processor):
        """Test conflict resolution between contradictory insights."""
        conflicting_insights = [
            {
                "type": "principle",
                "content": "Use recursion for better readability",
                "confidence": 0.6,
                "category": "readability",
            },
            {
                "type": "principle",
                "content": "Avoid recursion for better performance",
                "confidence": 0.8,
                "category": "performance",
            },
        ]

        resolved = await processor._resolve_insight_conflicts(conflicting_insights)

        # Should keep the higher confidence insight in case of conflict
        assert len(resolved) <= len(conflicting_insights)

    @pytest.mark.asyncio
    async def test_clustering_insights(self, processor):
        """Test insight clustering functionality."""
        similar_insights = [
            {
                "type": "principle",
                "content": "Use efficient algorithms for large datasets",
                "confidence": 0.7,
                "category": "performance",
            },
            {
                "type": "principle",
                "content": "Choose efficient algorithms when processing large data",
                "confidence": 0.8,
                "category": "performance",
            },
            {"type": "principle", "content": "Always validate user input", "confidence": 0.9, "category": "validation"},
        ]

        clustered = await processor._cluster_insights(similar_insights)

        # Should cluster similar insights together
        assert len(clustered) <= len(similar_insights)

        # Check for cluster metadata
        for insight in clustered:
            if "cluster_size" in insight:
                assert insight["cluster_size"] >= 1

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, processor):
        """Test confidence scoring algorithm."""
        insights = [
            {"type": "principle", "content": "Short principle", "source": "reflection", "category": "general"},
            {
                "type": "principle",
                "content": "This is a much longer and more detailed principle that should receive higher confidence score",
                "source": "evaluation",
                "category": "performance",
                "cluster_size": 3,
            },
        ]

        scored = await processor._score_insight_confidence(insights)

        assert len(scored) == len(insights)

        # Longer, evaluation-based, clustered insight should have higher confidence
        assert scored[0]["confidence"] > scored[1]["confidence"]

    def test_get_performance_report_disabled(self, model_config, temp_paths):
        """Test performance report when metrics are disabled."""
        feedback_path, prompt_path = temp_paths
        processor = PromptProcessor(
            model_config=model_config, feedback_path=feedback_path, prompt_path=prompt_path, enable_metrics=False
        )

        report = processor.get_performance_report()
        assert "message" in report
        assert "Performance monitoring is disabled" in report["message"]

    def test_get_performance_report_enabled(self, processor):
        """Test performance report when metrics are enabled."""
        report = processor.get_performance_report()

        # Should have either metrics or a report structure
        assert isinstance(report, dict)

    def test_get_principle_summary_empty(self, processor):
        """Test principle summary with no principles."""
        summary = processor.get_principle_summary()

        assert summary["total_principles"] == 0
        assert summary["active_principles"] == 0
        assert isinstance(summary["categories"], dict)

    @pytest.mark.asyncio
    async def test_get_principle_summary_with_data(self, processor):
        """Test principle summary with actual principles."""
        insights = [
            {"type": "principle", "content": "Test principle 1", "confidence": 0.8, "category": "performance"},
            {"type": "principle", "content": "Test principle 2", "confidence": 0.7, "category": "readability"},
        ]

        await processor._version_principles(insights)
        summary = processor.get_principle_summary()

        assert summary["total_principles"] == 2
        assert summary["active_principles"] == 2
        assert len(summary["categories"]) == 2
        assert "performance" in summary["categories"]
        assert "readability" in summary["categories"]


class TestPromptProcessorIntegration:
    """Integration tests for PromptProcessor."""

    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.3)

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            feedback_path = temp_path / "feedback"
            prompt_path = temp_path / "prompts"
            feedback_path.mkdir()
            prompt_path.mkdir()
            yield feedback_path, prompt_path

    @pytest.mark.asyncio
    async def test_full_processing_workflow(self, model_config, temp_paths):
        """Test complete processing workflow from feedback to updated prompt."""
        feedback_path, prompt_path = temp_paths

        processor = PromptProcessor(
            model_config=model_config,
            feedback_path=feedback_path,
            prompt_path=prompt_path,
            enable_metrics=True,
            min_confidence_threshold=0.5,
        )

        # Create sample feedback files
        feedback_data = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "problem": "Implement binary search",
                "solution": "def binary_search(arr, target): # implementation",
                "evaluation": {"correctness": 0.9, "efficiency": 0.8, "readability": 0.7},
                "reflection": "Binary search is efficient for sorted arrays. Time complexity is O(log n).",
            }
        ]

        for i, feedback in enumerate(feedback_data):
            file_path = feedback_path / f"feedback_{i}.json"
            file_path.write_text(json.dumps(feedback, indent=2))

        # Mock LLM responses
        with patch("ollama.generate") as mock_generate:
            mock_generate.side_effect = [
                # Response for principle extraction
                {
                    "response": json.dumps(
                        [
                            {
                                "principle": "Use binary search for sorted data structures",
                                "category": "algorithms",
                                "examples": "searching in sorted arrays, trees",
                            }
                        ]
                    )
                },
                # Response for prompt update
                {
                    "response": "You are a helpful coding assistant. Use binary search for sorted data structures when appropriate."
                },
            ]

            # Run full workflow
            insights = await processor.process_feedback()
            updated_prompt = await processor.update_system_prompt(insights)

            # Verify results
            assert len(insights) > 0
            assert len(updated_prompt) > 50
            assert "binary search" in updated_prompt.lower()

            # Verify files were created
            assert (prompt_path / "system_prompt.txt").exists()
            version_files = list(processor.versions_path.glob("system_prompt_*.txt"))
            assert len(version_files) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, model_config, temp_paths):
        """Test concurrent processing of multiple feedback files."""
        feedback_path, prompt_path = temp_paths

        processor = PromptProcessor(
            model_config=model_config, feedback_path=feedback_path, prompt_path=prompt_path, enable_metrics=False
        )

        # Create multiple feedback files
        for i in range(5):
            feedback = {
                "timestamp": f"2024-01-01T{10+i}:00:00",
                "problem": f"Problem {i}",
                "solution": f"def solution_{i}(): pass",
                "evaluation": {"correctness": 0.8, "efficiency": 0.7, "readability": 0.9},
                "reflection": f"Reflection for problem {i}",
            }
            file_path = feedback_path / f"feedback_{i}.json"
            file_path.write_text(json.dumps(feedback, indent=2))

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {
                "response": json.dumps(
                    [
                        {
                            "principle": "Write clear and concise code",
                            "category": "readability",
                            "examples": "function naming, variable naming",
                        }
                    ]
                )
            }

            insights = await processor.process_feedback()

            # Should process all files and extract insights
            assert len(insights) > 0

    @pytest.mark.asyncio
    async def test_error_resilience(self, model_config, temp_paths):
        """Test error resilience with corrupted feedback files."""
        feedback_path, prompt_path = temp_paths

        processor = PromptProcessor(model_config=model_config, feedback_path=feedback_path, prompt_path=prompt_path)

        # Create mix of valid and invalid feedback files
        valid_feedback = {
            "timestamp": "2024-01-01T10:00:00",
            "problem": "Valid problem",
            "solution": "def valid(): pass",
            "evaluation": {"correctness": 0.8},
            "reflection": "Valid reflection",
        }

        # Valid file
        (feedback_path / "feedback_valid.json").write_text(json.dumps(valid_feedback))

        # Invalid JSON file
        (feedback_path / "feedback_invalid.json").write_text("invalid json content")

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {
                "response": json.dumps(
                    [
                        {
                            "principle": "Handle errors gracefully",
                            "category": "reliability",
                            "examples": "exception handling, validation",
                        }
                    ]
                )
            }

            # Should handle errors gracefully and process valid files
            insights = await processor.process_feedback()

            # Should still extract insights from valid files
            assert isinstance(insights, list)
