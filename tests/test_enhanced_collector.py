"""
Tests for the enhanced FeedbackCollector with async support, caching, and multiple backends.
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from memento.config import EvaluationBackend, ModelConfig, ModelType
from memento.core.collector import FeedbackCollector
from memento.exceptions import EvaluationError, ReflectionError, ValidationError


class TestFeedbackCollector:
    """Test cases for enhanced FeedbackCollector."""

    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.7)

    @pytest.fixture
    def storage_path(self):
        """Create temporary storage path."""
        with TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def collector(self, model_config, storage_path):
        """Create FeedbackCollector instance."""
        return FeedbackCollector(
            model_config=model_config,
            storage_path=storage_path,
            enable_metrics=True,
            cache_evaluations=True,
            evaluation_backend=EvaluationBackend.LLM,
        )

    @pytest.fixture
    def test_problem(self):
        """Sample test problem."""
        return {
            "description": "Write a function to calculate fibonacci numbers",
            "solution": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        }

    @pytest.fixture
    def test_criteria(self):
        """Sample evaluation criteria."""
        return ["correctness", "efficiency", "readability"]

    def test_initialization(self, model_config, storage_path):
        """Test FeedbackCollector initialization."""
        collector = FeedbackCollector(
            model_config=model_config,
            storage_path=storage_path,
            enable_metrics=False,
            cache_evaluations=False,
            evaluation_backend=EvaluationBackend.AUTOMATED,
        )

        assert collector.model_config == model_config
        assert collector.storage_path == storage_path
        assert collector.cache_evaluations is False
        assert collector.evaluation_backend == EvaluationBackend.AUTOMATED
        assert collector.monitor is None
        assert collector.cache_path.exists()

    def test_initialization_with_metrics(self, model_config, storage_path):
        """Test FeedbackCollector initialization with metrics enabled."""
        collector = FeedbackCollector(model_config=model_config, storage_path=storage_path, enable_metrics=True)

        assert collector.monitor is not None
        assert collector.cache_evaluations is True
        assert collector.evaluation_backend == EvaluationBackend.LLM

    @pytest.mark.asyncio
    async def test_collect_solution_feedback_success(self, collector, test_problem, test_criteria):
        """Test successful feedback collection."""
        with patch("ollama.generate") as mock_generate:
            # Mock evaluation response
            mock_generate.side_effect = [
                {"response": '{"correctness": 0.8, "efficiency": 0.6, "readability": 0.9}'},
                {"response": "This solution uses recursion which is clear but inefficient for large numbers."},
            ]

            result = await collector.collect_solution_feedback(
                problem=test_problem["description"],
                solution=test_problem["solution"],
                evaluation_criteria=test_criteria,
            )

            assert "timestamp" in result
            assert result["problem"] == test_problem["description"]
            assert result["solution"] == test_problem["solution"]
            assert result["evaluation_criteria"] == test_criteria
            assert "evaluation" in result
            assert "reflection" in result
            assert result["backend"] == "llm"
            assert "model_config" in result

            # Check evaluation scores
            evaluation = result["evaluation"]
            assert evaluation["correctness"] == 0.8
            assert evaluation["efficiency"] == 0.6
            assert evaluation["readability"] == 0.9

    @pytest.mark.asyncio
    async def test_collect_solution_feedback_validation_error(self, collector):
        """Test feedback collection with validation errors."""
        # Test empty problem
        with pytest.raises(ValidationError, match="Problem cannot be empty"):
            await collector.collect_solution_feedback("", "solution", ["criteria"])

        # Test empty solution
        with pytest.raises(ValidationError, match="Solution cannot be empty"):
            await collector.collect_solution_feedback("problem", "", ["criteria"])

        # Test empty criteria
        with pytest.raises(ValidationError, match="Evaluation criteria cannot be empty"):
            await collector.collect_solution_feedback("problem", "solution", [])

    @pytest.mark.asyncio
    async def test_evaluate_solution_llm_backend(self, collector, test_problem, test_criteria):
        """Test solution evaluation with LLM backend."""
        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": '{"correctness": 0.85, "efficiency": 0.70, "readability": 0.90}'}

            result = await collector.evaluate_solution(test_problem["solution"], test_criteria)

            assert isinstance(result, dict)
            assert len(result) == len(test_criteria)
            assert all(0.0 <= score <= 1.0 for score in result.values())
            assert result["correctness"] == 0.85
            assert result["efficiency"] == 0.70
            assert result["readability"] == 0.90

    @pytest.mark.asyncio
    async def test_evaluate_solution_automated_backend(self, model_config, storage_path, test_problem, test_criteria):
        """Test solution evaluation with automated backend."""
        collector = FeedbackCollector(
            model_config=model_config, storage_path=storage_path, evaluation_backend=EvaluationBackend.AUTOMATED
        )

        result = await collector.evaluate_solution(test_problem["solution"], test_criteria)

        assert isinstance(result, dict)
        assert len(result) == len(test_criteria)
        assert all(0.0 <= score <= 1.0 for score in result.values())

    @pytest.mark.asyncio
    async def test_evaluate_solution_caching(self, collector, test_problem, test_criteria):
        """Test evaluation result caching."""
        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": '{"correctness": 0.85, "efficiency": 0.70, "readability": 0.90}'}

            # First call should use LLM
            result1 = await collector.evaluate_solution(test_problem["solution"], test_criteria)
            assert mock_generate.call_count == 1

            # Second call should use cache
            result2 = await collector.evaluate_solution(test_problem["solution"], test_criteria)
            assert mock_generate.call_count == 1  # No additional calls
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_evaluate_solution_invalid_json(self, collector, test_problem, test_criteria):
        """Test evaluation with invalid JSON response."""
        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": "invalid json"}

            with pytest.raises(EvaluationError, match="Invalid LLM evaluation response"):
                await collector.evaluate_solution(test_problem["solution"], test_criteria)

    @pytest.mark.asyncio
    async def test_generate_reflection_success(self, collector, test_problem):
        """Test successful reflection generation."""
        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {
                "response": "This recursive approach is clear but has exponential time complexity."
            }

            result = await collector.generate_reflection(test_problem["description"], test_problem["solution"])

            assert isinstance(result, str)
            assert len(result) > 0
            assert "recursive" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_reflection_caching(self, collector, test_problem):
        """Test reflection caching."""
        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {
                "response": "This recursive approach is clear but has exponential time complexity."
            }

            # First call should use LLM
            result1 = await collector.generate_reflection(test_problem["description"], test_problem["solution"])
            assert mock_generate.call_count == 1

            # Second call should use cache
            result2 = await collector.generate_reflection(test_problem["description"], test_problem["solution"])
            assert mock_generate.call_count == 1  # No additional calls
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_generate_reflection_empty_response(self, collector, test_problem):
        """Test reflection generation with empty response."""
        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": ""}

            with pytest.raises(ReflectionError, match="Generated reflection is empty"):
                await collector.generate_reflection(test_problem["description"], test_problem["solution"])

    @pytest.mark.asyncio
    async def test_batch_collect_feedback(self, collector, test_criteria):
        """Test batch feedback collection."""
        problems_solutions = [
            {"problem": "Problem 1", "solution": "Solution 1"},
            {"problem": "Problem 2", "solution": "Solution 2"},
        ]

        with patch("ollama.generate") as mock_generate:
            mock_generate.side_effect = [
                {"response": '{"correctness": 0.8, "efficiency": 0.6, "readability": 0.9}'},
                {"response": "Reflection 1"},
                {"response": '{"correctness": 0.7, "efficiency": 0.8, "readability": 0.85}'},
                {"response": "Reflection 2"},
            ]

            results = await collector.batch_collect_feedback(problems_solutions, test_criteria)

            assert len(results) == 2
            assert all("timestamp" in result for result in results)
            assert all("evaluation" in result for result in results)
            assert all("reflection" in result for result in results)

    @pytest.mark.asyncio
    async def test_batch_collect_feedback_partial_failure(self, collector, test_criteria):
        """Test batch feedback collection with partial failures."""
        problems_solutions = [
            {"problem": "Problem 1", "solution": "Solution 1"},
            {"problem": "", "solution": "Solution 2"},  # Invalid problem
        ]

        with patch("ollama.generate") as mock_generate:
            mock_generate.side_effect = [
                {"response": '{"correctness": 0.8, "efficiency": 0.6, "readability": 0.9}'},
                {"response": "Reflection 1"},
            ]

            results = await collector.batch_collect_feedback(problems_solutions, test_criteria)

            # Should have 1 successful result (first item), second should fail validation
            assert len(results) == 1
            assert results[0]["problem"] == "Problem 1"

    def test_clear_cache(self, collector):
        """Test cache clearing functionality."""
        # Add some items to cache
        collector._evaluation_cache["test"] = {"score": 0.8}
        collector._reflection_cache["test"] = "test reflection"

        assert len(collector._evaluation_cache) == 1
        assert len(collector._reflection_cache) == 1

        collector.clear_cache()

        assert len(collector._evaluation_cache) == 0
        assert len(collector._reflection_cache) == 0

    def test_get_performance_report_disabled(self, model_config, storage_path):
        """Test performance report when metrics are disabled."""
        collector = FeedbackCollector(model_config=model_config, storage_path=storage_path, enable_metrics=False)

        report = collector.get_performance_report()
        assert "message" in report
        assert "Performance monitoring is disabled" in report["message"]

    def test_get_performance_report_enabled(self, collector):
        """Test performance report when metrics are enabled."""
        report = collector.get_performance_report()

        # Should have either metrics or a report structure
        assert isinstance(report, dict)
        # Initially no metrics, so should have basic structure
        assert "generated_at" in report or "message" in report


class TestFeedbackCollectorIntegration:
    """Integration tests for FeedbackCollector."""

    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.5)

    @pytest.fixture
    def storage_path(self):
        """Create temporary storage path."""
        with TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_full_feedback_workflow_llm(self, model_config, storage_path):
        """Test complete feedback collection workflow with LLM backend."""
        collector = FeedbackCollector(
            model_config=model_config,
            storage_path=storage_path,
            evaluation_backend=EvaluationBackend.LLM,
            enable_metrics=True,
        )

        with patch("ollama.generate") as mock_generate:
            mock_generate.side_effect = [
                {"response": '{"correctness": 0.9, "efficiency": 0.7, "readability": 0.8}'},
                {"response": "The solution is correct but could be optimized for better performance."},
            ]

            feedback = await collector.collect_solution_feedback(
                problem="Calculate factorial of a number",
                solution="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                evaluation_criteria=["correctness", "efficiency", "readability"],
            )

            # Verify feedback structure
            assert feedback["evaluation"]["correctness"] == 0.9
            assert feedback["evaluation"]["efficiency"] == 0.7
            assert feedback["evaluation"]["readability"] == 0.8
            assert "optimized" in feedback["reflection"].lower()

            # Verify file was created
            feedback_files = list(storage_path.glob("feedback_*.json"))
            assert len(feedback_files) == 1

            # Verify file content
            with open(feedback_files[0]) as f:
                stored_data = json.load(f)
            assert stored_data["problem"] == feedback["problem"]
            assert stored_data["evaluation"] == feedback["evaluation"]

    @pytest.mark.asyncio
    async def test_full_feedback_workflow_automated(self, model_config, storage_path):
        """Test complete feedback collection workflow with automated backend."""
        collector = FeedbackCollector(
            model_config=model_config,
            storage_path=storage_path,
            evaluation_backend=EvaluationBackend.AUTOMATED,
            cache_evaluations=False,
        )

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": "The solution uses recursion which is readable but not optimal."}

            feedback = await collector.collect_solution_feedback(
                problem="Calculate factorial of a number",
                solution="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                evaluation_criteria=["correctness", "efficiency", "readability"],
            )

            # Verify automated evaluation
            assert all(0.0 <= score <= 1.0 for score in feedback["evaluation"].values())
            assert feedback["backend"] == "automated"

            # Verify reflection still uses LLM
            assert "recursion" in feedback["reflection"].lower()

    @pytest.mark.asyncio
    async def test_concurrent_feedback_collection(self, model_config, storage_path):
        """Test concurrent feedback collection operations."""
        collector = FeedbackCollector(
            model_config=model_config, storage_path=storage_path, evaluation_backend=EvaluationBackend.AUTOMATED
        )

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": "Good solution with room for improvement."}

            # Run multiple feedback collections concurrently
            tasks = []
            for i in range(3):
                task = collector.collect_solution_feedback(
                    problem=f"Problem {i}",
                    solution=f"def solution_{i}(): pass",
                    evaluation_criteria=["correctness", "readability"],
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all("evaluation" in result for result in results)
            assert all("reflection" in result for result in results)

            # Verify all files were created
            feedback_files = list(storage_path.glob("feedback_*.json"))
            assert len(feedback_files) == 3
