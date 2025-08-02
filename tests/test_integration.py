"""
Integration tests for the complete Memento framework workflow.

Tests the interaction between PromptLearner, FeedbackCollector, and PromptProcessor
to ensure the full feedback loop works correctly.
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from memento.config import EvaluationBackend, ModelConfig, ModelType
from memento.core.collector import FeedbackCollector
from memento.core.learner import PromptLearner
from memento.core.processor import PromptProcessor


class TestMementoIntegration:
    """Integration tests for the complete Memento workflow."""

    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return ModelConfig(model_type=ModelType.OLLAMA, model_name="llama3.2", temperature=0.7)

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing."""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            storage_path = temp_path / "storage"
            feedback_path = temp_path / "feedback"
            prompt_path = temp_path / "prompts"

            storage_path.mkdir()
            feedback_path.mkdir()
            prompt_path.mkdir()

            yield storage_path, feedback_path, prompt_path

    @pytest.fixture
    def memento_components(self, model_config, temp_paths):
        """Create all three Memento core components."""
        storage_path, feedback_path, prompt_path = temp_paths

        learner = PromptLearner(model_config=model_config, storage_path=storage_path, enable_metrics=True)

        collector = FeedbackCollector(
            model_config=model_config,
            storage_path=feedback_path,
            enable_metrics=True,
            evaluation_backend=EvaluationBackend.LLM,
        )

        processor = PromptProcessor(
            model_config=model_config,
            feedback_path=feedback_path,
            prompt_path=prompt_path,
            enable_metrics=True,
            min_confidence_threshold=0.6,
        )

        return learner, collector, processor

    @pytest.fixture
    def sample_problems(self):
        """Sample problems for testing."""
        return [
            {
                "description": "Write a function to calculate the factorial of a number",
                "solution": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "criteria": ["correctness", "efficiency", "readability"],
            },
            {
                "description": "Implement binary search algorithm",
                "solution": "def binary_search(arr, target): # binary search implementation",
                "criteria": ["correctness", "efficiency", "readability"],
            },
            {
                "description": "Create a function to reverse a string",
                "solution": "def reverse_string(s): return s[::-1]",
                "criteria": ["correctness", "readability", "simplicity"],
            },
        ]

    @pytest.mark.asyncio
    async def test_complete_feedback_loop(self, memento_components, sample_problems):
        """Test the complete feedback loop: Learning -> Collection -> Processing -> Update."""
        learner, collector, processor = memento_components

        # Mock LLM responses for all operations
        with patch("ollama.generate") as mock_generate:
            mock_responses = [
                # Learner evaluation responses
                {"response": '{"correctness": 0.8, "efficiency": 0.6, "readability": 0.9}'},
                {"response": "The recursive factorial is correct but inefficient for large numbers."},
                # Collector evaluation responses
                {"response": '{"correctness": 0.8, "efficiency": 0.6, "readability": 0.9}'},
                {"response": "Recursive approach has exponential time complexity."},
                # Processor principle extraction
                {
                    "response": json.dumps(
                        [
                            {
                                "principle": "Use iterative approaches for better performance with large inputs",
                                "category": "performance",
                                "examples": "factorial, fibonacci, tree traversal",
                            }
                        ]
                    )
                },
                # Processor prompt update
                {
                    "response": "You are a coding assistant. Use iterative approaches for better performance with large inputs."
                },
            ]
            mock_generate.side_effect = mock_responses

            problem = sample_problems[0]

            # Step 1: Use PromptLearner to evaluate and evolve prompt
            initial_prompt = "You are a helpful coding assistant."

            evaluation_result = await learner.evaluate_prompt_performance(
                prompt=initial_prompt, problem=problem["description"], criteria=problem["criteria"]
            )

            assert "evaluation" in evaluation_result
            assert "lessons" in evaluation_result

            # Step 2: Use FeedbackCollector to collect detailed feedback
            feedback = await collector.collect_solution_feedback(
                problem=problem["description"], solution=problem["solution"], evaluation_criteria=problem["criteria"]
            )

            assert "evaluation" in feedback
            assert "reflection" in feedback
            assert feedback["backend"] == "llm"

            # Step 3: Use PromptProcessor to extract insights and update prompt
            insights = await processor.extract_insights([feedback])
            updated_prompt = await processor.update_system_prompt(insights)

            assert len(insights) > 0
            assert len(updated_prompt) > len(initial_prompt)
            assert "iterative" in updated_prompt.lower()

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, memento_components, sample_problems):
        """Test batch processing across all components."""
        learner, collector, processor = memento_components

        with patch("ollama.generate") as mock_generate:
            # Create enough mock responses for batch processing
            evaluation_responses = [
                {"response": '{"correctness": 0.9, "efficiency": 0.7, "readability": 0.8}'},
                {"response": '{"correctness": 0.8, "efficiency": 0.9, "readability": 0.9}'},
                {"response": '{"correctness": 0.7, "efficiency": 0.6, "readability": 1.0}'},
            ]

            reflection_responses = [
                {"response": "Factorial recursion is clear but inefficient."},
                {"response": "Binary search is optimal for sorted arrays."},
                {"response": "String slicing is pythonic and efficient."},
            ]

            principle_response = {
                "response": json.dumps(
                    [
                        {
                            "principle": "Choose appropriate algorithms based on data characteristics",
                            "category": "algorithms",
                            "examples": "binary search for sorted data, hash tables for lookups",
                        }
                    ]
                )
            }

            prompt_update_response = {"response": "Updated system prompt with algorithmic insights."}

            mock_generate.side_effect = (
                evaluation_responses + reflection_responses + [principle_response, prompt_update_response]
            )

            # Batch collect feedback for all problems
            problems_solutions = [{"problem": p["description"], "solution": p["solution"]} for p in sample_problems]

            feedback_batch = await collector.batch_collect_feedback(
                problems_solutions, ["correctness", "efficiency", "readability"]
            )

            assert len(feedback_batch) == len(sample_problems)

            # Process all feedback together
            insights = await processor.extract_insights(feedback_batch)
            updated_prompt = await processor.update_system_prompt(insights)

            assert len(insights) > 0
            assert isinstance(updated_prompt, str)
            assert len(updated_prompt) > 50

    @pytest.mark.asyncio
    async def test_performance_metrics_integration(self, memento_components, sample_problems):
        """Test that performance metrics are collected across all components."""
        learner, collector, processor = memento_components

        with patch("ollama.generate") as mock_generate:
            mock_generate.side_effect = [
                {"response": '{"correctness": 0.8, "efficiency": 0.7, "readability": 0.9}'},
                {"response": "Good solution with room for optimization."},
                {"response": '{"correctness": 0.8, "efficiency": 0.7, "readability": 0.9}'},
                {"response": "Consider performance implications."},
                {"response": json.dumps([{"principle": "Test principle", "category": "general", "examples": ""}])},
                {"response": "Updated prompt."},
            ]

            problem = sample_problems[0]

            # Perform operations that should generate metrics
            await learner.evaluate_prompt_performance(
                prompt="Test prompt", problem=problem["description"], criteria=problem["criteria"]
            )

            await collector.collect_solution_feedback(
                problem=problem["description"], solution=problem["solution"], evaluation_criteria=problem["criteria"]
            )

            await processor.process_feedback()

            # Check that all components have metrics
            learner_report = learner.get_performance_report()
            collector_report = collector.get_performance_report()
            processor_report = processor.get_performance_report()

            assert isinstance(learner_report, dict)
            assert isinstance(collector_report, dict)
            assert isinstance(processor_report, dict)

    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, memento_components, sample_problems):
        """Test error handling and recovery across components."""
        learner, collector, processor = memento_components

        # Test with one failing LLM call
        with patch("ollama.generate") as mock_generate:

            def side_effect(*args, **kwargs):
                # First call fails, subsequent calls succeed
                if not hasattr(side_effect, "call_count"):
                    side_effect.call_count = 0
                side_effect.call_count += 1

                if side_effect.call_count == 1:
                    raise Exception("LLM service unavailable")
                else:
                    return {"response": '{"correctness": 0.8, "efficiency": 0.7, "readability": 0.9}'}

            mock_generate.side_effect = side_effect

            problem = sample_problems[0]

            # First call should fail
            with pytest.raises(Exception):
                await learner.evaluate_prompt_performance(
                    prompt="Test prompt", problem=problem["description"], criteria=problem["criteria"]
                )

            # Reset for successful call
            mock_generate.side_effect = [
                {"response": '{"correctness": 0.8, "efficiency": 0.7, "readability": 0.9}'},
                {"response": "Good solution."},
            ]

            # Second call should succeed
            result = await collector.collect_solution_feedback(
                problem=problem["description"], solution=problem["solution"], evaluation_criteria=problem["criteria"]
            )

            assert "evaluation" in result

    @pytest.mark.asyncio
    async def test_caching_across_components(self, memento_components, sample_problems):
        """Test that caching works effectively across components."""
        learner, collector, processor = memento_components

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": '{"correctness": 0.8, "efficiency": 0.7, "readability": 0.9}'}

            problem = sample_problems[0]

            # Make the same evaluation twice - should use cache on second call
            await collector.evaluate_solution(problem["solution"], problem["criteria"])
            await collector.evaluate_solution(problem["solution"], problem["criteria"])

            # Should only call LLM once due to caching
            assert mock_generate.call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memento_components, sample_problems):
        """Test concurrent operations across all components."""
        learner, collector, processor = memento_components

        with patch("ollama.generate") as mock_generate:
            mock_generate.return_value = {"response": '{"correctness": 0.8, "efficiency": 0.7, "readability": 0.9}'}

            # Run multiple operations concurrently
            tasks = []

            for problem in sample_problems:
                # Collector task
                task = collector.evaluate_solution(problem["solution"], problem["criteria"])
                tasks.append(task)

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            assert len(results) == len(sample_problems)
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_data_flow_consistency(self, memento_components, sample_problems):
        """Test that data flows consistently between components."""
        learner, collector, processor = memento_components

        with patch("ollama.generate") as mock_generate:
            mock_generate.side_effect = [
                # Collector responses
                {"response": '{"correctness": 0.8, "efficiency": 0.6, "readability": 0.9}'},
                {"response": "Recursive solution needs optimization for large inputs."},
                # Processor responses
                {
                    "response": json.dumps(
                        [
                            {
                                "principle": "Optimize recursive algorithms with memoization",
                                "category": "performance",
                                "examples": "fibonacci, factorial",
                            }
                        ]
                    )
                },
                {"response": "You are a coding assistant. Optimize recursive algorithms with memoization."},
            ]

            problem = sample_problems[0]

            # Step 1: Collect feedback
            feedback = await collector.collect_solution_feedback(
                problem=problem["description"], solution=problem["solution"], evaluation_criteria=problem["criteria"]
            )

            # Verify feedback structure
            assert feedback["problem"] == problem["description"]
            assert feedback["solution"] == problem["solution"]
            assert "evaluation" in feedback
            assert "reflection" in feedback

            # Step 2: Process feedback
            insights = await processor.extract_insights([feedback])

            # Verify insights reference the original feedback
            assert len(insights) > 0
            principle_insights = [i for i in insights if i.get("type") == "principle"]
            assert len(principle_insights) > 0

            # Step 3: Update prompt
            updated_prompt = await processor.update_system_prompt(insights)

            # Verify prompt incorporates insights
            assert "memoization" in updated_prompt.lower()

    @pytest.mark.asyncio
    async def test_configuration_consistency(self, temp_paths):
        """Test that all components use consistent configuration."""
        storage_path, feedback_path, prompt_path = temp_paths

        # Create components with same model config
        model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="llama2", temperature=0.5)

        learner = PromptLearner(model_config=model_config, storage_path=storage_path)
        collector = FeedbackCollector(model_config=model_config, storage_path=feedback_path)
        processor = PromptProcessor(model_config=model_config, feedback_path=feedback_path, prompt_path=prompt_path)

        # Verify all components use the same configuration
        assert learner.model_config.model_name == "llama2"
        assert collector.model_config.model_name == "llama2"
        assert processor.model_config.model_name == "llama2"

        assert learner.model_config.temperature == 0.5
        assert collector.model_config.temperature == 0.5
        assert processor.model_config.temperature == 0.5

    def test_storage_path_isolation(self, temp_paths):
        """Test that components maintain proper storage path isolation."""
        storage_path, feedback_path, prompt_path = temp_paths

        model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="llama3.2")

        learner = PromptLearner(model_config=model_config, storage_path=storage_path)
        collector = FeedbackCollector(model_config=model_config, storage_path=feedback_path)
        processor = PromptProcessor(model_config=model_config, feedback_path=feedback_path, prompt_path=prompt_path)

        # Verify path isolation
        assert learner.storage_path == storage_path
        assert collector.storage_path == feedback_path
        assert processor.feedback_path == feedback_path
        assert processor.prompt_path == prompt_path

        # Verify paths don't overlap
        assert learner.storage_path != collector.storage_path
        assert collector.storage_path != processor.prompt_path
