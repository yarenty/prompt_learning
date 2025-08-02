"""
Tests for the enhanced PromptLearner with async support and error handling.
"""

from unittest.mock import Mock, patch

import pytest

from memento.config import ModelConfig, ModelType
from memento.core import EvaluationError, PromptLearner, ValidationError


class TestPromptLearner:
    """Test cases for the enhanced PromptLearner."""

    @pytest.fixture
    def model_config(self):
        """Create a test model configuration."""
        return ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.7, max_tokens=2048)

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create a temporary storage path."""
        return tmp_path / "test_evolution"

    @pytest.fixture
    def learner(self, model_config, storage_path):
        """Create a test learner instance."""
        return PromptLearner(
            model_config=model_config, storage_path=storage_path, enable_metrics=False  # Disable metrics for testing
        )

    def test_initialization(self, learner, model_config, storage_path):
        """Test learner initialization."""
        assert learner.model_config == model_config
        assert learner.storage_path == storage_path
        assert learner.evolution_history == []
        assert learner.monitor is None  # Metrics disabled

    def test_initialization_with_metrics(self, model_config, storage_path):
        """Test learner initialization with metrics enabled."""
        learner = PromptLearner(model_config=model_config, storage_path=storage_path, enable_metrics=True)
        assert learner.monitor is not None

    def test_validate_model_config_invalid_type(self, storage_path):
        """Test validation with invalid model type."""
        # This test is now handled by Pydantic's built-in validation
        # The ModelConfig will raise a ValidationError before our custom validation
        with pytest.raises(Exception):  # Pydantic validation error
            ModelConfig(model_type="invalid_type", model_name="test")  # type: ignore

    def test_validate_model_config_empty_name(self, storage_path):
        """Test validation with empty model name."""
        # This test is now handled by Pydantic's built-in validation
        with pytest.raises(Exception):  # Pydantic validation error
            ModelConfig(model_type=ModelType.OLLAMA, model_name="")

    @pytest.mark.asyncio
    async def test_evaluate_prompt_performance_validation(self, learner):
        """Test input validation in evaluate_prompt_performance."""
        # Test with invalid prompt
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            await learner.evaluate_prompt_performance(
                prompt="", problem={"description": "test", "solution": "test"}, evaluation_criteria=["correctness"]
            )

        # Test with invalid problem
        with pytest.raises(ValidationError, match="Problem missing required field"):
            await learner.evaluate_prompt_performance(
                prompt="You are a programmer",
                problem={"description": "test"},  # Missing solution
                evaluation_criteria=["correctness"],
            )

        # Test with invalid criteria
        with pytest.raises(ValidationError, match="Evaluation criteria cannot be empty"):
            await learner.evaluate_prompt_performance(
                prompt="You are a programmer",
                problem={"description": "test", "solution": "test"},
                evaluation_criteria=[],
            )

    @pytest.mark.asyncio
    async def test_evolve_prompt_validation(self, learner):
        """Test input validation in evolve_prompt."""
        # Test with invalid prompt
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            await learner.evolve_prompt(current_prompt="", lessons=[{"criterion": "test", "lesson": "test"}])

        # Test with invalid lessons
        with pytest.raises(ValidationError, match="Lesson missing required field"):
            await learner.evolve_prompt(
                current_prompt="You are a programmer", lessons=[{"criterion": "test"}]
            )  # Missing lesson

    @pytest.mark.asyncio
    async def test_save_evolution_step_validation(self, learner):
        """Test input validation in save_evolution_step."""
        # Test with invalid prompt type
        with pytest.raises(ValidationError, match="Prompt type must be a non-empty string"):
            await learner.save_evolution_step(
                prompt_type="",
                current_prompt="You are a programmer",
                updated_prompt="You are a better programmer",
                evaluation_results=[],
            )

        # Test with invalid prompts
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            await learner.save_evolution_step(
                prompt_type="test",
                current_prompt="",
                updated_prompt="You are a better programmer",
                evaluation_results=[],
            )

    def test_create_evaluation_prompt(self, learner):
        """Test evaluation prompt creation."""
        prompt = "You are a programmer"
        problem = {"description": "Write a function", "solution": "def func(): pass"}
        criteria = ["correctness", "efficiency"]

        result = learner._create_evaluation_prompt(prompt, problem, criteria)

        assert "You are evaluating a system prompt's performance" in result
        assert prompt in result
        assert problem["description"] in result
        assert problem["solution"] in result
        assert "correctness, efficiency" in result
        assert "JSON format" in result

    def test_create_evolution_prompt(self, learner):
        """Test evolution prompt creation."""
        current_prompt = "You are a programmer"
        lessons = [
            {"criterion": "correctness", "lesson": "Always validate inputs"},
            {"criterion": "efficiency", "lesson": "Use appropriate data structures"},
        ]

        result = learner._create_evolution_prompt(current_prompt, lessons)

        assert "You are evolving a system prompt" in result
        assert current_prompt in result
        assert "correctness: Always validate inputs" in result
        assert "efficiency: Use appropriate data structures" in result

    def test_extract_lessons(self, learner):
        """Test lesson extraction from evaluation."""
        evaluation = {
            "correctness": {"score": 0.8, "explanation": "Good logic", "lesson": "Always validate inputs"},
            "efficiency": {"score": 0.9, "explanation": "Fast algorithm", "lesson": "Use appropriate data structures"},
        }

        lessons = learner._extract_lessons(evaluation)

        assert len(lessons) == 2
        assert lessons[0]["criterion"] == "correctness"
        assert lessons[0]["lesson"] == "Always validate inputs"
        assert lessons[1]["criterion"] == "efficiency"
        assert lessons[1]["lesson"] == "Use appropriate data structures"

    def test_calculate_problem_complexity(self, learner):
        """Test problem complexity calculation."""
        simple_problem = {"description": "Simple task", "solution": "Simple solution"}

        complex_problem = {"description": "A" * 500, "solution": "B" * 500}  # Long description  # Long solution

        simple_complexity = learner._calculate_problem_complexity(simple_problem)
        complex_complexity = learner._calculate_problem_complexity(complex_problem)

        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0
        assert complex_complexity > simple_complexity

    def test_parse_evaluation_response(self, learner):
        """Test evaluation response parsing."""
        response = '{"correctness": {"score": 0.8, "explanation": "Good", "lesson": "Test"}}'
        criteria = ["correctness", "efficiency"]

        result = learner._parse_evaluation_response(response, criteria)

        assert "correctness" in result
        assert result["correctness"]["score"] == 0.8
        assert "efficiency" in result  # Should be added with defaults
        assert result["efficiency"]["score"] == 0.0

    def test_parse_evaluation_response_invalid_json(self, learner):
        """Test evaluation response parsing with invalid JSON."""
        response = "invalid json"
        criteria = ["correctness"]

        with pytest.raises(EvaluationError, match="Invalid evaluation response"):
            learner._parse_evaluation_response(response, criteria)

    def test_get_performance_report_disabled(self, learner):
        """Test performance report when metrics are disabled."""
        report = learner.get_performance_report()
        assert report["message"] == "Performance monitoring is disabled"

    def test_get_performance_report_enabled(self, model_config, storage_path):
        """Test performance report when metrics are enabled."""
        learner = PromptLearner(model_config=model_config, storage_path=storage_path, enable_metrics=True)

        report = learner.get_performance_report()
        # When no metrics have been collected, it should return a message
        assert "message" in report or "generated_at" in report


class TestPromptLearnerIntegration:
    """Integration tests for PromptLearner with mocked LLM."""

    @pytest.fixture
    def model_config(self):
        """Create a test model configuration."""
        return ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.7, max_tokens=2048)

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create a temporary storage path."""
        return tmp_path / "test_evolution"

    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama response."""
        mock_response = Mock()
        mock_response.response = """
        {
            "correctness": {
                "score": 0.85,
                "explanation": "Good logic and error handling",
                "lesson": "Always validate inputs and handle edge cases"
            },
            "efficiency": {
                "score": 0.9,
                "explanation": "Optimal algorithm used",
                "lesson": "Choose appropriate data structures for the problem"
            }
        }
        """
        return mock_response

    @pytest.mark.asyncio
    async def test_full_evaluation_workflow(self, model_config, storage_path, mock_ollama_response):
        """Test the complete evaluation workflow."""
        with patch("ollama.generate", return_value=mock_ollama_response):
            learner = PromptLearner(model_config=model_config, storage_path=storage_path, enable_metrics=False)

            result = await learner.evaluate_prompt_performance(
                prompt="You are a Python programmer",
                problem={
                    "description": "Write a function to find the maximum element in a list",
                    "solution": "def find_max(lst): return max(lst) if lst else None",
                },
                evaluation_criteria=["correctness", "efficiency"],
            )

            assert "timestamp" in result
            assert "prompt" in result
            assert "problem" in result
            assert "evaluation" in result
            assert "lessons" in result

            evaluation = result["evaluation"]
            assert "correctness" in evaluation
            assert "efficiency" in evaluation
            assert evaluation["correctness"]["score"] == 0.85
            assert evaluation["efficiency"]["score"] == 0.9

            lessons = result["lessons"]
            assert len(lessons) == 2
            assert any("validate inputs" in lesson["lesson"] for lesson in lessons)
            assert any("data structures" in lesson["lesson"] for lesson in lessons)

    @pytest.mark.asyncio
    async def test_full_evolution_workflow(self, model_config, storage_path):
        """Test the complete evolution workflow."""
        mock_evolution_response = Mock()
        mock_evolution_response.response = (
            "You are an enhanced Python programmer who always validates inputs and uses appropriate data structures."
        )

        with patch("ollama.generate", return_value=mock_evolution_response):
            learner = PromptLearner(model_config=model_config, storage_path=storage_path, enable_metrics=False)

            lessons = [
                {
                    "criterion": "correctness",
                    "score": 0.8,
                    "explanation": "Good logic",
                    "lesson": "Always validate inputs",
                },
                {
                    "criterion": "efficiency",
                    "score": 0.9,
                    "explanation": "Fast algorithm",
                    "lesson": "Use appropriate data structures",
                },
            ]

            evolved_prompt = await learner.evolve_prompt(current_prompt="You are a Python programmer", lessons=lessons)

            assert (
                evolved_prompt
                == "You are an enhanced Python programmer who always validates inputs and uses appropriate data structures."
            )

    @pytest.mark.asyncio
    async def test_save_evolution_step(self, model_config, storage_path):
        """Test saving evolution step."""
        learner = PromptLearner(model_config=model_config, storage_path=storage_path, enable_metrics=False)

        await learner.save_evolution_step(
            prompt_type="python_programmer",
            current_prompt="You are a Python programmer",
            updated_prompt="You are an enhanced Python programmer",
            evaluation_results=[{"score": 0.8}],
        )

        # Check that step was added to history
        assert len(learner.evolution_history) == 1
        step = learner.evolution_history[0]
        assert step["prompt_type"] == "python_programmer"
        assert step["current_prompt"] == "You are a Python programmer"
        assert step["updated_prompt"] == "You are an enhanced Python programmer"

        # Check that file was created
        evolution_files = list(storage_path.glob("evolution_python_programmer_*.json"))
        assert len(evolution_files) == 1
