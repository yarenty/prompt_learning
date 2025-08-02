"""
Tests for feedback loop functionality.
"""

import json

import pytest

from feedback_loop.collector import FeedbackCollector


@pytest.fixture
def temp_feedback_dir(tmp_path):
    """Create a temporary directory for feedback storage."""
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    yield feedback_dir
    # shutil.rmtree(feedback_dir) # This line was removed as per the edit hint


@pytest.fixture
def collector(temp_feedback_dir):
    """Create a FeedbackCollector instance with temporary storage."""
    return FeedbackCollector(storage_path=str(temp_feedback_dir))


def test_collect_solution_feedback(collector, temp_feedback_dir):
    """Test collecting feedback for a solution."""
    problem = "Write a function to find the maximum element in a list"
    solution = """
    def find_max(lst):
        if not lst:
            return None
        return max(lst)
    """
    criteria = ["correctness", "efficiency", "readability"]

    feedback = collector.collect_solution_feedback(problem, solution, criteria)

    # Check feedback structure
    assert "timestamp" in feedback
    assert "problem" in feedback
    assert "solution" in feedback
    assert "evaluation" in feedback
    assert "reflection" in feedback

    # Check evaluation structure
    assert all(criterion in feedback["evaluation"] for criterion in criteria)

    # Check file storage
    feedback_files = list(temp_feedback_dir.glob("feedback_*.json"))
    assert len(feedback_files) == 1

    # Verify stored content
    with open(feedback_files[0]) as f:
        stored_feedback = json.load(f)
        assert stored_feedback == feedback


def test_evaluate_solution(collector):
    """Test solution evaluation."""
    solution = "def example(): pass"
    criteria = ["correctness", "efficiency"]

    evaluation = collector._evaluate_solution(solution, criteria)

    assert isinstance(evaluation, dict)
    assert all(criterion in evaluation for criterion in criteria)
    assert all(isinstance(score, float) for score in evaluation.values())


def test_generate_reflection(collector):
    """Test reflection generation."""
    problem = "Test problem"
    solution = "Test solution"

    reflection = collector._generate_reflection(problem, solution)

    assert isinstance(reflection, str)
    assert len(reflection) > 0
