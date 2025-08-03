"""Tests for task-specific metrics."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memento.benchmarking.evaluation.task_metrics import (
    MathematicsMetrics,
    ProgrammingMetrics,
    WritingMetrics,
)


class TestProgrammingMetrics:
    """Test programming-specific metrics."""

    @pytest.fixture
    def metrics(self):
        """Create programming metrics instance."""
        return ProgrammingMetrics()

    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.test_timeout == 5

    def test_calculate_pass_at_k_empty(self, metrics):
        """Test pass@k with empty test cases."""
        result = metrics.calculate_pass_at_k("print('hello')", [], k=1)
        assert result == 0.0

    def test_calculate_code_quality_valid_code(self, metrics):
        """Test code quality metrics with valid code."""
        code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

        quality = metrics.calculate_code_quality(code)

        assert isinstance(quality, dict)
        assert "complexity" in quality
        assert "maintainability" in quality
        assert "style_score" in quality
        assert "documentation_score" in quality
        assert "readability_score" in quality

        # All scores should be between 0 and 1
        for score in quality.values():
            assert 0 <= score <= 1

    def test_calculate_code_quality_invalid_syntax(self, metrics):
        """Test code quality with invalid syntax."""
        code = "def invalid_function("  # Syntax error

        quality = metrics.calculate_code_quality(code)

        # Should return zero scores for invalid syntax
        assert all(score == 0.0 for score in quality.values())

    def test_calculate_runtime_metrics_empty_inputs(self, metrics):
        """Test runtime metrics with empty inputs."""
        result = metrics.calculate_runtime_metrics("print('test')", [])

        assert result["average_time"] == 0.0
        assert result["memory_usage"] == 0.0

    def test_calculate_complexity(self, metrics):
        """Test complexity calculation."""
        import ast

        # Simple function
        simple_code = "def simple(): return 1"
        tree = ast.parse(simple_code)
        complexity = metrics._calculate_complexity(tree)
        assert 0 < complexity <= 1

        # Complex function with conditions
        complex_code = """
def complex_func(x):
    if x > 0:
        if x > 10:
            return x * 2
        else:
            return x
    elif x < 0:
        return -x
    else:
        return 0
"""
        tree = ast.parse(complex_code)
        complexity = metrics._calculate_complexity(tree)
        assert complexity > 0

    def test_calculate_style_score(self, metrics):
        """Test style score calculation."""
        # Good style
        good_code = '''
def calculate_sum(numbers):
    """Calculate sum of numbers."""
    return sum(numbers)
'''
        score = metrics._calculate_style_score(good_code)
        assert score > 0.5

        # Poor style
        poor_code = """
def x(a):return a+1 if a>0 else 0 if a==0 else -1 # very long line that exceeds reasonable length limits and should be penalized by the style checker
"""
        score = metrics._calculate_style_score(poor_code)
        assert score < 0.8  # Should be penalized

    def test_calculate_documentation_score(self, metrics):
        """Test documentation score calculation."""
        # Well documented
        documented_code = '''
def fibonacci(n):
    """Calculate fibonacci number.
    
    Args:
        n: The position in fibonacci sequence
        
    Returns:
        The fibonacci number at position n
    """
    # Base case
    if n <= 1:
        return n
    # Recursive case
    return fibonacci(n-1) + fibonacci(n-2)
'''
        score = metrics._calculate_documentation_score(documented_code)
        assert score > 0.3

        # No documentation
        undocumented_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        score = metrics._calculate_documentation_score(undocumented_code)
        assert score < 0.2


class TestMathematicsMetrics:
    """Test mathematics-specific metrics."""

    @pytest.fixture
    def metrics(self):
        """Create mathematics metrics instance."""
        return MathematicsMetrics()

    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics is not None

    def test_calculate_accuracy_exact_match(self, metrics):
        """Test accuracy with exact numerical match."""
        generated = "The answer is 42"
        reference = "#### 42"

        accuracy = metrics.calculate_accuracy(generated, reference)
        assert accuracy == 1.0

    def test_calculate_accuracy_close_match(self, metrics):
        """Test accuracy with close numerical match."""
        generated = "The result is 42.01"
        reference = "42"

        accuracy = metrics.calculate_accuracy(generated, reference)
        assert 0.5 <= accuracy < 1.0  # Should be good but not perfect

    def test_calculate_accuracy_no_numbers(self, metrics):
        """Test accuracy with no extractable numbers."""
        generated = "I don't know the answer"
        reference = "The answer is 42"

        accuracy = metrics.calculate_accuracy(generated, reference)
        assert accuracy == 0.0

    def test_extract_numerical_answer(self, metrics):
        """Test numerical answer extraction."""
        # Test various formats
        test_cases = [
            ("#### 42", 42.0),
            ("The answer is 3.14", 3.14),
            ("x = 100", 100.0),
            ("$25.50$", 25.50),
            ("No numbers here", None),
            ("Multiple 1 numbers 2 here 3", 3.0),  # Should take last
        ]

        for text, expected in test_cases:
            result = metrics._extract_numerical_answer(text)
            if expected is None:
                assert result is None
            else:
                assert abs(result - expected) < 1e-6

    def test_evaluate_reasoning(self, metrics):
        """Test reasoning evaluation."""
        solution = """
First, we need to find the derivative.
Then, we set it equal to zero.
Finally, we solve for x.
Therefore, the answer is x = 5.
"""

        reasoning = metrics.evaluate_reasoning(solution)

        assert isinstance(reasoning, dict)
        assert "step_count" in reasoning
        assert "clarity" in reasoning
        assert "logical_flow" in reasoning
        assert "completeness" in reasoning

        # All scores should be between 0 and 1
        for score in reasoning.values():
            assert 0 <= score <= 1

    def test_analyze_complexity(self, metrics):
        """Test complexity analysis."""
        solution = """
Using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a
We have a=1, b=-5, c=6
Therefore: x = (5 ± √(25-24)) / 2 = (5 ± 1) / 2
So x = 3 or x = 2
"""

        complexity = metrics.analyze_complexity(solution)

        assert isinstance(complexity, dict)
        assert "operation_count" in complexity
        assert "concept_diversity" in complexity
        assert "formula_complexity" in complexity
        assert "proof_structure" in complexity

    def test_extract_solution_steps(self, metrics):
        """Test solution step extraction."""
        solution = """
1. First, we identify the variables.
2. Then, we substitute the values.
3. Finally, we solve the equation.
"""

        steps = metrics._extract_solution_steps(solution)
        assert len(steps) >= 3
        assert any("identify" in step.lower() for step in steps)
        assert any("substitute" in step.lower() for step in steps)
        assert any("solve" in step.lower() for step in steps)

    def test_evaluate_step_clarity(self, metrics):
        """Test step clarity evaluation."""
        clear_steps = [
            "First, we substitute x = 2 into the equation y = 3x + 1",
            "Therefore, y = 3(2) + 1 = 7",
            "Hence, the point (2, 7) lies on the line",
        ]

        unclear_steps = ["Do this", "Then that", "Get answer"]

        clear_score = metrics._evaluate_step_clarity(clear_steps)
        unclear_score = metrics._evaluate_step_clarity(unclear_steps)

        assert clear_score > unclear_score
        assert 0 <= clear_score <= 1
        assert 0 <= unclear_score <= 1


class TestWritingMetrics:
    """Test writing-specific metrics."""

    @pytest.fixture
    def metrics(self):
        """Create writing metrics instance."""
        return WritingMetrics()

    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.rouge_scorer is not None

    def test_calculate_rouge(self, metrics):
        """Test ROUGE score calculation."""
        generated = "The quick brown fox jumps over the lazy dog"
        reference = "A quick brown fox jumps over a lazy dog"

        rouge_scores = metrics.calculate_rouge(generated, reference)

        assert isinstance(rouge_scores, dict)
        expected_keys = [
            "rouge1_precision",
            "rouge1_recall",
            "rouge1_fmeasure",
            "rouge2_precision",
            "rouge2_recall",
            "rouge2_fmeasure",
            "rougeL_precision",
            "rougeL_recall",
            "rougeL_fmeasure",
        ]

        for key in expected_keys:
            assert key in rouge_scores
            assert 0 <= rouge_scores[key] <= 1

    def test_evaluate_coherence(self, metrics):
        """Test coherence evaluation."""
        coherent_text = """
The weather today is sunny and warm. This makes it perfect for outdoor activities.
Many people will likely go to the park. The park offers various recreational facilities.
"""

        incoherent_text = """
The weather is sunny. Cats like fish. Mathematics is difficult. 
My favorite color is blue. The economy is complex.
"""

        coherent_scores = metrics.evaluate_coherence(coherent_text)
        incoherent_scores = metrics.evaluate_coherence(incoherent_text)

        # Coherent text should score higher
        assert coherent_scores["local_coherence"] > incoherent_scores["local_coherence"]
        assert coherent_scores["topic_consistency"] > incoherent_scores["topic_consistency"]

        # All scores should be between 0 and 1
        for scores in [coherent_scores, incoherent_scores]:
            for score in scores.values():
                assert 0 <= score <= 1

    def test_analyze_style(self, metrics):
        """Test style analysis."""
        formal_text = """
The analysis demonstrates significant improvements in performance metrics.
Furthermore, the statistical evaluation confirms the hypothesis.
Therefore, we conclude that the proposed method is effective.
"""

        informal_text = """
This stuff is really cool! The results are pretty awesome.
I think it's gonna work great. Super excited about this!
"""

        formal_style = metrics.analyze_style(formal_text)
        informal_style = metrics.analyze_style(informal_text)

        # Formal text should have higher formality score
        assert formal_style["formality"] > informal_style["formality"]

        # All scores should be between 0 and 1
        for style in [formal_style, informal_style]:
            for score in style.values():
                assert 0 <= score <= 1

    def test_split_into_sentences(self, metrics):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = metrics._split_into_sentences(text)

        assert len(sentences) == 4
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
        assert "Fourth sentence" in sentences[3]

    def test_calculate_readability(self, metrics):
        """Test readability calculation."""
        simple_text = "The cat sat on the mat. It was a nice day."
        complex_text = """
The implementation of sophisticated algorithmic methodologies necessitates
comprehensive understanding of computational complexity theoretical frameworks.
"""

        simple_score = metrics._calculate_readability(simple_text)
        complex_score = metrics._calculate_readability(complex_text)

        # Simple text should be more readable
        assert simple_score > complex_score
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1

    def test_calculate_lexical_diversity(self, metrics):
        """Test lexical diversity calculation."""
        repetitive_text = "The cat cat cat sat sat sat on on on the the the mat mat mat"
        diverse_text = "The feline creature gracefully positioned itself upon the woven surface"

        repetitive_score = metrics._calculate_lexical_diversity(repetitive_text)
        diverse_score = metrics._calculate_lexical_diversity(diverse_text)

        # Diverse text should have higher lexical diversity
        assert diverse_score > repetitive_score
        assert 0 <= repetitive_score <= 1
        assert 0 <= diverse_score <= 1

    def test_calculate_formality(self, metrics):
        """Test formality calculation."""
        formal_text = "The analysis demonstrates significant improvements in performance."
        informal_text = "This stuff is really cool and pretty awesome!"

        formal_score = metrics._calculate_formality(formal_text)
        informal_score = metrics._calculate_formality(informal_text)

        # Formal text should have higher formality score
        assert formal_score > informal_score

    def test_calculate_syntactic_complexity(self, metrics):
        """Test syntactic complexity calculation."""
        simple_text = "The cat sat. The dog ran."
        complex_text = """
        The cat, which was black and had been sitting quietly, 
        suddenly jumped when the dog that was running quickly 
        approached the area where it had been resting.
        """

        simple_score = metrics._calculate_syntactic_complexity(simple_text)
        complex_score = metrics._calculate_syntactic_complexity(complex_text)

        # Complex text should have higher complexity score
        assert complex_score > simple_score
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1

    def test_calculate_fluency(self, metrics):
        """Test fluency calculation."""
        fluent_text = "The weather today is beautiful and perfect for outdoor activities."
        disfluent_text = "The the weather weather today today is is beautiful beautiful and and"

        fluent_score = metrics._calculate_fluency(fluent_text)
        disfluent_score = metrics._calculate_fluency(disfluent_text)

        # Fluent text should have higher fluency score
        assert fluent_score > disfluent_score
        assert 0 <= fluent_score <= 1
        assert 0 <= disfluent_score <= 1


class TestIntegration:
    """Integration tests for all metrics."""

    def test_all_metrics_initialization(self):
        """Test that all metric classes can be initialized."""
        prog_metrics = ProgrammingMetrics()
        math_metrics = MathematicsMetrics()
        writing_metrics = WritingMetrics()

        assert prog_metrics is not None
        assert math_metrics is not None
        assert writing_metrics is not None

    def test_metrics_return_valid_ranges(self):
        """Test that all metrics return values in valid ranges."""
        prog_metrics = ProgrammingMetrics()
        math_metrics = MathematicsMetrics()
        writing_metrics = WritingMetrics()

        # Test programming metrics
        code = "def test(): return 1"
        quality = prog_metrics.calculate_code_quality(code)
        for score in quality.values():
            assert 0 <= score <= 1

        # Test math metrics
        solution = "The answer is 42"
        accuracy = math_metrics.calculate_accuracy(solution, "42")
        assert 0 <= accuracy <= 1

        # Test writing metrics
        text = "This is a test sentence."
        style = writing_metrics.analyze_style(text)
        for score in style.values():
            assert 0 <= score <= 1
