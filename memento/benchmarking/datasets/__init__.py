"""
Dataset integrations for benchmarking.

This module provides integrations with popular coding and reasoning datasets.
"""

from typing import Any, Dict, List


class HumanEvalDataset:
    """HumanEval dataset integration."""

    def __init__(self):
        self.problems = self._load_sample_problems()

    def _load_sample_problems(self) -> List[Dict[str, Any]]:
        """Load sample problems (placeholder for real HumanEval integration)."""
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": (
                    "def has_close_elements(numbers, threshold):\n"
                    '    """Check if any two numbers are closer than threshold."""\n'
                ),
                "canonical_solution": (
                    "    for idx, elem in enumerate(numbers):\n"
                    "        for idx2, elem2 in enumerate(numbers):\n"
                    "            if idx != idx2:\n"
                    "                distance = abs(elem - elem2)\n"
                    "                if distance < threshold:\n"
                    "                    return True\n"
                    "    return False"
                ),
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
                    "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False"
                ),
                "entry_point": "has_close_elements",
            }
        ]

    def get_problems(self) -> List[Dict[str, Any]]:
        """Get all problems."""
        return self.problems


class APPSDataset:
    """APPS dataset integration."""

    def __init__(self):
        self.problems = self._load_sample_problems()

    def _load_sample_problems(self) -> List[Dict[str, Any]]:
        """Load sample problems (placeholder for real APPS integration)."""
        return [
            {
                "problem_id": "apps_1",
                "description": "Given an array of integers, return the sum of all even numbers.",
                "difficulty": "introductory",
                "solutions": ["def sum_evens(arr): return sum(x for x in arr if x % 2 == 0)"],
                "input_output": {"input": "[1, 2, 3, 4, 5, 6]", "output": "12"},
            }
        ]

    def get_problems(self) -> List[Dict[str, Any]]:
        """Get all problems."""
        return self.problems


class CodeContestsDataset:
    """CodeContests dataset integration."""

    def __init__(self):
        self.problems = self._load_sample_problems()

    def _load_sample_problems(self) -> List[Dict[str, Any]]:
        """Load sample problems (placeholder for real CodeContests integration)."""
        return [
            {
                "name": "A+B Problem",
                "description": "Calculate the sum of two integers A and B.",
                "input_format": "Two integers A and B",
                "output_format": "Single integer representing A + B",
                "sample_input": "3 5",
                "sample_output": "8",
                "difficulty": 800,
            }
        ]

    def get_problems(self) -> List[Dict[str, Any]]:
        """Get all problems."""
        return self.problems


__all__ = ["HumanEvalDataset", "APPSDataset", "CodeContestsDataset"]
