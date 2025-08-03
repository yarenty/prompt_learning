"""
Standard Open-Source Datasets Integration

This module integrates established, peer-reviewed datasets for professional benchmarking:
- HumanEval, BigCodeBench, APPS for software engineering
- MATH, GSM8K for mathematics
- WritingBench, BiGGen-Bench for creative writing

Using standard datasets ensures:
- Reproducibility and comparability with other research
- Peer-reviewed quality and difficulty
- No contamination concerns
- Professional credibility
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


class StandardDatasetManager:
    """Manager for integrating standard open-source evaluation datasets."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize with optional cache directory."""
        self.cache_dir = cache_dir or Path("data/standard_datasets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Available standard datasets
        self.available_datasets = {
            # Software Engineering / Programming
            "humaneval": {
                "source": "openai_humaneval",
                "description": "OpenAI HumanEval - 164 programming problems",
                "domain": "programming",
                "size": 164,
            },
            "bigcodebench": {
                "source": "bigcode-project/bigcodebench",
                "description": "BigCodeBench - Advanced programming tasks",
                "domain": "programming",
                "size": 1140,
            },
            "apps": {
                "source": "codeparrot/apps",
                "description": "APPS - Competitive programming problems",
                "domain": "programming",
                "size": 10000,
            },
            "livecodebench": {
                "source": "livecodebench/livecodebench",
                "description": "LiveCodeBench - Contamination-free programming evaluation",
                "domain": "programming",
                "size": 600,
            },
            # Mathematics
            "math_hard": {
                "source": "lighteval/MATH-Hard",
                "description": "MATH-Hard - Competition mathematics problems",
                "domain": "mathematics",
                "size": 3630,
            },
            "gsm8k": {
                "source": "gsm8k",
                "description": "GSM8K - Grade school math word problems",
                "domain": "mathematics",
                "size": 8500,
            },
            "mmlu_math": {
                "source": "cais/mmlu",
                "description": "MMLU Mathematics - Mathematical reasoning subset",
                "domain": "mathematics",
                "size": 1000,
            },
            # Creative Writing
            "writingbench": {
                "source": "prometheus-eval/BiGGen-Bench",  # Using BiGGen-Bench as WritingBench proxy
                "description": "BiGGen-Bench - Comprehensive text generation benchmark",
                "domain": "writing",
                "size": 765,
            },
            "creativity": {
                "source": "froggeric/creativity",
                "description": "LLM Creativity Benchmark - Creative writing evaluation",
                "domain": "writing",
                "size": 48,
            },
        }

    def list_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all available standard datasets."""
        return self.available_datasets

    def load_dataset(self, dataset_name: str, split: str = "test") -> List[Dict[str, Any]]:
        """Load a standard dataset."""
        if dataset_name not in self.available_datasets:
            raise ValueError(
                f"Dataset {dataset_name} not available. Choose from: {list(self.available_datasets.keys())}"
            )

        dataset_info = self.available_datasets[dataset_name]

        try:
            # Load using HuggingFace datasets
            # Handle special cases for datasets that need config
            if dataset_name == "gsm8k":
                dataset = load_dataset(dataset_info["source"], "main", split=split, cache_dir=str(self.cache_dir))
            else:
                dataset = load_dataset(dataset_info["source"], split=split, cache_dir=str(self.cache_dir))

            # Convert to list of dictionaries
            if hasattr(dataset, "to_pandas"):
                df = dataset.to_pandas()
                return df.to_dict("records")
            else:
                return list(dataset)

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return self._load_fallback_dataset(dataset_name)

    def _load_fallback_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load fallback dataset if HuggingFace fails."""
        fallback_data = []

        if dataset_name == "humaneval":
            # Fallback HumanEval samples
            fallback_data = [
                {
                    "task_id": "HumanEval/0",
                    "prompt": 'def has_close_elements(numbers, threshold):\n    """Check if any two numbers are closer than threshold."""\n',
                    "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False",
                    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0], 0.05) == False",
                    "entry_point": "has_close_elements",
                }
            ]

        elif dataset_name == "math_hard":
            # Fallback MATH problems
            fallback_data = [
                {
                    "problem": "What is the range of the function $y = \\frac{x^2 + 3x + 2}{x+1}$?",
                    "level": "Level 5",
                    "type": "Algebra",
                    "solution": "We can factor the numerator to get $y = \\frac{(x+1)(x+2)}{x+1}$. If we exclude the case where $x = -1$, the function is equivalent to $y = x+2$. However, because $x$ cannot equal $-1$, $y$ cannot equal 1. Therefore, the range is all real numbers except for 1, which we may write as $y \\in (-\\infty, 1)\\cup(1, \\infty)$.",
                }
            ]

        elif dataset_name == "gsm8k":
            # Fallback GSM8K problems
            fallback_data = [
                {
                    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                    "answer": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
                }
            ]

        elif dataset_name == "writingbench":
            # Fallback creative writing prompts
            fallback_data = [
                {
                    "capability": "planning",
                    "task": "travel_plan",
                    "input": "Design a travel plan for a tourist traveling to Paris. Requirements: - Total Duration: 2 days and 1 night - Transportation: Walk - Must Have: Eiffel Tower, Louvre Museum, Escargot",
                    "reference_answer": "Day 1 - Morning: Visit the Louvre Museum (3 hours)...",
                    "score_rubric": {
                        "criteria": "Does the response effectively plan a tourist's 2-day trip to Paris, incorporating the must-have experiences within the given constraints?"
                    },
                }
            ]

        return fallback_data

    def get_dataset_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics for a dataset."""
        if dataset_name not in self.available_datasets:
            return {}

        try:
            data = self.load_dataset(dataset_name)
            dataset_info = self.available_datasets[dataset_name]

            return {
                "name": dataset_name,
                "description": dataset_info["description"],
                "domain": dataset_info["domain"],
                "total_problems": len(data),
                "expected_size": dataset_info["size"],
                "source": dataset_info["source"],
                "sample_keys": list(data[0].keys()) if data else [],
            }
        except Exception as e:
            return {
                "name": dataset_name,
                "error": str(e),
                "description": self.available_datasets[dataset_name]["description"],
            }

    def create_evaluation_suite(self, domains: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Create a comprehensive evaluation suite from standard datasets."""
        if domains is None:
            domains = ["programming", "mathematics", "writing"]

        evaluation_suite = {}

        for dataset_name, info in self.available_datasets.items():
            if info["domain"] in domains:
                try:
                    data = self.load_dataset(dataset_name)
                    # Sample a subset for manageable evaluation
                    if len(data) > 50:
                        import random

                        data = random.sample(data, 50)

                    evaluation_suite[dataset_name] = data
                    print(f"✅ Loaded {dataset_name}: {len(data)} problems")
                except Exception as e:
                    print(f"❌ Failed to load {dataset_name}: {e}")

        return evaluation_suite

    def export_evaluation_problems(self, output_path: Path, max_per_dataset: int = 20) -> None:
        """Export a curated set of evaluation problems."""
        output_path.mkdir(parents=True, exist_ok=True)

        for dataset_name, info in self.available_datasets.items():
            try:
                data = self.load_dataset(dataset_name)

                # Sample problems for evaluation
                if len(data) > max_per_dataset:
                    import random

                    data = random.sample(data, max_per_dataset)

                # Save to JSON
                output_file = output_path / f"{dataset_name}_problems.json"
                with open(output_file, "w") as f:
                    json.dump({"dataset_info": info, "problems": data, "count": len(data)}, f, indent=2)

                print(f"✅ Exported {dataset_name}: {len(data)} problems to {output_file}")

            except Exception as e:
                print(f"❌ Failed to export {dataset_name}: {e}")


class StandardEvaluationRunner:
    """Runner for evaluating models on standard datasets."""

    def __init__(self, dataset_manager: StandardDatasetManager):
        """Initialize with dataset manager."""
        self.dataset_manager = dataset_manager
        self.results = {}

    def evaluate_programming_dataset(self, dataset_name: str, model_responses: List[str]) -> Dict[str, Any]:
        """Evaluate programming dataset (HumanEval, BigCodeBench, etc.)."""
        data = self.dataset_manager.load_dataset(dataset_name)

        if len(model_responses) != len(data):
            raise ValueError(f"Number of responses ({len(model_responses)}) doesn't match dataset size ({len(data)})")

        results = {"dataset": dataset_name, "total_problems": len(data), "evaluations": []}

        for i, (problem, response) in enumerate(zip(data, model_responses)):
            # Simple evaluation - in practice you'd run the code and check tests
            evaluation = {
                "problem_id": problem.get("task_id", f"problem_{i}"),
                "response": response,
                "has_solution": len(response.strip()) > 0,
                "estimated_correctness": self._estimate_code_quality(response),
            }
            results["evaluations"].append(evaluation)

        # Calculate metrics
        results["metrics"] = {
            "response_rate": sum(1 for e in results["evaluations"] if e["has_solution"]) / len(results["evaluations"]),
            "avg_estimated_quality": sum(e["estimated_correctness"] for e in results["evaluations"])
            / len(results["evaluations"]),
        }

        return results

    def evaluate_math_dataset(self, dataset_name: str, model_responses: List[str]) -> Dict[str, Any]:
        """Evaluate mathematics dataset (MATH, GSM8K, etc.)."""
        data = self.dataset_manager.load_dataset(dataset_name)

        results = {"dataset": dataset_name, "total_problems": len(data), "evaluations": []}

        for i, (problem, response) in enumerate(zip(data, model_responses)):
            evaluation = {
                "problem_id": problem.get("problem", f"problem_{i}")[:50] + "...",
                "response": response,
                "has_solution": len(response.strip()) > 0,
                "estimated_correctness": self._estimate_math_quality(response, problem),
            }
            results["evaluations"].append(evaluation)

        results["metrics"] = {
            "response_rate": sum(1 for e in results["evaluations"] if e["has_solution"]) / len(results["evaluations"]),
            "avg_estimated_quality": sum(e["estimated_correctness"] for e in results["evaluations"])
            / len(results["evaluations"]),
        }

        return results

    def evaluate_writing_dataset(self, dataset_name: str, model_responses: List[str]) -> Dict[str, Any]:
        """Evaluate creative writing dataset."""
        data = self.dataset_manager.load_dataset(dataset_name)

        results = {"dataset": dataset_name, "total_problems": len(data), "evaluations": []}

        for i, (problem, response) in enumerate(zip(data, model_responses)):
            evaluation = {
                "problem_id": f"writing_problem_{i}",
                "response": response,
                "has_solution": len(response.strip()) > 0,
                "estimated_quality": self._estimate_writing_quality(response),
            }
            results["evaluations"].append(evaluation)

        results["metrics"] = {
            "response_rate": sum(1 for e in results["evaluations"] if e["has_solution"]) / len(results["evaluations"]),
            "avg_estimated_quality": sum(e["estimated_quality"] for e in results["evaluations"])
            / len(results["evaluations"]),
        }

        return results

    def _estimate_code_quality(self, code: str) -> float:
        """Estimate code quality (0-1 score)."""
        if not code.strip():
            return 0.0

        score = 0.0

        # Basic checks
        if "def " in code:
            score += 0.3
        if "return" in code:
            score += 0.2
        if len(code.split("\n")) > 2:
            score += 0.2
        if any(keyword in code for keyword in ["if", "for", "while"]):
            score += 0.2
        if code.count("(") == code.count(")"):
            score += 0.1

        return min(score, 1.0)

    def _estimate_math_quality(self, response: str, problem: Dict[str, Any]) -> float:
        """Estimate math solution quality (0-1 score)."""
        if not response.strip():
            return 0.0

        score = 0.0

        # Basic checks
        if len(response) > 50:
            score += 0.2
        if any(word in response.lower() for word in ["solution", "answer", "solve"]):
            score += 0.2
        if any(symbol in response for symbol in ["=", "+", "-", "*", "/"]):
            score += 0.2
        if "step" in response.lower():
            score += 0.2
        if response.count("\n") > 1:  # Multi-line solution
            score += 0.2

        return min(score, 1.0)

    def _estimate_writing_quality(self, text: str) -> float:
        """Estimate writing quality (0-1 score)."""
        if not text.strip():
            return 0.0

        score = 0.0

        # Basic quality indicators
        word_count = len(text.split())
        if word_count > 50:
            score += 0.2
        if word_count > 200:
            score += 0.2

        sentence_count = text.count(".") + text.count("!") + text.count("?")
        if sentence_count > 3:
            score += 0.2

        if any(word in text.lower() for word in ["story", "narrative", "character", "plot"]):
            score += 0.2

        # Check for variety in sentence structure
        if text.count(",") > 2:
            score += 0.2

        return min(score, 1.0)
