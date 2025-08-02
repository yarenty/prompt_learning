"""
Dataset Manager

Centralized management of all Memento evaluation datasets.
Provides unified access to software engineering, mathematics, and creative writing problems.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .creative_writing import CreativeWritingDataset, CreativeWritingProblem
from .mathematics import MathematicsDataset, MathematicsProblem
from .software_engineering import SoftwareEngineeringDataset, SoftwareEngineeringProblem

# Type alias for any problem type
Problem = Union[SoftwareEngineeringProblem, MathematicsProblem, CreativeWritingProblem]


class DatasetManager:
    """Centralized manager for all evaluation datasets."""

    def __init__(self, base_storage_path: Optional[Path] = None):
        """Initialize dataset manager."""
        self.base_storage_path = base_storage_path or Path("data")
        self.base_storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize individual datasets
        self.software_engineering = SoftwareEngineeringDataset(
            storage_path=self.base_storage_path / "software_engineering"
        )
        self.mathematics = MathematicsDataset(storage_path=self.base_storage_path / "mathematics")
        self.creative_writing = CreativeWritingDataset(storage_path=self.base_storage_path / "creative_writing")

        self.datasets = {
            "software_engineering": self.software_engineering,
            "mathematics": self.mathematics,
            "creative_writing": self.creative_writing,
        }

    def get_dataset(self, domain: str) -> Union[SoftwareEngineeringDataset, MathematicsDataset, CreativeWritingDataset]:
        """Get dataset by domain name."""
        if domain not in self.datasets:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.datasets.keys())}")
        return self.datasets[domain]

    def get_all_problems(self, domain: Optional[str] = None) -> List[Problem]:
        """Get all problems from specified domain or all domains."""
        if domain:
            return self.get_dataset(domain).problems

        all_problems = []
        for dataset in self.datasets.values():
            all_problems.extend(dataset.problems)
        return all_problems

    def get_problems_by_difficulty(self, difficulty: str, domain: Optional[str] = None) -> List[Problem]:
        """Get problems filtered by difficulty across domains."""
        if domain:
            return self.get_dataset(domain).get_problems_by_difficulty(difficulty)

        problems = []
        for dataset in self.datasets.values():
            problems.extend(dataset.get_problems_by_difficulty(difficulty))
        return problems

    def get_random_problems(
        self, count: int, domain: Optional[str] = None, difficulty: Optional[str] = None, category: Optional[str] = None
    ) -> List[Problem]:
        """Get random sample of problems with optional filters."""
        import random

        # Get base set of problems
        if domain:
            available = self.get_dataset(domain).problems
        else:
            available = self.get_all_problems()

        # Apply filters
        if difficulty:
            available = [p for p in available if p.difficulty == difficulty]

        if category:
            # Handle category filtering for different problem types
            filtered = []
            for p in available:
                if hasattr(p, "category") and p.category == category:
                    filtered.append(p)
                elif hasattr(p, "domain") and p.domain == category:
                    filtered.append(p)
            available = filtered

        return random.sample(available, min(count, len(available)))

    def create_balanced_sample(
        self, problems_per_domain: int = 10, difficulty_distribution: Optional[Dict[str, float]] = None
    ) -> List[Problem]:
        """Create a balanced sample across all domains and difficulties."""
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

        balanced_sample = []

        for domain_name, dataset in self.datasets.items():
            domain_problems = []

            for difficulty, ratio in difficulty_distribution.items():
                count = int(problems_per_domain * ratio)
                if count > 0:
                    difficulty_problems = dataset.get_problems_by_difficulty(difficulty)
                    if difficulty_problems:
                        import random

                        selected = random.sample(difficulty_problems, min(count, len(difficulty_problems)))
                        domain_problems.extend(selected)

            balanced_sample.extend(domain_problems)

        return balanced_sample

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all datasets."""
        total_stats = {
            "total_problems": 0,
            "domains": {},
            "difficulties": {"easy": 0, "medium": 0, "hard": 0},
            "categories": {},
            "average_time": 0,
        }

        total_time = 0

        for domain_name, dataset in self.datasets.items():
            stats = dataset.get_statistics()

            # Aggregate totals
            total_stats["total_problems"] += stats["total_problems"]
            total_stats["domains"][domain_name] = stats["total_problems"]
            total_time += stats["average_time"] * stats["total_problems"]

            # Aggregate difficulties
            if "difficulties" in stats:
                for diff, count in stats["difficulties"].items():
                    total_stats["difficulties"][diff] += count

            # Aggregate categories/domains
            if "categories" in stats:
                for cat, count in stats["categories"].items():
                    key = f"{domain_name}_{cat}"
                    total_stats["categories"][key] = count
            elif "domains" in stats:
                for dom, count in stats["domains"].items():
                    key = f"{domain_name}_{dom}"
                    total_stats["categories"][key] = count

        # Calculate overall average time
        if total_stats["total_problems"] > 0:
            total_stats["average_time"] = total_time / total_stats["total_problems"]

        return total_stats

    def export_dataset_summary(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export comprehensive dataset summary."""
        if output_path is None:
            output_path = self.base_storage_path / "dataset_summary.json"

        summary = {
            "metadata": {
                "total_datasets": len(self.datasets),
                "domains": list(self.datasets.keys()),
                "version": "1.0.0",
                "description": "Comprehensive evaluation dataset for prompt evolution methods",
            },
            "statistics": self.get_comprehensive_statistics(),
            "domain_details": {},
        }

        # Add detailed statistics for each domain
        for domain_name, dataset in self.datasets.items():
            summary["domain_details"][domain_name] = dataset.get_statistics()

        # Save to file
        try:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Error saving dataset summary: {e}")

        return summary

    def validate_datasets(self) -> Dict[str, Any]:
        """Validate all datasets for completeness and consistency."""
        validation_results = {"valid": True, "issues": [], "domain_validation": {}}

        for domain_name, dataset in self.datasets.items():
            domain_issues = []

            # Check if dataset has problems
            if not dataset.problems:
                domain_issues.append("No problems found")

            # Check for expected number of problems (150 per domain)
            expected_count = 150
            if len(dataset.problems) != expected_count:
                domain_issues.append(f"Expected {expected_count} problems, found {len(dataset.problems)}")

            # Check for required fields in problems
            for i, problem in enumerate(dataset.problems[:5]):  # Check first 5 problems
                if not problem.id:
                    domain_issues.append(f"Problem {i} missing ID")
                if not problem.title:
                    domain_issues.append(f"Problem {i} missing title")
                if hasattr(problem, "difficulty") and problem.difficulty not in ["easy", "medium", "hard"]:
                    domain_issues.append(f"Problem {i} has invalid difficulty: {problem.difficulty}")

            validation_results["domain_validation"][domain_name] = {
                "valid": len(domain_issues) == 0,
                "issues": domain_issues,
                "problem_count": len(dataset.problems),
            }

            if domain_issues:
                validation_results["valid"] = False
                validation_results["issues"].extend([f"{domain_name}: {issue}" for issue in domain_issues])

        return validation_results

    def get_problem_by_id(self, problem_id: str) -> Optional[Problem]:
        """Find a problem by its ID across all datasets."""
        for dataset in self.datasets.values():
            for problem in dataset.problems:
                if problem.id == problem_id:
                    return problem
        return None

    def search_problems(self, query: str, domain: Optional[str] = None) -> List[Problem]:
        """Search for problems containing the query string."""
        results = []

        datasets_to_search = [self.get_dataset(domain)] if domain else self.datasets.values()

        for dataset in datasets_to_search:
            for problem in dataset.problems:
                # Search in title and description/prompt/statement
                searchable_text = problem.title.lower()

                if hasattr(problem, "description"):
                    searchable_text += " " + problem.description.lower()
                elif hasattr(problem, "prompt"):
                    searchable_text += " " + problem.prompt.lower()
                elif hasattr(problem, "statement"):
                    searchable_text += " " + problem.statement.lower()

                if query.lower() in searchable_text:
                    results.append(problem)

        return results
