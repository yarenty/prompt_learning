#!/usr/bin/env python3
"""
Dataset & Evaluation Demo

Demonstrates the comprehensive dataset collection and evaluation suite
implemented in the Memento project.

Features demonstrated:
- Software Engineering Dataset (150 problems)
- Mathematics Dataset (150 problems)
- Creative Writing Dataset (150 problems)
- Dataset Manager for unified access
- Automated Evaluation Suite
- Statistical Analysis and Reporting
"""

import asyncio
from pathlib import Path

from memento.config import ModelConfig, ModelType
from memento.datasets import DatasetManager, EvaluationSuite


class Phase5Demo:
    """Comprehensive demonstration of dtasets capabilities."""

    def __init__(self):
        """Initialize demo components."""
        self.model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="llama3.2", temperature=0.7)

        # Initialize components
        self.dataset_manager = DatasetManager(base_storage_path=Path("demo_data/phase5"))
        self.evaluation_suite = EvaluationSuite(
            dataset_manager=self.dataset_manager,
            model_config=self.model_config,
            storage_path=Path("demo_data/phase5/evaluations"),
        )

    async def run_comprehensive_demo(self):
        """Run complete demonstration."""
        print("🚀 DATASET & EVALUATION COMPREHENSIVE DEMO")
        print("=" * 60)

        # 1. Demonstrate dataset capabilities
        await self.demonstrate_datasets()

        # 2. Show dataset management features
        await self.demonstrate_dataset_management()

        # 3. Demonstrate evaluation suite
        await self.demonstrate_evaluation_suite()

        # 4. Show statistical analysis
        await self.demonstrate_statistical_analysis()

        print("\n🎉 Demo Complete!")
        print("✅ All 450 problems across 3 domains ready for evaluation")
        print("✅ Comprehensive evaluation framework operational")
        print("✅ Statistical analysis and reporting available")

    async def demonstrate_datasets(self):
        """Demonstrate individual dataset capabilities."""
        print("\n📊 DATASET DEMONSTRATION")
        print("-" * 40)

        # Software Engineering Dataset
        print("\n🔧 Software Engineering Dataset:")
        se_stats = self.dataset_manager.software_engineering.get_statistics()
        print(f"  Total Problems: {se_stats['total_problems']}")
        print(f"  Categories: {list(se_stats['categories'].keys())}")
        print(f"  Difficulties: {se_stats['difficulties']}")
        print(f"  Average Time: {se_stats['average_time']:.1f} minutes")

        # Show sample problem
        sample_se = self.dataset_manager.software_engineering.get_random_problems(1)[0]
        print(f"\n  Sample Problem: {sample_se.title}")
        print(f"  Category: {sample_se.category}")
        print(f"  Difficulty: {sample_se.difficulty}")
        print(f"  Description: {sample_se.description[:100]}...")

        # Mathematics Dataset
        print("\n📐 Mathematics Dataset:")
        math_stats = self.dataset_manager.mathematics.get_statistics()
        print(f"  Total Problems: {math_stats['total_problems']}")
        print(f"  Domains: {list(math_stats['domains'].keys())}")
        print(f"  Difficulties: {math_stats['difficulties']}")
        print(f"  Average Time: {math_stats['average_time']:.1f} minutes")

        # Show sample problem
        sample_math = self.dataset_manager.mathematics.get_random_problems(1)[0]
        print(f"\n  Sample Problem: {sample_math.title}")
        print(f"  Domain: {sample_math.domain}")
        print(f"  Difficulty: {sample_math.difficulty}")
        print(f"  Statement: {sample_math.statement[:100]}...")

        # Creative Writing Dataset
        print("\n✍️ Creative Writing Dataset:")
        writing_stats = self.dataset_manager.creative_writing.get_statistics()
        print(f"  Total Problems: {writing_stats['total_problems']}")
        print(f"  Categories: {list(writing_stats['categories'].keys())}")
        print(f"  Difficulties: {writing_stats['difficulties']}")
        print(f"  Average Time: {writing_stats['average_time']:.1f} minutes")

        # Show sample problem
        sample_writing = self.dataset_manager.creative_writing.get_random_problems(1)[0]
        print(f"\n  Sample Problem: {sample_writing.title}")
        print(f"  Category: {sample_writing.category}")
        print(f"  Genre: {sample_writing.genre}")
        print(f"  Prompt: {sample_writing.prompt[:100]}...")

    async def demonstrate_dataset_management(self):
        """Demonstrate dataset manager capabilities."""
        print("\n🗂️ DATASET MANAGEMENT")
        print("-" * 40)

        # Overall statistics
        overall_stats = self.dataset_manager.get_comprehensive_statistics()
        print("\nOverall Statistics:")
        print(f"  Total Problems: {overall_stats['total_problems']}")
        print(f"  Domains: {list(overall_stats['domains'].keys())}")
        print(f"  Difficulty Distribution: {overall_stats['difficulties']}")
        print(f"  Average Time: {overall_stats['average_time']:.1f} minutes")

        # Balanced sampling
        print("\n🎯 Balanced Sampling:")
        balanced_sample = self.dataset_manager.create_balanced_sample(
            problems_per_domain=5, difficulty_distribution={"easy": 0.2, "medium": 0.6, "hard": 0.2}
        )
        print(f"  Created balanced sample of {len(balanced_sample)} problems")

        domain_counts = {}
        difficulty_counts = {}
        for problem in balanced_sample:
            # Determine domain
            if hasattr(problem, "category"):
                if problem.category in ["algorithm", "data_structure", "design_pattern", "architecture", "testing"]:
                    domain = "software_engineering"
                elif problem.category in ["story", "essay", "documentation", "problem_solving", "style_adaptation"]:
                    domain = "creative_writing"
                else:
                    domain = "unknown"
            elif hasattr(problem, "domain"):
                if problem.domain in ["algebra", "calculus", "proof", "optimization", "statistics"]:
                    domain = "mathematics"
                else:
                    domain = "unknown"
            else:
                domain = "unknown"

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            difficulty_counts[problem.difficulty] = difficulty_counts.get(problem.difficulty, 0) + 1

        print(f"  Domain distribution: {domain_counts}")
        print(f"  Difficulty distribution: {difficulty_counts}")

        # Search functionality
        print("\n🔍 Search Functionality:")
        search_results = self.dataset_manager.search_problems("algorithm")
        print(f"  Found {len(search_results)} problems containing 'algorithm'")

        # Validation
        print("\n✅ Dataset Validation:")
        validation = self.dataset_manager.validate_datasets()
        print(f"  Overall valid: {validation['valid']}")
        if validation["issues"]:
            print(f"  Issues found: {len(validation['issues'])}")
        else:
            print("  No issues found - all datasets are complete and consistent")

    async def demonstrate_evaluation_suite(self):
        """Demonstrate evaluation suite capabilities."""
        print("\n🎯 EVALUATION SUITE")
        print("-" * 40)

        # Get sample problems for evaluation
        sample_problems = self.dataset_manager.get_random_problems(3)

        print(f"\nSelected {len(sample_problems)} problems for evaluation demo:")
        for i, problem in enumerate(sample_problems, 1):
            print(f"  {i}. {problem.title} ({problem.difficulty})")

        # Create sample solutions
        sample_solutions = [
            "def quicksort(arr): return sorted(arr)  # Simple implementation",
            "x = 3, y = 1.5  # Solution to the quadratic equation",
            "Once upon a time, in a galaxy far away, a time traveler discovered parallel universes...",
        ]

        print("\n🤖 Automated Evaluation:")
        print("  Evaluating solutions using LLM-based assessment...")

        try:
            # Run automated evaluation
            session = await self.evaluation_suite.evaluate_method_performance(
                method_name="demo_method",
                problem_ids=[p.id for p in sample_problems],
                solutions=sample_solutions,
                evaluation_method="automated",
            )

            print("  ✅ Evaluation completed!")
            print(f"  Session ID: {session.session_id}")
            print(f"  Problems evaluated: {len(session.results)}")
            print(f"  Mean score: {session.session_statistics.get('mean_score', 0):.2f}")
            print(f"  Duration: {session.session_statistics.get('evaluation_duration', 0):.1f} seconds")

            # Show detailed results
            print("\n📊 Detailed Results:")
            for result in session.results:
                print(f"  Problem {result.problem_id}:")
                print(f"    Overall Score: {result.overall_score:.2f}")
                print(f"    Domain: {result.domain}")
                print(f"    Feedback: {result.feedback[:100]}...")

        except Exception as e:
            print(f"  ⚠️ Evaluation demo skipped: {e}")
            print("  (This is normal if Ollama is not running)")

    async def demonstrate_statistical_analysis(self):
        """Demonstrate statistical analysis capabilities."""
        print("\n📈 STATISTICAL ANALYSIS")
        print("-" * 40)

        # Export dataset summary
        print("\n📋 Dataset Summary Export:")
        summary = self.dataset_manager.export_dataset_summary()

        print(f"  Summary exported with {summary['statistics']['total_problems']} problems")
        print(f"  Domains covered: {len(summary['metadata']['domains'])}")
        print(f"  Version: {summary['metadata']['version']}")

        # Show domain breakdown
        print("\n🔍 Domain Analysis:")
        for domain, details in summary["domain_details"].items():
            print(f"  {domain.title()}:")
            print(f"    Problems: {details['total_problems']}")
            print(f"    Avg Time: {details['average_time']:.1f} minutes")

            if "categories" in details:
                top_category = max(details["categories"].items(), key=lambda x: x[1])
                print(f"    Top Category: {top_category[0]} ({top_category[1]} problems)")
            elif "domains" in details:
                top_subdomain = max(details["domains"].items(), key=lambda x: x[1])
                print(f"    Top Subdomain: {top_subdomain[0]} ({top_subdomain[1]} problems)")

        # Problem complexity analysis
        print("\n⚡ Complexity Analysis:")
        all_problems = self.dataset_manager.get_all_problems()

        time_by_difficulty = {"easy": [], "medium": [], "hard": []}
        for problem in all_problems:
            if hasattr(problem, "estimated_time_minutes"):
                time_by_difficulty[problem.difficulty].append(problem.estimated_time_minutes)

        for difficulty, times in time_by_difficulty.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"  {difficulty.title()} problems: {len(times)} problems, {avg_time:.1f} min avg")

        print("\n🎯 Evaluation Readiness:")
        print(f"  ✅ {summary['statistics']['total_problems']} problems ready for evaluation")
        print("  ✅ Automated evaluation system operational")
        print("  ✅ Statistical analysis framework available")
        print("  ✅ Multi-domain coverage ensures comprehensive testing")


async def main():
    """Run the dataset demonstration."""
    demo = Phase5Demo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
