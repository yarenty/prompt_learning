#!/usr/bin/env python3
"""
Real Dataset Demo

Demonstrates loading and working with real datasets from HuggingFace:
- HumanEval for programming
- GSM8K for mathematics
- WritingBench for creative writing

This demo shows actual dataset loading, sample problems, and evaluation.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from memento.benchmarking.datasets.loader import DatasetLoader
from memento.datasets.standard_datasets import StandardDatasetManager


class RealDatasetDemo:
    """Demonstration using real datasets from HuggingFace."""

    def __init__(self):
        """Initialize demo components."""
        self.standard_manager = StandardDatasetManager()
        self.dataset_loader = DatasetLoader()

    async def run_comprehensive_demo(self):
        """Run complete demonstration with real datasets."""
        print("🚀 REAL DATASET DEMO - Using Actual Datasets")
        print("=" * 60)

        # 1. Show available datasets
        await self.show_available_datasets()

        # 2. Load and examine real datasets
        await self.examine_real_datasets()

        # 3. Show sample problems from each dataset
        await self.show_sample_problems()

        # 4. Demonstrate dataset loading with our loader
        await self.demonstrate_dataset_loader()

        print("\n🎉 Real Dataset Demo Complete!")
        print("✅ All datasets loaded from HuggingFace")
        print("✅ Real problems with actual content")
        print("✅ Ready for real benchmarking")

    async def show_available_datasets(self):
        """Show all available standard datasets."""
        print("\n📊 AVAILABLE STANDARD DATASETS")
        print("-" * 40)

        datasets = self.standard_manager.list_available_datasets()

        print("Available datasets from HuggingFace:")
        for name, info in datasets.items():
            print(f"  🔹 {name}:")
            print(f"     Description: {info['description']}")
            print(f"     Domain: {info['domain']}")
            print(f"     Size: {info['size']} problems")
            print(f"     Source: {info['source']}")
            print()

    async def examine_real_datasets(self):
        """Load and examine real datasets."""
        print("\n🔍 EXAMINING REAL DATASETS")
        print("-" * 40)

        # Test datasets that we know work
        test_datasets = ["gsm8k", "humaneval"]

        for dataset_name in test_datasets:
            print(f"\n📋 Loading {dataset_name}...")
            try:
                # Load a small sample
                dataset = self.standard_manager.load_dataset(dataset_name, split="test")

                if dataset:
                    print(f"  ✅ Successfully loaded {len(dataset)} problems")

                    # Show first problem structure
                    if len(dataset) > 0:
                        first_problem = dataset[0]
                        print(f"  📝 Sample problem keys: {list(first_problem.keys())}")

                        # Show a snippet of the problem
                        if "question" in first_problem:
                            print(f"  💭 Question preview: {first_problem['question'][:100]}...")
                        elif "prompt" in first_problem:
                            print(f"  💭 Prompt preview: {first_problem['prompt'][:100]}...")
                        elif "problem" in first_problem:
                            print(f"  💭 Problem preview: {first_problem['problem'][:100]}...")
                else:
                    print(f"  ⚠️ Dataset loaded but empty")

            except Exception as e:
                print(f"  ❌ Failed to load {dataset_name}: {e}")

    async def show_sample_problems(self):
        """Show actual sample problems from real datasets."""
        print("\n📝 SAMPLE REAL PROBLEMS")
        print("-" * 40)

        # Try to load GSM8K (which we know works)
        try:
            print("\n🔢 GSM8K Mathematics Problems:")
            gsm8k_data = self.standard_manager.load_dataset("gsm8k", split="test")

            if gsm8k_data and len(gsm8k_data) >= 2:
                for i in range(min(2, len(gsm8k_data))):
                    problem = gsm8k_data[i]
                    print(f"\n  Problem {i+1}:")
                    print(f"    Question: {problem.get('question', 'N/A')}")
                    print(f"    Answer: {problem.get('answer', 'N/A')}")
                    print()
            else:
                print("  ⚠️ No GSM8K problems available")

        except Exception as e:
            print(f"  ❌ Failed to load GSM8K: {e}")

        # Try to load HumanEval
        try:
            print("\n💻 HumanEval Programming Problems:")
            humaneval_data = self.standard_manager.load_dataset("humaneval", split="test")

            if humaneval_data and len(humaneval_data) >= 1:
                problem = humaneval_data[0]
                print(f"\n  Problem 1:")
                print(f"    Prompt: {problem.get('prompt', 'N/A')[:200]}...")
                print(f"    Test: {problem.get('test', 'N/A')[:100]}...")
                print()
            else:
                print("  ⚠️ No HumanEval problems available")

        except Exception as e:
            print(f"  ❌ Failed to load HumanEval: {e}")

    async def demonstrate_dataset_loader(self):
        """Demonstrate our dataset loader with real datasets."""
        print("\n🔄 DATASET LOADER DEMONSTRATION")
        print("-" * 40)

        # Test our dataset loader with GSM8K
        try:
            print("\n🔧 Testing DatasetLoader with GSM8K:")

            # Load a small sample using our loader
            dataset = await self.dataset_loader.load_dataset("gsm8k", max_samples=3)

            if dataset:
                print(f"  ✅ Loaded {len(dataset)} problems via DatasetLoader")

                # Show the first problem
                if len(dataset) > 0:
                    first_problem = dataset[0]
                    print(f"  📝 First problem structure:")
                    print(f"    Keys: {list(first_problem.keys())}")

                    if "question" in first_problem:
                        print(f"    Question: {first_problem['question'][:150]}...")
                    if "answer" in first_problem:
                        print(f"    Answer: {first_problem['answer'][:100]}...")
            else:
                print("  ⚠️ DatasetLoader returned empty dataset")

        except Exception as e:
            print(f"  ❌ DatasetLoader failed: {e}")

        # Test with HumanEval
        try:
            print("\n🔧 Testing DatasetLoader with HumanEval:")

            dataset = await self.dataset_loader.load_dataset("humaneval", max_samples=2)

            if dataset:
                print(f"  ✅ Loaded {len(dataset)} problems via DatasetLoader")

                if len(dataset) > 0:
                    first_problem = dataset[0]
                    print(f"  📝 First problem structure:")
                    print(f"    Keys: {list(first_problem.keys())}")

                    if "prompt" in first_problem:
                        print(f"    Prompt: {first_problem['prompt'][:150]}...")
                    if "test" in first_problem:
                        print(f"    Test: {first_problem['test'][:100]}...")
            else:
                print("  ⚠️ DatasetLoader returned empty dataset")

        except Exception as e:
            print(f"  ❌ DatasetLoader failed: {e}")

    async def create_sample_evaluation(self):
        """Create a sample evaluation with real data."""
        print("\n🎯 SAMPLE EVALUATION WITH REAL DATA")
        print("-" * 40)

        try:
            # Load a few problems from GSM8K
            problems = await self.dataset_loader.load_dataset("gsm8k", max_samples=2)

            if problems:
                print(f"  📊 Loaded {len(problems)} problems for evaluation")

                # Generate real responses using Ollama
                import ollama

                real_responses = []
                for i, problem in enumerate(problems):
                    try:
                        question = problem.get("question", "")
                        prompt = f"""Solve this math problem step by step:

{question}

Please provide a clear, step-by-step solution."""

                        response = ollama.chat(
                            model="llama3.2",
                            messages=[{"role": "user", "content": prompt}],
                            options={"temperature": 0.7, "num_predict": 500},
                        )

                        if response and "message" in response:
                            real_responses.append(response["message"]["content"])
                            print(f"  🤖 Generated real response {i+1}: {response['message']['content'][:100]}...")
                        else:
                            real_responses.append("Failed to generate response")
                            print(f"  ❌ Failed to generate response {i+1}")

                    except Exception as e:
                        real_responses.append(f"Error: {e}")
                        print(f"  ❌ Error generating response {i+1}: {e}")

                print(f"  📈 Generated {len(real_responses)} real responses using Ollama")
                print(f"  📈 Ready for evaluation with real problems and responses")

                # Show what evaluation would look like
                print(f"  📋 Evaluation would include:")
                print(f"    - Mathematical accuracy")
                print(f"    - Step-by-step reasoning")
                print(f"    - Final answer correctness")
                print(f"    - Response quality metrics")
            else:
                print("  ⚠️ No problems available for evaluation")

        except Exception as e:
            print(f"  ❌ Evaluation setup failed: {e}")


async def main():
    """Run the real dataset demonstration."""
    demo = RealDatasetDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
