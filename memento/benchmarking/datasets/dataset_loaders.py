"""Dataset loaders for benchmarking."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiohttp
import datasets
from tqdm import tqdm


class DatasetLoader:
    """Loader for benchmark datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize dataset loader.

        Args:
            cache_dir: Directory to cache datasets. Defaults to ~/.cache/memento/datasets
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/memento/datasets")
        os.makedirs(self.cache_dir, exist_ok=True)

    async def load_humaneval(self) -> List[Dict]:
        """Load OpenAI HumanEval dataset.

        Returns:
            List of problems with:
                - prompt: Problem description and function signature
                - test: Test cases
                - entry_point: Function name to test
        """
        cache_path = Path(self.cache_dir) / "humaneval.json"

        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        # Download from OpenAI's GitHub
        async with aiohttp.ClientSession() as session:
            url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl"
            async with session.get(url) as response:
                data = await response.text()

        problems = [json.loads(line) for line in data.splitlines()]

        # Cache for future use
        with open(cache_path, "w") as f:
            json.dump(problems, f)

        return problems

    async def load_gsm8k(self, split: str = "test") -> List[Dict]:
        """Load GSM8K dataset.

        Args:
            split: Dataset split ('train' or 'test')

        Returns:
            List of problems with:
                - question: Math word problem
                - answer: Step-by-step solution with final answer
        """
        dataset = datasets.load_dataset("openai/gsm8k", split=split)

        return [{"question": item["question"], "answer": item["answer"]} for item in dataset]

    async def load_apps(self, split: str = "test", max_samples: int = 5000) -> List[Dict]:
        """Load APPS dataset.

        Args:
            split: Dataset split ('train' or 'test')
            max_samples: Maximum number of samples to load

        Returns:
            List of problems with:
                - question: Programming problem description
                - solution: Reference solution
                - tests: Test cases
        """
        dataset = datasets.load_dataset("codeparrot/apps", split=split).shuffle().select(range(max_samples))

        return [
            {"question": item["question"], "solution": item["solution"], "tests": item["tests"]} for item in dataset
        ]

    async def load_mmlu_math(self, split: str = "test") -> List[Dict]:
        """Load MMLU Math subset.

        Args:
            split: Dataset split ('train' or 'test')

        Returns:
            List of problems with:
                - question: Math question
                - choices: Multiple choice options
                - answer: Correct answer index
        """
        dataset = datasets.load_dataset("hendrycks/mmlu", "mathematics", split=split)

        return [
            {"question": item["question"], "choices": item["choices"], "answer": item["answer"]} for item in dataset
        ]

    async def load_writingbench(self, split: str = "train") -> List[Dict]:
        """Load WritingBench dataset.

        Args:
            split: Dataset split ('train' only)

        Returns:
            List of problems with:
                - prompt: Writing prompt
                - reference: Reference solution
                - rubric: Evaluation criteria
        """
        dataset = datasets.load_dataset("writing_bench", split=split)

        return [
            {"prompt": item["prompt"], "reference": item["reference"], "rubric": item["rubric"]} for item in dataset
        ]

    async def load_creativity(self, split: str = "train") -> List[Dict]:
        """Load Creativity dataset.

        Args:
            split: Dataset split ('train' only)

        Returns:
            List of problems with:
                - prompt: Creative writing prompt
                - examples: Example responses
                - criteria: Evaluation criteria
        """
        dataset = datasets.load_dataset("creativity_bench", split=split)

        return [
            {"prompt": item["prompt"], "examples": item["examples"], "criteria": item["criteria"]} for item in dataset
        ]

    async def load_all(
        self, datasets: Optional[List[str]] = None, max_samples: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[Dict]]:
        """Load multiple datasets.

        Args:
            datasets: List of dataset names to load. Defaults to all.
            max_samples: Maximum samples per dataset.

        Returns:
            Dictionary mapping dataset names to problem lists.
        """
        available_datasets = {
            "humaneval": self.load_humaneval,
            "gsm8k": self.load_gsm8k,
            "apps": self.load_apps,
            "mmlu_math": self.load_mmlu_math,
            "writingbench": self.load_writingbench,
            "creativity": self.load_creativity,
        }

        datasets = datasets or list(available_datasets.keys())
        max_samples = max_samples or {}

        results = {}
        for name in tqdm(datasets, desc="Loading datasets"):
            loader = available_datasets[name]
            data = await loader()

            if name in max_samples:
                data = data[: max_samples[name]]

            results[name] = data

        return results
