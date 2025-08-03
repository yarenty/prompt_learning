"""Dataset loader with caching support."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Dataset loader with caching support."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, use_auth_token: Optional[str] = None):
        """Initialize dataset loader.

        Args:
            cache_dir: Directory for caching datasets
            use_auth_token: Optional Hugging Face auth token
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "memento"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_auth_token = use_auth_token

        # Configure dataset loading
        self.datasets = {
            "humaneval": {"load": self._load_humaneval, "splits": ["train", "test"], "metrics": ["pass@k", "accuracy"]},
            "gsm8k": {
                "load": self._load_gsm8k,
                "splits": ["train", "test"],
                "metrics": ["accuracy", "reasoning_steps"],
            },
            "apps": {"load": self._load_apps, "splits": ["train", "test"], "metrics": ["pass@k", "code_quality"]},
            "mmlu_math": {
                "load": self._load_mmlu_math,
                "splits": ["train", "validation", "test"],
                "metrics": ["accuracy", "confidence"],
            },
            "writingbench": {
                "load": self._load_writingbench,
                "splits": ["train", "validation"],
                "metrics": ["readability", "coherence", "style"],
            },
        }

    async def load_dataset(
        self, name: str, split: Optional[str] = None, max_samples: Optional[int] = None, **kwargs
    ) -> Union[Dataset, DatasetDict]:
        """Load dataset with caching.

        Args:
            name: Dataset name
            split: Optional split name
            max_samples: Maximum number of samples
            **kwargs: Additional loading arguments

        Returns:
            Loaded dataset

        Raises:
            ValueError: If dataset not found or invalid split
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found")

        # Generate cache key
        cache_key = self._get_cache_key(name, split, max_samples, kwargs)
        cache_path = self.cache_dir / f"{cache_key}.json"

        # Try loading from cache
        if cache_path.exists():
            logger.info(f"Loading {name} from cache")
            return self._load_from_cache(cache_path)

        # Load dataset
        logger.info(f"Loading {name} from source")
        dataset = await self.datasets[name]["load"](split=split, max_samples=max_samples, **kwargs)

        # Save to cache
        self._save_to_cache(dataset, cache_path)

        return dataset

    def _get_cache_key(
        self, name: str, split: Optional[str], max_samples: Optional[int], kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for dataset.

        Args:
            name: Dataset name
            split: Optional split name
            max_samples: Maximum number of samples
            kwargs: Additional loading arguments

        Returns:
            Cache key string
        """
        # Create key components
        components = [
            f"name={name}",
            f"split={split}" if split else "split=None",
            f"max_samples={max_samples}" if max_samples else "max_samples=None",
        ]

        # Add sorted kwargs
        for key, value in sorted(kwargs.items()):
            components.append(f"{key}={value}")

        # Generate hash
        key_string = "_".join(components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _load_from_cache(self, cache_path: Path) -> Union[Dataset, DatasetDict]:
        """Load dataset from cache.

        Args:
            cache_path: Path to cached dataset

        Returns:
            Loaded dataset
        """
        with open(cache_path) as f:
            data = json.load(f)

        if isinstance(data, dict) and "type" in data:
            if data["type"] == "Dataset":
                return Dataset.from_dict(data["data"])
            elif data["type"] == "DatasetDict":
                return DatasetDict({k: Dataset.from_dict(v) for k, v in data["data"].items()})

        raise ValueError(f"Invalid cache format in {cache_path}")

    def _save_to_cache(self, dataset: Union[Dataset, DatasetDict], cache_path: Path) -> None:
        """Save dataset to cache.

        Args:
            dataset: Dataset to save
            cache_path: Cache file path
        """
        if isinstance(dataset, Dataset):
            data = {"type": "Dataset", "data": dataset.to_dict()}
        else:
            data = {"type": "DatasetDict", "data": {k: v.to_dict() for k, v in dataset.items()}}

        with open(cache_path, "w") as f:
            json.dump(data, f)

    async def _load_humaneval(
        self, split: Optional[str] = None, max_samples: Optional[int] = None, **kwargs
    ) -> Dataset:
        """Load HumanEval dataset.

        Args:
            split: Optional split name
            max_samples: Maximum number of samples
            **kwargs: Additional loading arguments

        Returns:
            Loaded dataset
        """
        dataset = datasets.load_dataset("openai/human-eval", split=split, use_auth_token=self.use_auth_token)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    async def _load_gsm8k(self, split: Optional[str] = None, max_samples: Optional[int] = None, **kwargs) -> Dataset:
        """Load GSM8K dataset.

        Args:
            split: Optional split name
            max_samples: Maximum number of samples
            **kwargs: Additional loading arguments

        Returns:
            Loaded dataset
        """
        dataset = datasets.load_dataset("gsm8k", "main", split=split, use_auth_token=self.use_auth_token)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    async def _load_apps(self, split: Optional[str] = None, max_samples: Optional[int] = None, **kwargs) -> Dataset:
        """Load APPS dataset.

        Args:
            split: Optional split name
            max_samples: Maximum number of samples
            **kwargs: Additional loading arguments

        Returns:
            Loaded dataset
        """
        dataset = datasets.load_dataset("codeparrot/apps", split=split, use_auth_token=self.use_auth_token)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    async def _load_mmlu_math(
        self, split: Optional[str] = None, max_samples: Optional[int] = None, **kwargs
    ) -> Dataset:
        """Load MMLU Math dataset.

        Args:
            split: Optional split name
            max_samples: Maximum number of samples
            **kwargs: Additional loading arguments

        Returns:
            Loaded dataset
        """
        dataset = datasets.load_dataset("cais/mmlu", "mathematics", split=split, use_auth_token=self.use_auth_token)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    async def _load_writingbench(
        self, split: Optional[str] = None, max_samples: Optional[int] = None, **kwargs
    ) -> Dataset:
        """Load WritingBench dataset.

        Args:
            split: Optional split name
            max_samples: Maximum number of samples
            **kwargs: Additional loading arguments

        Returns:
            Loaded dataset
        """
        dataset = datasets.load_dataset("writing_bench", split=split, use_auth_token=self.use_auth_token)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset
