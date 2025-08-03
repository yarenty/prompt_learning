"""Tests for dataset loader."""

import os
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from memento.benchmarking.datasets.loader import DatasetLoader

@pytest.fixture
def loader():
    """Create temporary dataset loader."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield DatasetLoader(cache_dir=tmpdir)

@pytest.fixture
def sample_dataset():
    """Create sample dataset."""
    return Dataset.from_dict({
        "text": ["Sample text 1", "Sample text 2"],
        "label": [0, 1]
    })

@pytest.fixture
def sample_dataset_dict():
    """Create sample dataset dictionary."""
    dataset = Dataset.from_dict({
        "text": ["Sample text 1", "Sample text 2"],
        "label": [0, 1]
    })
    return DatasetDict({
        "train": dataset,
        "test": dataset
    })

def test_loader_initialization(loader):
    """Test loader initialization."""
    assert isinstance(loader.cache_dir, Path)
    assert loader.cache_dir.exists()
    assert loader.use_auth_token is None
    
    # Check dataset configurations
    assert "humaneval" in loader.datasets
    assert "gsm8k" in loader.datasets
    assert "apps" in loader.datasets
    assert "mmlu_math" in loader.datasets
    assert "writingbench" in loader.datasets
    
    for config in loader.datasets.values():
        assert "load" in config
        assert "splits" in config
        assert "metrics" in config

def test_cache_key_generation(loader):
    """Test cache key generation."""
    # Test with minimal parameters
    key1 = loader._get_cache_key("humaneval", None, None, {})
    assert isinstance(key1, str)
    
    # Test with split
    key2 = loader._get_cache_key("humaneval", "test", None, {})
    assert key2 != key1
    
    # Test with max_samples
    key3 = loader._get_cache_key("humaneval", "test", 100, {})
    assert key3 != key2
    
    # Test with kwargs
    key4 = loader._get_cache_key("humaneval", "test", 100, {"shuffle": True})
    assert key4 != key3
    
    # Test deterministic
    key5 = loader._get_cache_key("humaneval", "test", 100, {"shuffle": True})
    assert key5 == key4

def test_cache_operations(loader, sample_dataset, sample_dataset_dict):
    """Test cache operations."""
    # Test saving and loading Dataset
    cache_path = loader.cache_dir / "test_dataset.json"
    loader._save_to_cache(sample_dataset, cache_path)
    loaded_dataset = loader._load_from_cache(cache_path)
    
    assert isinstance(loaded_dataset, Dataset)
    assert len(loaded_dataset) == len(sample_dataset)
    assert loaded_dataset.column_names == sample_dataset.column_names
    
    # Test saving and loading DatasetDict
    cache_path = loader.cache_dir / "test_dataset_dict.json"
    loader._save_to_cache(sample_dataset_dict, cache_path)
    loaded_dataset_dict = loader._load_from_cache(cache_path)
    
    assert isinstance(loaded_dataset_dict, DatasetDict)
    assert loaded_dataset_dict.keys() == sample_dataset_dict.keys()
    for split in loaded_dataset_dict:
        assert len(loaded_dataset_dict[split]) == len(sample_dataset_dict[split])
        assert (loaded_dataset_dict[split].column_names == 
                sample_dataset_dict[split].column_names)

@pytest.mark.asyncio
async def test_load_dataset(loader):
    """Test dataset loading."""
    # Test loading HumanEval
    dataset = await loader.load_dataset("humaneval", max_samples=5)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 5
    
    # Test loading from cache
    cached_dataset = await loader.load_dataset("humaneval", max_samples=5)
    assert isinstance(cached_dataset, Dataset)
    assert len(cached_dataset) == 5
    
    # Test invalid dataset
    with pytest.raises(ValueError):
        await loader.load_dataset("invalid_dataset")

@pytest.mark.asyncio
async def test_humaneval_loading(loader):
    """Test HumanEval dataset loading."""
    dataset = await loader._load_humaneval(max_samples=5)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 5
    assert "prompt" in dataset.column_names
    assert "canonical_solution" in dataset.column_names

@pytest.mark.asyncio
async def test_gsm8k_loading(loader):
    """Test GSM8K dataset loading."""
    dataset = await loader._load_gsm8k(max_samples=5)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 5
    assert "question" in dataset.column_names
    assert "answer" in dataset.column_names

@pytest.mark.asyncio
async def test_apps_loading(loader):
    """Test APPS dataset loading."""
    dataset = await loader._load_apps(max_samples=5)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 5
    assert "problem" in dataset.column_names
    assert "solutions" in dataset.column_names

@pytest.mark.asyncio
async def test_mmlu_math_loading(loader):
    """Test MMLU Math dataset loading."""
    dataset = await loader._load_mmlu_math(max_samples=5)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 5
    assert "question" in dataset.column_names
    assert "choices" in dataset.column_names
    assert "answer" in dataset.column_names

@pytest.mark.asyncio
async def test_writingbench_loading(loader):
    """Test WritingBench dataset loading."""
    dataset = await loader._load_writingbench(max_samples=5)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 5
    assert "prompt" in dataset.column_names
    assert "response" in dataset.column_names

@pytest.mark.integration
async def test_dataset_integration(loader):
    """Test dataset loading integration."""
    # Load all datasets with small samples
    results = {}
    for name in loader.datasets:
        dataset = await loader.load_dataset(name, max_samples=2)
        results[name] = dataset
        
    # Check results
    for name, dataset in results.items():
        assert isinstance(dataset, (Dataset, DatasetDict))
        if isinstance(dataset, Dataset):
            assert len(dataset) == 2
        else:
            for split in dataset.values():
                assert len(split) == 2

@pytest.mark.stress
async def test_dataset_stress(loader):
    """Test dataset loading under stress."""
    # Load same dataset multiple times
    for _ in range(10):
        dataset = await loader.load_dataset("humaneval", max_samples=5)
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 5
        
    # Load multiple datasets concurrently
    import asyncio
    tasks = []
    for name in loader.datasets:
        task = asyncio.create_task(loader.load_dataset(name, max_samples=5))
        tasks.append(task)
        
    results = await asyncio.gather(*tasks)
    
    for dataset in results:
        assert isinstance(dataset, (Dataset, DatasetDict))
        if isinstance(dataset, Dataset):
            assert len(dataset) == 5
        else:
            for split in dataset.values():
                assert len(split) == 5 