#!/usr/bin/env python3
"""
Test script for the comprehensive benchmark system.
This script tests the basic functionality without requiring external datasets.
"""

import asyncio
import tempfile
from pathlib import Path

from memento.benchmarking.comprehensive_benchmark import ComprehensiveBenchmark
from memento.config.models import ModelConfig, ModelType


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


async def create_mock_datasets():
    """Create mock datasets for testing."""

    # Programming dataset
    programming_samples = [
        {
            "id": 1,
            "problem": "Write a function to calculate factorial of n",
            "expected": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "test_cases": [
                {"inputs": [5], "expected": 120},
                {"inputs": [0], "expected": 1},
                {"inputs": [3], "expected": 6},
            ],
        },
        {
            "id": 2,
            "problem": "Write a function to check if a number is prime",
            "expected": "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
            "test_cases": [
                {"inputs": [7], "expected": True},
                {"inputs": [4], "expected": False},
                {"inputs": [2], "expected": True},
            ],
        },
    ]

    # Mathematics dataset
    math_samples = [
        {
            "id": 1,
            "problem": "Solve: 2x + 5 = 11",
            "expected": "x = 3",
            "solution_steps": ["2x + 5 = 11", "2x = 11 - 5", "2x = 6", "x = 3"],
        },
        {
            "id": 2,
            "problem": "What is the area of a circle with radius 5?",
            "expected": "78.54",
            "solution_steps": ["A = πr²", "A = π × 5²", "A = 25π", "A ≈ 78.54"],
        },
    ]

    # Writing dataset
    writing_samples = [
        {
            "id": 1,
            "problem": "Write a brief summary of the benefits of renewable energy",
            "expected": "Renewable energy sources like solar and wind power offer numerous benefits including reduced carbon emissions, energy independence, and long-term cost savings.",
            "criteria": ["clarity", "completeness", "accuracy"],
        },
        {
            "id": 2,
            "problem": "Describe the process of photosynthesis in simple terms",
            "expected": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "criteria": ["scientific accuracy", "simplicity", "coherence"],
        },
    ]

    return {
        "programming_test": MockDataset(programming_samples),
        "math_test": MockDataset(math_samples),
        "writing_test": MockDataset(writing_samples),
    }


async def test_basic_functionality():
    """Test basic benchmark functionality."""
    print("🧪 Testing Basic Benchmark Functionality")
    print("=" * 50)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_benchmark"

        # Create model configuration (using a real model name)
        model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="llama3.2")  # Use a common model name

        print(f"📁 Output directory: {output_dir}")
        print(f"🤖 Model: {model_config.model_name} ({model_config.model_type.value})")

        # Initialize benchmark system
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=output_dir,
            enable_dashboard=False,  # Disable for testing
            enable_resource_monitoring=True,
        )

        print("✅ Initialized benchmark system")

        # Create mock datasets
        mock_datasets = await create_mock_datasets()

        # Patch the dataset loader to return mock data
        async def mock_load_dataset(name, **kwargs):
            if name in mock_datasets:
                dataset = mock_datasets[name]
                max_samples = kwargs.get("max_samples")
                if max_samples:
                    samples = list(dataset)[:max_samples]
                    return MockDataset(samples)
                return dataset
            else:
                raise ValueError(f"Unknown mock dataset: {name}")

        # Replace the dataset loader method
        benchmark.dataset_loader.load_dataset = mock_load_dataset

        print("✅ Set up mock datasets")

        # Test individual components
        await test_resource_monitor()
        await test_model_evaluator(model_config, output_dir)

        # Test full benchmark (with very limited scope)
        print("\n🏁 Running Mini Benchmark...")
        try:
            results = await benchmark.run_benchmark(
                datasets=["programming_test"],  # Just one dataset
                models=["memento"],  # Just one model
                baseline=None,  # No baseline comparison
                max_samples_per_dataset=1,  # Just 1 sample
            )

            print("✅ Mini benchmark completed!")
            print(f"📊 Results keys: {list(results.keys())}")

            # Check results structure
            assert "metadata" in results
            assert "dataset_results" in results
            assert "model_results" in results

            print("✅ Results structure validated")

            # Print summary
            if "dataset_results" in results and "programming_test" in results["dataset_results"]:
                prog_results = results["dataset_results"]["programming_test"]
                if "memento" in prog_results:
                    memento_results = prog_results["memento"]
                    print(f"🤖 Memento accuracy: {memento_results.get('accuracy', 'N/A')}")
                    print(f"⏱️  Memento latency: {memento_results.get('latency', 'N/A')}")
                    print(f"⭐ Quality score: {memento_results.get('quality_score', 'N/A')}")

        except Exception as e:
            print(f"❌ Mini benchmark failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    return True


async def test_resource_monitor():
    """Test resource monitoring."""
    print("\n💻 Testing Resource Monitor...")

    from memento.benchmarking.comprehensive_benchmark import ResourceMonitor

    monitor = ResourceMonitor()

    # Test basic functionality
    monitor.start_monitoring()
    assert monitor.monitoring is True

    monitor.collect_metrics()
    assert len(monitor.metrics) > 0

    monitor.stop_monitoring()
    assert monitor.monitoring is False

    # Check summary
    summary = monitor.get_summary()
    assert isinstance(summary, dict)

    print("✅ Resource monitor working")


async def test_model_evaluator(model_config, output_dir):
    """Test model evaluator."""
    print("\n🔬 Testing Model Evaluator...")

    from memento.benchmarking.evaluation.model_evaluator import ModelEvaluator

    evaluator = ModelEvaluator(model_config)

    # Test task type detection
    assert evaluator._detect_task_type("humaneval") == "programming"
    assert evaluator._detect_task_type("gsm8k") == "mathematics"
    assert evaluator._detect_task_type("writingbench") == "writing"

    # Test sample preparation
    mock_data = [
        {"id": 1, "problem": "test problem", "expected": "test answer"},
        {"id": 2, "description": "another problem", "solution": "another answer"},
    ]

    samples = evaluator._prepare_dataset_samples(mock_data, max_samples=1)
    assert len(samples) == 1
    assert samples[0]["problem"] == "test problem"

    print("✅ Model evaluator working")


async def test_task_metrics():
    """Test task-specific metrics."""
    print("\n📊 Testing Task Metrics...")

    from memento.benchmarking.evaluation.task_metrics import MathematicsMetrics, ProgrammingMetrics, WritingMetrics

    # Test programming metrics
    prog_metrics = ProgrammingMetrics()
    code = "def hello(): return 'world'"
    quality = prog_metrics.calculate_code_quality(code)
    assert isinstance(quality, dict)
    assert "complexity" in quality
    print("✅ Programming metrics working")

    # Test math metrics
    math_metrics = MathematicsMetrics()
    accuracy = math_metrics.calculate_accuracy("The answer is 42", "42")
    assert accuracy > 0.5
    print("✅ Mathematics metrics working")

    # Test writing metrics
    writing_metrics = WritingMetrics()
    text1 = "This is a test sentence."
    text2 = "This is another test sentence."
    rouge = writing_metrics.calculate_rouge(text1, text2)
    assert isinstance(rouge, dict)
    assert "rouge1_fmeasure" in rouge
    print("✅ Writing metrics working")


async def main():
    """Main test function."""
    print("🎯 Comprehensive Benchmark System Test")
    print("=" * 55)

    try:
        # Test individual components first
        await test_task_metrics()

        # Test full system
        success = await test_basic_functionality()

        if success:
            print("\n🎉 All tests passed!")
            print("✅ The benchmark system is working correctly")
            print("\n🔗 Next steps:")
            print("  1. Try running with real datasets:")
            print("     python -m memento.cli benchmark run --datasets humaneval --models memento --max-samples 5")
            print("  2. Test with multiple models:")
            print("     python -m memento.cli benchmark run --models memento promptbreeder --baseline promptbreeder")
            print("  3. Generate reports:")
            print("     python -m memento.cli benchmark report --results-dir ./comprehensive_benchmark_results")
        else:
            print("\n❌ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
