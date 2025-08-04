#!/usr/bin/env python3
"""Test script to verify benchmarking system components."""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        from memento.config.models import ModelConfig, ModelType
        print("✅ ModelConfig and ModelType imported")
        
        from memento.benchmarking.comprehensive_benchmark import ComprehensiveBenchmark
        print("✅ ComprehensiveBenchmark imported")
        
        from memento.benchmarking.main import ComprehensiveBenchmark as MainBenchmark
        print("✅ Main ComprehensiveBenchmark imported")
        
        from memento.benchmarking.datasets.loader import DatasetLoader
        print("✅ DatasetLoader imported")
        
        from memento.benchmarking.evaluation.model_evaluator import ModelEvaluator
        print("✅ ModelEvaluator imported")
        
        from memento.benchmarking.baselines.promptbreeder import PromptBreeder
        print("✅ PromptBreeder imported")
        
        from memento.benchmarking.baselines.self_evolving_gpt import SelfEvolvingGPT
        print("✅ SelfEvolvingGPT imported")
        
        from memento.benchmarking.baselines.auto_evolve import AutoEvolve
        print("✅ AutoEvolve imported")
        
        from memento.benchmarking.evaluation.task_metrics import ProgrammingMetrics, MathematicsMetrics, WritingMetrics
        print("✅ Task metrics imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_config():
    """Test model configuration creation."""
    print("\n🔍 Testing model configuration...")
    
    try:
        from memento.config.models import ModelConfig, ModelType
        
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama3.2",
            temperature=0.7,
            max_tokens=2048
        )
        
        print(f"✅ ModelConfig created: {config.model_name}")
        return True
        
    except Exception as e:
        print(f"❌ ModelConfig creation failed: {e}")
        return False

def test_dataset_loader():
    """Test dataset loader initialization."""
    print("\n🔍 Testing dataset loader...")
    
    try:
        from memento.benchmarking.datasets.loader import DatasetLoader
        
        loader = DatasetLoader()
        available_datasets = list(loader.datasets.keys())
        
        print(f"✅ DatasetLoader created with {len(available_datasets)} datasets: {available_datasets}")
        return True
        
    except Exception as e:
        print(f"❌ DatasetLoader creation failed: {e}")
        return False

def test_benchmark_creation():
    """Test benchmark system creation."""
    print("\n🔍 Testing benchmark creation...")
    
    try:
        from memento.config.models import ModelConfig, ModelType
        from memento.benchmarking.comprehensive_benchmark import ComprehensiveBenchmark
        
        model_config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama3.2",
            temperature=0.7,
            max_tokens=2048
        )
        
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir="test_output",
            enable_dashboard=False,
            enable_resource_monitoring=False
        )
        
        print("✅ ComprehensiveBenchmark created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark creation failed: {e}")
        return False

def test_baseline_models():
    """Test baseline model creation."""
    print("\n🔍 Testing baseline models...")
    
    try:
        from memento.config.models import ModelConfig, ModelType
        from memento.benchmarking.baselines.promptbreeder import PromptBreeder
        from memento.benchmarking.baselines.self_evolving_gpt import SelfEvolvingGPT
        from memento.benchmarking.baselines.auto_evolve import AutoEvolve
        
        model_config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama3.2",
            temperature=0.7,
            max_tokens=2048
        )
        
        # Test PromptBreeder
        promptbreeder = PromptBreeder(
            model_config=model_config,
            storage_path=Path("test_promptbreeder")
        )
        print("✅ PromptBreeder created")
        
        # Test SelfEvolvingGPT
        self_evolving = SelfEvolvingGPT(
            model_config=model_config,
            storage_path=Path("test_self_evolving")
        )
        print("✅ SelfEvolvingGPT created")
        
        # Test AutoEvolve
        auto_evolve = AutoEvolve(
            model_config=model_config,
            storage_path=Path("test_auto_evolve")
        )
        print("✅ AutoEvolve created")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline model creation failed: {e}")
        return False

def test_task_metrics():
    """Test task metrics creation."""
    print("\n🔍 Testing task metrics...")
    
    try:
        from memento.benchmarking.evaluation.task_metrics import ProgrammingMetrics, MathematicsMetrics, WritingMetrics
        
        programming_metrics = ProgrammingMetrics()
        print("✅ ProgrammingMetrics created")
        
        math_metrics = MathematicsMetrics()
        print("✅ MathematicsMetrics created")
        
        writing_metrics = WritingMetrics()
        print("✅ WritingMetrics created")
        
        return True
        
    except Exception as e:
        print(f"❌ Task metrics creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Memento Benchmarking System")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_config,
        test_dataset_loader,
        test_benchmark_creation,
        test_baseline_models,
        test_task_metrics,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The benchmarking system is ready to use.")
        print("\n💡 You can now run benchmarks with:")
        print("   python -m memento.cli.benchmark run --datasets humaneval --models memento --max-samples 5")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
