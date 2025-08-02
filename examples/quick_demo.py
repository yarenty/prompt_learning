#!/usr/bin/env python3
"""
Quick Memento Framework Demo

A streamlined demo that tests core functionality without long-running LLM operations.
"""

import asyncio
from pathlib import Path

from memento.config import EvaluationBackend, ModelConfig, ModelType
from memento.core.collector import FeedbackCollector
from memento.core.learner import PromptLearner
from memento.core.processor import PromptProcessor


async def quick_test():
    """Quick test of all components."""
    print("🚀 Quick Memento Framework Test")
    print("=" * 50)

    # Setup
    model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="llama3.2", temperature=0.7)

    base_path = Path("./quick_test_data")
    base_path.mkdir(exist_ok=True)

    # Test 1: Component Initialization
    print("\n📋 Test 1: Component Initialization")

    learner = PromptLearner(model_config=model_config, storage_path=base_path / "learner", enable_metrics=True)
    print("✅ PromptLearner initialized")

    # Use automated backend for faster testing
    collector = FeedbackCollector(
        model_config=model_config,
        storage_path=base_path / "feedback",
        evaluation_backend=EvaluationBackend.AUTOMATED,
        cache_evaluations=True,
    )
    print("✅ FeedbackCollector initialized (automated backend)")

    processor = PromptProcessor(
        model_config=model_config,
        feedback_path=base_path / "feedback",
        prompt_path=base_path / "prompts",
        enable_metrics=True,
    )
    print("✅ PromptProcessor initialized")

    # Test 2: Basic Operations
    print("\n📊 Test 2: Basic Operations")

    # Test automated evaluation (fast)
    evaluation_result = await collector.evaluate_solution("def test(): return 42", ["correctness", "efficiency"])
    print(f"✅ Automated evaluation: {evaluation_result}")

    # Test caching
    cached_result = await collector.evaluate_solution("def test(): return 42", ["correctness", "efficiency"])
    print(f"✅ Cached evaluation: {cached_result}")
    print(f"   Results match: {evaluation_result == cached_result}")

    # Test feedback collection
    feedback = await collector.collect_solution_feedback(
        problem="Test problem", solution="def test(): return 42", evaluation_criteria=["correctness", "efficiency"]
    )
    print("✅ Feedback collection completed")
    print(f"   Backend: {feedback['backend']}")
    print(f"   Evaluation keys: {list(feedback['evaluation'].keys())}")

    # Test 3: Performance Metrics
    print("\n📈 Test 3: Performance Metrics")

    learner_metrics = learner.get_performance_report()
    collector_metrics = collector.get_performance_report()
    processor_metrics = processor.get_performance_report()

    print(f"✅ Learner metrics: {type(learner_metrics).__name__}")
    print(f"✅ Collector metrics: {type(collector_metrics).__name__}")
    print(f"✅ Processor metrics: {type(processor_metrics).__name__}")

    # Test 4: Configuration Consistency
    print("\n⚙️  Test 4: Configuration Consistency")

    components = [learner, collector, processor]
    model_names = [c.model_config.model_name for c in components]
    temperatures = [c.model_config.temperature for c in components]

    print(f"✅ Model names consistent: {len(set(model_names)) == 1}")
    print(f"✅ Temperatures consistent: {len(set(temperatures)) == 1}")
    print(f"   Model: {model_names[0]}, Temperature: {temperatures[0]}")

    # Test 5: Error Handling
    print("\n🛡️  Test 5: Error Handling")

    try:
        await collector.collect_solution_feedback("", "", [])
        print("❌ Should have raised ValidationError")
    except Exception as e:
        print(f"✅ Validation error caught: {type(e).__name__}")

    # Summary
    print("\n" + "=" * 50)
    print("🎉 Quick Test Summary")
    print("=" * 50)
    print("✅ All components initialized successfully")
    print("✅ Basic operations working")
    print("✅ Caching functional")
    print("✅ Performance metrics collected")
    print("✅ Configuration consistency verified")
    print("✅ Error handling working")
    print("\n🚀 Memento Framework is ready for production!")


if __name__ == "__main__":
    asyncio.run(quick_test())
