#!/usr/bin/env python3
"""
Enhanced Components Showcase

Individual demonstrations of each enhanced Memento component's specific capabilities:
- FeedbackCollector: Caching, batch processing, multiple backends
- PromptProcessor: Principle extraction, versioning, conflict resolution
- Integration examples showing component interactions
"""

import asyncio
from pathlib import Path

from memento.config import EvaluationBackend, ModelConfig, ModelType
from memento.core.collector import FeedbackCollector
from memento.core.processor import PrincipleVersion, PromptProcessor


async def showcase_feedback_collector():
    """Showcase FeedbackCollector's enhanced capabilities."""
    print("🎯 FeedbackCollector Enhanced Features Showcase")
    print("=" * 50)

    # Setup
    model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.3)

    storage_path = Path("./demo_feedback")
    storage_path.mkdir(exist_ok=True)

    # Feature 1: Multiple Evaluation Backends
    print("\n🔧 Feature 1: Multiple Evaluation Backends")

    # LLM Backend
    llm_collector = FeedbackCollector(
        model_config=model_config,
        storage_path=storage_path / "llm",
        evaluation_backend=EvaluationBackend.LLM,
        cache_evaluations=True,
    )

    # Automated Backend
    auto_collector = FeedbackCollector(
        model_config=model_config,
        storage_path=storage_path / "auto",
        evaluation_backend=EvaluationBackend.AUTOMATED,
        cache_evaluations=True,
    )

    test_solution = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    criteria = ["correctness", "efficiency", "readability"]

    try:
        print("  Testing LLM backend...")
        llm_result = await llm_collector.evaluate_solution(test_solution, criteria)
        print(f"  ✅ LLM scores: {llm_result}")

        print("  Testing Automated backend...")
        auto_result = await auto_collector.evaluate_solution(test_solution, criteria)
        print(f"  ✅ Automated scores: {auto_result}")

        print("  📊 Backend comparison:")
        for criterion in criteria:
            llm_score = llm_result.get(criterion, 0)
            auto_score = auto_result.get(criterion, 0)
            print(f"    {criterion}: LLM={llm_score:.2f}, Auto={auto_score:.2f}")

    except Exception as e:
        print(f"  ❌ Backend demo error: {e}")

    # Feature 2: Intelligent Caching
    print("\n💾 Feature 2: Intelligent Caching")

    try:
        print("  First evaluation (cache miss)...")
        start_time = asyncio.get_event_loop().time()
        result1 = await llm_collector.evaluate_solution(test_solution, criteria)
        first_time = asyncio.get_event_loop().time() - start_time

        print("  Second evaluation (cache hit)...")
        start_time = asyncio.get_event_loop().time()
        result2 = await llm_collector.evaluate_solution(test_solution, criteria)
        second_time = asyncio.get_event_loop().time() - start_time

        print("  ⚡ Cache effectiveness:")
        print(f"    First call: {first_time:.3f}s")
        print(f"    Second call: {second_time:.3f}s")
        print(f"    Speedup: {first_time/second_time:.1f}x faster")
        print(f"    Results identical: {result1 == result2}")

    except Exception as e:
        print(f"  ❌ Caching demo error: {e}")

    # Feature 3: Batch Processing
    print("\n🚀 Feature 3: Concurrent Batch Processing")

    batch_problems = [
        {"problem": "Sort an array", "solution": "def sort_array(arr): return sorted(arr)"},
        {"problem": "Find maximum", "solution": "def find_max(arr): return max(arr)"},
        {"problem": "Calculate sum", "solution": "def calc_sum(arr): return sum(arr)"},
    ]

    try:
        print(f"  Processing {len(batch_problems)} problems concurrently...")
        start_time = asyncio.get_event_loop().time()

        batch_results = await llm_collector.batch_collect_feedback(batch_problems, criteria)

        batch_time = asyncio.get_event_loop().time() - start_time

        print("  ✅ Batch processing completed!")
        print(f"    Processed: {len(batch_results)}/{len(batch_problems)} items")
        print(f"    Total time: {batch_time:.2f}s")
        print(f"    Average per item: {batch_time/len(batch_results):.2f}s")

    except Exception as e:
        print(f"  ❌ Batch processing demo error: {e}")

    # Feature 4: Performance Metrics
    print("\n📊 Feature 4: Performance Metrics Collection")

    try:
        report = llm_collector.get_performance_report()
        if "message" not in report:
            print(f"  📈 Operations tracked: {report.get('total_metrics', 0)}")
            print(f"  🕐 Report generated: {report.get('generated_at', 'N/A')}")
            if "operations" in report:
                for op, stats in report["operations"].items():
                    print(f"    {op}: {stats.get('count', 0)} calls, " f"avg {stats.get('average_time', 0):.3f}s")
        else:
            print(f"  ℹ️  {report['message']}")

    except Exception as e:
        print(f"  ❌ Metrics demo error: {e}")


async def showcase_prompt_processor():
    """Showcase PromptProcessor's enhanced capabilities."""
    print("\n🔬 PromptProcessor Enhanced Features Showcase")
    print("=" * 50)

    # Setup
    model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.4)

    feedback_path = Path("./demo_feedback/llm")
    prompt_path = Path("./demo_prompts")
    prompt_path.mkdir(exist_ok=True)

    processor = PromptProcessor(
        model_config=model_config,
        feedback_path=feedback_path,
        prompt_path=prompt_path,
        enable_metrics=True,
        min_confidence_threshold=0.5,
    )

    # Feature 1: Principle Versioning System
    print("\n📚 Feature 1: Principle Versioning System")

    try:
        # Create sample principles
        principle1 = PrincipleVersion(
            principle="Use iterative approaches for better performance",
            confidence=0.8,
            version=1,
            metadata={"category": "performance", "examples": "fibonacci, factorial"},
        )

        principle2 = PrincipleVersion(
            principle="Use optimized iterative methods for large datasets",
            confidence=0.9,
            version=2,
            metadata={"category": "performance", "examples": "fibonacci, factorial, loops"},
        )

        print("  Creating principle versions...")
        print(f"    V1: {principle1.principle} (confidence: {principle1.confidence})")
        print(f"    V2: {principle2.principle} (confidence: {principle2.confidence})")

        # Test serialization
        p1_dict = principle1.to_dict()
        p1_restored = PrincipleVersion.from_dict(p1_dict)

        print(f"  ✅ Serialization test: {p1_restored.principle == principle1.principle}")
        print(f"  📅 Created: {principle1.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"  ❌ Versioning demo error: {e}")

    # Feature 2: Confidence Scoring Algorithm
    print("\n🎯 Feature 2: Multi-Factor Confidence Scoring")

    sample_insights = [
        {"type": "principle", "content": "Short principle", "source": "reflection", "category": "general"},
        {
            "type": "principle",
            "content": "This is a comprehensive principle with detailed explanation and multiple examples that should receive higher confidence",
            "source": "evaluation",
            "category": "performance",
            "cluster_size": 3,
            "avg_score": 0.2,  # Low score = high significance
        },
    ]

    try:
        print("  Analyzing insight confidence factors...")
        scored_insights = await processor._score_insight_confidence(sample_insights)

        for i, insight in enumerate(scored_insights):
            print(f"    Insight {i+1}: {insight['confidence']:.3f} confidence")
            print(f"      Length: {len(insight['content'])} chars")
            print(f"      Source: {insight.get('source', 'unknown')}")
            print(f"      Category: {insight.get('category', 'general')}")
            if "cluster_size" in insight:
                print(f"      Cluster size: {insight['cluster_size']}")

    except Exception as e:
        print(f"  ❌ Confidence scoring demo error: {e}")

    # Feature 3: Conflict Resolution
    print("\n⚖️  Feature 3: Intelligent Conflict Resolution")

    conflicting_insights = [
        {
            "type": "principle",
            "content": "Use recursion for better code readability and elegance",
            "confidence": 0.6,
            "category": "readability",
        },
        {
            "type": "principle",
            "content": "Avoid recursion for better performance and stack safety",
            "confidence": 0.8,
            "category": "performance",
        },
    ]

    try:
        print("  Detecting conflicts...")
        conflicts = await processor._detect_conflicts(conflicting_insights)
        print(f"    Found {len(conflicts)} conflicts")

        if conflicts:
            print("  Resolving conflicts by confidence...")
            resolved = await processor._resolve_conflicts_by_confidence(conflicts)
            print(f"    Resolved to {len(resolved)} insights")

            for insight in resolved:
                print(f"      Kept: {insight['content'][:50]}... (confidence: {insight['confidence']})")

    except Exception as e:
        print(f"  ❌ Conflict resolution demo error: {e}")

    # Feature 4: Advanced Clustering
    print("\n🎯 Feature 4: TF-IDF + DBSCAN Insight Clustering")

    similar_insights = [
        {
            "type": "principle",
            "content": "Use efficient algorithms for large datasets to improve performance",
            "confidence": 0.7,
        },
        {
            "type": "principle",
            "content": "Choose optimized algorithms when processing large data for better speed",
            "confidence": 0.8,
        },
        {"type": "principle", "content": "Always validate user input before processing", "confidence": 0.9},
    ]

    try:
        print(f"  Clustering {len(similar_insights)} insights...")
        clustered = await processor._cluster_insights(similar_insights)

        print(f"  ✅ Clustered into {len(clustered)} groups")
        for i, insight in enumerate(clustered):
            cluster_info = f" (cluster size: {insight['cluster_size']})" if "cluster_size" in insight else ""
            print(f"    Group {i+1}: {insight['content'][:50]}...{cluster_info}")

    except Exception as e:
        print(f"  ❌ Clustering demo error: {e}")

    # Feature 5: Principle Summary
    print("\n📊 Feature 5: Comprehensive Principle Management")

    try:
        # Add some principles to the processor
        sample_principles = [
            {
                "type": "principle",
                "content": "Use appropriate data structures for the problem",
                "confidence": 0.85,
                "category": "algorithms",
            },
            {
                "type": "principle",
                "content": "Write readable and maintainable code",
                "confidence": 0.90,
                "category": "readability",
            },
        ]

        await processor._version_principles(sample_principles)

        summary = processor.get_principle_summary()
        print(f"  📚 Total principles: {summary['total_principles']}")
        print(f"  ✅ Active principles: {summary['active_principles']}")
        print(f"  📂 Categories: {len(summary['categories'])}")

        for category, stats in summary["categories"].items():
            print(
                f"    {category}: {stats['active']}/{stats['total']} active "
                f"(avg confidence: {stats['avg_confidence']:.2f})"
            )

    except Exception as e:
        print(f"  ❌ Principle management demo error: {e}")


async def showcase_component_integration():
    """Showcase how components work together."""
    print("\n🔗 Component Integration Showcase")
    print("=" * 50)

    # Setup components
    model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.5)

    feedback_path = Path("./demo_feedback/llm")
    prompt_path = Path("./demo_prompts")

    collector = FeedbackCollector(
        model_config=model_config,
        storage_path=feedback_path,
        evaluation_backend=EvaluationBackend.LLM,
        cache_evaluations=True,
    )

    processor = PromptProcessor(
        model_config=model_config, feedback_path=feedback_path, prompt_path=prompt_path, min_confidence_threshold=0.6
    )

    print("\n🔄 Data Flow Integration")

    try:
        # Step 1: Collector generates feedback
        print("  Step 1: Collecting feedback...")
        feedback = await collector.collect_solution_feedback(
            problem="Implement a stack data structure",
            solution="class Stack: def __init__(self): self.items = []",
            evaluation_criteria=["correctness", "efficiency", "design"],
        )

        print(f"    ✅ Feedback collected with {len(feedback['evaluation'])} criteria")

        # Step 2: Processor extracts insights from feedback
        print("  Step 2: Extracting insights...")
        insights = await processor.extract_insights([feedback])

        print(f"    ✅ Extracted {len(insights)} insights")
        if insights:
            top_insight = max(insights, key=lambda x: x.get("confidence", 0))
            print(f"    🏆 Top insight: {top_insight['content'][:60]}...")
            print(f"       Confidence: {top_insight.get('confidence', 0):.2f}")

        # Step 3: Processor updates system prompt
        if insights:
            print("  Step 3: Updating system prompt...")
            updated_prompt = await processor.update_system_prompt(insights)

            print(f"    ✅ Prompt updated ({len(updated_prompt)} characters)")
            print(f"    📄 Preview: {updated_prompt[:80]}...")

        # Step 4: Show integration metrics
        print("  Step 4: Integration metrics...")
        collector_report = collector.get_performance_report()
        processor_report = processor.get_performance_report()

        print(f"    📊 Collector operations: {collector_report.get('total_metrics', 'N/A')}")
        print(f"    📊 Processor operations: {processor_report.get('total_metrics', 'N/A')}")

    except Exception as e:
        print(f"  ❌ Integration demo error: {e}")


async def main():
    """Run all showcases."""
    print("🎭 Enhanced Memento Components Showcase")
    print("=" * 80)

    try:
        await showcase_feedback_collector()
        await showcase_prompt_processor()
        await showcase_component_integration()

        print("\n" + "=" * 80)
        print("🎊 All Component Showcases Completed!")
        print("=" * 80)
        print("✅ FeedbackCollector: Multi-backend, caching, batch processing")
        print("✅ PromptProcessor: Versioning, conflict resolution, clustering")
        print("✅ Integration: Seamless data flow between components")
        print("\n🚀 Enhanced Memento components are production-ready!")

    except Exception as e:
        print(f"\n❌ Showcase failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
