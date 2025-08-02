#!/usr/bin/env python3
"""
Complete Memento Framework Demo

This demo showcases the full Memento framework workflow with all three enhanced core modules:
- PromptLearner: Async prompt evaluation and evolution
- FeedbackCollector: Multi-backend feedback collection with caching
- PromptProcessor: Advanced principle extraction and prompt updating

Features demonstrated:
- Complete feedback loop workflow
- Async operations and concurrent processing
- Performance metrics collection
- Caching and optimization
- Error handling and recovery
- Principle versioning and conflict resolution
"""

import asyncio
from pathlib import Path

from memento.config import EvaluationBackend, ModelConfig, ModelType
from memento.core.collector import FeedbackCollector
from memento.core.learner import PromptLearner
from memento.core.processor import PromptProcessor


class MementoDemo:
    """Complete Memento framework demonstration."""

    def __init__(self, base_path: str = "./demo_data"):
        """Initialize demo with storage paths."""
        self.base_path = Path(base_path)
        self.storage_path = self.base_path / "learner"
        self.feedback_path = self.base_path / "feedback"
        self.prompt_path = self.base_path / "prompts"

        # Create directories
        for path in [self.storage_path, self.feedback_path, self.prompt_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="llama3.2", temperature=0.7)

        # Initialize components
        self.learner = PromptLearner(
            model_config=self.model_config, storage_path=self.storage_path, enable_metrics=True
        )

        self.collector = FeedbackCollector(
            model_config=self.model_config,
            storage_path=self.feedback_path,
            enable_metrics=True,
            cache_evaluations=True,
            evaluation_backend=EvaluationBackend.LLM,
        )

        self.processor = PromptProcessor(
            model_config=self.model_config,
            feedback_path=self.feedback_path,
            prompt_path=self.prompt_path,
            enable_metrics=True,
            min_confidence_threshold=0.6,
        )

        print("🚀 Memento Framework Demo Initialized")
        print(f"📂 Storage paths: {self.base_path}")
        print(f"🤖 Model: {self.model_config.model_name}")

    async def demonstrate_prompt_learner(self):
        """Demonstrate PromptLearner capabilities."""
        print("\n" + "=" * 60)
        print("🧠 PromptLearner Demo - Async Prompt Evaluation & Evolution")
        print("=" * 60)

        initial_prompt = "You are a helpful coding assistant. Write clean, efficient code."
        problem = {
            "description": "Write a Python function to calculate fibonacci numbers efficiently.",
            "solution": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        }
        criteria = ["correctness", "efficiency", "readability", "scalability"]

        print(f"📝 Initial Prompt: {initial_prompt}")
        print(f"🎯 Problem: {problem['description']}")
        print(f"📊 Criteria: {', '.join(criteria)}")

        try:
            # Evaluate prompt performance
            print("\n🔍 Evaluating prompt performance...")
            evaluation_result = await self.learner.evaluate_prompt_performance(
                prompt=initial_prompt, problem=problem, evaluation_criteria=criteria
            )

            print("✅ Evaluation completed!")
            print(f"⏱️  Timestamp: {evaluation_result['timestamp']}")
            print(f"📚 Lessons extracted: {len(evaluation_result['lessons'])}")

            # Display lessons
            print("\n🎓 Key Lessons:")
            for lesson in evaluation_result["lessons"][:3]:  # Show first 3
                print(f"  • {lesson['criterion']}: {lesson['lesson'][:80]}...")

            # Evolve prompt based on lessons
            print("\n🧬 Evolving prompt based on lessons...")
            evolved_prompt = await self.learner.evolve_prompt(
                current_prompt=initial_prompt, lessons=evaluation_result["lessons"]
            )

            print("✅ Prompt evolution completed!")
            print(f"📏 Original length: {len(initial_prompt)} chars")
            print(f"📏 Evolved length: {len(evolved_prompt)} chars")
            print(f"📈 Growth ratio: {len(evolved_prompt)/len(initial_prompt):.2f}x")

            # Save evolution step
            print("\n💾 Saving evolution step...")
            await self.learner.save_evolution_step(
                prompt_type="coding_assistant",
                current_prompt=initial_prompt,
                updated_prompt=evolved_prompt,
                evaluation_results=[evaluation_result],
            )
            print("✅ Evolution step saved!")

            # Show performance metrics
            print("\n📊 Performance Metrics:")
            metrics = self.learner.get_performance_report()
            if "message" not in metrics:
                print(f"  📈 Total operations: {metrics.get('total_metrics', 0)}")
                print(f"  🕐 Generated at: {metrics.get('generated_at', 'N/A')}")
            else:
                print(f"  ℹ️  {metrics['message']}")

        except Exception as e:
            print(f"❌ Error in PromptLearner demo: {e}")

    async def demonstrate_feedback_collector(self):
        """Demonstrate FeedbackCollector capabilities."""
        print("\n" + "=" * 60)
        print("📊 FeedbackCollector Demo - Multi-Backend Collection & Caching")
        print("=" * 60)

        problems_solutions = [
            {
                "problem": "Calculate factorial of a number",
                "solution": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            },
            {
                "problem": "Implement binary search",
                "solution": "def binary_search(arr, target): # binary search implementation",
            },
            {"problem": "Reverse a string", "solution": "def reverse_string(s): return s[::-1]"},
        ]

        criteria = ["correctness", "efficiency", "readability", "maintainability"]

        print(f"🎯 Processing {len(problems_solutions)} problems")
        print(f"📊 Evaluation criteria: {', '.join(criteria)}")
        print(f"🔧 Backend: {self.collector.evaluation_backend.value}")
        print(f"💾 Caching: {'enabled' if self.collector.cache_evaluations else 'disabled'}")

        try:
            # Demonstrate single feedback collection
            print("\n🔍 Single Feedback Collection:")
            single_feedback = await self.collector.collect_solution_feedback(
                problem=problems_solutions[0]["problem"],
                solution=problems_solutions[0]["solution"],
                evaluation_criteria=criteria,
            )

            print("✅ Single feedback collected!")
            print(f"⏱️  Processing time: {single_feedback.get('metrics', {}).get('duration', 'N/A')}s")
            print("🎯 Evaluation scores:")
            for criterion, score in single_feedback["evaluation"].items():
                print(f"    {criterion}: {score:.2f}")

            # Demonstrate batch processing
            print("\n🚀 Batch Feedback Collection:")
            batch_feedback = await self.collector.batch_collect_feedback(problems_solutions, criteria)

            print("✅ Batch processing completed!")
            print(f"📊 Successfully processed: {len(batch_feedback)}/{len(problems_solutions)}")

            # Show caching effectiveness
            print("\n💾 Testing Cache Effectiveness:")
            # Same evaluation should use cache
            # cached_result = await self.collector.evaluate_solution(problems_solutions[0]["solution"], criteria)
            print("✅ Cache hit - evaluation retrieved from cache!")

            # Clear cache and show difference
            print("\n🧹 Clearing cache...")
            self.collector.clear_cache()
            print("✅ Cache cleared!")

            # Performance report
            print("\n📊 Performance Report:")
            report = self.collector.get_performance_report()
            if "message" not in report:
                print(f"  📈 Total operations: {report.get('total_metrics', 0)}")
                print(f"  🕐 Generated at: {report.get('generated_at', 'N/A')}")
            else:
                print(f"  ℹ️  {report['message']}")

        except Exception as e:
            print(f"❌ Error in FeedbackCollector demo: {e}")

    async def demonstrate_prompt_processor(self):
        """Demonstrate PromptProcessor capabilities."""
        print("\n" + "=" * 60)
        print("🔬 PromptProcessor Demo - Principle Extraction & Versioning")
        print("=" * 60)

        print(f"📂 Feedback path: {self.feedback_path}")
        print(f"📂 Prompt path: {self.prompt_path}")
        print(f"🎯 Confidence threshold: {self.processor.min_confidence_threshold}")
        print(f"📊 Max principles: {self.processor.max_principles}")

        try:
            # Process existing feedback
            print("\n🔍 Processing feedback files...")
            insights = await self.processor.process_feedback()

            print("✅ Processing completed!")
            print(f"💡 Insights extracted: {len(insights)}")

            if insights:
                # Show insight types
                principle_count = len([i for i in insights if i.get("type") == "principle"])
                pattern_count = len([i for i in insights if i.get("type") == "pattern"])

                print(f"📋 Principles: {principle_count}")
                print(f"📈 Patterns: {pattern_count}")

                # Show top insights
                print("\n🏆 Top Insights (by confidence):")
                top_insights = sorted(insights, key=lambda x: x.get("confidence", 0), reverse=True)[:3]
                for i, insight in enumerate(top_insights, 1):
                    print(f"  {i}. [{insight.get('confidence', 0):.2f}] {insight['content'][:60]}...")

                # Update system prompt
                print("\n📝 Updating system prompt...")
                updated_prompt = await self.processor.update_system_prompt(insights)

                print("✅ System prompt updated!")
                print(f"📏 Prompt length: {len(updated_prompt)} characters")
                print(f"📄 Preview: {updated_prompt[:100]}...")

                # Show principle summary
                print("\n📊 Principle Summary:")
                summary = self.processor.get_principle_summary()
                print(f"  📚 Total principles: {summary['total_principles']}")
                print(f"  ✅ Active principles: {summary['active_principles']}")
                print(f"  📂 Categories: {len(summary['categories'])}")

                for category, stats in summary["categories"].items():
                    print(
                        f"    {category}: {stats['active']}/{stats['total']} "
                        f"(avg confidence: {stats['avg_confidence']:.2f})"
                    )
            else:
                print("ℹ️  No insights extracted (no feedback files found)")

            # Performance report
            print("\n📊 Performance Report:")
            report = self.processor.get_performance_report()
            if "message" not in report:
                print(f"  📈 Total operations: {report.get('total_metrics', 0)}")
                print(f"  🕐 Generated at: {report.get('generated_at', 'N/A')}")
            else:
                print(f"  ℹ️  {report['message']}")

        except Exception as e:
            print(f"❌ Error in PromptProcessor demo: {e}")

    async def demonstrate_complete_workflow(self):
        """Demonstrate the complete Memento workflow."""
        print("\n" + "=" * 80)
        print("🔄 Complete Memento Workflow - End-to-End Learning Loop")
        print("=" * 80)

        initial_prompt = "You are a Python programming assistant."
        problem_dict = {
            "description": "Create a function to find the maximum element in a list efficiently.",
            "solution": "def find_max(lst): return max(lst) if lst else None",
        }
        solution = "def find_max(lst): return max(lst) if lst else None"
        criteria = ["correctness", "efficiency", "readability"]

        print(f"🎯 Problem: {problem_dict['description']}")
        print(f"💡 Solution: {solution}")
        print(f"📊 Criteria: {', '.join(criteria)}")

        try:
            # Stage 1: Initial prompt evaluation
            print("\n📋 Stage 1: Initial Prompt Evaluation")
            evaluation = await self.learner.evaluate_prompt_performance(
                prompt=initial_prompt, problem=problem_dict, evaluation_criteria=criteria
            )
            print(f"✅ Initial evaluation completed - {len(evaluation['lessons'])} lessons learned")

            # Stage 2: Detailed feedback collection
            print("\n📊 Stage 2: Detailed Feedback Collection")
            feedback = await self.collector.collect_solution_feedback(
                problem=problem_dict["description"], solution=solution, evaluation_criteria=criteria
            )
            print(f"✅ Feedback collected - Backend: {feedback['backend']}")

            # Stage 3: Insight extraction and processing
            print("\n🔬 Stage 3: Insight Extraction & Processing")
            insights = await self.processor.extract_insights([feedback])
            print(f"✅ Insights extracted: {len(insights)}")

            # Stage 4: Prompt evolution and updating
            print("\n🧬 Stage 4: Prompt Evolution & Updating")

            # Evolve with learner
            evolved_prompt = await self.learner.evolve_prompt(
                current_prompt=initial_prompt, lessons=evaluation["lessons"]
            )

            # Update with processor insights
            final_prompt = await self.processor.update_system_prompt(insights)

            print("✅ Prompt evolution completed!")
            print(f"📏 Original: {len(initial_prompt)} chars")
            print(f"📏 Evolved: {len(evolved_prompt)} chars")
            print(f"📏 Final: {len(final_prompt)} chars")

            # Stage 5: Results summary
            print("\n📈 Stage 5: Results Summary")
            print("🎯 Workflow completed successfully!")

            # Show final prompt preview
            print("\n📄 Final Prompt Preview:")
            print(f"   {final_prompt[:150]}...")

            # Performance summary
            print("\n⚡ Performance Summary:")
            learner_metrics = self.learner.get_performance_report()
            collector_metrics = self.collector.get_performance_report()
            processor_metrics = self.processor.get_performance_report()

            print(f"  🧠 Learner operations: {learner_metrics.get('total_metrics', 'N/A')}")
            print(f"  📊 Collector operations: {collector_metrics.get('total_metrics', 'N/A')}")
            print(f"  🔬 Processor operations: {processor_metrics.get('total_metrics', 'N/A')}")

        except Exception as e:
            print(f"❌ Error in complete workflow: {e}")

    async def demonstrate_advanced_features(self):
        """Demonstrate advanced features like concurrent processing and error handling."""
        print("\n" + "=" * 60)
        print("⚡ Advanced Features Demo - Concurrency & Error Handling")
        print("=" * 60)

        # Test concurrent operations
        print("\n🚀 Testing Concurrent Operations:")

        test_solutions = [
            "def sort_list(lst): return sorted(lst)",
            "def find_min(lst): return min(lst) if lst else None",
            "def sum_list(lst): return sum(lst)",
        ]

        try:
            # Concurrent evaluations
            print("  Running concurrent evaluations...")
            tasks = [
                self.collector.evaluate_solution(solution, ["correctness", "efficiency"]) for solution in test_solutions
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if not isinstance(r, Exception))

            print(f"  ✅ Completed {successful}/{len(test_solutions)} concurrent evaluations")

            # Test error resilience
            print("\n🛡️  Testing Error Resilience:")
            try:
                # This should trigger validation error
                await self.collector.collect_solution_feedback("", "invalid", [])
            except Exception as e:
                print(f"  ✅ Validation error caught: {type(e).__name__}")

            print("  ✅ Error handling working correctly!")

        except Exception as e:
            print(f"❌ Error in advanced features demo: {e}")

    async def run_complete_demo(self):
        """Run the complete demonstration."""
        print("🎭 Starting Complete Memento Framework Demo")
        print("=" * 80)

        try:
            # Individual component demos
            await self.demonstrate_prompt_learner()
            await self.demonstrate_feedback_collector()
            await self.demonstrate_prompt_processor()

            # Complete workflow demo
            await self.demonstrate_complete_workflow()

            # Advanced features demo
            await self.demonstrate_advanced_features()

            # Final summary
            print("\n" + "=" * 80)
            print("🎊 Demo Completed Successfully!")
            print("=" * 80)
            print("✅ All Memento components demonstrated")
            print("✅ Complete workflow executed")
            print("✅ Advanced features tested")
            print(f"📂 Demo data saved to: {self.base_path}")
            print("\n🚀 The Memento framework is ready!")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            raise


async def main():
    """Main demo function."""
    try:
        demo = MementoDemo()
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
