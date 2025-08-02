#!/usr/bin/env python3
"""
Enhanced PromptLearner Demo

This demo showcases the enhanced PromptLearner with async support,
error handling, validation, and performance metrics.
"""

import asyncio

from memento.config import ModelConfig, ModelType
from memento.core import PromptLearner
from memento.exceptions import ValidationError


async def demo_enhanced_learner():
    """Demonstrate the enhanced PromptLearner capabilities."""
    print("🚀 Enhanced PromptLearner Demonstration")
    print("=" * 50)

    # Create model configuration
    model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama", temperature=0.7, max_tokens=2048)

    # Create learner with metrics enabled
    learner = PromptLearner(model_config=model_config, storage_path="data/enhanced_demo", enable_metrics=True)

    print(f"✅ Initialized learner with model: {model_config.model_name}")
    print(f"📊 Performance metrics: {'Enabled' if learner.monitor else 'Disabled'}")

    # Test problem
    problem = {
        "description": "Write a Python function to find the maximum element in a list",
        "solution": """
def find_max(lst):
    if not lst:
        return None
    return max(lst)

# Test cases
print(find_max([1, 2, 3, 4, 5]))  # 5
print(find_max([]))  # None
print(find_max([-1, -2, -3]))  # -1
        """,
        "category": "algorithms",
        "difficulty": "beginner",
    }

    # Test prompt
    prompt = "You are a Python programmer who writes clean, efficient, and well-documented code."

    # Evaluation criteria
    criteria = ["correctness", "efficiency", "readability", "maintainability"]

    print("\n📝 Testing prompt evaluation...")
    print(f"   Problem: {problem['description']}")
    print(f"   Criteria: {', '.join(criteria)}")

    try:
        # Evaluate prompt performance
        result = await learner.evaluate_prompt_performance(prompt=prompt, problem=problem, evaluation_criteria=criteria)

        print("✅ Evaluation completed successfully!")
        print(f"   Timestamp: {result['timestamp']}")
        print(f"   Lessons extracted: {len(result['lessons'])}")

        # Show evaluation results
        evaluation = result["evaluation"]
        print("\n📊 Evaluation Results:")
        for criterion, data in evaluation.items():
            if isinstance(data, dict):
                score = data.get("score", 0.0)
                explanation = data.get("explanation", "No explanation")
                print(f"   {criterion}: {score:.2f} - {explanation}")

        # Show lessons learned
        print("\n🎓 Lessons Learned:")
        for lesson in result["lessons"]:
            print(f"   • {lesson['criterion']}: {lesson['lesson']}")

        # Test prompt evolution
        print("\n🔄 Testing prompt evolution...")
        evolved_prompt = await learner.evolve_prompt(current_prompt=prompt, lessons=result["lessons"])

        print("✅ Prompt evolution completed!")
        print(f"   Original length: {len(prompt)} characters")
        print(f"   Evolved length: {len(evolved_prompt)} characters")
        print(f"   Evolution ratio: {len(evolved_prompt)/len(prompt):.2f}x")

        # Save evolution step
        print("\n💾 Saving evolution step...")
        await learner.save_evolution_step(
            prompt_type="python_programmer",
            current_prompt=prompt,
            updated_prompt=evolved_prompt,
            evaluation_results=[result],
        )
        print("✅ Evolution step saved!")

        # Show performance report
        print("\n📈 Performance Report:")
        report = learner.get_performance_report()
        if "message" in report:
            print(f"   {report['message']}")
        else:
            print(f"   Generated at: {report['generated_at']}")
            print(f"   Total metrics: {report['total_metrics']}")
            print(f"   Operations: {list(report['operations'].keys())}")

        # Test validation
        print("\n🔍 Testing validation...")
        try:
            # This should raise a ValidationError
            await learner.evaluate_prompt_performance(
                prompt="", problem=problem, evaluation_criteria=criteria
            )  # Empty prompt
        except ValidationError as e:
            print(f"✅ Validation caught error: {e}")

        try:
            # This should raise a ValidationError
            await learner.evaluate_prompt_performance(
                prompt=prompt, problem={"description": "test"}, evaluation_criteria=criteria  # Missing solution
            )
        except ValidationError as e:
            print(f"✅ Validation caught error: {e}")

        print("\n🎉 Demonstration completed successfully!")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        raise


async def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n🛡️ Error Handling Demonstration")
    print("=" * 40)

    # Create learner
    model_config = ModelConfig(model_type=ModelType.OLLAMA, model_name="codellama")

    learner = PromptLearner(model_config=model_config, storage_path="data/error_demo", enable_metrics=False)

    # Test various error scenarios
    error_scenarios = [
        {
            "name": "Empty prompt",
            "prompt": "",
            "problem": {"description": "test", "solution": "test"},
            "criteria": ["correctness"],
        },
        {
            "name": "Invalid problem (missing solution)",
            "prompt": "You are a programmer",
            "problem": {"description": "test"},
            "criteria": ["correctness"],
        },
        {
            "name": "Empty evaluation criteria",
            "prompt": "You are a programmer",
            "problem": {"description": "test", "solution": "test"},
            "criteria": [],
        },
    ]

    for scenario in error_scenarios:
        print(f"\n🔍 Testing: {scenario['name']}")
        try:
            await learner.evaluate_prompt_performance(
                prompt=scenario["prompt"], problem=scenario["problem"], evaluation_criteria=scenario["criteria"]
            )
            print("❌ Expected error but got success")
        except ValidationError as e:
            print(f"✅ Caught ValidationError: {e}")
        except Exception as e:
            print(f"⚠️ Caught unexpected error: {e}")


async def main():
    """Main demonstration function."""
    print("🎯 Memento Enhanced PromptLearner Demo")
    print("=" * 60)

    try:
        # Run main demonstration
        await demo_enhanced_learner()

        # Run error handling demonstration
        await demo_error_handling()

        print("\n🎊 All demonstrations completed successfully!")
        print("📁 Check 'data/enhanced_demo' and 'data/error_demo' for generated files")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
