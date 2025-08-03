"""
Comprehensive Benchmark System Demo

This script demonstrates the new comprehensive benchmarking capabilities
of the Memento framework, including:
- Performance monitoring and resource tracking
- Task-specific metrics (programming, math, writing)
- Real-time dashboard visualization
- Statistical analysis and comparison
- Professional reporting
"""

import asyncio
import tempfile
from pathlib import Path

from memento.benchmarking.comprehensive_benchmark import ComprehensiveBenchmark
from memento.benchmarking.evaluation.task_metrics import (
    MathematicsMetrics,
    ProgrammingMetrics,
    WritingMetrics,
)
from memento.config.models import ModelConfig, ModelType


async def demo_comprehensive_benchmark():
    """Demonstrate comprehensive benchmark system."""
    print("🚀 Memento Comprehensive Benchmark Demo")
    print("=" * 50)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "benchmark_demo"
        
        # Create model configuration
        model_config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            temperature=0.7
        )
        
        print(f"📁 Output directory: {output_dir}")
        print(f"🤖 Model: {model_config.model_name} ({model_config.model_type.value})")
        print()
        
        # Initialize comprehensive benchmark
        benchmark = ComprehensiveBenchmark(
            model_config=model_config,
            output_dir=output_dir,
            enable_dashboard=False,  # Disable for demo
            enable_resource_monitoring=True
        )
        
        print("✅ Initialized comprehensive benchmark system")
        print("📊 Resource monitoring: Enabled")
        print("📱 Dashboard: Disabled (for demo)")
        print()
        
        # Demo dataset configuration
        datasets = ["humaneval"]  # Start with one dataset for demo
        models = ["memento", "promptbreeder"]
        baseline = "promptbreeder"
        
        print(f"📋 Datasets: {', '.join(datasets)}")
        print(f"🤖 Models: {', '.join(models)}")
        print(f"📈 Baseline: {baseline}")
        print()
        
        print("🏁 Starting benchmark run...")
        
        try:
            # Run comprehensive benchmark
            results = await benchmark.run_benchmark(
                datasets=datasets,
                models=models,
                baseline=baseline,
                max_samples_per_dataset=5  # Small number for demo
            )
            
            print("✅ Benchmark completed successfully!")
            print()
            
            # Display results summary
            display_results_summary(results)
            
            # Show generated files
            print("📁 Generated Files:")
            print("-" * 20)
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(output_dir)
                    print(f"  📄 {relative_path}")
            print()
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            raise


def demo_task_specific_metrics():
    """Demonstrate task-specific metrics."""
    print("📊 Task-Specific Metrics Demo")
    print("=" * 40)
    
    # Programming Metrics Demo
    print("\n🔧 Programming Metrics:")
    prog_metrics = ProgrammingMetrics()
    
    sample_code = '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
    
def fibonacci_iterative(n):
    """Calculate fibonacci number iteratively."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''
    
    quality_metrics = prog_metrics.calculate_code_quality(sample_code)
    print("  Code Quality Metrics:")
    for metric, score in quality_metrics.items():
        print(f"    {metric}: {score:.3f}")
    
    # Mathematics Metrics Demo
    print("\n📐 Mathematics Metrics:")
    math_metrics = MathematicsMetrics()
    
    sample_solution = '''
To solve this quadratic equation ax² + bx + c = 0, we use the quadratic formula.
First, we identify the coefficients: a = 1, b = -5, c = 6
Then, we apply the formula: x = (-b ± √(b²-4ac)) / 2a
Substituting: x = (5 ± √(25-24)) / 2 = (5 ± 1) / 2
Therefore: x = 3 or x = 2
#### The solutions are x = 2 and x = 3
'''
    
    reasoning_metrics = math_metrics.evaluate_reasoning(sample_solution)
    print("  Reasoning Quality Metrics:")
    for metric, score in reasoning_metrics.items():
        print(f"    {metric}: {score:.3f}")
    
    accuracy = math_metrics.calculate_accuracy(sample_solution, "x = 2 and x = 3")
    print(f"  Numerical Accuracy: {accuracy:.3f}")
    
    # Writing Metrics Demo
    print("\n✍️  Writing Metrics:")
    writing_metrics = WritingMetrics()
    
    sample_text = '''
The implementation of the new algorithm demonstrates significant improvements 
    over existing approaches. Furthermore, the comprehensive evaluation across 
    multiple datasets confirms the effectiveness of our method. The results 
    indicate a substantial enhancement in both accuracy and efficiency metrics.
    
    However, there are certain limitations that should be acknowledged. 
    Nevertheless, the overall performance gains justify the increased complexity. 
    Therefore, we conclude that this approach represents a meaningful 
    advancement in the field.
'''
    
    style_metrics = writing_metrics.analyze_style(sample_text)
    print("  Style Analysis Metrics:")
    for metric, score in style_metrics.items():
        print(f"    {metric}: {score:.3f}")
    
    coherence_metrics = writing_metrics.evaluate_coherence(sample_text)
    print("  Coherence Analysis Metrics:")
    for metric, score in coherence_metrics.items():
        print(f"    {metric}: {score:.3f}")


def display_results_summary(results):
    """Display benchmark results summary."""
    print("📊 Benchmark Results Summary:")
    print("-" * 35)
    
    # Metadata
    metadata = results.get("metadata", {})
    if metadata:
        print(f"⏱️  Duration: {metadata.get('duration_seconds', 0):.2f} seconds")
        print(f"📋 Datasets: {len(metadata.get('datasets', []))}")
        print(f"🤖 Models: {len(metadata.get('models', []))}")
        print()
    
    # Performance Analysis
    perf = results.get("performance_analysis", {})
    if perf and isinstance(perf, dict):
        print("⚡ Performance Analysis:")
        if "samples_per_second" in perf:
            print(f"  Throughput: {perf['samples_per_second']:.2f} samples/sec")
        if "average_latency" in perf:
            print(f"  Avg Latency: {perf['average_latency']:.3f} seconds")
        if "total_samples" in perf:
            print(f"  Total Samples: {perf['total_samples']}")
        print()
    
    # Resource Analysis
    resources = results.get("resource_analysis", {})
    if resources and isinstance(resources, dict):
        print("💻 Resource Usage:")
        if "peak_memory_mb" in resources:
            print(f"  Peak Memory: {resources['peak_memory_mb']:.1f} MB")
        if "avg_cpu_percent" in resources:
            print(f"  Avg CPU: {resources['avg_cpu_percent']:.1f}%")
        if "duration_seconds" in resources:
            print(f"  Duration: {resources['duration_seconds']:.2f} seconds")
        if "efficiency_score" in resources:
            print(f"  Efficiency: {resources['efficiency_score']:.3f}")
        print()
    
    # Model Results
    model_results = results.get("model_results", {})
    if model_results:
        print("🤖 Model Performance:")
        for model, metrics in model_results.items():
            if isinstance(metrics, dict):
                print(f"  {model}:")
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        mean = metric_data["mean"]
                        std = metric_data.get("std", 0)
                        print(f"    {metric_name}: {mean:.3f} ± {std:.3f}")
        print()
    
    # Comparison Results
    comparisons = results.get("comparison_results", {})
    if comparisons:
        baseline = results.get("metadata", {}).get("baseline", "baseline")
        print(f"📈 Improvements over {baseline}:")
        
        for model, improvements in comparisons.items():
            if isinstance(improvements, dict):
                print(f"  {model}:")
                
                # Calculate average improvements across datasets
                all_improvements = {}
                for dataset_improvements in improvements.values():
                    if isinstance(dataset_improvements, dict):
                        for metric, improvement in dataset_improvements.items():
                            if isinstance(improvement, (int, float)):
                                if metric not in all_improvements:
                                    all_improvements[metric] = []
                                all_improvements[metric].append(improvement)
                
                # Display average improvements
                for metric, improvement_list in all_improvements.items():
                    if improvement_list:
                        avg_improvement = sum(improvement_list) / len(improvement_list)
                        print(f"    {metric}: {avg_improvement:+.1f}%")
        print()


def demo_cli_usage():
    """Demonstrate CLI usage examples."""
    print("💻 CLI Usage Examples:")
    print("=" * 30)
    
    print("1. Basic benchmark run:")
    print("   python -m memento.cli benchmark run \\")
    print("     --datasets humaneval gsm8k writingbench \\")
    print("     --models memento promptbreeder \\")
    print("     --baseline promptbreeder")
    print()
    
    print("2. Advanced benchmark with configuration:")
    print("   python -m memento.cli benchmark run \\")
    print("     --config-file benchmark_config.yaml \\")
    print("     --max-samples 100 \\")
    print("     --dashboard")
    print()
    
    print("3. Generate configuration template:")
    print("   python -m memento.cli benchmark init-config \\")
    print("     --template research \\")
    print("     --output research_config.yaml")
    print()
    
    print("4. Validate datasets:")
    print("   python -m memento.cli benchmark validate-datasets \\")
    print("     --datasets humaneval gsm8k \\")
    print("     --max-samples 5")
    print()
    
    print("5. Generate report from results:")
    print("   python -m memento.cli benchmark report \\")
    print("     --results-dir ./benchmark_results \\")
    print("     --output-format html")
    print()
    
    print("6. Run dashboard only:")
    print("   python -m memento.cli benchmark dashboard-only \\")
    print("     --host 0.0.0.0 --port 8050")
    print()


async def main():
    """Main demo function."""
    print("🎯 Memento Comprehensive Benchmarking System")
    print("=" * 55)
    print()
    
    # Demo 1: Task-specific metrics
    demo_task_specific_metrics()
    print()
    
    # Demo 2: CLI usage examples
    demo_cli_usage()
    print()
    
    # Demo 3: Comprehensive benchmark (commented out to avoid long execution)
    print("🚀 Comprehensive Benchmark Demo:")
    print("   (Uncomment the line below to run full benchmark)")
    print("   # await demo_comprehensive_benchmark()")
    print()
    
    # Uncomment to run actual benchmark:
    # await demo_comprehensive_benchmark()
    
    print("✅ Demo completed!")
    print()
    print("🔗 Next Steps:")
    print("  1. Try the CLI commands shown above")
    print("  2. Create a configuration file for your specific needs")
    print("  3. Run benchmarks on your datasets")
    print("  4. Analyze results with the built-in visualizations")
    print("  5. Generate professional reports for publication")


if __name__ == "__main__":
    asyncio.run(main()) 