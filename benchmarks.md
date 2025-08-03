# Memento Benchmarking Guide

## Quick Start

The fastest way to run benchmarks is using the CLI interface from your shell:

```bash
# Quick benchmark with default settings
python -m memento.cli.main benchmark run --datasets humaneval gsm8k --models memento promptbreeder

# Generate report from results
python -m memento.cli.main benchmark report --input-dir benchmark_results



python -m memento.cli benchmark run --datasets gsm8k --models memento --max-samples 1 --no-dashboard --model-name llama3.2

python -m memento.cli benchmark run --datasets gsm8k --models memento --max-samples 5 --model-name llama3.2
```

## Detailed Guide

### 1. Setup

First, make sure you have the environment set up:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Memento with benchmarking dependencies
pip install -e ".[benchmark]"

# Verify installation
python -m memento.cli.main --version
```

### 2. Available Commands

```bash
# Show all available commands
python -m memento.cli.main --help

# Show benchmark specific commands
python -m memento.cli.main benchmark --help

# Show dataset commands
python -m memento.cli.main datasets --help
```

### 3. Running Benchmarks

#### Basic Usage

```bash
# Run benchmark on specific datasets
python -m memento.cli.main benchmark run \
    --datasets humaneval gsm8k \
    --models memento promptbreeder \
    --output-dir my_benchmark_results

# Run with specific model configurations
python -m memento.cli.main benchmark run \
    --datasets humaneval \
    --models memento \
    --model-config model_type=ollama,model_name=llama2,temperature=0.7 \
    --output-dir programming_results
```

#### Task-Specific Benchmarks

```bash
# Programming benchmark
python -m memento.cli.main benchmark run \
    --focus programming \
    --datasets humaneval apps \
    --models memento promptbreeder \
    --metrics pass@1 pass@10 code_quality \
    --output-dir programming_results

# Mathematics benchmark
python -m memento.cli.main benchmark run \
    --focus mathematics \
    --datasets gsm8k mmlu_math \
    --models memento promptbreeder \
    --metrics accuracy reasoning_steps \
    --output-dir math_results
```

#### Advanced Options

```bash
# Run with specific dataset splits and sizes
python -m memento.cli.main benchmark run \
    --datasets humaneval:max_samples=164 gsm8k:split=test,max_samples=500 \
    --models memento promptbreeder \
    --parallel-tasks 3 \
    --save-intermediates \
    --output-dir full_results

# Run with custom evaluation settings
python -m memento.cli.main benchmark run \
    --datasets humaneval \
    --models memento \
    --batch-size 32 \
    --timeout 300 \
    --cache-dir ~/.cache/memento \
    --output-dir custom_results
```

### 4. Managing Datasets

```bash
# List available datasets
python -m memento.cli.main datasets list

# Show dataset details
python -m memento.cli.main datasets info humaneval
python -m memento.cli.main datasets info gsm8k

# Download specific datasets
python -m memento.cli.main datasets download humaneval gsm8k

# Clear dataset cache
python -m memento.cli.main datasets clear-cache
```

### 5. Generating Reports

```bash
# Generate comprehensive report
python -m memento.cli.main benchmark report \
    --input-dir benchmark_results \
    --include-plots \
    --include-error-analysis \
    --output-format html

# Generate comparison report
python -m memento.cli.main benchmark compare \
    --results-dirs run1_results run2_results \
    --output-dir comparison_results
```

### 6. Configuration Files

You can also use configuration files to avoid long command lines:

```yaml
# benchmark_config.yaml
datasets:
  - name: humaneval
    max_samples: 164
  - name: gsm8k
    split: test
    max_samples: 500

models:
  - name: memento
    config:
      model_type: ollama
      model_name: llama2
      temperature: 0.7
  - name: promptbreeder
    config:
      model_type: openai
      model_name: gpt-4
      temperature: 0.7

output_dir: benchmark_results
parallel_tasks: 3
save_intermediates: true
```

Then run with:

```bash
python -m memento.cli.main benchmark run --config benchmark_config.yaml
```

### 7. Troubleshooting

Common issues and solutions:

1. **Dataset Size Mismatch**
```bash
# Get correct dataset size
python -m memento.cli.main datasets info humaneval

# Use correct size in benchmark
python -m memento.cli.main benchmark run \
    --datasets humaneval:max_samples=164 \
    --models memento
```

2. **Split Handling**
```bash
# Check available splits
python -m memento.cli.main datasets info gsm8k --show-splits

# Use correct split
python -m memento.cli.main benchmark run \
    --datasets gsm8k:split=test \
    --models memento
```

3. **Memory Management**
```bash
# Run with memory optimization
python -m memento.cli.main benchmark run \
    --datasets apps \
    --models memento \
    --batch-size 16 \
    --use-disk-cache \
    --gc-interval 1000
```

### 8. Environment Variables

You can set these environment variables to configure the benchmarking:

```bash
# Set up environment
export MEMENTO_CACHE_DIR=~/.cache/memento
export MEMENTO_OUTPUT_DIR=./benchmark_results
export MEMENTO_LOG_LEVEL=INFO

# Model-specific settings
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

# Run benchmark with environment settings
python -m memento.cli.main benchmark run --datasets humaneval gsm8k
```

### 9. Logging and Debugging

```bash
# Enable debug logging
python -m memento.cli.main benchmark run \
    --datasets humaneval \
    --models memento \
    --log-level DEBUG \
    --output-dir debug_results

# Save logs to file
python -m memento.cli.main benchmark run \
    --datasets humaneval \
    --models memento \
    --log-file benchmark.log
```

### 10. Example Shell Script

Here's a complete example script for running a comprehensive benchmark:

```bash
#!/bin/bash

# benchmark.sh
set -e

# Setup
source venv/bin/activate
export MEMENTO_LOG_LEVEL=INFO
export MEMENTO_CACHE_DIR=~/.cache/memento

# Clear previous results
rm -rf benchmark_results

# Download datasets
python -m memento.cli.main datasets download humaneval gsm8k apps

# Run programming benchmarks
python -m memento.cli.main benchmark run \
    --focus programming \
    --datasets humaneval:max_samples=164 apps:split=test,max_samples=1000 \
    --models memento promptbreeder \
    --metrics pass@1 pass@10 code_quality \
    --output-dir benchmark_results/programming

# Run math benchmarks
python -m memento.cli.main benchmark run \
    --focus mathematics \
    --datasets gsm8k:split=test,max_samples=500 \
    --models memento promptbreeder \
    --metrics accuracy reasoning_steps \
    --output-dir benchmark_results/mathematics

# Generate reports
python -m memento.cli.main benchmark report \
    --input-dir benchmark_results \
    --include-plots \
    --include-error-analysis

echo "Benchmark complete! Results in benchmark_results/"
```

Run the script:
```bash
chmod +x benchmark.sh
./benchmark.sh
``` 