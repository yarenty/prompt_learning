# System Prompt Learning Framework

A framework for enhancing AI problem-solving capabilities through system prompt learning, using Ollama instead of OpenAI.

## Project Structure

```
.
├── feedback_loop/
│   ├── __init__.py
│   ├── collector.py
│   └── evaluator.py
├── prompt_integration/
│   ├── __init__.py
│   ├── processor.py
│   └── insights.py
├── examples/
│   ├── __init__.py
│   ├── problems.py
│   ├── system_prompts.py
│   ├── run_problems.py
│   └── logger.py
├── tests/
│   ├── __init__.py
│   ├── test_collector.py
│   ├── test_processor.py
│   └── test_insights.py
├── logs/
│   ├── prompt_learning_*.log
│   └── run_info_*.json
├── requirements.txt
└── README.md
```

## Components

### 1. Feedback Loop System
- Collects feedback on problem solutions
- Evaluates solutions based on multiple criteria
- Generates detailed feedback and reflections

### 2. Prompt Integration
- Processes feedback to extract insights
- Updates system prompts based on insights
- Maintains prompt evolution history

### 3. Example Problems
- Basic Algorithms (List Filtering, String Palindrome)
- Data Structures (Tree Traversal)
- Concurrency (Task Processing)
- Design Patterns (Connection Pool, Caching, Middleware)

### 4. System Prompts
The framework includes various initial system prompts to test prompt evolution:
- Beginner Python Programmer
- Expert Python Programmer
- Music Composer
- Mathematician
- Creative Writer
- System Architect
- Security Expert
- Data Scientist
- Game Developer
- Ethical Hacker
- AI Researcher
- Embedded Systems Programmer
- Web Developer
- Mobile Developer
- DevOps Engineer

Each prompt type approaches programming from a different perspective, allowing us to study how different starting points affect prompt evolution.

### 5. Logging System
- Detailed logging of evaluation process
- Tracking of prompt evolution
- Storage of run information and insights
- Separate log files for each prompt type

## Setup

0. Get ollama and model
```
ollama pull codellama
```

1. Create a virtual environment:
```bash
python -m venv venv
# or
uv venv

source venv/bin/activate  # On Unix/macOS
# or 
source venv/bin/activate.fish  # On Unix/macOS - fish shell

# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or 
uv init
uv add -r requirements.txt
```

## Usage

1. Start feedback collection:
```bash
python -m feedback_loop.collect
```

2. Process and integrate feedback:
```bash
python -m prompt_integration.process
```

3. Run example problems with different system prompts:
```bash
python -m examples.run_problems
```

This will:
- Run all problems with each system prompt type
- Generate detailed logs for each run
- Save run information in JSON format
- Track prompt evolution across different perspectives

4. View logs:
- Check `logs/prompt_learning_*.log` for detailed logs
- Check `logs/run_info_*.json` for run summaries
- Check `logs/combined_results_*.json` for comparison across prompt types

## Logging Output

The system provides detailed logging of the evaluation process:

1. Evaluation Results:
```
Evaluation Results:
------------------------------
correctness     : 0.85
efficiency      : 0.90
readability     : 0.95
maintainability : 0.88
error_handling  : 0.82
documentation   : 0.90
------------------------------
```

2. Insights:
```
Extracted Insights:
==================================================
Insight: Consider edge cases in input validation
Support Count: 3
Confidence: 0.85
------------------------------
```

3. Prompt Evolution:
```
System Prompt Evolution:
==================================================
New Insights Added:
+ Consider edge cases in input validation
+ Implement proper error handling
+ Add comprehensive documentation

Full Updated Prompt:
------------------------------
[Updated system prompt content]
------------------------------
```

## Prompt Evolution Analysis

The framework tracks how different initial prompts evolve when solving the same set of problems. This allows us to:

1. Compare how different perspectives approach similar problems
2. Identify common patterns in prompt evolution
3. Understand which initial prompts lead to better solutions
4. Study the convergence of different perspectives

Each run generates:
- Individual log files for each prompt type
- Combined results showing evolution across all prompts
- Detailed insights and evaluation metrics
- Prompt evolution history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 