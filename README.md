# System Prompt Learning Framework

This project implements a framework for learning and evolving system prompts through a structured feedback loop, focusing on coding domain challenges.

## Project Structure

```
.
├── feedback_loop/           # Feedback collection and evaluation
├── prompt_integration/      # System prompt evolution and management
├── examples/               # Example problems and run scripts
│   ├── problems.py        # Collection of example problems
│   ├── run_problems.py    # Script to run problems
│   └── logger.py          # Logging configuration
├── tests/                 # Test cases and evaluation framework
├── logs/                  # Generated logs and run information
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Components

1. **Feedback Loop**
   - Solution quality evaluation
   - Reflection collection
   - Quality filtering
   - Detailed logging of evaluation process

2. **Prompt Integration**
   - Reflection consolidation
   - System prompt evolution
   - Performance tracking
   - Insight clustering and synthesis

3. **Example Problems**
   - Basic algorithms (list filtering, palindrome checking)
   - Data structures (tree traversal)
   - Concurrency (async task processing)
   - Design patterns (caching, error handling)
   - Each problem includes:
     - Clear description
     - Implemented solution
     - Evaluation criteria

4. **Logging System**
   - Detailed evaluation logging
   - Problem-solving insights
   - System prompt evolution tracking
   - Run information storage
   - Two logging levels:
     - Console: User-friendly output
     - File: Detailed debug information

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

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Run tests:
```bash
pytest tests/
```

2. Start feedback collection:
```bash
python -m feedback_loop.collect
```

3. Process and integrate feedback:
```bash
python -m prompt_integration.process
```

4. More problems
```bash
python -m examples.run_problems
```



This will:
- Process all example problems
- Generate evaluations and reflections
- Extract insights from feedback
- Update the system prompt
- Save detailed logs in the `logs` directory

5. View logs:
- Check `logs/prompt_learning_*.log` for detailed logs
- Check `logs/run_info.json` for run summary

## Logging Output

The system provides detailed logging of:

1. Problem Evaluation:
```
Problem 1: List Filtering
==================================================
Description: [Problem description]

Evaluation Results:
------------------------------
correctness      : 0.95
efficiency       : 0.90
readability      : 0.85
maintainability  : 0.80
error_handling   : 0.75
documentation    : 0.70
------------------------------

Reflection:
------------------------------
[Detailed reflection on the solution]
------------------------------
```

2. Insights and Prompt Evolution:
```
Extracted Insights:
==================================================
[Insight details with support counts]

System Prompt Evolution:
==================================================
[Changes to the system prompt]
```

## Contributing

This is a research project exploring system prompt learning. Contributions and ideas are welcome! 