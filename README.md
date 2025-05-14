# System Prompt Learning Framework

This project implements a framework for learning and evolving system prompts through a structured feedback loop, focusing on coding domain challenges.

## Project Structure

```
.
├── feedback_loop/           # Feedback collection and evaluation
├── prompt_integration/      # System prompt evolution and management
├── tests/                  # Test cases and evaluation framework
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Components

1. **Feedback Loop**
   - Solution quality evaluation
   - Reflection collection
   - Quality filtering

2. **Prompt Integration**
   - Reflection consolidation
   - System prompt evolution
   - Performance tracking

3. **Testing Framework**
   - Problem sets
   - Performance metrics
   - Baseline comparisons

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
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

## Contributing

This is a research project exploring system prompt learning. Contributions and ideas are welcome! 