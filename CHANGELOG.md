# Changelog

## [0.4.0] - Core Module Enhancements

### Enhanced FeedbackCollector
- **Async Support**: All LLM operations now use async/await for better performance and concurrency
- **Multiple Evaluation Backends**: LLM, automated heuristics, and human evaluation support
- **Intelligent Caching**: Evaluation and reflection results cached to reduce redundant LLM calls
- **Batch Processing**: Concurrent processing of multiple feedback items with error resilience
- **Comprehensive Validation**: Input validation for problems, solutions, and evaluation criteria
- **Performance Metrics**: Optional metrics collection with timing and operation tracking

### Enhanced PromptProcessor  
- **Advanced Principle Extraction**: LLM-powered extraction of actionable principles from reflections
- **Confidence Scoring**: Multi-factor confidence assessment for extracted insights
- **Principle Versioning**: Full versioning system tracking principle evolution over time
- **Conflict Resolution**: Automatic detection and resolution of contradictory insights
- **Pattern Analysis**: Statistical analysis of evaluation scores to identify trends
- **Insight Clustering**: TF-IDF and DBSCAN clustering to group similar insights

## [0.3.0] - Core Module Enhancement

- **Abstract Base Classes**: Created comprehensive interfaces for `BaseLearner`, `BaseCollector`, and `BaseProcessor`
- **Custom Exception Hierarchy**: Added 11 custom exception classes for better error categorization and handling
- **Comprehensive Validation**: Input validation for all parameters with detailed error messages
- **Performance Metrics**: Optional metrics collection with timing context managers and performance reports
- **Async Support**: All LLM operations now support async/await for better performance
- **Pydantic V2 Migration**: Updated to modern Pydantic features with field validators
- **Testing Infrastructure**: 18 comprehensive test cases with async support and mocking
- **Circular Import Resolution**: Clean module structure with proper separation of concerns
- **Virtual Environment**: Set up with `uv` for clean dependency management
- **Production Quality**: Professional-grade code with comprehensive error handling and validation

## [0.2.0] - Project Architecture Overhaul

- Refactored the entire codebase into a professional Python package structure (`memento/`).
- Centralized configuration management using Pydantic for robust, validated settings.
- Implemented a unified logging system with JSON and colored output options.
- Introduced a modern CLI using Typer and Rich for user-friendly command-line interaction.
- Added utility modules for validation, helpers, and logging.
- Integrated code quality tools: Black, isort, flake8, mypy, and pytest.
- Established pre-commit hooks for automated formatting, linting, and type checking.
- Updated and expanded documentation, including a new README and setup script.
- Fixed all major linting and formatting issues for a clean, maintainable codebase.

## [0.1.0] - Initial Code PoC