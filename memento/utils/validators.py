"""
Validation utilities for the Memento framework.

This module provides validation functions for configuration, data, and
other framework components.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import get_settings


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, raises ValueError if invalid

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ["model", "evaluation", "storage"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")

    # Validate model configuration
    if "model" in config:
        validate_model_config(config["model"])

    # Validate evaluation configuration
    if "evaluation" in config:
        validate_evaluation_config(config["evaluation"])

    # Validate storage configuration
    if "storage" in config:
        validate_storage_config(config["storage"])

    return True


def validate_model_config(model_config: Dict[str, Any]) -> bool:
    """
    Validate model configuration.

    Args:
        model_config: Model configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ["model_type", "model_name"]

    for field in required_fields:
        if field not in model_config:
            raise ValueError(f"Missing required model configuration field: {field}")

    # Validate model type
    valid_types = ["ollama", "openai", "anthropic", "local"]
    if model_config["model_type"] not in valid_types:
        raise ValueError(f"Invalid model type: {model_config['model_type']}")

    # Validate model name
    if not model_config["model_name"] or not model_config["model_name"].strip():
        raise ValueError("Model name cannot be empty")

    # Validate API key for cloud models
    if model_config["model_type"] in ["openai", "anthropic"]:
        if not model_config.get("api_key"):
            raise ValueError(
                f"API key required for {model_config['model_type']} models"
            )

    # Validate temperature
    if "temperature" in model_config:
        temp = model_config["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            raise ValueError("Temperature must be a number between 0 and 2")

    # Validate max tokens
    if "max_tokens" in model_config:
        max_tokens = model_config["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Max tokens must be a positive integer")

    return True


def validate_evaluation_config(eval_config: Dict[str, Any]) -> bool:
    """
    Validate evaluation configuration.

    Args:
        eval_config: Evaluation configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate criteria
    if "criteria" not in eval_config:
        raise ValueError("Missing evaluation criteria")

    criteria = eval_config["criteria"]
    if not isinstance(criteria, list) or len(criteria) == 0:
        raise ValueError("Evaluation criteria must be a non-empty list")

    for criterion in criteria:
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError("Each evaluation criterion must be a non-empty string")

    # Validate backend
    if "backend" in eval_config:
        valid_backends = ["llm", "human", "automated"]
        if eval_config["backend"] not in valid_backends:
            raise ValueError(f"Invalid evaluation backend: {eval_config['backend']}")

    # Validate batch size
    if "batch_size" in eval_config:
        batch_size = eval_config["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")

    return True


def validate_storage_config(storage_config: Dict[str, Any]) -> bool:
    """
    Validate storage configuration.

    Args:
        storage_config: Storage configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate paths
    path_fields = [
        "base_path",
        "feedback_path",
        "evolution_path",
        "logs_path",
        "cache_path",
    ]

    for field in path_fields:
        if field in storage_config:
            path = storage_config[field]
            if not isinstance(path, (str, Path)):
                raise ValueError(f"{field} must be a string or Path object")

    return True


def validate_prompt(prompt: str) -> bool:
    """
    Validate system prompt.

    Args:
        prompt: System prompt to validate

    Returns:
        True if valid

    Raises:
        ValueError: If prompt is invalid
    """
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string")

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if len(prompt) > 10000:
        raise ValueError("Prompt too long (max 10000 characters)")

    return True


def validate_problem(problem: Dict[str, Any]) -> bool:
    """
    Validate problem dictionary.

    Args:
        problem: Problem dictionary to validate

    Returns:
        True if valid

    Raises:
        ValueError: If problem is invalid
    """
    required_fields = ["description", "solution"]

    for field in required_fields:
        if field not in problem:
            raise ValueError(f"Missing required problem field: {field}")

    # Validate description
    if (
        not isinstance(problem["description"], str)
        or not problem["description"].strip()
    ):
        raise ValueError("Problem description must be a non-empty string")

    # Validate solution
    if not isinstance(problem["solution"], str) or not problem["solution"].strip():
        raise ValueError("Problem solution must be a non-empty string")

    return True


def validate_evaluation_result(result: Dict[str, Any]) -> bool:
    """
    Validate evaluation result.

    Args:
        result: Evaluation result dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If result is invalid
    """
    if not isinstance(result, dict):
        raise ValueError("Evaluation result must be a dictionary")

    # Check for required fields
    if "timestamp" not in result:
        raise ValueError("Missing timestamp in evaluation result")

    if "scores" not in result:
        raise ValueError("Missing scores in evaluation result")

    # Validate scores
    scores = result["scores"]
    if not isinstance(scores, dict):
        raise ValueError("Scores must be a dictionary")

    for criterion, score in scores.items():
        if not isinstance(criterion, str):
            raise ValueError("Criterion names must be strings")

        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            raise ValueError(f"Score for {criterion} must be a number between 0 and 1")

    return True


def validate_file_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate file path.

    Args:
        path: File path to validate
        must_exist: Whether the file must exist

    Returns:
        Path object

    Raises:
        ValueError: If path is invalid
    """
    if isinstance(path, str):
        path = Path(path)

    if not isinstance(path, Path):
        raise ValueError("Path must be a string or Path object")

    if must_exist and not path.exists():
        raise ValueError(f"File does not exist: {path}")

    return path


def validate_json_string(json_str: str) -> bool:
    """
    Validate JSON string.

    Args:
        json_str: JSON string to validate

    Returns:
        True if valid

    Raises:
        ValueError: If JSON is invalid
    """
    if not isinstance(json_str, str):
        raise ValueError("Input must be a string")

    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid

    Raises:
        ValueError: If email is invalid
    """
    if not isinstance(email, str):
        raise ValueError("Email must be a string")

    # Basic email regex pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    if not re.match(pattern, email):
        raise ValueError(f"Invalid email format: {email}")

    return True


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid

    Raises:
        ValueError: If URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    # Basic URL regex pattern
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"

    if not re.match(pattern, url):
        raise ValueError(f"Invalid URL format: {url}")

    return True
