"""
Validation utilities for core Memento framework components.

This module provides validation functions for inputs, configurations,
and data structures used throughout the framework.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from ..exceptions import ValidationError


def validate_prompt(prompt: str) -> str:
    """
    Validate a system prompt.

    Args:
        prompt: The prompt to validate

    Returns:
        The validated prompt

    Raises:
        ValidationError: If prompt is invalid
    """
    if not isinstance(prompt, str):
        raise ValidationError("Prompt must be a string")

    if not prompt.strip():
        raise ValidationError("Prompt cannot be empty")

    if len(prompt) > 10000:
        raise ValidationError("Prompt is too long (max 10000 characters)")

    return prompt.strip()


def validate_problem(problem: Dict[str, str]) -> Dict[str, str]:
    """
    Validate a problem dictionary.

    Args:
        problem: The problem dictionary to validate

    Returns:
        The validated problem dictionary

    Raises:
        ValidationError: If problem is invalid
    """
    if not isinstance(problem, dict):
        raise ValidationError("Problem must be a dictionary")

    required_fields = ["description", "solution"]
    for field in required_fields:
        if field not in problem:
            raise ValidationError(f"Problem missing required field: {field}")

        if not isinstance(problem[field], str):
            raise ValidationError(f"Problem field '{field}' must be a string")

        if not problem[field].strip():
            raise ValidationError(f"Problem field '{field}' cannot be empty")

    # Validate optional fields
    optional_fields = ["category", "difficulty", "tags"]
    for field in optional_fields:
        if field in problem and not isinstance(problem[field], str):
            raise ValidationError(f"Problem field '{field}' must be a string")

    return {
        "description": problem["description"].strip(),
        "solution": problem["solution"].strip(),
        **{k: v.strip() for k, v in problem.items() if k in optional_fields and v},
    }


def validate_evaluation_criteria(criteria: List[str]) -> List[str]:
    """
    Validate evaluation criteria list.

    Args:
        criteria: The criteria list to validate

    Returns:
        The validated criteria list

    Raises:
        ValidationError: If criteria are invalid
    """
    if not isinstance(criteria, list):
        raise ValidationError("Evaluation criteria must be a list")

    if not criteria:
        raise ValidationError("Evaluation criteria cannot be empty")

    valid_criteria = {
        "correctness",
        "efficiency",
        "readability",
        "maintainability",
        "error_handling",
        "documentation",
        "clarity",
        "creativity",
        "robustness",
        "scalability",
        "security",
        "performance",
    }

    validated_criteria = []
    for criterion in criteria:
        if not isinstance(criterion, str):
            raise ValidationError("Each criterion must be a string")

        criterion = criterion.lower().strip()
        if not criterion:
            raise ValidationError("Criterion cannot be empty")

        if criterion not in valid_criteria:
            raise ValidationError(f"Invalid criterion: {criterion}")

        validated_criteria.append(criterion)

    return validated_criteria


def validate_evaluation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an evaluation result.

    Args:
        result: The evaluation result to validate

    Returns:
        The validated evaluation result

    Raises:
        ValidationError: If result is invalid
    """
    if not isinstance(result, dict):
        raise ValidationError("Evaluation result must be a dictionary")

    # Validate scores
    for criterion, score in result.items():
        if not isinstance(criterion, str):
            raise ValidationError("Criterion names must be strings")

        if isinstance(score, (int, float)):
            if not 0.0 <= score <= 1.0:
                raise ValidationError(f"Score for '{criterion}' must be between 0.0 and 1.0")
        elif isinstance(score, dict):
            # Handle detailed evaluation format
            if "score" in score:
                score_val = score["score"]
                if not isinstance(score_val, (int, float)) or not 0.0 <= score_val <= 1.0:
                    raise ValidationError(f"Score for '{criterion}' must be between 0.0 and 1.0")
        else:
            raise ValidationError(f"Score for '{criterion}' must be a number or detailed dict")

    return result


def validate_lesson(lesson: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a lesson learned.

    Args:
        lesson: The lesson to validate

    Returns:
        The validated lesson

    Raises:
        ValidationError: If lesson is invalid
    """
    if not isinstance(lesson, dict):
        raise ValidationError("Lesson must be a dictionary")

    required_fields = ["criterion", "lesson"]
    for field in required_fields:
        if field not in lesson:
            raise ValidationError(f"Lesson missing required field: {field}")

        if not isinstance(lesson[field], str):
            raise ValidationError(f"Lesson field '{field}' must be a string")

        if not lesson[field].strip():
            raise ValidationError(f"Lesson field '{field}' cannot be empty")

    # Validate optional fields
    if "score" in lesson:
        score = lesson["score"]
        if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            raise ValidationError("Lesson score must be between 0.0 and 1.0")

    if "explanation" in lesson and not isinstance(lesson["explanation"], str):
        raise ValidationError("Lesson explanation must be a string")

    return {
        "criterion": lesson["criterion"].strip(),
        "lesson": lesson["lesson"].strip(),
        "score": lesson.get("score", 0.0),
        "explanation": lesson.get("explanation", "").strip(),
    }


def validate_insight(insight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an insight.

    Args:
        insight: The insight to validate

    Returns:
        The validated insight

    Raises:
        ValidationError: If insight is invalid
    """
    if not isinstance(insight, dict):
        raise ValidationError("Insight must be a dictionary")

    required_fields = ["insight"]
    for field in required_fields:
        if field not in insight:
            raise ValidationError(f"Insight missing required field: {field}")

        if not isinstance(insight[field], str):
            raise ValidationError(f"Insight field '{field}' must be a string")

        if not insight[field].strip():
            raise ValidationError(f"Insight field '{field}' cannot be empty")

    # Validate optional fields
    if "support_count" in insight:
        support_count = insight["support_count"]
        if not isinstance(support_count, int) or support_count < 1:
            raise ValidationError("Support count must be a positive integer")

    if "cluster_id" in insight and not isinstance(insight["cluster_id"], (str, int)):
        raise ValidationError("Cluster ID must be a string or integer")

    return {
        "insight": insight["insight"].strip(),
        "support_count": insight.get("support_count", 1),
        "cluster_id": insight.get("cluster_id", "unknown"),
    }


def validate_storage_path(path: Union[str, Path]) -> Path:
    """
    Validate a storage path.

    Args:
        path: The path to validate

    Returns:
        The validated Path object

    Raises:
        ValidationError: If path is invalid
    """
    try:
        path_obj = Path(path)
    except Exception as e:
        raise ValidationError(f"Invalid path: {e}")

    # Check if path is absolute or can be made absolute
    try:
        path_obj.resolve()
    except Exception as e:
        raise ValidationError(f"Cannot resolve path: {e}")

    return path_obj


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model configuration.

    Args:
        config: The model configuration to validate

    Returns:
        The validated configuration

    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Model configuration must be a dictionary")

    required_fields = ["model_type", "model_name"]
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Model configuration missing required field: {field}")

        if not isinstance(config[field], str):
            raise ValidationError(f"Model configuration field '{field}' must be a string")

        if not config[field].strip():
            raise ValidationError(f"Model configuration field '{field}' cannot be empty")

    # Validate model type
    valid_types = {"ollama", "openai", "anthropic", "local"}
    model_type = config["model_type"].lower()
    if model_type not in valid_types:
        raise ValidationError(f"Invalid model type: {model_type}")

    # Validate optional fields
    if "temperature" in config:
        temp = config["temperature"]
        if not isinstance(temp, (int, float)) or not 0.0 <= temp <= 2.0:
            raise ValidationError("Temperature must be between 0.0 and 2.0")

    if "max_tokens" in config:
        max_tokens = config["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValidationError("Max tokens must be a positive integer")

    if "timeout" in config:
        timeout = config["timeout"]
        if not isinstance(timeout, (int, float)) or timeout < 1:
            raise ValidationError("Timeout must be a positive number")

    return {
        "model_type": model_type,
        "model_name": config["model_name"].strip(),
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 2048),
        "timeout": config.get("timeout", 30),
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
    }


def validate_json_response(response: str) -> Dict[str, Any]:
    """
    Validate and parse a JSON response.

    Args:
        response: The JSON response string to validate

    Returns:
        The parsed JSON object

    Raises:
        ValidationError: If response is invalid JSON
    """
    if not isinstance(response, str):
        raise ValidationError("Response must be a string")

    if not response.strip():
        raise ValidationError("Response cannot be empty")

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON response: {e}")

    if not isinstance(parsed, dict):
        raise ValidationError("JSON response must be an object")

    return parsed


def validate_feedback_data(feedback: Dict[str, Any]) -> None:
    """
    Validate comprehensive feedback data structure.

    Args:
        feedback: Feedback data dictionary

    Raises:
        ValidationError: If feedback data is invalid
    """
    if not isinstance(feedback, dict):
        raise ValidationError("Feedback data must be a dictionary")

    required_fields = ["timestamp", "problem", "solution", "evaluation", "reflection"]
    for field in required_fields:
        if field not in feedback:
            raise ValidationError(f"Missing required field: {field}")

    # Validate timestamp format
    try:
        from datetime import datetime

        datetime.fromisoformat(feedback["timestamp"])
    except ValueError as e:
        raise ValidationError(f"Invalid timestamp format: {e}") from e

    # Validate evaluation structure
    evaluation = feedback["evaluation"]
    if not isinstance(evaluation, dict):
        raise ValidationError("Evaluation must be a dictionary")

    for criterion, score in evaluation.items():
        if not isinstance(score, (int, float)):
            raise ValidationError(f"Evaluation score for '{criterion}' must be numeric")
        if not 0.0 <= score <= 1.0:
            raise ValidationError(f"Evaluation score for '{criterion}' must be between 0.0 and 1.0")

    # Validate reflection
    if not isinstance(feedback["reflection"], str) or not feedback["reflection"].strip():
        raise ValidationError("Reflection must be a non-empty string")
