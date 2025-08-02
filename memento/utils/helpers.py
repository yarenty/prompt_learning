"""
Helper utilities for the Memento framework.

This module provides common helper functions used throughout the framework.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def format_timestamp(timestamp: Optional[Union[str, datetime]] = None) -> str:
    """
    Format timestamp in ISO format.

    Args:
        timestamp: Timestamp to format (defaults to current time)

    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    return timestamp.isoformat()


def safe_json_load(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with error handling.

    Args:
        json_str: JSON string to parse
        default: Default value to return on error

    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dump(obj: Any, default: str = "{}") -> str:
    """
    Safely dump object to JSON string with error handling.

    Args:
        obj: Object to serialize
        default: Default string to return on error

    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default


def generate_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of content.

    Args:
        content: Content to hash
        algorithm: Hash algorithm to use

    Returns:
        Hexadecimal hash string
    """
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    return hash_func(content.encode("utf-8")).hexdigest()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"

    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        sanitized = name[: 255 - len(ext) - 1] + ("." + ext if ext else "")

    return sanitized


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Unflatten dictionary with dot notation.

    Args:
        d: Flattened dictionary
        sep: Separator for nested keys

    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    return result


def merge_dicts(
    dict1: Dict[str, Any], dict2: Dict[str, Any], deep: bool = True
) -> Dict[str, Any]:
    """
    Merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        deep: Whether to perform deep merge

    Returns:
        Merged dictionary
    """
    if not deep:
        return {**dict1, **dict2}

    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    return result


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def remove_duplicates(lst: List[Any], key: Optional[callable] = None) -> List[Any]:
    """
    Remove duplicates from list while preserving order.

    Args:
        lst: List to deduplicate
        key: Function to extract key for comparison

    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    for item in lst:
        item_key = key(item) if key else item
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    return result


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to specified length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def extract_code_blocks(text: str) -> List[str]:
    """
    Extract code blocks from text.

    Args:
        text: Text containing code blocks

    Returns:
        List of code blocks
    """
    # Pattern for markdown code blocks
    pattern = r"```(?:\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_functions_from_code(code: str, language: str = "python") -> List[str]:
    """
    Extract function definitions from code.

    Args:
        code: Source code
        language: Programming language

    Returns:
        List of function definitions
    """
    if language.lower() == "python":
        pattern = r"def\s+\w+\s*\([^)]*\)\s*:.*?(?=def|\Z)"
        matches = re.findall(pattern, code, re.DOTALL)
        return [match.strip() for match in matches]

    # Add patterns for other languages as needed
    return []


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    # Tokenize texts
    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))

    if not tokens1 and not tokens2:
        return 1.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union) if union else 0.0


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0.0

    return file_path.stat().st_size / (1024 * 1024)


def is_valid_python_code(code: str) -> bool:
    """
    Check if code is valid Python syntax.

    Args:
        code: Python code to validate

    Returns:
        True if valid Python code
    """
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False
