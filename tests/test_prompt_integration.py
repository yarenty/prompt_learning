"""
Test cases for the prompt integration system.
"""
import pytest
from pathlib import Path
import json
import shutil
from prompt_integration.processor import PromptProcessor

@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for feedback and prompts."""
    feedback_dir = tmp_path / "feedback"
    prompt_dir = tmp_path / "prompts"
    feedback_dir.mkdir()
    prompt_dir.mkdir()
    yield feedback_dir, prompt_dir
    shutil.rmtree(feedback_dir)
    shutil.rmtree(prompt_dir)

@pytest.fixture
def processor(temp_dirs):
    """Create a PromptProcessor instance with temporary storage."""
    feedback_dir, prompt_dir = temp_dirs
    return PromptProcessor(
        feedback_path=str(feedback_dir),
        prompt_path=str(prompt_dir)
    )

def test_process_feedback(processor, temp_dirs):
    """Test processing feedback and extracting insights."""
    feedback_dir, _ = temp_dirs
    
    # Create sample feedback files
    sample_feedback = [
        {
            "timestamp": "2024-01-01T00:00:00",
            "problem": "Test problem 1",
            "solution": "Test solution 1",
            "evaluation": {"correctness": 0.8},
            "reflection": "When dealing with list operations, consider edge cases first."
        },
        {
            "timestamp": "2024-01-01T00:01:00",
            "problem": "Test problem 2",
            "solution": "Test solution 2",
            "evaluation": {"correctness": 0.9},
            "reflection": "Always check for empty lists before processing."
        }
    ]
    
    for i, feedback in enumerate(sample_feedback):
        file_path = feedback_dir / f"feedback_20240101_000{i}.json"
        with open(file_path, "w") as f:
            json.dump(feedback, f)
    
    insights = processor.process_feedback()
    
    assert isinstance(insights, list)
    assert len(insights) > 0
    assert all("cluster_id" in insight for insight in insights)
    assert all("insight" in insight for insight in insights)
    assert all("support_count" in insight for insight in insights)

def test_update_system_prompt(processor, temp_dirs):
    """Test updating the system prompt with new insights."""
    _, prompt_dir = temp_dirs
    
    # Create initial prompt
    initial_prompt = "# Initial System Prompt\n\nBase instructions."
    current_prompt_file = prompt_dir / "current_prompt.txt"
    current_prompt_file.write_text(initial_prompt)
    
    # Test insights
    insights = [
        {
            "cluster_id": 0,
            "insight": "Test insight 1",
            "support_count": 3
        },
        {
            "cluster_id": 1,
            "insight": "Test insight 2",
            "support_count": 1  # Should be filtered out
        }
    ]
    
    updated_prompt = processor.update_system_prompt(insights)
    
    # Check prompt content
    assert initial_prompt in updated_prompt
    assert "Test insight 1" in updated_prompt
    assert "Test insight 2" not in updated_prompt
    
    # Check file storage
    prompt_files = list(prompt_dir.glob("prompt_*.txt"))
    assert len(prompt_files) == 1
    assert (prompt_dir / "current_prompt.txt").exists()

def test_load_current_prompt(processor, temp_dirs):
    """Test loading the current system prompt."""
    _, prompt_dir = temp_dirs
    
    # Test with no existing prompt
    prompt = processor._load_current_prompt()
    assert prompt == "# Initial System Prompt\n\n"
    
    # Test with existing prompt
    test_prompt = "# Test Prompt\n\nTest content."
    current_prompt_file = prompt_dir / "current_prompt.txt"
    current_prompt_file.write_text(test_prompt)
    
    prompt = processor._load_current_prompt()
    assert prompt == test_prompt 