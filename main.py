"""
Example script demonstrating the system prompt learning framework with Ollama.
"""
from feedback_loop.collector import FeedbackCollector
from prompt_integration.processor import PromptProcessor
import json
from pathlib import Path

def initialize_system_prompt(prompt_path: str = "data/prompts"):
    """Initialize the system prompt with base content."""
    prompt_dir = Path(prompt_path)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    initial_prompt = """# System Prompt for Code Generation

You are an expert programmer who writes clean, efficient, and maintainable code.
Your solutions should be:
1. Correct and handle edge cases
2. Efficient in terms of time and space complexity
3. Readable with clear variable names and comments
4. Maintainable and follow best practices

When solving problems:
1. First understand the requirements thoroughly
2. Consider edge cases and error handling
3. Choose appropriate data structures and algorithms
4. Write clear, self-documenting code
5. Test your solution with various inputs
"""
    
    current_prompt_file = prompt_dir / "current_prompt.txt"
    current_prompt_file.write_text(initial_prompt)
    return initial_prompt

def main():
    # Initialize system prompt
    initialize_system_prompt()
    
    # Initialize components with CodeLlama model
    collector = FeedbackCollector(model="codellama")
    processor = PromptProcessor(model="codellama")
    
    # Example problem and solution
    problem = """
    Write a function that takes a list of numbers and returns a new list
    containing only the even numbers, maintaining their original order.
    """
    
    solution = """
    def get_even_numbers(numbers):
        return [num for num in numbers if num % 2 == 0]
    """
    
    # Collect feedback
    evaluation_criteria = [
        "correctness",
        "efficiency",
        "readability",
        "maintainability"
    ]
    
    print("Collecting feedback...")
    feedback = collector.collect_solution_feedback(
        problem=problem,
        solution=solution,
        evaluation_criteria=evaluation_criteria
    )
    
    print("\nFeedback collected:")
    print(f"Evaluation: {feedback['evaluation']}")
    print(f"Reflection: {feedback['reflection']}")
    
    # Process feedback and update system prompt
    print("\nProcessing feedback...")
    insights = processor.process_feedback()
    
    print("\nExtracted insights:")
    for insight in insights:
        print(f"- {insight['insight']} (supported by {insight['support_count']} cases)")
    
    # Format insights for prompt update
    principles_text = "\n\n".join([
        f"# Insight from {insight['support_count']} similar cases:\n{insight['insight']}"
        for insight in insights if insight['support_count'] >= 2
    ])
    
    print("\nUpdating system prompt...")
    update_prompt = f"""
    You are updating an AI system's prompt to incorporate new problem-solving principles.
    
    CURRENT SYSTEM PROMPT:
    {processor._load_current_prompt()}
    
    NEW PRINCIPLES TO INCORPORATE:
    {principles_text}
    
    Please create an updated system prompt that smoothly integrates these new principles.
    The updated prompt should be coherent, well-structured, and maintain the original prompt's purpose and tone.
    Do not simply append the principles at the end - integrate them naturally into the appropriate sections.
    """
    
    updated_prompt = processor.update_system_prompt(insights)
    
    print("\nUpdated system prompt:")
    print(updated_prompt)

if __name__ == "__main__":
    main() 
    
    
    