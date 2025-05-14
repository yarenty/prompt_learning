"""
Example script demonstrating the system prompt learning framework.
"""
from feedback_loop.collector import FeedbackCollector
from prompt_integration.processor import PromptProcessor

def main():
    # Initialize components
    collector = FeedbackCollector()
    processor = PromptProcessor()
    
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
    
    print("\nUpdating system prompt...")
    updated_prompt = processor.update_system_prompt(insights)
    
    print("\nUpdated system prompt:")
    print(updated_prompt)

if __name__ == "__main__":
    main() 