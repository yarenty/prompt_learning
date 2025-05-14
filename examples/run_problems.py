"""
Script to run example problems through the system prompt learning framework.
"""
from feedback_loop.collector import FeedbackCollector
from prompt_integration.processor import PromptProcessor
from examples.problems import get_all_problems
from tqdm import tqdm
import time

def run_problems():
    # Initialize components
    collector = FeedbackCollector(model="codellama")
    processor = PromptProcessor(model="codellama")
    
    # Get all problems
    problems = get_all_problems()
    
    # Common evaluation criteria
    evaluation_criteria = [
        "correctness",
        "efficiency",
        "readability",
        "maintainability",
        "error_handling",
        "documentation"
    ]
    
    print(f"Running {len(problems)} problems through the system...")
    
    # Process each problem
    for i, problem in enumerate(tqdm(problems, desc="Processing problems")):
        print(f"\n\nProblem {i+1}: {problem['name']}")
        print("-" * 50)
        print(f"Description: {problem['description'].strip()}")
        
        # Collect feedback
        feedback = collector.collect_solution_feedback(
            problem=problem['description'],
            solution=problem['solution'],
            evaluation_criteria=evaluation_criteria
        )
        
        print("\nFeedback collected:")
        print(f"Evaluation: {feedback['evaluation']}")
        print(f"Reflection: {feedback['reflection']}")
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Process all feedback and update system prompt
    print("\n\nProcessing all feedback...")
    insights = processor.process_feedback()
    
    print("\nExtracted insights:")
    for insight in insights:
        print(f"- {insight['insight']} (supported by {insight['support_count']} cases)")
    
    print("\nUpdating system prompt...")
    updated_prompt = processor.update_system_prompt(insights)
    
    print("\nFinal system prompt:")
    print("-" * 50)
    print(updated_prompt)

if __name__ == "__main__":
    run_problems() 