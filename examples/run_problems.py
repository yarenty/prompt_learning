"""
Script to run example problems through the system prompt learning framework.
"""
from feedback_loop.collector import FeedbackCollector
from prompt_integration.processor import PromptProcessor
from examples.problems import get_all_problems
from examples.logger import logger
from tqdm import tqdm
import time
import json
from datetime import datetime

def log_evaluation(evaluation: dict, criteria: list):
    """Log detailed evaluation results."""
    logger.info("\nEvaluation Results:")
    logger.info("-" * 30)
    for criterion in criteria:
        score = evaluation.get(criterion, 0.0)
        logger.info(f"{criterion:15} : {score:.2f}")
    logger.info("-" * 30)

def log_reflection(reflection: str):
    """Log reflection with formatting."""
    logger.info("\nReflection:")
    logger.info("-" * 30)
    logger.info(reflection)
    logger.info("-" * 30)

def log_insights(insights: list):
    """Log insights with detailed information."""
    logger.info("\nExtracted Insights:")
    logger.info("=" * 50)
    for insight in insights:
        logger.info(f"\nInsight #{insight['cluster_id']}")
        logger.info(f"Support Count: {insight['support_count']}")
        logger.info("-" * 30)
        logger.info(insight['insight'])
        logger.info("-" * 30)

def log_prompt_evolution(current_prompt: str, updated_prompt: str):
    """Log the evolution of the system prompt."""
    logger.info("\nSystem Prompt Evolution:")
    logger.info("=" * 50)
    
    # Log the changes
    current_lines = set(current_prompt.split('\n'))
    updated_lines = set(updated_prompt.split('\n'))
    
    added_lines = updated_lines - current_lines
    if added_lines:
        logger.info("\nNew Insights Added:")
        for line in added_lines:
            if line.strip():
                logger.info(f"+ {line}")
    
    logger.info("\nFull Updated Prompt:")
    logger.info("=" * 50)
    logger.info(updated_prompt)

def run_problems():
    # Initialize components
    logger.info("Initializing system components...")
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
    
    logger.info(f"Starting evaluation of {len(problems)} problems...")
    
    # Process each problem
    for i, problem in enumerate(tqdm(problems, desc="Processing problems")):
        logger.info(f"\n\nProblem {i+1}: {problem['name']}")
        logger.info("=" * 50)
        logger.info(f"Description: {problem['description'].strip()}")
        
        # Log the solution being evaluated
        logger.debug("\nSolution being evaluated:")
        logger.debug(problem['solution'])
        
        # Collect feedback
        logger.info("\nCollecting feedback...")
        feedback = collector.collect_solution_feedback(
            problem=problem['description'],
            solution=problem['solution'],
            evaluation_criteria=evaluation_criteria
        )
        
        # Log detailed feedback
        log_evaluation(feedback['evaluation'], evaluation_criteria)
        log_reflection(feedback['reflection'])
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Process all feedback and update system prompt
    logger.info("\n\nProcessing all feedback...")
    insights = processor.process_feedback()
    
    # Log insights
    log_insights(insights)
    
    # Get current prompt before update
    current_prompt = processor._load_current_prompt()
    
    # Update system prompt
    logger.info("\nUpdating system prompt...")
    updated_prompt = processor.update_system_prompt(insights)
    
    # Log prompt evolution
    log_prompt_evolution(current_prompt, updated_prompt)
    
    # Save detailed run information
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "problems_evaluated": len(problems),
        "insights_generated": len(insights),
        "evaluation_criteria": evaluation_criteria,
        "final_insights": insights
    }
    
    with open("logs/run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

if __name__ == "__main__":
    run_problems() 