"""
Script to run example problems through the system prompt learning framework.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from feedback_loop.collector import FeedbackCollector
from prompt_integration.processor import PromptProcessor
from examples.problems import get_all_problems
from examples.system_prompts import get_all_prompts
from examples.logger import logger
from tqdm import tqdm
import time

def log_evaluation(evaluation: dict, criteria: list):
    """Log detailed evaluation results."""
    logger.info("Evaluation Results:")
    logger.info("-" * 30)
    for criterion in criteria:
        score = evaluation.get(criterion, 0)
        logger.info(f"{criterion:<15}: {score:.2f}")
    logger.info("-" * 30)

def log_reflection(reflection: str):
    """Log reflection with formatting."""
    logger.info("\nReflection:")
    logger.info("-" * 30)
    logger.info(reflection)
    logger.info("-" * 30)

def log_insights(insights: list):
    """Log extracted insights with detailed information."""
    logger.info("\nExtracted Insights:")
    logger.info("=" * 50)
    for insight in insights:
        logger.info(f"\nInsight: {insight['insight']}")
        logger.info(f"Support Count: {insight['support_count']}")
        logger.info("-" * 30)

def log_prompt_evolution(current_prompt: str, updated_prompt: str):
    """Log the evolution of the system prompt."""
    logger.info("\nSystem Prompt Evolution:")
    logger.info("=" * 50)
    
    # Find new lines in the updated prompt
    current_lines = set(current_prompt.split('\n'))
    updated_lines = set(updated_prompt.split('\n'))
    new_lines = updated_lines - current_lines
    
    if new_lines:
        logger.info("\nNew Insights Added:")
        for line in new_lines:
            if line.strip():
                logger.info(f"+ {line}")
    else:
        logger.info("\nNo new insights added in this iteration")
    
    logger.info("\nFull Updated Prompt:")
    logger.info("-" * 30)
    logger.info(updated_prompt)
    logger.info("-" * 30)

async def run_problems_with_prompt(initial_prompt: str, prompt_type: str):
    """Run problems with a specific initial prompt."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting evaluation with prompt type: {prompt_type}")
    logger.info(f"{'='*50}\n")
    
    # Initialize components
    feedback_collector = FeedbackCollector()
    prompt_processor = PromptProcessor()
    
    # Get all problems
    problems = get_all_problems()
    
    # Evaluation criteria
    criteria = [
        "correctness",
        "efficiency",
        "readability",
        "maintainability",
        "error_handling",
        "documentation"
    ]
    
    # Process each problem
    for i, problem in enumerate(problems, 1):
        logger.info(f"\nProblem {i}: {problem['name']}")
        logger.info("=" * 50)
        logger.info(f"Description: {problem['description']}")
        
        # Collect feedback
        feedback = feedback_collector.collect_solution_feedback(
            problem=problem['description'],
            solution=problem['solution'],
            evaluation_criteria=criteria
        )
        
        # Log evaluation results
        log_evaluation(feedback['evaluation'], criteria)
        
        # Log reflection
        log_reflection(feedback['reflection'])
        
        # Process feedback and update system prompt
        insights = prompt_processor.process_feedback()
        updated_prompt = prompt_processor.update_system_prompt(insights)
        
        # Log insights
        log_insights(insights)
        
        # Log prompt evolution
        current_prompt = prompt_processor._load_current_prompt()
        log_prompt_evolution(current_prompt, updated_prompt)
        
        # Add delay to avoid rate limiting
        await asyncio.sleep(1)
    
    # Save run information
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "prompt_type": prompt_type,
        "initial_prompt": initial_prompt,
        "final_prompt": prompt_processor._load_current_prompt(),
        "problems_evaluated": len(problems),
        "insights_generated": len(insights),
        "evaluation_criteria": criteria,
        "final_insights": insights
    }
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Save run information
    run_file = log_dir / f"run_info_{prompt_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(run_file, "w") as f:
        json.dump(run_info, f, indent=2)
    
    logger.info(f"\nRun information saved to {run_file}")
    return run_info

async def run_all_prompts():
    """Run problems with all initial prompts."""
    all_prompts = get_all_prompts()
    results = {}
    
    for prompt_type, initial_prompt in all_prompts.items():
        try:
            result = await run_problems_with_prompt(initial_prompt, prompt_type)
            results[prompt_type] = result
        except Exception as e:
            logger.error(f"Error running problems with prompt type {prompt_type}: {str(e)}")
            results[prompt_type] = {"error": str(e)}
    
    # Save combined results
    log_dir = Path("logs")
    combined_file = log_dir / f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nCombined results saved to {combined_file}")
    return results

if __name__ == "__main__":
    asyncio.run(run_all_prompts()) 