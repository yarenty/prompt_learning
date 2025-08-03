"""Evaluation system for benchmarking."""

import ast
import asyncio
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


class Evaluator:
    """Evaluator for different types of tasks."""

    def __init__(self):
        """Initialize evaluator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    async def evaluate_programming(self, generated: str, reference: Dict, timeout: int = 5) -> Dict[str, float]:
        """Evaluate programming solution.

        Args:
            generated: Generated code
            reference: Reference with test cases
            timeout: Execution timeout in seconds

        Returns:
            Metrics including:
                - correctness: Fraction of passing tests
                - efficiency: Runtime performance
                - quality: Code quality metrics
        """
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            f.write(generated.encode())
            f.flush()

            # Run tests
            results = []
            for test in reference["test"]:
                try:
                    # Execute with timeout
                    proc = await asyncio.create_subprocess_exec(
                        "python",
                        f.name,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    stdout, stderr = await asyncio.wait_for(proc.communicate(test["input"].encode()), timeout=timeout)

                    # Check output
                    output = stdout.decode().strip()
                    expected = test["output"].strip()
                    results.append(output == expected)

                except (asyncio.TimeoutError, Exception):
                    results.append(False)

        # Code quality metrics
        try:
            tree = ast.parse(generated)
            quality = {
                "cyclomatic_complexity": self._calculate_complexity(tree),
                "maintainability": self._calculate_maintainability(tree),
            }
        except:
            quality = {"cyclomatic_complexity": 0, "maintainability": 0}

        return {
            "correctness": sum(results) / len(results),
            "efficiency": 1.0 if all(results) else 0.0,
            "quality": quality,
        }

    async def evaluate_math(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate math solution.

        Args:
            generated: Generated solution
            reference: Reference solution

        Returns:
            Metrics including:
                - correctness: Answer matches
                - step_accuracy: Step-by-step accuracy
        """
        # Extract final answer
        gen_answer = self._extract_answer(generated)
        ref_answer = self._extract_answer(reference)

        # Compare answers
        try:
            gen_num = float(gen_answer)
            ref_num = float(ref_answer)
            correct = math.isclose(gen_num, ref_num, rel_tol=1e-3)
        except:
            correct = gen_answer == ref_answer

        # Compare solution steps
        gen_steps = self._extract_steps(generated)
        ref_steps = self._extract_steps(reference)

        step_scores = []
        for gen, ref in zip(gen_steps, ref_steps):
            step_scores.append(self._compare_steps(gen, ref))

        return {"correctness": float(correct), "step_accuracy": np.mean(step_scores) if step_scores else 0.0}

    async def evaluate_writing(
        self, generated: str, reference: str, criteria: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Evaluate writing solution.

        Args:
            generated: Generated text
            reference: Reference text
            criteria: Evaluation criteria

        Returns:
            Metrics including:
                - relevance: Content relevance
                - fluency: Language fluency
                - creativity: Creative aspects
        """
        # Calculate BLEU score
        bleu = sentence_bleu([reference.split()], generated.split())

        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, generated)

        # Evaluate against criteria if provided
        criteria_scores = {}
        if criteria:
            for criterion, rubric in criteria.items():
                score = self._evaluate_criterion(generated, rubric)
                criteria_scores[criterion] = score

        return {
            "relevance": rouge_scores["rougeL"].fmeasure,
            "fluency": bleu,
            "creativity": np.mean(list(criteria_scores.values())) if criteria_scores else 0.0,
        }

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
                complexity += 1
        return complexity

    def _calculate_maintainability(self, tree: ast.AST) -> float:
        """Calculate maintainability index."""
        # Simplified version
        loc = len(ast.unparse(tree).split("\n"))
        cc = self._calculate_complexity(tree)
        return max(0, (171 - 5.2 * math.log(loc) - 0.23 * cc - 16.2) / 171)

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from solution."""
        # Look for patterns like "#### 42" or "The answer is 42"
        patterns = [r"####\s*(\d+\.?\d*)", r"answer\s*(?:is|:)\s*(\d+\.?\d*)", r"=\s*(\d+\.?\d*)$"]

        for pattern in patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                return match.group(1)

        return text.strip()

    def _extract_steps(self, text: str) -> List[str]:
        """Extract solution steps."""
        # Split on common step markers
        steps = re.split(r"\d+\.|Step|First|Then|Finally|\n\s*\n", text)
        return [s.strip() for s in steps if s.strip()]

    def _compare_steps(self, step1: str, step2: str) -> float:
        """Compare similarity of solution steps."""
        # Use ROUGE-L for step comparison
        score = self.rouge_scorer.score(step1, step2)
        return score["rougeL"].fmeasure

    def _evaluate_criterion(self, text: str, rubric: Dict) -> float:
        """Evaluate text against a criterion rubric."""
        # Simple keyword/pattern matching for now
        # Could be replaced with more sophisticated NLP
        score = 0
        for level, descriptors in rubric.items():
            for descriptor in descriptors:
                if re.search(descriptor, text, re.IGNORECASE):
                    score = float(level) / len(rubric)
        return score
