"""Task-specific metrics for different evaluation domains."""

import ast
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


class ProgrammingMetrics:
    """Programming-specific metrics."""
    
    def __init__(self):
        """Initialize programming metrics."""
        self.test_timeout = 5  # seconds
        
    def calculate_pass_at_k(
        self,
        generated_code: str,
        test_cases: List[Dict[str, Any]],
        k: int = 1
    ) -> float:
        """Calculate pass@k metric for code evaluation.
        
        Args:
            generated_code: Generated code solution
            test_cases: List of test cases with inputs and expected outputs
            k: Number of attempts
            
        Returns:
            Pass@k score (fraction of successful runs)
        """
        if not test_cases:
            return 0.0
            
        success_count = 0
        
        for _ in range(k):
            passed_tests = 0
            
            for test_case in test_cases:
                if self._run_test_case(generated_code, test_case):
                    passed_tests += 1
                    
            # Consider successful if all tests pass
            if passed_tests == len(test_cases):
                success_count += 1
                
        return success_count / k
        
    def calculate_code_quality(self, code: str) -> Dict[str, float]:
        """Calculate code quality metrics.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            tree = ast.parse(code)
            
            return {
                "complexity": self._calculate_complexity(tree),
                "maintainability": self._calculate_maintainability(tree),
                "style_score": self._calculate_style_score(code),
                "documentation_score": self._calculate_documentation_score(code),
                "readability_score": self._calculate_readability_score(code)
            }
        except SyntaxError:
            return {
                "complexity": 0.0,
                "maintainability": 0.0,
                "style_score": 0.0,
                "documentation_score": 0.0,
                "readability_score": 0.0
            }
            
    def calculate_runtime_metrics(
        self,
        code: str,
        test_inputs: List[Any]
    ) -> Dict[str, float]:
        """Calculate runtime performance metrics.
        
        Args:
            code: Code to analyze
            test_inputs: List of test inputs
            
        Returns:
            Runtime performance metrics
        """
        if not test_inputs:
            return {"average_time": 0.0, "memory_usage": 0.0}
            
        times = []
        memory_usages = []
        
        for test_input in test_inputs:
            runtime_info = self._measure_runtime(code, test_input)
            if runtime_info:
                times.append(runtime_info["time"])
                memory_usages.append(runtime_info["memory"])
                
        return {
            "average_time": np.mean(times) if times else 0.0,
            "max_time": np.max(times) if times else 0.0,
            "min_time": np.min(times) if times else 0.0,
            "average_memory": np.mean(memory_usages) if memory_usages else 0.0,
            "max_memory": np.max(memory_usages) if memory_usages else 0.0
        }
        
    def _run_test_case(self, code: str, test_case: Dict[str, Any]) -> bool:
        """Run a single test case."""
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Prepare test code
                test_code = f"""
{code}

# Test case
try:
    inputs = {test_case.get('inputs', [])}
    expected = {test_case.get('expected')}
    
    # Extract function name from code (simple heuristic)
    import ast
    tree = ast.parse('''{code}''')
    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break
    
    if func_name:
        result = globals()[func_name](*inputs) if inputs else globals()[func_name]()
        print("PASS" if result == expected else "FAIL")
    else:
        print("FAIL")  # No function found
        
except Exception as e:
    print("FAIL")
"""
                f.write(test_code)
                f.flush()
                
                # Run the test
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout
                )
                
                Path(f.name).unlink()  # Clean up
                
                return "PASS" in result.stdout
                
        except Exception:
            return False
            
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
                
        # Normalize to 0-1 scale (assuming max complexity of 20)
        return min(1.0, complexity / 20.0)
        
    def _calculate_maintainability(self, tree: ast.AST) -> float:
        """Calculate maintainability index."""
        try:
            # Lines of code
            loc = len(ast.unparse(tree).split('\n'))
            
            # Cyclomatic complexity
            cc = self._calculate_complexity(tree) * 20  # Denormalize
            
            # Halstead volume (simplified)
            operators = 0
            operands = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.operator):
                    operators += 1
                elif isinstance(node, (ast.Name, ast.Constant)):
                    operands += 1
                    
            if operators + operands == 0:
                hv = 0
            else:
                hv = (operators + operands) * math.log2(operators + operands + 1)
                
            # Maintainability Index formula
            mi = max(0, (171 - 5.2 * math.log(hv + 1) - 0.23 * cc - 16.2 * math.log(loc + 1)) / 171)
            
            return mi
            
        except Exception:
            return 0.5  # Default moderate maintainability
            
    def _calculate_style_score(self, code: str) -> float:
        """Calculate code style score."""
        score = 1.0
        lines = code.split('\n')
        
        # Check for basic style issues
        for line in lines:
            # Long lines
            if len(line) > 100:
                score -= 0.1
                
            # Inconsistent indentation
            if line.startswith(' ') and not line.startswith('    '):
                score -= 0.1
                
        # Check for naming conventions
        if not re.search(r'def [a-z_][a-z0-9_]*\(', code):
            score -= 0.2  # No proper function names
            
        return max(0.0, min(1.0, score))
        
    def _calculate_documentation_score(self, code: str) -> float:
        """Calculate documentation score."""
        lines = code.split('\n')
        total_lines = len([l for l in lines if l.strip()])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        docstring_lines = len([l for l in lines if '"""' in l or "'''" in l])
        
        if total_lines == 0:
            return 0.0
            
        doc_ratio = (comment_lines + docstring_lines) / total_lines
        return min(1.0, doc_ratio * 2)  # Scale so 50% documentation = 1.0
        
    def _calculate_readability_score(self, code: str) -> float:
        """Calculate readability score."""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if not non_empty_lines:
            return 0.0
            
        # Average line length
        avg_line_length = np.mean([len(l) for l in non_empty_lines])
        line_length_score = max(0, 1 - (avg_line_length - 50) / 100)  # Optimal ~50 chars
        
        # Nesting depth
        max_indent = max([len(l) - len(l.lstrip()) for l in non_empty_lines])
        nesting_score = max(0, 1 - max_indent / 32)  # Penalize deep nesting
        
        # Variable name quality (simple heuristic)
        var_names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        long_names = [n for n in var_names if len(n) > 2]
        name_score = len(long_names) / len(var_names) if var_names else 0
        
        return (line_length_score + nesting_score + name_score) / 3
        
    def _measure_runtime(self, code: str, test_input: Any) -> Optional[Dict[str, float]]:
        """Measure runtime performance."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                runtime_code = f"""
import time
import psutil
import os

{code}

# Measure execution
process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss
start_time = time.perf_counter()

try:
    # Extract and call function
    import ast
    tree = ast.parse('''{code}''')
    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break
    
    if func_name:
        if isinstance({test_input}, list):
            result = globals()[func_name](*{test_input})
        else:
            result = globals()[func_name]({test_input})
            
    end_time = time.perf_counter()
    end_memory = process.memory_info().rss
    
    print(f"TIME:{{end_time - start_time}}")
    print(f"MEMORY:{{end_memory - start_memory}}")
    
except Exception as e:
    print(f"ERROR:{{e}}")
"""
                f.write(runtime_code)
                f.flush()
                
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout
                )
                
                Path(f.name).unlink()
                
                # Parse results
                time_match = re.search(r'TIME:([\d.]+)', result.stdout)
                memory_match = re.search(r'MEMORY:([\d.]+)', result.stdout)
                
                if time_match and memory_match:
                    return {
                        "time": float(time_match.group(1)),
                        "memory": float(memory_match.group(1))
                    }
                    
        except Exception:
            pass
            
        return None


class MathematicsMetrics:
    """Mathematics-specific metrics."""
    
    def __init__(self):
        """Initialize mathematics metrics."""
        pass
        
    def calculate_accuracy(self, generated: str, reference: str) -> float:
        """Calculate numerical accuracy.
        
        Args:
            generated: Generated solution
            reference: Reference solution
            
        Returns:
            Accuracy score (0-1)
        """
        gen_answer = self._extract_numerical_answer(generated)
        ref_answer = self._extract_numerical_answer(reference)
        
        if gen_answer is None or ref_answer is None:
            return 0.0
            
        # Check for exact match first
        if abs(gen_answer - ref_answer) < 1e-6:
            return 1.0
            
        # Calculate relative error
        if ref_answer == 0:
            return 1.0 if gen_answer == 0 else 0.0
            
        relative_error = abs(gen_answer - ref_answer) / abs(ref_answer)
        
        # Convert to accuracy score
        if relative_error < 0.01:  # 1% error
            return 0.9
        elif relative_error < 0.05:  # 5% error
            return 0.7
        elif relative_error < 0.1:   # 10% error
            return 0.5
        else:
            return 0.0
            
    def evaluate_reasoning(self, solution: str) -> Dict[str, float]:
        """Evaluate mathematical reasoning quality.
        
        Args:
            solution: Solution text to evaluate
            
        Returns:
            Reasoning quality metrics
        """
        steps = self._extract_solution_steps(solution)
        
        return {
            "step_count": min(1.0, len(steps) / 10),  # Normalize to 10 steps max
            "clarity": self._evaluate_step_clarity(steps),
            "logical_flow": self._evaluate_logical_flow(steps),
            "completeness": self._evaluate_completeness(solution)
        }
        
    def analyze_complexity(self, solution: str) -> Dict[str, Any]:
        """Analyze solution complexity.
        
        Args:
            solution: Solution to analyze
            
        Returns:
            Complexity analysis
        """
        return {
            "operation_count": self._count_mathematical_operations(solution),
            "concept_diversity": self._analyze_concepts_used(solution),
            "formula_complexity": self._analyze_formula_complexity(solution),
            "proof_structure": self._analyze_proof_structure(solution)
        }
        
    def _extract_numerical_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        # Common answer patterns
        patterns = [
            r'####\s*([-+]?\d*\.?\d+)',  # #### 42
            r'answer\s*(?:is|=)\s*([-+]?\d*\.?\d+)',  # answer is 42
            r'=\s*([-+]?\d*\.?\d+)\s*$',  # = 42
            r'\$\s*([-+]?\d*\.?\d+)\s*\$',  # $42$
            r'(?:^|\s)([-+]?\d*\.?\d+)(?:\s|$)'  # standalone number
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
                    
        return None
        
    def _extract_solution_steps(self, solution: str) -> List[str]:
        """Extract solution steps from text."""
        # Split on common step indicators
        step_patterns = [
            r'\d+\.',  # 1. 2. 3.
            r'Step\s+\d+',  # Step 1
            r'First,?|Then,?|Next,?|Finally,?',  # Sequence words
            r'\n\s*\n'  # Double newlines
        ]
        
        # Combine patterns
        pattern = '|'.join(f'({p})' for p in step_patterns)
        steps = re.split(pattern, solution, flags=re.IGNORECASE)
        
        # Clean and filter steps
        clean_steps = []
        for step in steps:
            if step and not re.match(r'^\s*\d+\.\s*$|^Step\s+\d+\s*$', step.strip()):
                clean_step = step.strip()
                if len(clean_step) > 10:  # Minimum meaningful step length
                    clean_steps.append(clean_step)
                    
        return clean_steps
        
    def _evaluate_step_clarity(self, steps: List[str]) -> float:
        """Evaluate clarity of solution steps."""
        if not steps:
            return 0.0
            
        clarity_scores = []
        
        for step in steps:
            # Check for mathematical notation
            has_math = bool(re.search(r'[=+\-*/(){}[\]]|\d+', step))
            
            # Check for explanatory words
            has_explanation = bool(re.search(r'\b(because|since|therefore|thus|so|hence)\b', step, re.IGNORECASE))
            
            # Check for appropriate length
            length_score = 1.0 if 20 <= len(step) <= 200 else 0.5
            
            step_score = (int(has_math) + int(has_explanation) + length_score) / 3
            clarity_scores.append(step_score)
            
        return np.mean(clarity_scores)
        
    def _evaluate_logical_flow(self, steps: List[str]) -> float:
        """Evaluate logical flow between steps."""
        if len(steps) < 2:
            return 1.0 if steps else 0.0
            
        # Check for transition words between steps
        transitions = 0
        transition_words = ['therefore', 'thus', 'hence', 'so', 'then', 'next', 'since', 'because']
        
        for i in range(1, len(steps)):
            step = steps[i].lower()
            if any(word in step for word in transition_words):
                transitions += 1
                
        return transitions / (len(steps) - 1)
        
    def _evaluate_completeness(self, solution: str) -> float:
        """Evaluate completeness of solution."""
        completeness_indicators = [
            r'therefore|thus|hence',  # Conclusion words
            r'answer\s*(?:is|=)',     # Final answer
            r'####',                   # Answer marker
            r'solution:|result:',      # Solution indicators
        ]
        
        score = 0.0
        for pattern in completeness_indicators:
            if re.search(pattern, solution, re.IGNORECASE):
                score += 0.25
                
        return min(1.0, score)
        
    def _count_mathematical_operations(self, solution: str) -> int:
        """Count mathematical operations in solution."""
        operations = re.findall(r'[+\-*/=<>≤≥∑∏∫∂]', solution)
        return len(operations)
        
    def _analyze_concepts_used(self, solution: str) -> float:
        """Analyze diversity of mathematical concepts."""
        concepts = [
            r'\b(?:algebra|equation|variable)\b',
            r'\b(?:geometry|triangle|circle|angle)\b',
            r'\b(?:calculus|derivative|integral|limit)\b',
            r'\b(?:probability|statistics|mean|median)\b',
            r'\b(?:trigonometry|sin|cos|tan)\b',
            r'\b(?:logarithm|exponential|log)\b'
        ]
        
        found_concepts = 0
        for concept_pattern in concepts:
            if re.search(concept_pattern, solution, re.IGNORECASE):
                found_concepts += 1
                
        return found_concepts / len(concepts)
        
    def _analyze_formula_complexity(self, solution: str) -> float:
        """Analyze complexity of formulas used."""
        # Count nested parentheses
        max_nesting = 0
        current_nesting = 0
        
        for char in solution:
            if char == '(':
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == ')':
                current_nesting -= 1
                
        # Count mathematical symbols
        math_symbols = len(re.findall(r'[∑∏∫∂√±×÷∞∝∈∉⊂⊃∪∩]', solution))
        
        # Normalize complexity
        complexity = (max_nesting + math_symbols) / 10
        return min(1.0, complexity)
        
    def _analyze_proof_structure(self, solution: str) -> float:
        """Analyze proof structure quality."""
        proof_elements = [
            r'\b(?:given|assume|suppose)\b',      # Assumptions
            r'\b(?:prove|show|demonstrate)\b',     # Goal
            r'\b(?:therefore|thus|hence|qed)\b',   # Conclusions
            r'\b(?:by|using|applying)\b',          # Methods
            r'\b(?:case|if|when)\b'                # Case analysis
        ]
        
        found_elements = 0
        for element_pattern in proof_elements:
            if re.search(element_pattern, solution, re.IGNORECASE):
                found_elements += 1
                
        return found_elements / len(proof_elements)


class WritingMetrics:
    """Writing-specific metrics."""
    
    def __init__(self):
        """Initialize writing metrics."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores.
        
        Args:
            generated: Generated text
            reference: Reference text
            
        Returns:
            ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            "rouge1_precision": scores['rouge1'].precision,
            "rouge1_recall": scores['rouge1'].recall,
            "rouge1_fmeasure": scores['rouge1'].fmeasure,
            "rouge2_precision": scores['rouge2'].precision,
            "rouge2_recall": scores['rouge2'].recall,
            "rouge2_fmeasure": scores['rouge2'].fmeasure,
            "rougeL_precision": scores['rougeL'].precision,
            "rougeL_recall": scores['rougeL'].recall,
            "rougeL_fmeasure": scores['rougeL'].fmeasure,
        }
        
    def evaluate_coherence(self, text: str) -> Dict[str, float]:
        """Evaluate text coherence.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Coherence metrics
        """
        sentences = self._split_into_sentences(text)
        
        return {
            "local_coherence": self._evaluate_local_coherence(sentences),
            "global_coherence": self._evaluate_global_coherence(text),
            "topic_consistency": self._evaluate_topic_consistency(sentences),
            "transition_quality": self._evaluate_transitions(sentences)
        }
        
    def analyze_style(self, text: str) -> Dict[str, float]:
        """Analyze writing style.
        
        Args:
            text: Text to analyze
            
        Returns:
            Style metrics
        """
        return {
            "readability": self._calculate_readability(text),
            "lexical_diversity": self._calculate_lexical_diversity(text),
            "formality": self._calculate_formality(text),
            "complexity": self._calculate_syntactic_complexity(text),
            "fluency": self._calculate_fluency(text)
        }
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _evaluate_local_coherence(self, sentences: List[str]) -> float:
        """Evaluate local coherence between adjacent sentences."""
        if len(sentences) < 2:
            return 1.0 if sentences else 0.0
            
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            current = sentences[i].lower().split()
            next_sent = sentences[i + 1].lower().split()
            
            # Calculate word overlap
            current_set = set(current)
            next_set = set(next_sent)
            
            if not current_set or not next_set:
                coherence_scores.append(0.0)
                continue
                
            overlap = len(current_set.intersection(next_set))
            union = len(current_set.union(next_set))
            
            # Jaccard similarity
            similarity = overlap / union if union > 0 else 0.0
            coherence_scores.append(similarity)
            
        return np.mean(coherence_scores)
        
    def _evaluate_global_coherence(self, text: str) -> float:
        """Evaluate global coherence of the entire text."""
        # Check for structural elements
        structure_indicators = [
            r'\b(?:first|firstly|initially)\b',
            r'\b(?:second|secondly|then|next)\b',
            r'\b(?:finally|lastly|in conclusion)\b',
            r'\b(?:however|but|although|despite)\b',
            r'\b(?:therefore|thus|consequently)\b'
        ]
        
        found_indicators = 0
        for pattern in structure_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                found_indicators += 1
                
        return min(1.0, found_indicators / 3)  # Normalize to reasonable expectation
        
    def _evaluate_topic_consistency(self, sentences: List[str]) -> float:
        """Evaluate topic consistency across sentences."""
        if not sentences:
            return 0.0
            
        # Extract content words (nouns, verbs, adjectives)
        content_words = []
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            content_words.extend(words)
            
        if not content_words:
            return 0.0
            
        # Calculate word frequency
        word_freq = {}
        for word in content_words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Find recurring themes (words appearing multiple times)
        recurring_words = [word for word, freq in word_freq.items() if freq > 1]
        
        return len(recurring_words) / len(set(content_words)) if content_words else 0.0
        
    def _evaluate_transitions(self, sentences: List[str]) -> float:
        """Evaluate quality of transitions between sentences."""
        if len(sentences) < 2:
            return 1.0 if sentences else 0.0
            
        transition_words = [
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'consequently', 'additionally', 'similarly', 'in contrast',
            'on the other hand', 'for example', 'in fact', 'indeed'
        ]
        
        transitions_found = 0
        for sentence in sentences[1:]:  # Skip first sentence
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in transition_words):
                transitions_found += 1
                
        return transitions_found / (len(sentences) - 1)
        
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        sentences = self._split_into_sentences(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
            
        # Count syllables (simplified)
        syllables = 0
        for word in words:
            # Simple syllable counting heuristic
            word = word.lower().strip('.,!?;:"')
            syllable_count = len(re.findall(r'[aeiou]', word))
            syllables += max(1, syllable_count)  # At least 1 syllable per word
            
        # Flesch Reading Ease formula (simplified)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Normalize to 0-1 scale
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        normalized_score = max(0, min(100, flesch_score)) / 100
        
        return normalized_score
        
    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (Type-Token Ratio)."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return 0.0
            
        unique_words = set(words)
        return len(unique_words) / len(words)
        
    def _calculate_formality(self, text: str) -> float:
        """Calculate formality score."""
        formal_indicators = [
            r'\b(?:therefore|however|furthermore|moreover|nevertheless)\b',
            r'\b(?:analysis|evaluation|investigation|examination)\b',
            r'\b(?:significant|substantial|considerable|extensive)\b'
        ]
        
        informal_indicators = [
            r'\b(?:really|pretty|quite|very|super)\b',
            r'\b(?:stuff|things|guys|folks)\b',
            r'[!]{2,}|[?]{2,}',  # Multiple punctuation
            r'\b(?:gonna|wanna|gotta)\b'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                          for pattern in formal_indicators)
        informal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                            for pattern in informal_indicators)
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5  # Neutral
            
        return formal_count / total_indicators
        
    def _calculate_syntactic_complexity(self, text: str) -> float:
        """Calculate syntactic complexity."""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return 0.0
            
        complexity_scores = []
        
        for sentence in sentences:
            words = sentence.split()
            if not words:
                complexity_scores.append(0.0)
                continue
                
            # Count subordinate clauses
            subordinate_markers = ['that', 'which', 'who', 'where', 'when', 'because', 'although', 'if']
            subordinate_count = sum(1 for word in words if word.lower() in subordinate_markers)
            
            # Calculate complexity based on length and subordination
            length_factor = min(1.0, len(words) / 20)  # Normalize to 20 words
            subordination_factor = min(1.0, subordinate_count / 3)  # Normalize to 3 clauses
            
            complexity_scores.append((length_factor + subordination_factor) / 2)
            
        return np.mean(complexity_scores)
        
    def _calculate_fluency(self, text: str) -> float:
        """Calculate fluency score based on grammatical patterns."""
        # Simple fluency indicators
        fluency_score = 1.0
        
        # Check for repetitive patterns
        words = text.lower().split()
        if len(words) > 1:
            repetitions = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
            fluency_score -= (repetitions / len(words)) * 0.5
            
        # Check for incomplete sentences
        sentences = self._split_into_sentences(text)
        incomplete_sentences = sum(1 for s in sentences if len(s.split()) < 3)
        if sentences:
            fluency_score -= (incomplete_sentences / len(sentences)) * 0.3
            
        # Check for proper capitalization
        if not text[0].isupper() if text else True:
            fluency_score -= 0.1
            
        return max(0.0, fluency_score) 