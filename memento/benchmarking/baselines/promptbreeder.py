"""
PromptBreeder Implementation

Based on the PromptBreeder paper, this implements evolutionary prompt optimization
using genetic algorithms with crossover, mutation, and selection operations.

Reference: "PromptBreeder: Self-Referential Self-Improvement Via Prompt Evolution"
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ollama

from ...config import ModelConfig
from ...utils.validation import validate_prompt


@dataclass
class Individual:
    """Represents an individual prompt in the population."""

    prompt: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None
    mutations: List[str] = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.mutations is None:
            self.mutations = []

    @property
    def id(self) -> str:
        """Generate unique ID for this individual."""
        return f"gen{self.generation}_{hash(self.prompt) % 10000:04d}"


class PromptBreeder:
    """
    PromptBreeder implementation using evolutionary algorithms.

    This class implements the core PromptBreeder algorithm with:
    - Population-based evolution
    - Crossover and mutation operations
    - Fitness-based selection
    - Elitism preservation
    """

    def __init__(
        self,
        model_config: ModelConfig,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elitism_ratio: float = 0.2,
        max_generations: int = 10,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize PromptBreeder.

        Args:
            model_config: Model configuration for LLM calls
            population_size: Size of the prompt population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_ratio: Fraction of best individuals to preserve
            max_generations: Maximum number of generations
            storage_path: Path to store evolution history
        """
        self.model_config = model_config
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.max_generations = max_generations

        self.storage_path = storage_path or Path("./promptbreeder_data")
        self.storage_path.mkdir(exist_ok=True)

        # Evolution state
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.evolution_history: List[Dict[str, Any]] = []

        # Mutation operators
        self.mutation_operators = [
            self._mutate_add_instruction,
            self._mutate_rephrase,
            self._mutate_add_example,
            self._mutate_change_style,
            self._mutate_add_constraint,
        ]

    async def evolve(
        self,
        initial_prompt: str,
        evaluation_function: callable,
        problem_set: List[Dict[str, Any]],
    ) -> Individual:
        """
        Run the evolutionary optimization process.

        Args:
            initial_prompt: Starting prompt for evolution
            evaluation_function: Function to evaluate prompt fitness
            problem_set: Set of problems for evaluation

        Returns:
            Best individual found during evolution
        """
        # Initialize population
        await self._initialize_population(initial_prompt)

        for generation in range(self.max_generations):
            self.generation = generation

            # Evaluate population
            await self._evaluate_population(evaluation_function, problem_set)

            # Track best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best

            # Log generation results
            self._log_generation()

            # Create next generation
            if generation < self.max_generations - 1:
                await self._create_next_generation()

        # Save final results
        await self._save_results()

        return self.best_individual

    async def _initialize_population(self, initial_prompt: str) -> None:
        """Initialize the population with variations of the initial prompt."""
        validate_prompt(initial_prompt)

        # Add the original prompt
        self.population = [Individual(prompt=initial_prompt, generation=0)]

        # Generate variations
        for i in range(self.population_size - 1):
            try:
                variant = await self._generate_variant(initial_prompt)
                self.population.append(Individual(prompt=variant, generation=0))
            except Exception:
                # Fallback to slight modifications if generation fails
                variant = await self._simple_variation(initial_prompt, i)
                self.population.append(Individual(prompt=variant, generation=0))

    async def _generate_variant(self, base_prompt: str) -> str:
        """Generate a variant of the base prompt using LLM."""
        variation_prompt = f"""
        Create a variation of the following system prompt that maintains the same core purpose
        but uses different wording, structure, or approach:

        Original prompt: {base_prompt}

        Generate a single alternative version that:
        1. Keeps the same fundamental goal
        2. Uses different phrasing or structure
        3. May add or modify instructions slightly
        4. Remains clear and actionable

        Return only the new prompt without any explanation.
        """

        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=variation_prompt,
            options={"temperature": 0.8, "num_predict": 200},
        )

        return response["response"].strip()

    async def _simple_variation(self, base_prompt: str, seed: int) -> str:
        """Create simple variations when LLM generation fails."""
        variations = [
            f"Please {base_prompt.lower()}",
            f"{base_prompt} Be thorough and accurate.",
            f"{base_prompt} Provide detailed explanations.",
            f"Task: {base_prompt}",
            f"{base_prompt} Use clear, professional language.",
        ]
        return variations[seed % len(variations)]

    async def _evaluate_population(self, evaluation_function: callable, problem_set: List[Dict[str, Any]]) -> None:
        """Evaluate fitness of all individuals in the population."""
        for individual in self.population:
            try:
                fitness = await evaluation_function(individual.prompt, problem_set)
                individual.fitness = fitness
            except Exception:
                # Assign low fitness to failed individuals
                individual.fitness = 0.0

    async def _create_next_generation(self) -> None:
        """Create the next generation using selection, crossover, and mutation."""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Preserve elite individuals
        elite_count = int(self.population_size * self.elitism_ratio)
        next_generation = self.population[:elite_count].copy()

        # Update generation number for elites
        for individual in next_generation:
            individual.generation = self.generation + 1

        # Generate offspring to fill remaining slots
        while len(next_generation) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                child = await self._crossover(parent1, parent2)
            else:
                child = Individual(prompt=parent1.prompt, generation=self.generation + 1, parent_ids=[parent1.id])

            # Mutation
            if random.random() < self.mutation_rate:
                child = await self._mutate(child)

            next_generation.append(child)

        self.population = next_generation[: self.population_size]

    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select an individual using tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    async def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create offspring by combining two parent prompts."""
        crossover_prompt = f"""
        Combine these two system prompts into a single, coherent prompt that incorporates
        the best elements of both:

        Prompt 1: {parent1.prompt}

        Prompt 2: {parent2.prompt}

        Create a new prompt that:
        1. Combines the strengths of both prompts
        2. Maintains clarity and coherence
        3. Avoids redundancy
        4. Results in a single, actionable system prompt

        Return only the combined prompt without explanation.
        """

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_config.model_name,
                prompt=crossover_prompt,
                options={"temperature": 0.6, "num_predict": 300},
            )

            child_prompt = response["response"].strip()

            return Individual(prompt=child_prompt, generation=self.generation + 1, parent_ids=[parent1.id, parent2.id])
        except Exception:
            # Fallback to simple combination
            combined = f"{parent1.prompt} {parent2.prompt}"
            return Individual(prompt=combined, generation=self.generation + 1, parent_ids=[parent1.id, parent2.id])

    async def _mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual."""
        mutation_op = random.choice(self.mutation_operators)

        try:
            mutated_prompt = await mutation_op(individual.prompt)
            individual.prompt = mutated_prompt
            individual.mutations.append(mutation_op.__name__)
        except Exception:
            # If mutation fails, keep original
            pass

        return individual

    async def _mutate_add_instruction(self, prompt: str) -> str:
        """Add a new instruction to the prompt."""
        mutation_prompt = f"""
        Add a helpful instruction to this system prompt:

        {prompt}

        Add one clear, specific instruction that would improve the prompt's effectiveness.
        Return the complete modified prompt.
        """

        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=mutation_prompt,
            options={"temperature": 0.7, "num_predict": 250},
        )

        return response["response"].strip()

    async def _mutate_rephrase(self, prompt: str) -> str:
        """Rephrase part of the prompt."""
        mutation_prompt = f"""
        Rephrase this system prompt to make it clearer and more effective:

        {prompt}

        Keep the same meaning but improve the wording, structure, or clarity.
        Return only the rephrased prompt.
        """

        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=mutation_prompt,
            options={"temperature": 0.6, "num_predict": 250},
        )

        return response["response"].strip()

    async def _mutate_add_example(self, prompt: str) -> str:
        """Add an example to the prompt."""
        mutation_prompt = f"""
        Add a brief, helpful example to this system prompt:

        {prompt}

        Add one concrete example that demonstrates what's expected.
        Return the complete prompt with the example added.
        """

        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=mutation_prompt,
            options={"temperature": 0.7, "num_predict": 300},
        )

        return response["response"].strip()

    async def _mutate_change_style(self, prompt: str) -> str:
        """Change the style or tone of the prompt."""
        styles = ["more professional", "more conversational", "more detailed", "more concise", "more encouraging"]
        style = random.choice(styles)

        mutation_prompt = f"""
        Modify this system prompt to be {style}:

        {prompt}

        Keep the same core instructions but adjust the tone and style.
        Return only the modified prompt.
        """

        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=mutation_prompt,
            options={"temperature": 0.6, "num_predict": 250},
        )

        return response["response"].strip()

    async def _mutate_add_constraint(self, prompt: str) -> str:
        """Add a constraint or requirement to the prompt."""
        mutation_prompt = f"""
        Add a helpful constraint or requirement to this system prompt:

        {prompt}

        Add one specific constraint that would improve the quality or consistency of outputs.
        Return the complete modified prompt.
        """

        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_config.model_name,
            prompt=mutation_prompt,
            options={"temperature": 0.7, "num_predict": 250},
        )

        return response["response"].strip()

    def _log_generation(self) -> None:
        """Log statistics for the current generation."""
        fitnesses = [ind.fitness for ind in self.population]

        generation_stats = {
            "generation": self.generation,
            "best_fitness": max(fitnesses),
            "average_fitness": np.mean(fitnesses),
            "worst_fitness": min(fitnesses),
            "fitness_std": np.std(fitnesses),
            "best_prompt": max(self.population, key=lambda x: x.fitness).prompt,
            "timestamp": time.time(),
        }

        self.evolution_history.append(generation_stats)

        print(
            f"Generation {self.generation}: Best={generation_stats['best_fitness']:.3f}, "
            f"Avg={generation_stats['average_fitness']:.3f}, "
            f"Std={generation_stats['fitness_std']:.3f}"
        )

    async def _save_results(self) -> None:
        """Save evolution results to storage."""
        results = {
            "best_individual": {
                "prompt": self.best_individual.prompt,
                "fitness": self.best_individual.fitness,
                "generation": self.best_individual.generation,
                "id": self.best_individual.id,
            },
            "evolution_history": self.evolution_history,
            "final_population": [
                {"prompt": ind.prompt, "fitness": ind.fitness, "generation": ind.generation, "id": ind.id}
                for ind in self.population
            ],
            "parameters": {
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elitism_ratio": self.elitism_ratio,
                "max_generations": self.max_generations,
            },
        }

        results_file = self.storage_path / f"promptbreeder_results_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_file}")

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution process."""
        if not self.evolution_history:
            return {"status": "No evolution history available"}

        initial_fitness = self.evolution_history[0]["best_fitness"]
        final_fitness = self.evolution_history[-1]["best_fitness"]
        improvement = final_fitness - initial_fitness

        return {
            "total_generations": len(self.evolution_history),
            "initial_fitness": initial_fitness,
            "final_fitness": final_fitness,
            "improvement": improvement,
            "improvement_percentage": (improvement / initial_fitness * 100) if initial_fitness > 0 else 0,
            "best_prompt": self.best_individual.prompt if self.best_individual else None,
            "convergence_generation": self._find_convergence_generation(),
        }

    def _find_convergence_generation(self) -> int:
        """Find the generation where the algorithm converged."""
        if len(self.evolution_history) < 3:
            return len(self.evolution_history)

        # Look for when improvement becomes minimal
        for i in range(2, len(self.evolution_history)):
            recent_improvements = [
                self.evolution_history[j]["best_fitness"] - self.evolution_history[j - 1]["best_fitness"]
                for j in range(max(0, i - 2), i + 1)
            ]

            if all(imp < 0.01 for imp in recent_improvements):
                return i

        return len(self.evolution_history)
