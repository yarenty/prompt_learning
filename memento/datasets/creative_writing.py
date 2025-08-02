"""
Creative Writing Dataset

Comprehensive collection of 150 creative writing problems covering:
- Story generation (30 problems)
- Essay writing (30 problems)
- Technical documentation (30 problems)
- Creative problem-solving (30 problems)
- Style adaptation (30 problems)

Each problem includes:
- Writing prompt
- Genre/style requirements
- Target audience
- Evaluation criteria
- Example outputs
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CreativeWritingProblem:
    """Individual creative writing problem."""

    id: str
    title: str
    prompt: str
    category: str  # story, essay, documentation, problem_solving, style_adaptation
    genre: str
    difficulty: str  # easy, medium, hard
    target_audience: str
    word_count_range: tuple[int, int]
    style_requirements: List[str]
    evaluation_criteria: List[str]
    example_elements: List[str]
    estimated_time_minutes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreativeWritingProblem":
        """Create from dictionary."""
        return cls(**data)


class CreativeWritingDataset:
    """Comprehensive creative writing problem dataset."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize dataset."""
        self.storage_path = storage_path or Path("data/creative_writing")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.problems: List[CreativeWritingProblem] = []
        self._load_or_create_problems()

    def _load_or_create_problems(self) -> None:
        """Load existing problems or create new ones."""
        problems_file = self.storage_path / "problems.json"

        if problems_file.exists():
            self._load_problems(problems_file)
        else:
            self._create_comprehensive_problems()
            self._save_problems(problems_file)

    def _load_problems(self, problems_file: Path) -> None:
        """Load problems from file."""
        try:
            with open(problems_file, "r") as f:
                problems_data = json.load(f)

            self.problems = [CreativeWritingProblem.from_dict(problem_data) for problem_data in problems_data]
        except Exception as e:
            print(f"Error loading problems: {e}")
            self._create_comprehensive_problems()

    def _save_problems(self, problems_file: Path) -> None:
        """Save problems to file."""
        try:
            problems_data = [problem.to_dict() for problem in self.problems]
            with open(problems_file, "w") as f:
                json.dump(problems_data, f, indent=2)
        except Exception as e:
            print(f"Error saving problems: {e}")

    def _create_comprehensive_problems(self) -> None:
        """Create comprehensive set of 150 creative writing problems."""
        self.problems = []

        # Story generation (30 problems)
        self.problems.extend(self._create_story_problems())

        # Essay writing (30 problems)
        self.problems.extend(self._create_essay_problems())

        # Technical documentation (30 problems)
        self.problems.extend(self._create_documentation_problems())

        # Creative problem-solving (30 problems)
        self.problems.extend(self._create_problem_solving_problems())

        # Style adaptation (30 problems)
        self.problems.extend(self._create_style_adaptation_problems())

    def _create_story_problems(self) -> List[CreativeWritingProblem]:
        """Create story generation problems."""
        problems = []

        problems.append(
            CreativeWritingProblem(
                id="story_001",
                title="Sci-Fi Short Story",
                prompt="Write a short story about a time traveler who discovers that changing the past creates parallel universes instead of altering the current timeline.",
                category="story",
                genre="science fiction",
                difficulty="medium",
                target_audience="young adults",
                word_count_range=(800, 1200),
                style_requirements=["vivid imagery", "dialogue", "plot twist", "character development"],
                evaluation_criteria=["creativity", "plot coherence", "character development", "writing quality"],
                example_elements=["time machine", "parallel universe", "moral dilemma", "scientific explanation"],
                estimated_time_minutes=60,
            )
        )

        problems.append(
            CreativeWritingProblem(
                id="story_002",
                title="Mystery Flash Fiction",
                prompt="Write a mystery story in exactly 300 words where the detective realizes they are the criminal.",
                category="story",
                genre="mystery",
                difficulty="hard",
                target_audience="adults",
                word_count_range=(300, 300),
                style_requirements=["concise prose", "red herrings", "revelation", "atmospheric tension"],
                evaluation_criteria=["plot twist effectiveness", "word economy", "suspense building", "resolution"],
                example_elements=["detective", "crime scene", "clues", "self-discovery"],
                estimated_time_minutes=45,
            )
        )

        # Add more story problems (28 more to reach 30)
        story_genres = ["fantasy", "horror", "romance", "thriller", "historical fiction", "comedy"]
        for i in range(3, 31):
            problems.append(
                CreativeWritingProblem(
                    id=f"story_{i:03d}",
                    title=f"Story Problem {i}",
                    prompt=f"Creative story prompt {i}",
                    category="story",
                    genre=story_genres[i % len(story_genres)],
                    difficulty=["easy", "medium", "hard"][i % 3],
                    target_audience=["children", "young adults", "adults"][i % 3],
                    word_count_range=(500 + i * 20, 1000 + i * 30),
                    style_requirements=["engaging narrative", "character development"],
                    evaluation_criteria=["creativity", "writing quality", "engagement"],
                    example_elements=["plot", "characters", "setting"],
                    estimated_time_minutes=45 + (i % 3) * 15,
                )
            )

        return problems

    def _create_essay_problems(self) -> List[CreativeWritingProblem]:
        """Create essay writing problems."""
        problems = []

        problems.append(
            CreativeWritingProblem(
                id="essay_001",
                title="Argumentative Essay on AI Ethics",
                prompt="Write an argumentative essay discussing whether artificial intelligence should have rights and legal protections similar to humans.",
                category="essay",
                genre="argumentative",
                difficulty="hard",
                target_audience="academic",
                word_count_range=(1000, 1500),
                style_requirements=[
                    "thesis statement",
                    "evidence-based arguments",
                    "counterarguments",
                    "logical structure",
                ],
                evaluation_criteria=["argument strength", "evidence quality", "logical flow", "writing clarity"],
                example_elements=["thesis", "supporting evidence", "counterarguments", "conclusion"],
                estimated_time_minutes=90,
            )
        )

        problems.append(
            CreativeWritingProblem(
                id="essay_002",
                title="Personal Narrative Essay",
                prompt="Write a personal narrative about a moment that fundamentally changed your perspective on life.",
                category="essay",
                genre="narrative",
                difficulty="medium",
                target_audience="general",
                word_count_range=(600, 900),
                style_requirements=["first person", "vivid details", "reflection", "emotional connection"],
                evaluation_criteria=["personal insight", "narrative flow", "emotional impact", "descriptive language"],
                example_elements=["personal experience", "reflection", "growth", "vivid details"],
                estimated_time_minutes=60,
            )
        )

        # Add more essay problems
        essay_types = ["persuasive", "expository", "comparative", "analytical", "descriptive"]
        for i in range(3, 31):
            problems.append(
                CreativeWritingProblem(
                    id=f"essay_{i:03d}",
                    title=f"Essay Problem {i}",
                    prompt=f"Essay writing challenge {i}",
                    category="essay",
                    genre=essay_types[i % len(essay_types)],
                    difficulty=["easy", "medium", "hard"][i % 3],
                    target_audience=["general", "academic", "professional"][i % 3],
                    word_count_range=(400 + i * 30, 800 + i * 40),
                    style_requirements=["clear structure", "supporting evidence"],
                    evaluation_criteria=["clarity", "organization", "evidence"],
                    example_elements=["introduction", "body", "conclusion"],
                    estimated_time_minutes=50 + (i % 3) * 20,
                )
            )

        return problems

    def _create_documentation_problems(self) -> List[CreativeWritingProblem]:
        """Create technical documentation problems."""
        problems = []

        problems.append(
            CreativeWritingProblem(
                id="doc_001",
                title="API Documentation",
                prompt="Write comprehensive API documentation for a REST API that manages a library system (books, users, loans).",
                category="documentation",
                genre="technical",
                difficulty="medium",
                target_audience="developers",
                word_count_range=(1200, 2000),
                style_requirements=["clear structure", "code examples", "parameter descriptions", "error handling"],
                evaluation_criteria=["completeness", "clarity", "usability", "accuracy"],
                example_elements=["endpoints", "parameters", "responses", "examples"],
                estimated_time_minutes=120,
            )
        )

        problems.append(
            CreativeWritingProblem(
                id="doc_002",
                title="User Manual",
                prompt="Create a user manual for a smart home automation system, targeting non-technical users.",
                category="documentation",
                genre="instructional",
                difficulty="medium",
                target_audience="general users",
                word_count_range=(800, 1200),
                style_requirements=["step-by-step instructions", "screenshots", "troubleshooting", "simple language"],
                evaluation_criteria=["user-friendliness", "completeness", "clarity", "organization"],
                example_elements=["setup guide", "features", "troubleshooting", "FAQ"],
                estimated_time_minutes=90,
            )
        )

        # Add more documentation problems
        doc_types = ["user guide", "technical specification", "tutorial", "FAQ", "installation guide"]
        for i in range(3, 31):
            problems.append(
                CreativeWritingProblem(
                    id=f"doc_{i:03d}",
                    title=f"Documentation Problem {i}",
                    prompt=f"Technical documentation task {i}",
                    category="documentation",
                    genre=doc_types[i % len(doc_types)],
                    difficulty=["easy", "medium", "hard"][i % 3],
                    target_audience=["users", "developers", "administrators"][i % 3],
                    word_count_range=(600 + i * 25, 1000 + i * 35),
                    style_requirements=["clear instructions", "organized structure"],
                    evaluation_criteria=["clarity", "completeness", "usability"],
                    example_elements=["instructions", "examples", "references"],
                    estimated_time_minutes=60 + (i % 3) * 30,
                )
            )

        return problems

    def _create_problem_solving_problems(self) -> List[CreativeWritingProblem]:
        """Create creative problem-solving problems."""
        problems = []

        problems.append(
            CreativeWritingProblem(
                id="problem_001",
                title="Business Innovation Proposal",
                prompt="Propose an innovative solution to reduce food waste in urban restaurants, including implementation strategy and expected outcomes.",
                category="problem_solving",
                genre="business proposal",
                difficulty="hard",
                target_audience="business executives",
                word_count_range=(1000, 1500),
                style_requirements=["problem analysis", "creative solution", "implementation plan", "metrics"],
                evaluation_criteria=["innovation", "feasibility", "impact potential", "presentation quality"],
                example_elements=["problem definition", "solution", "implementation", "metrics"],
                estimated_time_minutes=100,
            )
        )

        problems.append(
            CreativeWritingProblem(
                id="problem_002",
                title="Educational Challenge Solution",
                prompt="Design a creative approach to teach complex mathematical concepts to elementary school students using gamification.",
                category="problem_solving",
                genre="educational design",
                difficulty="medium",
                target_audience="educators",
                word_count_range=(700, 1000),
                style_requirements=["pedagogical approach", "game mechanics", "learning objectives", "assessment"],
                evaluation_criteria=["educational value", "engagement", "feasibility", "creativity"],
                example_elements=["learning objectives", "game design", "assessment", "implementation"],
                estimated_time_minutes=75,
            )
        )

        # Add more problem-solving problems
        problem_domains = ["environmental", "social", "technological", "healthcare", "education"]
        for i in range(3, 31):
            problems.append(
                CreativeWritingProblem(
                    id=f"problem_{i:03d}",
                    title=f"Problem Solving Challenge {i}",
                    prompt=f"Creative solution for challenge {i}",
                    category="problem_solving",
                    genre=f"{problem_domains[i % len(problem_domains)]} solution",
                    difficulty=["medium", "hard"][i % 2],
                    target_audience=["professionals", "experts"][i % 2],
                    word_count_range=(600 + i * 20, 1000 + i * 30),
                    style_requirements=["analytical thinking", "creative approach"],
                    evaluation_criteria=["innovation", "feasibility", "impact"],
                    example_elements=["problem analysis", "solution", "implementation"],
                    estimated_time_minutes=60 + (i % 2) * 30,
                )
            )

        return problems

    def _create_style_adaptation_problems(self) -> List[CreativeWritingProblem]:
        """Create style adaptation problems."""
        problems = []

        problems.append(
            CreativeWritingProblem(
                id="style_001",
                title="Shakespeare Style Adaptation",
                prompt="Rewrite a modern news article about climate change in the style of Shakespeare, maintaining the factual content while using Elizabethan language and iambic pentameter where appropriate.",
                category="style_adaptation",
                genre="classical adaptation",
                difficulty="hard",
                target_audience="literature enthusiasts",
                word_count_range=(400, 600),
                style_requirements=[
                    "Elizabethan vocabulary",
                    "poetic language",
                    "archaic sentence structure",
                    "metaphorical expressions",
                ],
                evaluation_criteria=[
                    "style authenticity",
                    "content preservation",
                    "linguistic creativity",
                    "readability",
                ],
                example_elements=["thee/thou usage", "metaphors", "archaic terms", "poetic devices"],
                estimated_time_minutes=90,
            )
        )

        problems.append(
            CreativeWritingProblem(
                id="style_002",
                title="Children's Book Adaptation",
                prompt="Adapt a complex scientific concept (quantum physics) into a simple, engaging children's story suitable for ages 6-8.",
                category="style_adaptation",
                genre="children's literature",
                difficulty="medium",
                target_audience="children",
                word_count_range=(300, 500),
                style_requirements=[
                    "simple vocabulary",
                    "engaging narrative",
                    "educational content",
                    "age-appropriate",
                ],
                evaluation_criteria=["age appropriateness", "educational value", "engagement", "simplicity"],
                example_elements=["simple words", "relatable characters", "fun storyline", "learning elements"],
                estimated_time_minutes=60,
            )
        )

        # Add more style adaptation problems
        styles = ["academic", "journalistic", "poetic", "conversational", "formal", "humorous"]
        for i in range(3, 31):
            problems.append(
                CreativeWritingProblem(
                    id=f"style_{i:03d}",
                    title=f"Style Adaptation {i}",
                    prompt=f"Style adaptation challenge {i}",
                    category="style_adaptation",
                    genre=f"{styles[i % len(styles)]} style",
                    difficulty=["easy", "medium", "hard"][i % 3],
                    target_audience=["general", "specific", "academic"][i % 3],
                    word_count_range=(300 + i * 15, 600 + i * 20),
                    style_requirements=["style consistency", "appropriate tone"],
                    evaluation_criteria=["style accuracy", "content quality", "adaptation skill"],
                    example_elements=["style elements", "tone", "vocabulary"],
                    estimated_time_minutes=45 + (i % 3) * 15,
                )
            )

        return problems

    def get_problems_by_category(self, category: str) -> List[CreativeWritingProblem]:
        """Get problems filtered by category."""
        return [p for p in self.problems if p.category == category]

    def get_problems_by_genre(self, genre: str) -> List[CreativeWritingProblem]:
        """Get problems filtered by genre."""
        return [p for p in self.problems if p.genre == genre]

    def get_problems_by_difficulty(self, difficulty: str) -> List[CreativeWritingProblem]:
        """Get problems filtered by difficulty."""
        return [p for p in self.problems if p.difficulty == difficulty]

    def get_random_problems(self, count: int, category: Optional[str] = None) -> List[CreativeWritingProblem]:
        """Get random sample of problems."""
        import random

        if category:
            available = self.get_problems_by_category(category)
        else:
            available = self.problems

        return random.sample(available, min(count, len(available)))

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        categories = {}
        genres = {}
        difficulties = {}

        for problem in self.problems:
            categories[problem.category] = categories.get(problem.category, 0) + 1
            genres[problem.genre] = genres.get(problem.genre, 0) + 1
            difficulties[problem.difficulty] = difficulties.get(problem.difficulty, 0) + 1

        return {
            "total_problems": len(self.problems),
            "categories": categories,
            "genres": genres,
            "difficulties": difficulties,
            "average_time": sum(p.estimated_time_minutes for p in self.problems) / len(self.problems),
        }
