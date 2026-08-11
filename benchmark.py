"""
benchmark.py — Benchmark loader for the synergy expansion experiment.

Reads question_dataset_v2.json, validates schema, and provides a clean
interface to iterate over questions filtered by domain (math/coding).

Schema (per question_dataset_v2.json):
  Common fields (all questions):
    "id": str
    "domain": "math" | "coding"
    "type": str
    "source": str
    "difficulty_prior": str
    "contains_trap": bool
    "requires_multistep": bool
    "requires_symbolic": bool
    "answer_type": str
    "question": str
    "answer": str

  Math-only field:
    "final_answer": str   — ground-truth numeric/symbolic answer

  Coding-only field:
    "test_cases": dict    — test cases to validate generated code against

Usage:
    from benchmark import BenchmarkLoader

    loader = BenchmarkLoader("question_dataset_v2.json")
    
    # All questions (100 total: 60 math + 40 coding)
    all_qs = loader.load_questions()
    
    # Filtered by domain
    math_qs = loader.load_questions(domain="math")
    coding_qs = loader.load_questions(domain="coding")
    
    # Access individual question
    for q in math_qs:
        print(q.id, q.question, q.final_answer)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Question:
    """Represents a single benchmark question."""
    id: str
    domain: str  # "math" or "coding"
    type: str
    source: str
    difficulty_prior: str
    contains_trap: bool
    requires_multistep: bool
    requires_symbolic: bool
    answer_type: str
    question: str
    answer: str  # Full worked solution
    final_answer: Optional[str] = None  # Math: ground-truth answer
    test_cases: Optional[dict] = None  # Coding: test case dict


class BenchmarkLoader:
    """Loads and serves questions from question_dataset_v2.json."""

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Path to question_dataset_v2.json
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        self.data = self._load_json()
        self.metadata = self.data.get("metadata", {})
        self.questions_raw = self.data.get("questions", [])
        
        # Validate and parse
        self.questions = self._validate_and_parse_questions()

    def _load_json(self) -> dict:
        """Load and parse the JSON dataset."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)

    def _validate_and_parse_questions(self) -> List[Question]:
        """Parse raw question dicts into Question objects, validating schema."""
        questions = []
        common_keys = {
            "id", "domain", "type", "source", "difficulty_prior",
            "contains_trap", "requires_multistep", "requires_symbolic",
            "answer_type", "question", "answer"
        }
        
        for i, raw_q in enumerate(self.questions_raw):
            # Validate common schema
            missing = common_keys - set(raw_q.keys())
            if missing:
                raise ValueError(
                    f"Question {i} (id={raw_q.get('id', '?')}) missing common keys: {missing}"
                )
            
            # Validate domain
            domain = raw_q["domain"]
            if domain not in ("math", "coding"):
                raise ValueError(
                    f"Question {i} has invalid domain: {domain}. Must be 'math' or 'coding'."
                )
            
            # Some coding questions (e.g. debugging tasks) use final_answer
            # instead of test_cases. Accept whichever is present; require at
            # least one.
            final_answer = raw_q.get("final_answer")
            test_cases = raw_q.get("test_cases")
            if final_answer is None and test_cases is None:
                raise ValueError(
                    f"Question {i} (id={raw_q['id']}) has neither "
                    f"'final_answer' nor 'test_cases'"
                )
            
            # Parse into Question object
            q = Question(
                id=raw_q["id"],
                domain=domain,
                type=raw_q["type"],
                source=raw_q["source"],
                difficulty_prior=raw_q["difficulty_prior"],
                contains_trap=raw_q["contains_trap"],
                requires_multistep=raw_q["requires_multistep"],
                requires_symbolic=raw_q["requires_symbolic"],
                answer_type=raw_q["answer_type"],
                question=raw_q["question"],
                answer=raw_q["answer"],
                final_answer=final_answer,
                test_cases=test_cases,
            )
            questions.append(q)
        
        return questions

    def load_questions(self, domain: Optional[str] = None) -> List[Question]:
        """
        Load questions, optionally filtered by domain.
        
        Args:
            domain: "math", "coding", or None for all
        
        Returns:
            List of Question objects
        """
        if domain is None:
            return self.questions
        
        if domain not in ("math", "coding"):
            raise ValueError(f"Invalid domain: {domain}")
        
        return [q for q in self.questions if q.domain == domain]

    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Look up a single question by ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def summary(self) -> dict:
        """Return summary statistics about the dataset."""
        math_qs = self.load_questions(domain="math")
        coding_qs = self.load_questions(domain="coding")
        
        return {
            "total_questions": len(self.questions),
            "math_count": len(math_qs),
            "coding_count": len(coding_qs),
            "metadata": self.metadata,
        }


if __name__ == "__main__":
    # Smoke test: load and summarize dataset
    loader = BenchmarkLoader("question_dataset_v2.json")
    
    summary = loader.summary()
    print("Dataset Summary:")
    print(f"  Total questions: {summary['total_questions']}")
    print(f"  Math: {summary['math_count']}")
    print(f"  Coding: {summary['coding_count']}")
    print()
    
    # Show a few examples from each domain
    print("Sample math question:")
    m = loader.load_questions(domain="math")[0]
    print(f"  [{m.id}] {m.question[:60]}...")
    print(f"    Final answer: {m.final_answer}")
    print()
    
    print("Sample coding question:")
    c = loader.load_questions(domain="coding")[0]
    print(f"  [{c.id}] {c.question[:60]}...")
    print(f"    Test cases: {c.test_cases}")
