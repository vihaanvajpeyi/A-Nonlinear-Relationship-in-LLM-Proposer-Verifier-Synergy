"""
analyze_phase1_by_domain.py — Domain-split analysis for Phase 1 results.

Splits per-model accuracy into math vs. coding subsets, to distinguish
"weak overall" from "weak on one domain, masked by pooled average" — e.g.
a code-specialized model that's strong on coding but drags its average
down with poor math performance.

Usage:
    python3 analyze_phase1_by_domain.py
"""

import json
from collections import defaultdict

RESULTS_PATH = "phase1_results.jsonl"
DATASET_PATH = "question_dataset_v2.json"


def load_results(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_question_domains(path: str) -> dict:
    """Map question_id -> domain ('math' or 'coding')."""
    with open(path, "r") as f:
        data = json.load(f)
    return {q["id"]: q["domain"] for q in data["questions"]}


def analyze(rows: list, question_domains: dict):
    # per_model[model][domain] = {total, scored, correct, parse_failures}
    per_model = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "scored": 0, "correct": 0.0, "parse_failures": 0
    }))

    for row in rows:
        if row["phase"] != 1:
            continue
        model = row["model_id"]
        qid = row["question_id"]
        domain = question_domains.get(qid, "unknown")
        stats = per_model[model][domain]
        stats["total"] += 1

        if row["parse_failure"]:
            stats["parse_failures"] += 1
            continue

        stats["scored"] += 1
        score = row["score"]
        if score is True or (isinstance(score, (int, float)) and score >= 1.0):
            stats["correct"] += 1
        elif isinstance(score, (int, float)) and 0 < score < 1.0:
            stats["correct"] += score

    return per_model


def print_report(per_model: dict):
    print("=" * 100)
    print(f"{'Model':<22} {'Math Acc':>10} {'Math PF':>9} "
          f"{'Coding Acc':>11} {'Coding PF':>10} {'Gap (C-M)':>10}")
    print("=" * 100)

    def acc(stats):
        return stats["correct"] / stats["scored"] if stats["scored"] > 0 else 0.0

    def pf_rate(stats):
        return stats["parse_failures"] / stats["total"] if stats["total"] > 0 else 0.0

    rows_out = []
    for model, domains in per_model.items():
        math_stats = domains.get("math", {"total": 0, "scored": 0, "correct": 0.0, "parse_failures": 0})
        coding_stats = domains.get("coding", {"total": 0, "scored": 0, "correct": 0.0, "parse_failures": 0})
        math_acc = acc(math_stats)
        coding_acc = acc(coding_stats)
        gap = coding_acc - math_acc
        rows_out.append((model, math_acc, pf_rate(math_stats), coding_acc, pf_rate(coding_stats), gap))

    # Sort by gap descending (biggest "coding stronger than math" first)
    rows_out.sort(key=lambda r: -r[5])

    for model, math_acc, math_pf, coding_acc, coding_pf, gap in rows_out:
        print(f"{model:<22} {math_acc:>9.1%} {math_pf:>8.1%} "
              f"{coding_acc:>10.1%} {coding_pf:>9.1%} {gap:>+9.1%}")

    print("=" * 100)
    print("\nPositive gap = stronger on coding than math. Negative = stronger on math.")


if __name__ == "__main__":
    rows = load_results(RESULTS_PATH)
    question_domains = load_question_domains(DATASET_PATH)
    per_model = analyze(rows, question_domains)
    print_report(per_model)
