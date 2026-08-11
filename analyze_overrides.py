"""
analyze_overrides.py — Breaks down verifier behavior by outcome type, to
explain the source of negative average synergy found in analyze_synergy.py.

For every Phase 2 row with both a scored proposer answer and a scored
verifier answer, classifies into one of four outcomes:

  ECHO_CORRECT:      proposer correct, verifier keeps it correct
  HARMFUL_OVERRIDE:  proposer correct, verifier changes it to incorrect
  HELPFUL_CORRECTION: proposer incorrect, verifier corrects it
  NO_HELP:           proposer incorrect, verifier leaves/changes it, still incorrect

Reports overall rates and per-verifier rates, so we can see whether
negative synergy is driven by verifiers actively breaking correct answers
(HARMFUL_OVERRIDE) vs. simply failing to fix wrong ones (NO_HELP).

Proposer correctness is recomputed from raw proposer_answer against ground
truth (not logged directly in Phase 2 schema, since Phase 2 proposer calls
are independent model runs, not reused from Phase 1).

Usage:
    python3 analyze_overrides.py
"""

import json
from collections import defaultdict

from benchmark import BenchmarkLoader
from evaluator import score_standalone, score_code

DATASET_PATH = "question_dataset_v2.json"
PHASE2_PATH = "phase2_results.jsonl"

CODE_GENERATION_TYPES = {"code_generation"}


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_func_name(question_text: str):
    import re
    m = re.search(r'`(\w+)\s*\(', question_text)
    return m.group(1) if m else None


def classify_row(row, question, questions_by_id):
    """
    Returns one of: 'ECHO_CORRECT', 'HARMFUL_OVERRIDE', 'HELPFUL_CORRECTION',
    'NO_HELP', or None if the row can't be classified (missing data).
    """
    if row["proposer_answer"] is None or row["verifier_answer"] is None:
        return None

    is_code = question.domain == "coding" and question.type in CODE_GENERATION_TYPES

    if is_code:
        func_name = _extract_func_name(question.question)
        proposer_score = score_code(row["proposer_answer"], question.test_cases,
                                     expected_func_name=func_name)
        proposer_correct = proposer_score >= 1.0
        # verifier correctness: use collaboration_score directly (already
        # the verifier's score per protocol 8.2), treating >=1.0 as correct.
        if row["collaboration_score"] is None:
            return None
        verifier_correct = row["collaboration_score"] >= 1.0
    else:
        proposer_result = score_standalone(row["proposer_answer"], question.final_answer)
        if proposer_result is None:
            return None
        proposer_correct = proposer_result

        if row["collaboration_score"] is None:
            return None
        verifier_correct = bool(row["collaboration_score"])

    if proposer_correct and verifier_correct:
        return "ECHO_CORRECT"
    elif proposer_correct and not verifier_correct:
        return "HARMFUL_OVERRIDE"
    elif not proposer_correct and verifier_correct:
        return "HELPFUL_CORRECTION"
    else:
        return "NO_HELP"


def main():
    loader = BenchmarkLoader(DATASET_PATH)
    questions_by_id = {q.id: q for q in loader.load_questions()}

    phase2_rows = load_jsonl(PHASE2_PATH)

    overall_counts = defaultdict(int)
    per_verifier_counts = defaultdict(lambda: defaultdict(int))
    unclassifiable = 0

    for row in phase2_rows:
        question = questions_by_id.get(row["question_id"])
        if question is None:
            unclassifiable += 1
            continue

        outcome = classify_row(row, question, questions_by_id)
        if outcome is None:
            unclassifiable += 1
            continue

        overall_counts[outcome] += 1
        per_verifier_counts[row["verifier_id"]][outcome] += 1

    total_classified = sum(overall_counts.values())

    print("=" * 70)
    print("OVERALL OUTCOME BREAKDOWN")
    print("=" * 70)
    print(f"Total rows: {len(phase2_rows)}")
    print(f"Classified: {total_classified}")
    print(f"Unclassifiable (missing data): {unclassifiable}\n")

    for outcome in ["ECHO_CORRECT", "HARMFUL_OVERRIDE", "HELPFUL_CORRECTION", "NO_HELP"]:
        count = overall_counts[outcome]
        pct = count / total_classified if total_classified > 0 else 0
        print(f"  {outcome:<20} {count:>6} ({pct:.1%})")

    harmful = overall_counts["HARMFUL_OVERRIDE"]
    helpful = overall_counts["HELPFUL_CORRECTION"]
    print(f"\nHarmful overrides vs helpful corrections: {harmful} vs {helpful}")
    if harmful > helpful:
        print("=> Verifiers break MORE correct answers than they fix incorrect ones. "
              "This directly explains negative average synergy.")
    else:
        print("=> Verifiers fix more than they break, in aggregate. "
              "Negative synergy (if present) may come from a different source.")

    print("\n" + "=" * 70)
    print("PER-VERIFIER BREAKDOWN (sorted by harmful override rate, worst first)")
    print("=" * 70)
    print(f"{'Model':<22} {'Echo':>6} {'Harmful':>8} {'Helpful':>8} {'NoHelp':>7} "
          f"{'HarmfulRate':>12} {'HelpfulRate':>12}")

    rows_out = []
    for verifier, counts in per_verifier_counts.items():
        total = sum(counts.values())
        harmful_rate = counts["HARMFUL_OVERRIDE"] / total if total > 0 else 0
        helpful_rate = counts["HELPFUL_CORRECTION"] / total if total > 0 else 0
        rows_out.append((verifier, counts, total, harmful_rate, helpful_rate))

    rows_out.sort(key=lambda r: -r[3])

    for verifier, counts, total, harmful_rate, helpful_rate in rows_out:
        print(f"{verifier:<22} {counts['ECHO_CORRECT']:>6} "
              f"{counts['HARMFUL_OVERRIDE']:>8} {counts['HELPFUL_CORRECTION']:>8} "
              f"{counts['NO_HELP']:>7} {harmful_rate:>11.1%} {helpful_rate:>11.1%}")


if __name__ == "__main__":
    main()
