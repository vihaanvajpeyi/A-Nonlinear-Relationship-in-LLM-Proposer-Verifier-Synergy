"""
analyze_phase1.py — Post-run analysis for Phase 1 standalone results.

Reads phase1_results.jsonl and reports:
  - Per-model accuracy (excluding parse failures from the denominator, per
    protocol_v2.md 8.1 — parse failures are null, not 0)
  - Per-model parse failure rate
  - Separation of genuine parse failures (model responded, output didn't
    match expected format) vs. model-call failures (timeout/connection
    error — logged as parse failures but for an unrelated reason)

Usage:
    python3 analyze_phase1.py
"""

import json
from collections import defaultdict

RESULTS_PATH = "phase1_results.jsonl"


def load_results(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze(rows: list):
    per_model = defaultdict(lambda: {
        "total": 0,
        "scored": 0,          # non-null score (successfully parsed)
        "correct": 0,         # score == True or score == 1.0
        "genuine_parse_failures": 0,
        "infra_failures": 0,
    })

    for row in rows:
        if row["phase"] != 1:
            continue
        model = row["model_id"]
        stats = per_model[model]
        stats["total"] += 1

        is_infra_failure = (
            row["parse_failure"]
            and isinstance(row["raw_response"], str)
            and row["raw_response"].startswith("[MODEL CALL FAILED]")
        )

        if is_infra_failure:
            stats["infra_failures"] += 1
        elif row["parse_failure"]:
            stats["genuine_parse_failures"] += 1
        else:
            stats["scored"] += 1
            score = row["score"]
            # Handle both bool (math/debugging) and float (code) scores.
            if score is True or (isinstance(score, (int, float)) and score >= 1.0):
                stats["correct"] += 1
            elif isinstance(score, (int, float)) and 0 < score < 1.0:
                # Partial credit (code generation, some test cases passed)
                stats["correct"] += score  # fractional credit toward "correct"

    return per_model


def print_report(per_model: dict):
    print("=" * 90)
    print(f"{'Model':<22} {'Total':>6} {'Scored':>7} {'Accuracy':>9} "
          f"{'Parse Fail':>11} {'Infra Fail':>11} {'PF Rate':>8}")
    print("=" * 90)

    # Sort by accuracy descending for readability
    def accuracy(stats):
        return stats["correct"] / stats["scored"] if stats["scored"] > 0 else 0.0

    for model, stats in sorted(per_model.items(), key=lambda kv: -accuracy(kv[1])):
        acc = accuracy(stats)
        pf_rate = (stats["genuine_parse_failures"] + stats["infra_failures"]) / stats["total"]
        print(f"{model:<22} {stats['total']:>6} {stats['scored']:>7} "
              f"{acc:>8.1%} {stats['genuine_parse_failures']:>11} "
              f"{stats['infra_failures']:>11} {pf_rate:>7.1%}")

    print("=" * 90)

    total_genuine_pf = sum(s["genuine_parse_failures"] for s in per_model.values())
    total_infra = sum(s["infra_failures"] for s in per_model.values())
    total_rows = sum(s["total"] for s in per_model.values())

    print(f"\nTotal rows: {total_rows}")
    print(f"Genuine parse failures (model responded, bad format): {total_genuine_pf}")
    print(f"Infra failures (timeout/connection error): {total_infra}")
    print(f"Combined failure rate: {(total_genuine_pf + total_infra) / total_rows:.1%}")

    if total_infra > 0:
        print(f"\nNOTE: {total_infra} infra failures found. These are NOT genuine model "
              f"behavior — consider whether to re-run just these specific "
              f"(model, question) combinations, since they reflect plumbing issues "
              f"rather than model capability. This would be a targeted re-run, not a "
              f"violation of the no-retries policy (which is about not cherry-picking "
              f"better model outputs).")


if __name__ == "__main__":
    rows = load_results(RESULTS_PATH)
    per_model = analyze(rows)
    print_report(per_model)
