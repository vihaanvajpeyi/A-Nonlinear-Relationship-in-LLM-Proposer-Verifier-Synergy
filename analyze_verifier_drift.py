"""
analyze_verifier_drift.py — Directly compares each verifier's Phase 2
verifying accuracy against its own Phase 1 standalone accuracy, on the
identical set of questions, to test whether reviewing another model's
answer degrades accuracy relative to answering fresh.

Methodology:
  For each verifier model V and each question Q:
    - Phase 1 standalone score: V's single score answering Q fresh
      (from phase1_results_reparsed.jsonl).
    - Phase 2 verifying scores: V's score(s) as verifier on Q, pooled
      across all 14 proposers it was paired with (from phase2_results.jsonl).
      Each proposer pairing is a separate attempt at the same question, so
      this gives up to 14 verifying attempts per question vs. 1 standalone
      attempt.

  Only questions with a non-null Phase 1 score AND at least one non-null
  Phase 2 verifying score are included (matched sample) — this keeps the
  comparison apples-to-apples rather than penalizing/crediting a model for
  parse failures on one side but not the other.

  Two comparisons are reported per verifier:
    1. Per-question paired comparison: for each matched question, is the
       average Phase 2 verifying accuracy (across proposers) higher, lower,
       or equal to the single Phase 1 standalone score? Reports counts of
       "worse", "same", "better", plus a paired t-test on the differences.
    2. Pooled comparison: overall Phase 1 standalone accuracy vs overall
       Phase 2 verifying accuracy (all matched rows pooled), giving a
       single "accuracy drops by X points" headline number per verifier.

Usage:
    python3 analyze_verifier_drift.py
"""

import json
from collections import defaultdict

import numpy as np
from scipy import stats

PHASE1_PATH = "phase1_results_reparsed.jsonl"
PHASE2_PATH = "phase2_results.jsonl"


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _numeric(score):
    """Convert a score (bool, float, or None) to a numeric value in [0,1],
    or None if null."""
    if score is None:
        return None
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    return float(score)


def build_phase1_lookup(phase1_rows: list) -> dict:
    """(model_id, question_id) -> numeric score, non-null only."""
    lookup = {}
    for row in phase1_rows:
        if row["phase"] != 1:
            continue
        score = _numeric(row["score"])
        if score is not None:
            lookup[(row["model_id"], row["question_id"])] = score
    return lookup


def build_phase2_verifying_scores(phase2_rows: list) -> dict:
    """(verifier_id, question_id) -> list of numeric collaboration scores
    (non-null only), one per proposer pairing."""
    lookup = defaultdict(list)
    for row in phase2_rows:
        score = _numeric(row["collaboration_score"])
        if score is not None:
            key = (row["verifier_id"], row["question_id"])
            lookup[key].append(score)
    return lookup


def analyze_verifier(model_id: str, all_question_ids: list,
                      phase1_lookup: dict, phase2_lookup: dict) -> dict:
    """Compute the matched-sample comparison for a single verifier model."""
    per_question_diffs = []
    phase1_matched_scores = []
    phase2_matched_scores_pooled = []
    matched_question_ids = []

    for qid in all_question_ids:
        p1_score = phase1_lookup.get((model_id, qid))
        p2_scores = phase2_lookup.get((model_id, qid))

        if p1_score is None or not p2_scores:
            continue  # not in matched sample

        matched_question_ids.append(qid)
        p2_avg = sum(p2_scores) / len(p2_scores)

        per_question_diffs.append(p2_avg - p1_score)
        phase1_matched_scores.append(p1_score)
        phase2_matched_scores_pooled.extend(p2_scores)

    n_matched_questions = len(matched_question_ids)
    if n_matched_questions == 0:
        return {
            "model_id": model_id,
            "n_matched_questions": 0,
            "phase1_accuracy": None,
            "phase2_accuracy": None,
            "drift": None,
            "relative_drift": None,
            "worse": 0, "same": 0, "better": 0,
            "paired_ttest_p": None,
        }

    phase1_accuracy = float(np.mean(phase1_matched_scores))
    phase2_accuracy = float(np.mean(phase2_matched_scores_pooled))
    drift = phase2_accuracy - phase1_accuracy
    relative_drift = drift / phase1_accuracy if phase1_accuracy > 0 else None

    diffs = np.array(per_question_diffs)
    worse = int(np.sum(diffs < -1e-9))
    same = int(np.sum(np.abs(diffs) <= 1e-9))
    better = int(np.sum(diffs > 1e-9))

    paired_p = None
    if n_matched_questions >= 2 and np.std(diffs) > 0:
        t_stat, paired_p = stats.ttest_1samp(diffs, 0.0)

    return {
        "model_id": model_id,
        "n_matched_questions": n_matched_questions,
        "n_phase2_rows_pooled": len(phase2_matched_scores_pooled),
        "phase1_accuracy": phase1_accuracy,
        "phase2_accuracy": phase2_accuracy,
        "drift": drift,
        "relative_drift": relative_drift,
        "worse": worse, "same": same, "better": better,
        "paired_ttest_p": paired_p,
    }


def main():
    phase1_rows = load_jsonl(PHASE1_PATH)
    phase2_rows = load_jsonl(PHASE2_PATH)

    phase1_lookup = build_phase1_lookup(phase1_rows)
    phase2_lookup = build_phase2_verifying_scores(phase2_rows)

    all_models = sorted(set(model_id for model_id, _ in phase1_lookup.keys()))
    all_question_ids = sorted(set(qid for _, qid in phase1_lookup.keys()))

    results = [
        analyze_verifier(model_id, all_question_ids, phase1_lookup, phase2_lookup)
        for model_id in all_models
    ]

    # Sort by drift ascending (biggest degradation first)
    results_sorted = sorted(
        [r for r in results if r["drift"] is not None],
        key=lambda r: r["drift"]
    )

    print("=" * 100)
    print(f"{'Model':<22} {'N Qs':>5} {'Phase1 Acc':>11} {'Phase2 Acc':>11} "
          f"{'Drift':>8} {'RelDrift':>9} {'Worse':>6} {'Same':>5} {'Better':>7} {'p-value':>9}")
    print("=" * 100)

    for r in results_sorted:
        p_str = f"{r['paired_ttest_p']:.4f}" if r["paired_ttest_p"] is not None else "N/A"
        rel_str = f"{r['relative_drift']:+.1%}" if r["relative_drift"] is not None else "N/A"
        print(f"{r['model_id']:<22} {r['n_matched_questions']:>5} "
              f"{r['phase1_accuracy']:>10.1%} {r['phase2_accuracy']:>10.1%} "
              f"{r['drift']:>+7.1%} {rel_str:>9} {r['worse']:>6} {r['same']:>5} {r['better']:>7} "
              f"{p_str:>9}")

    print("=" * 100)

    # --- Relative drift table, sorted separately ---
    results_by_relative = sorted(
        [r for r in results if r["relative_drift"] is not None],
        key=lambda r: r["relative_drift"]
    )

    print("\n" + "=" * 70)
    print("SORTED BY RELATIVE DRIFT (drift as % of standalone accuracy)")
    print("=" * 70)
    print("A large NEGATIVE relative drift for a HIGH standalone-accuracy model")
    print("is evidence of genuine behavioral degradation, not just a ceiling")
    print("effect (i.e. 'had more room to fall so fell more').\n")
    print(f"{'Model':<22} {'Phase1 Acc':>11} {'RelDrift':>10}")
    for r in results_by_relative:
        print(f"{r['model_id']:<22} {r['phase1_accuracy']:>10.1%} "
              f"{r['relative_drift']:>+9.1%}")

    # --- Correlation: does relative drift depend on standalone accuracy? ---
    print("\n" + "=" * 70)
    print("CEILING-EFFECT CHECK: correlation between standalone accuracy")
    print("and relative drift")
    print("=" * 70)
    print("If relative drift is roughly CONSTANT across standalone accuracy")
    print("levels, degradation is a real, capability-independent behavioral")
    print("effect. If relative drift gets LESS negative as standalone")
    print("accuracy rises, some of the earlier absolute-drift pattern was a")
    print("ceiling effect rather than genuine anchoring.\n")

    if len(results_by_relative) >= 3:
        p1_vals = np.array([r["phase1_accuracy"] for r in results_by_relative])
        rel_vals = np.array([r["relative_drift"] for r in results_by_relative])
        corr, corr_p = stats.pearsonr(p1_vals, rel_vals)
        print(f"Pearson correlation (standalone accuracy vs relative drift): "
              f"r={corr:.3f}, p={corr_p:.4f}")
        if corr_p < 0.05:
            if corr > 0:
                print("=> Significant POSITIVE correlation: stronger models show LESS "
                      "relative degradation. Suggests some of the absolute-drift pattern "
                      "reflects a ceiling effect, though weaker models still show real "
                      "(and proportionally larger) degradation.")
            else:
                print("=> Significant NEGATIVE correlation: stronger models show MORE "
                      "relative degradation, even accounting for having more room to "
                      "fall. This is strong evidence of genuine behavioral anchoring "
                      "on the proposer's answer, not just arithmetic ceiling effects.")
        else:
            print("=> No significant correlation between standalone accuracy and "
                  "relative drift — degradation appears roughly capability-independent, "
                  "consistent with a genuine behavioral effect rather than a pure "
                  "ceiling artifact.")
    else:
        print("Too few data points for correlation test.")

    # Overall headline number: pooled across all verifiers
    all_p1 = []
    all_p2 = []
    for r in results:
        if r["drift"] is None:
            continue
        all_p1.append(r["phase1_accuracy"])
        all_p2.append(r["phase2_accuracy"])

    if all_p1:
        overall_p1 = np.mean(all_p1)
        overall_p2 = np.mean(all_p2)
        print(f"\nOverall (averaged across {len(all_p1)} verifiers with matched data):")
        print(f"  Phase 1 standalone accuracy: {overall_p1:.1%}")
        print(f"  Phase 2 verifying accuracy:  {overall_p2:.1%}")
        print(f"  Drift: {overall_p2 - overall_p1:+.1%}")

        # Paired t-test across verifier-level means (n = number of verifiers)
        if len(all_p1) >= 2:
            t_stat, p_val = stats.ttest_rel(all_p2, all_p1)
            print(f"  Paired t-test across verifiers: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                direction = "DECREASE" if overall_p2 < overall_p1 else "INCREASE"
                print(f"  => Statistically significant {direction} in accuracy "
                      f"when verifying vs answering standalone.")
            else:
                print(f"  => Not statistically significant at p<0.05.")


if __name__ == "__main__":
    main()
