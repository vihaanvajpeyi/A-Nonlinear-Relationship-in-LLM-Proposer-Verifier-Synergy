"""
analyze_synergy_v3.py — Two additional robustness analyses building on
analyze_synergy_v2.py:

1. MULTI-SEED SPLIT-HALF VALIDATION
   A single split-half test (one random seed) could look strong or weak by
   chance. This reruns the split-half independence test (verifying_accuracy
   from a random half, avg_synergy from the disjoint other half) across N
   different random seeds, and reports the distribution of quadratic
   coefficients and p-values — e.g. "significant in 14/20 splits" is a much
   harder claim to dispute than a single split's result.

2. DOMAIN DECOMPOSITION
   Quantifies precisely how much the pooled inverted-U reflects coding vs.
   math data, by computing the correlation (across the 15 verifier models)
   between each model's POOLED avg_synergy and its MATH-only vs CODING-only
   avg_synergy. A high pooled-vs-coding correlation and low pooled-vs-math
   correlation would numerically confirm "the pooled effect is substantially
   driven by coding-domain data" rather than leaving that claim qualitative.

Usage:
    python3 analyze_synergy_v3.py
"""

import json
import random
from collections import defaultdict

import numpy as np
import statsmodels.api as sm
from scipy import stats

from benchmark import BenchmarkLoader

PHASE2_PATH = "phase2_results.jsonl"
DATASET_PATH = "question_dataset_v2.json"
N_SEEDS = 20


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_domain_stats_splithalf(phase2_rows: list, question_domains: dict, seed: int) -> dict:
    """Same logic as analyze_synergy_v2.py's version, reproduced here so
    this script is self-contained and doesn't depend on import order."""
    rng = random.Random(seed)

    grouped = defaultdict(list)
    for row in phase2_rows:
        verifier = row["verifier_id"]
        domain = question_domains.get(row["question_id"], "unknown")
        if domain == "unknown":
            continue
        grouped[(verifier, domain)].append(row)
        grouped[(verifier, "pooled")].append(row)

    result = {}
    for key, rows in grouped.items():
        shuffled = rows[:]
        rng.shuffle(shuffled)
        midpoint = len(shuffled) // 2
        half_a = shuffled[:midpoint]
        half_b = shuffled[midpoint:]

        a_scored = [r for r in half_a if r["collaboration_score"] is not None]
        if a_scored:
            a_vals = [1.0 if r["collaboration_score"] is True else
                      (0.0 if r["collaboration_score"] is False else float(r["collaboration_score"]))
                      for r in a_scored]
            verifying_accuracy = sum(a_vals) / len(a_vals)
        else:
            verifying_accuracy = None

        b_synergy = [r["synergy"] for r in half_b if r["synergy"] is not None]
        avg_synergy = sum(b_synergy) / len(b_synergy) if b_synergy else None

        result[key] = {"verifying_accuracy": verifying_accuracy, "avg_synergy": avg_synergy}
    return result


def compute_domain_stats_full(phase2_rows: list, question_domains: dict) -> dict:
    """Standard (non-split) per-(verifier, domain) avg_synergy, for the
    domain decomposition correlation check."""
    buckets = defaultdict(lambda: {"synergy_sum": 0.0, "synergy_n": 0})
    for row in phase2_rows:
        verifier = row["verifier_id"]
        domain = question_domains.get(row["question_id"], "unknown")
        if domain == "unknown":
            continue
        for key_domain in (domain, "pooled"):
            key = (verifier, key_domain)
            if row["synergy"] is not None:
                buckets[key]["synergy_sum"] += row["synergy"]
                buckets[key]["synergy_n"] += 1

    result = {}
    for key, b in buckets.items():
        result[key] = (b["synergy_sum"] / b["synergy_n"]) if b["synergy_n"] > 0 else None
    return result


def fit_quadratic_and_test(x: np.ndarray, y: np.ndarray):
    X = np.column_stack([x, x ** 2])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.params[2], model.pvalues[2], model.rsquared


def run_multiseed_splithalf(phase2_rows, question_domains, all_models, domain, n_seeds):
    print(f"\n{'=' * 70}")
    print(f"MULTI-SEED SPLIT-HALF — {domain.upper()} (n_seeds={n_seeds})")
    print(f"{'=' * 70}")

    coefs, pvalues, r2s, significant_negative = [], [], [], 0

    for seed in range(n_seeds):
        stats_by_key = compute_domain_stats_splithalf(phase2_rows, question_domains, seed=seed)

        x_vals, y_vals = [], []
        for model in all_models:
            d = stats_by_key.get((model, domain))
            if d is None or d["verifying_accuracy"] is None or d["avg_synergy"] is None:
                continue
            x_vals.append(d["verifying_accuracy"])
            y_vals.append(d["avg_synergy"])

        if len(x_vals) < 4:
            continue

        x = np.array(x_vals)
        y = np.array(y_vals)
        coef, pval, r2 = fit_quadratic_and_test(x, y)
        coefs.append(coef)
        pvalues.append(pval)
        r2s.append(r2)
        if coef < 0 and pval < 0.05:
            significant_negative += 1

    n_runs = len(coefs)
    if n_runs == 0:
        print("No valid runs (insufficient data in every split).")
        return

    coefs = np.array(coefs)
    pvalues = np.array(pvalues)
    r2s = np.array(r2s)

    print(f"Valid splits: {n_runs}/{n_seeds}")
    print(f"Quadratic coefficient: mean={coefs.mean():.3f}, "
          f"std={coefs.std():.3f}, range=[{coefs.min():.3f}, {coefs.max():.3f}]")
    print(f"Fraction of splits with negative coefficient: "
          f"{np.mean(coefs < 0):.1%}")
    print(f"Fraction of splits significant (p<0.05) AND negative: "
          f"{significant_negative}/{n_runs} ({significant_negative/n_runs:.1%})")
    print(f"Median p-value across splits: {np.median(pvalues):.4f}")
    print(f"Mean R-squared across splits: {r2s.mean():.3f}")

    if significant_negative / n_runs >= 0.5:
        print("=> MAJORITY of independent splits show a significant inverted-U. "
              "Strong evidence this is a real effect, not a single-split artifact.")
    elif np.mean(coefs < 0) >= 0.75:
        print("=> Most splits show a negative (inverted-U-consistent) coefficient, "
              "even if not always individually significant — direction is consistent, "
              "but this sample size limits per-split statistical power.")
    else:
        print("=> Inconsistent sign/significance across splits — treat the single-split "
              "result with real caution; may not be a robust effect.")


def run_domain_decomposition(all_models, full_domain_stats):
    print(f"\n{'=' * 70}")
    print("DOMAIN DECOMPOSITION: what drives the pooled effect?")
    print(f"{'=' * 70}")
    print("Correlation (across 15 verifier models) between each model's POOLED")
    print("avg_synergy and its MATH-only vs CODING-only avg_synergy. Higher")
    print("correlation = that domain's per-model pattern more closely tracks,")
    print("and therefore more strongly drives, the pooled result.\n")

    pooled_vals, math_vals, coding_vals, labels = [], [], [], []
    for model in all_models:
        p = full_domain_stats.get((model, "pooled"))
        m = full_domain_stats.get((model, "math"))
        c = full_domain_stats.get((model, "coding"))
        if p is None or m is None or c is None:
            continue
        pooled_vals.append(p)
        math_vals.append(m)
        coding_vals.append(c)
        labels.append(model)

    n = len(pooled_vals)
    print(f"Models with complete data: {n}\n")
    if n < 4:
        print("Too few models for correlation analysis.")
        return

    pooled_arr = np.array(pooled_vals)
    math_arr = np.array(math_vals)
    coding_arr = np.array(coding_vals)

    r_math, p_math = stats.pearsonr(pooled_arr, math_arr)
    r_coding, p_coding = stats.pearsonr(pooled_arr, coding_arr)

    print(f"Pooled vs Math-only synergy:   r={r_math:.3f}, p={p_math:.4f}")
    print(f"Pooled vs Coding-only synergy: r={r_coding:.3f}, p={p_coding:.4f}")
    print()

    if r_coding > r_math:
        diff = r_coding - r_math
        print(f"=> Coding correlates MORE strongly with pooled synergy "
              f"(difference: {diff:.3f}). Numerically confirms the pooled "
              f"inverted-U is substantially driven by coding-domain data.")
    else:
        diff = r_math - r_coding
        print(f"=> Math correlates MORE strongly with pooled synergy "
              f"(difference: {diff:.3f}). Pooled effect is NOT primarily "
              f"a coding artifact — reconsider the domain-driver claim.")


if __name__ == "__main__":
    phase2_rows = load_jsonl(PHASE2_PATH)
    loader = BenchmarkLoader(DATASET_PATH)
    question_domains = {q.id: q.domain for q in loader.load_questions()}

    all_models = sorted(set(row["verifier_id"] for row in phase2_rows))

    # --- Part 1: multi-seed split-half validation ---
    run_multiseed_splithalf(phase2_rows, question_domains, all_models, "pooled", N_SEEDS)
    run_multiseed_splithalf(phase2_rows, question_domains, all_models, "coding", N_SEEDS)
    run_multiseed_splithalf(phase2_rows, question_domains, all_models, "math", N_SEEDS)

    # --- Part 2: domain decomposition ---
    full_domain_stats = compute_domain_stats_full(phase2_rows, question_domains)
    run_domain_decomposition(all_models, full_domain_stats)
