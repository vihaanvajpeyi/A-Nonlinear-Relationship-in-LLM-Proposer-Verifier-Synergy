"""
analyze_synergy_v2.py — Refit of the inverted-U hypothesis using the
theoretically correct predictor (verification accuracy, not standalone
accuracy), and split by domain (math vs. coding).

Motivation:
  The original hypothesis (and original 8-model study) used standalone
  accuracy to predict synergy. analyze_verifier_drift.py showed standalone
  and verifying accuracy diverge substantially and significantly (p=0.0065)
  for this 15-model roster — so predicting synergy from standalone accuracy
  is testing a mis-specified model. This script refits using each model's
  own Phase 2 verifying accuracy (matched sample, same methodology as
  analyze_verifier_drift.py) as the predictor instead.

  Domain split: math and coding verification are plausibly different tasks
  (numeric/symbolic answer-checking vs. code correctness reasoning). This
  script fits the regression separately within each domain, in addition to
  the pooled (both domains) version, so domain-specific patterns aren't
  washed out by pooling.

Four regressions are reported:
  1. Pooled: synergy ~ verifying_accuracy + verifying_accuracy^2
  2. Math only: synergy ~ verifying_accuracy (math) + ^2
  3. Coding only: synergy ~ verifying_accuracy (coding) + ^2
  4. Side-by-side comparison table of all three

Each includes LOO-CV, consistent with analyze_synergy.py's methodology.

Usage:
    python3 analyze_synergy_v2.py
"""

import json
from collections import defaultdict

import numpy as np
import statsmodels.api as sm

from benchmark import BenchmarkLoader

PHASE2_PATH = "phase2_results.jsonl"
DATASET_PATH = "question_dataset_v2.json"
COVERAGE_THRESHOLD = 0.50


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_domain_stats(phase2_rows: list, question_domains: dict) -> dict:
    """
    Per (verifier, domain) -> {verifying_accuracy, avg_synergy, coverage,
    n_rows}, computed directly from Phase 2 rows (no Phase 1 involved —
    this IS the verifying accuracy, not standalone).

    domain is 'math', 'coding', or 'pooled' (all rows regardless of domain).
    """
    # buckets[(verifier, domain)] = {"total":..., "score_sum":..., "scored":...,
    #                                 "synergy_sum":..., "synergy_n":...}
    buckets = defaultdict(lambda: {
        "total": 0, "score_sum": 0.0, "scored": 0,
        "synergy_sum": 0.0, "synergy_n": 0,
    })

    for row in phase2_rows:
        verifier = row["verifier_id"]
        domain = question_domains.get(row["question_id"], "unknown")
        if domain == "unknown":
            continue

        for key_domain in (domain, "pooled"):
            key = (verifier, key_domain)
            b = buckets[key]
            b["total"] += 1

            score = row["collaboration_score"]
            if score is not None:
                numeric_score = 1.0 if score is True else (
                    0.0 if score is False else float(score)
                )
                b["score_sum"] += numeric_score
                b["scored"] += 1

            if row["synergy"] is not None:
                b["synergy_sum"] += row["synergy"]
                b["synergy_n"] += 1

    result = {}
    for (verifier, domain), b in buckets.items():
        verifying_accuracy = (b["score_sum"] / b["scored"]) if b["scored"] > 0 else None
        avg_synergy = (b["synergy_sum"] / b["synergy_n"]) if b["synergy_n"] > 0 else None
        coverage = (b["synergy_n"] / b["total"]) if b["total"] > 0 else 0.0
        result[(verifier, domain)] = {
            "verifying_accuracy": verifying_accuracy,
            "avg_synergy": avg_synergy,
            "coverage": coverage,
            "n_rows": b["total"],
        }
    return result


def compute_domain_stats_splithalf(phase2_rows: list, question_domains: dict, seed: int = 42) -> dict:
    """
    Split-half version: for each (verifier, domain), randomly split that
    verifier's rows into two halves. verifying_accuracy is computed from
    half A, avg_synergy from half B (or vice versa, symmetric — we use A
    for accuracy, B for synergy consistently). This breaks the shared-
    metric dependency present in compute_domain_stats, where both
    verifying_accuracy and synergy are derived from the SAME
    collaboration_score values on the SAME rows (synergy directly
    includes collaboration_correct as a term, so predictor and target
    are not independent when computed from identical rows).

    This is the methodologically correct test: does verifying accuracy,
    measured on one set of interactions, predict synergy on a DIFFERENT
    set of interactions for the same model? If yes, that's genuine
    predictive signal, not an artifact of shared arithmetic.
    """
    import random as _random
    rng = _random.Random(seed)

    # Group rows by (verifier, domain), keeping 'pooled' as all rows too.
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

        # verifying_accuracy from half A
        a_scored = [r for r in half_a if r["collaboration_score"] is not None]
        if a_scored:
            a_vals = [1.0 if r["collaboration_score"] is True else
                      (0.0 if r["collaboration_score"] is False else float(r["collaboration_score"]))
                      for r in a_scored]
            verifying_accuracy = sum(a_vals) / len(a_vals)
        else:
            verifying_accuracy = None

        # avg_synergy from half B
        b_synergy = [r["synergy"] for r in half_b if r["synergy"] is not None]
        avg_synergy = sum(b_synergy) / len(b_synergy) if b_synergy else None

        result[key] = {
            "verifying_accuracy": verifying_accuracy,
            "avg_synergy": avg_synergy,
            "coverage": len(b_synergy) / len(half_b) if half_b else 0.0,
            "n_rows": len(rows),
            "n_half_a": len(a_scored),
            "n_half_b": len(b_synergy),
        }
    return result


def fit_quadratic_ols(x: np.ndarray, y: np.ndarray):
    X = np.column_stack([x, x ** 2])
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()


def loo_cv_r2(x: np.ndarray, y: np.ndarray):
    n = len(x)
    if n < 4:
        return None, "Too few points for meaningful LOO-CV (need >= 4)"

    predictions = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = np.column_stack([x[mask], x[mask] ** 2])
        X_train = sm.add_constant(X_train)
        y_train = y[mask]
        model = sm.OLS(y_train, X_train).fit()
        x_test = np.array([1.0, x[i], x[i] ** 2])
        predictions[i] = model.predict(x_test)[0]

    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return r2, None


def run_domain_analysis(label: str, domain: str, all_models: list, domain_stats: dict,
                         min_coverage: float = 0.0):
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    x_vals, y_vals, labels = [], [], []
    for model in all_models:
        stats = domain_stats.get((model, domain))
        if stats is None:
            continue
        if stats["verifying_accuracy"] is None or stats["avg_synergy"] is None:
            continue
        if stats["coverage"] < min_coverage:
            continue
        x_vals.append(stats["verifying_accuracy"])
        y_vals.append(stats["avg_synergy"])
        labels.append(model)

    n = len(x_vals)
    print(f"\nData points: {n} (min coverage required: {min_coverage:.0%})")
    if n < 4:
        print("Too few data points for regression. Skipping.")
        return None

    print(f"\n{'Model':<22} {'Verifying Acc':>14} {'Avg Synergy':>13} {'Coverage':>10}")
    for model, x, y in zip(labels, x_vals, y_vals):
        cov = domain_stats[(model, domain)]["coverage"]
        print(f"{model:<22} {x:>13.1%} {y:>13.3f} {cov:>9.1%}")

    x = np.array(x_vals)
    y = np.array(y_vals)

    model = fit_quadratic_ols(x, y)
    quad_coef = model.params[2]
    quad_pvalue = model.pvalues[2]

    print(f"\nQuadratic coefficient: {quad_coef:.4f} (p = {quad_pvalue:.4f})")
    if quad_coef < 0 and quad_pvalue < 0.05:
        print("=> SIGNIFICANT negative quadratic term: supports inverted-U shape.")
    elif quad_coef < 0:
        print("=> Negative (consistent with inverted-U), not significant at p<0.05.")
    else:
        print("=> Quadratic term not negative — does NOT support inverted-U shape.")

    if quad_coef < 0:
        peak_x = -model.params[1] / (2 * quad_coef)
        print(f"Estimated peak verifying accuracy: {peak_x:.1%}")

    print(f"R-squared (in-sample): {model.rsquared:.4f}")

    r2_loo, error = loo_cv_r2(x, y)
    if error:
        print(f"LOO-CV: {error}")
    else:
        print(f"LOO-CV R-squared: {r2_loo:.4f}")

    return {
        "n": n, "quad_coef": quad_coef, "quad_pvalue": quad_pvalue,
        "r2": model.rsquared, "loo_r2": r2_loo,
    }


if __name__ == "__main__":
    phase2_rows = load_jsonl(PHASE2_PATH)
    loader = BenchmarkLoader(DATASET_PATH)
    question_domains = {q.id: q.domain for q in loader.load_questions()}

    domain_stats = compute_domain_stats(phase2_rows, question_domains)
    all_models = sorted(set(v for v, d in domain_stats.keys()))

    print("Per-model verifying accuracy and coverage by domain:")
    print(f"{'Model':<22} {'Pooled VerAcc':>14} {'Math VerAcc':>13} {'Coding VerAcc':>14} "
          f"{'Pooled Cov':>11}")
    for model in all_models:
        pooled = domain_stats.get((model, "pooled"), {})
        math = domain_stats.get((model, "math"), {})
        coding = domain_stats.get((model, "coding"), {})
        p_acc = pooled.get("verifying_accuracy")
        m_acc = math.get("verifying_accuracy")
        c_acc = coding.get("verifying_accuracy")
        p_cov = pooled.get("coverage", 0.0)
        p_str = f"{p_acc:.1%}" if p_acc is not None else "N/A"
        m_str = f"{m_acc:.1%}" if m_acc is not None else "N/A"
        c_str = f"{c_acc:.1%}" if c_acc is not None else "N/A"
        print(f"{model:<22} {p_str:>14} {m_str:>13} {c_str:>14} {p_cov:>10.1%}")

    results = {}
    results["pooled"] = run_domain_analysis(
        "POOLED — synergy ~ verifying_accuracy (all questions)",
        "pooled", all_models, domain_stats
    )
    results["math"] = run_domain_analysis(
        "MATH ONLY — synergy ~ verifying_accuracy (math questions)",
        "math", all_models, domain_stats
    )
    results["coding"] = run_domain_analysis(
        "CODING ONLY — synergy ~ verifying_accuracy (coding questions)",
        "coding", all_models, domain_stats
    )

    print(f"\n{'=' * 70}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Domain':<10} {'N':>4} {'Quad Coef':>11} {'p-value':>9} {'R2':>8} {'LOO-CV R2':>11}")
    for domain, r in results.items():
        if r is None:
            print(f"{domain:<10} insufficient data")
            continue
        loo_str = f"{r['loo_r2']:.4f}" if r['loo_r2'] is not None else "N/A"
        print(f"{domain:<10} {r['n']:>4} {r['quad_coef']:>11.4f} {r['quad_pvalue']:>9.4f} "
              f"{r['r2']:>8.4f} {loo_str:>11}")

    # --- Robustness check: math regression excluding low-coverage points ---
    print(f"\n\n{'#' * 70}")
    print("ROBUSTNESS CHECK: MATH REGRESSION, EXCLUDING LOW-COVERAGE VERIFIERS")
    print(f"{'#' * 70}")
    print("Excludes verifiers with <10% math coverage (e.g. falcon3:3b at 0.6%,")
    print("llama3.2:1b at 0.1%), whose avg_synergy is based on essentially 1 row")
    print("and could distort the regression shape.")

    run_domain_analysis(
        "MATH ONLY, coverage >= 10%",
        "math", all_models, domain_stats, min_coverage=0.10
    )

    # --- CRITICAL: split-half validation to break predictor/target dependency ---
    print(f"\n\n{'#' * 70}")
    print("SPLIT-HALF VALIDATION")
    print(f"{'#' * 70}")
    print("IMPORTANT METHODOLOGICAL NOTE: in the analysis above, verifying_accuracy")
    print("and avg_synergy are BOTH computed from collaboration_score on the SAME")
    print("rows. Since synergy = collaboration_correct - max(...), these two")
    print("quantities share a term by construction, not just by an underlying")
    print("phenomenon. Part of the R^2 above is a mechanical artifact of this")
    print("shared dependency, not purely a measure of independent predictive power.")
    print()
    print("This section recomputes verifying_accuracy from a RANDOM HALF of each")
    print("verifier's rows, and avg_synergy from the OTHER (disjoint) half — a")
    print("genuine independence test. If the inverted-U survives here, it's real")
    print("predictive signal. If it collapses, the pooled/coding result above was")
    print("substantially inflated by shared-metric dependency.\n")

    splithalf_stats = compute_domain_stats_splithalf(phase2_rows, question_domains, seed=42)

    run_domain_analysis(
        "SPLIT-HALF — POOLED (verifying_acc from half A, synergy from half B)",
        "pooled", all_models, splithalf_stats
    )
    run_domain_analysis(
        "SPLIT-HALF — CODING (verifying_acc from half A, synergy from half B)",
        "coding", all_models, splithalf_stats
    )
    run_domain_analysis(
        "SPLIT-HALF — MATH (verifying_acc from half A, synergy from half B)",
        "math", all_models, splithalf_stats
    )
