"""
analyze_synergy.py — Regression analysis testing whether verifier standalone
accuracy predicts collaboration synergy (the inverted-U hypothesis), matching
the methodology of the original 8-model study (OLS with quadratic term,
LOO-CV for generalization).

Aggregation: one data point per verifier model (15 total) — average synergy
across all (proposer, question) pairs where that model served as verifier,
using only non-null synergy rows. Verifier standalone accuracy comes from
Phase 1 (phase1_results_reparsed.jsonl).

Model: synergy ~ standalone_accuracy + standalone_accuracy^2
A significant negative quadratic coefficient supports an inverted-U shape
(synergy rises then falls as verifier capability increases).

Inclusion threshold: verifiers with pathologically high parse-failure rates
(from diagnose_phase2.py) have very few non-null synergy rows, making their
average synergy unreliable. This script reports BOTH:
  - Full analysis (all 15 verifiers)
  - Sensitivity analysis (excluding verifiers with <50% synergy-row coverage)
so the effect of this exclusion decision is transparent rather than baked
in silently.

Usage:
    python3 analyze_synergy.py
"""

import json
from collections import defaultdict

import numpy as np
import statsmodels.api as sm

PHASE1_PATH = "phase1_results_reparsed.jsonl"
PHASE2_PATH = "phase2_results.jsonl"
COVERAGE_THRESHOLD = 0.50  # for sensitivity analysis exclusion


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_standalone_accuracy(phase1_rows: list) -> dict:
    """Per-model standalone accuracy across all 100 questions (Phase 1),
    using non-null scores only."""
    per_model = defaultdict(lambda: {"scored": 0, "correct": 0.0})

    for row in phase1_rows:
        if row["phase"] != 1:
            continue
        model = row["model_id"]
        score = row["score"]
        if score is None:
            continue
        per_model[model]["scored"] += 1
        if score is True or (isinstance(score, (int, float)) and score >= 1.0):
            per_model[model]["correct"] += 1
        elif isinstance(score, (int, float)) and 0 < score < 1.0:
            per_model[model]["correct"] += score

    return {
        model: (stats["correct"] / stats["scored"] if stats["scored"] > 0 else None)
        for model, stats in per_model.items()
    }


def compute_verifier_synergy(phase2_rows: list) -> dict:
    """Per-verifier average AND median synergy (non-null rows only), plus
    coverage (fraction of rows with non-null synergy, i.e. NOT a parse
    failure).

    Median is included alongside mean because per-row synergy can be
    dominated by a small number of pathological rows (e.g. harmful
    overrides), and mean-based aggregation is sensitive to those outliers
    in a way median is not. Comparing the two regression fits (mean vs
    median target) tells us whether the earlier weak/null inverted-U
    result was an artifact of outlier-sensitive averaging.
    """
    per_verifier_values = defaultdict(list)
    per_verifier_totals = defaultdict(int)

    for row in phase2_rows:
        verifier = row["verifier_id"]
        per_verifier_totals[verifier] += 1
        if row["synergy"] is not None:
            per_verifier_values[verifier].append(row["synergy"])

    result = {}
    for verifier, total in per_verifier_totals.items():
        values = per_verifier_values.get(verifier, [])
        non_null = len(values)
        avg_synergy = float(np.mean(values)) if non_null > 0 else None
        median_synergy = float(np.median(values)) if non_null > 0 else None
        coverage = non_null / total if total > 0 else 0.0
        result[verifier] = {
            "avg_synergy": avg_synergy,
            "median_synergy": median_synergy,
            "coverage": coverage,
            "non_null_rows": non_null,
            "total_rows": total,
        }
    return result


def fit_quadratic_ols(x: np.ndarray, y: np.ndarray):
    """Fit synergy ~ x + x^2. Returns fitted model and design matrix info."""
    X = np.column_stack([x, x ** 2])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def loo_cv_r2(x: np.ndarray, y: np.ndarray):
    """Leave-one-out cross-validation R^2 for the quadratic model."""
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


def run_analysis(label: str, models: list, standalone_acc: dict, verifier_synergy: dict,
                  target_field: str = "avg_synergy"):
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    x_vals, y_vals, labels = [], [], []
    for model in models:
        acc = standalone_acc.get(model)
        syn_data = verifier_synergy.get(model)
        if acc is None or syn_data is None or syn_data[target_field] is None:
            print(f"  SKIPPING {model}: missing standalone accuracy or synergy data")
            continue
        x_vals.append(acc)
        y_vals.append(syn_data[target_field])
        labels.append(model)

    n = len(x_vals)
    print(f"\nData points (verifiers with usable data): {n}")
    if n < 4:
        print("Too few data points for regression. Skipping.")
        return

    print(f"\n{'Model':<22} {'Standalone Acc':>15} {'Target Synergy':>15} {'Coverage':>10}")
    for model, x, y in zip(labels, x_vals, y_vals):
        cov = verifier_synergy[model]["coverage"]
        print(f"{model:<22} {x:>14.1%} {y:>15.3f} {cov:>9.1%}")

    x = np.array(x_vals)
    y = np.array(y_vals)

    print("\n--- OLS Regression: synergy ~ standalone_accuracy + standalone_accuracy^2 ---")
    model = fit_quadratic_ols(x, y)
    print(model.summary())

    quad_coef = model.params[2]
    quad_pvalue = model.pvalues[2]
    print(f"\nQuadratic coefficient: {quad_coef:.4f} (p = {quad_pvalue:.4f})")
    if quad_coef < 0 and quad_pvalue < 0.05:
        print("=> SIGNIFICANT negative quadratic term: supports inverted-U shape.")
    elif quad_coef < 0:
        print("=> Negative quadratic term (consistent with inverted-U), "
              "but not statistically significant at p<0.05.")
    else:
        print("=> Quadratic term is not negative — does NOT support inverted-U shape.")

    # Peak location (vertex of the parabola), if quadratic term is negative
    if quad_coef < 0:
        linear_coef = model.params[1]
        peak_x = -linear_coef / (2 * quad_coef)
        print(f"Estimated peak (vertex) standalone accuracy: {peak_x:.1%}")

    print(f"\nR-squared (in-sample): {model.rsquared:.4f}")

    print("\n--- Leave-One-Out Cross-Validation ---")
    r2_loo, error = loo_cv_r2(x, y)
    if error:
        print(error)
    else:
        print(f"LOO-CV R-squared: {r2_loo:.4f}")
        print("(Compare against in-sample R-squared above — a much lower "
              "LOO-CV R-squared indicates the fitted curve does not "
              "generalize well to held-out models, consistent with prior "
              "findings on this small a sample.)")


if __name__ == "__main__":
    phase1_rows = load_jsonl(PHASE1_PATH)
    phase2_rows = load_jsonl(PHASE2_PATH)

    standalone_acc = compute_standalone_accuracy(phase1_rows)
    verifier_synergy = compute_verifier_synergy(phase2_rows)

    all_models = list(standalone_acc.keys())

    print("Per-model summary:")
    print(f"{'Model':<22} {'Standalone Acc':>15} {'Verifier Coverage':>18}")
    for model in all_models:
        acc = standalone_acc.get(model)
        cov = verifier_synergy.get(model, {}).get("coverage", 0.0)
        acc_str = f"{acc:.1%}" if acc is not None else "N/A"
        print(f"{model:<22} {acc_str:>15} {cov:>17.1%}")

    # Full analysis: all 15 verifiers, mean-based
    run_analysis("FULL ANALYSIS — MEAN SYNERGY (all 15 verifiers)", all_models,
                 standalone_acc, verifier_synergy, target_field="avg_synergy")

    # Full analysis: all 15 verifiers, median-based (robust to outlier rows)
    run_analysis("FULL ANALYSIS — MEDIAN SYNERGY (all 15 verifiers, outlier-robust)",
                 all_models, standalone_acc, verifier_synergy, target_field="median_synergy")

    # Sensitivity analysis: exclude low-coverage verifiers
    high_coverage_models = [
        m for m in all_models
        if verifier_synergy.get(m, {}).get("coverage", 0.0) >= COVERAGE_THRESHOLD
    ]
    excluded = [m for m in all_models if m not in high_coverage_models]
    print(f"\n\nModels excluded from sensitivity analysis "
          f"(verifier coverage < {COVERAGE_THRESHOLD:.0%}): {excluded}")

    run_analysis(
        f"SENSITIVITY ANALYSIS — MEAN SYNERGY (coverage >= {COVERAGE_THRESHOLD:.0%}, "
        f"n={len(high_coverage_models)})",
        high_coverage_models, standalone_acc, verifier_synergy, target_field="avg_synergy"
    )

    run_analysis(
        f"SENSITIVITY ANALYSIS — MEDIAN SYNERGY (coverage >= {COVERAGE_THRESHOLD:.0%}, "
        f"outlier-robust, n={len(high_coverage_models)})",
        high_coverage_models, standalone_acc, verifier_synergy, target_field="median_synergy"
    )
