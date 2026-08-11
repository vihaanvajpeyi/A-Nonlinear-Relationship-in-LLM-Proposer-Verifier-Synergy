"""
generate_plots.py — Produces the 4 result figures for the paper:

  1. fig1_inverted_u.png — verifying accuracy vs avg synergy, math | coding
     panels, with fitted quadratic curve overlaid.
  2. fig2_standalone_vs_verifying.png — paired slope plot: standalone
     accuracy vs verifying accuracy per model.
  3. fig3_outcome_breakdown.png — stacked bar of ECHO_CORRECT /
     HARMFUL_OVERRIDE / HELPFUL_CORRECTION / NO_HELP per verifier,
     sorted by harmful-override rate.
  4. fig4_splithalf_robustness.png — strip/histogram plot of the 20
     quadratic coefficients from multi-seed split-half validation,
     coding vs math.

Reads: phase1_results_reparsed.jsonl, phase2_results.jsonl,
       question_dataset_v2.json
Writes: fig1-4 PNGs to the current directory.

Usage:
    python3 generate_plots.py
"""

import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from benchmark import BenchmarkLoader
from evaluator import score_standalone, score_code

PHASE1_PATH = "phase1_results_reparsed.jsonl"
PHASE2_PATH = "phase2_results.jsonl"
DATASET_PATH = "question_dataset_v2.json"
N_SEEDS = 20

CODE_GENERATION_TYPES = {"code_generation"}


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _numeric(score):
    if score is None:
        return None
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    return float(score)


def _extract_func_name(question_text: str):
    import re
    m = re.search(r'`(\w+)\s*\(', question_text)
    return m.group(1) if m else None


def fit_quadratic(x, y):
    X = sm.add_constant(np.column_stack([x, x ** 2]))
    return sm.OLS(y, X).fit()


# ============================================================
# Data loading (shared across figures)
# ============================================================

def load_all_data():
    phase1_rows = load_jsonl(PHASE1_PATH)
    phase2_rows = load_jsonl(PHASE2_PATH)
    loader = BenchmarkLoader(DATASET_PATH)
    questions_by_id = {q.id: q for q in loader.load_questions()}
    question_domains = {qid: q.domain for qid, q in questions_by_id.items()}
    return phase1_rows, phase2_rows, questions_by_id, question_domains


def compute_domain_verifying_stats(phase2_rows, question_domains):
    """(verifier, domain) -> {verifying_accuracy, avg_synergy}, domain in
    {'math','coding','pooled'}."""
    buckets = defaultdict(lambda: {"score_sum": 0.0, "scored": 0,
                                    "synergy_sum": 0.0, "synergy_n": 0})
    for row in phase2_rows:
        verifier = row["verifier_id"]
        domain = question_domains.get(row["question_id"], "unknown")
        if domain == "unknown":
            continue
        for key_domain in (domain, "pooled"):
            b = buckets[(verifier, key_domain)]
            score = row["collaboration_score"]
            if score is not None:
                b["score_sum"] += _numeric(score)
                b["scored"] += 1
            if row["synergy"] is not None:
                b["synergy_sum"] += row["synergy"]
                b["synergy_n"] += 1

    result = {}
    for key, b in buckets.items():
        result[key] = {
            "verifying_accuracy": (b["score_sum"] / b["scored"]) if b["scored"] > 0 else None,
            "avg_synergy": (b["synergy_sum"] / b["synergy_n"]) if b["synergy_n"] > 0 else None,
        }
    return result


def compute_standalone_accuracy(phase1_rows):
    per_model = defaultdict(lambda: {"scored": 0, "correct": 0.0})
    for row in phase1_rows:
        if row["phase"] != 1:
            continue
        score = _numeric(row["score"])
        if score is None:
            continue
        per_model[row["model_id"]]["scored"] += 1
        per_model[row["model_id"]]["correct"] += score
    return {m: (s["correct"] / s["scored"] if s["scored"] > 0 else None)
            for m, s in per_model.items()}


# ============================================================
# Figure 1: Inverted-U scatter, math | coding panels
# ============================================================

def make_fig1(phase2_rows, question_domains, all_models):
    domain_stats = compute_domain_verifying_stats(phase2_rows, question_domains)

    # Track n_rows per (model, domain) so we can flag low-coverage points
    coverage_counts = defaultdict(lambda: defaultdict(int))
    for row in phase2_rows:
        verifier = row["verifier_id"]
        domain = question_domains.get(row["question_id"], "unknown")
        if domain == "unknown":
            continue
        if row["synergy"] is not None:
            coverage_counts[(verifier, domain)]["synergy_n"] += 1
        coverage_counts[(verifier, domain)]["total"] += 1

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, domain, title in zip(
        axes, ["pooled", "math", "coding"],
        ["Pooled (primary result)", "Math", "Coding"]
    ):
        xs, ys, names, low_coverage = [], [], [], []
        for model in all_models:
            d = domain_stats.get((model, domain))
            if d is None or d["verifying_accuracy"] is None or d["avg_synergy"] is None:
                continue
            xs.append(d["verifying_accuracy"])
            ys.append(d["avg_synergy"])
            names.append(model)
            cov = coverage_counts.get((model, domain), {})
            n_syn = cov.get("synergy_n", 0)
            low_coverage.append(n_syn < 10)  # fewer than 10 non-null rows -> flag

        xs = np.array(xs)
        ys = np.array(ys)

        colors = ["darkorange" if lc else "steelblue" for lc in low_coverage]
        ax.scatter(xs, ys, s=60, alpha=0.85, edgecolor="black", zorder=3, c=colors)
        for x, y, name, lc in zip(xs, ys, names, low_coverage):
            label = f"{name}*" if lc else name
            ax.annotate(label, (x, y), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

        if len(xs) >= 4:
            model_fit = fit_quadratic(xs, ys)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            X_line = sm.add_constant(np.column_stack([x_line, x_line ** 2]))
            y_line = model_fit.predict(X_line)
            ax.plot(x_line, y_line, color="crimson", linewidth=2, zorder=2,
                    label=f"quadratic fit (p={model_fit.pvalues[2]:.3f})")
            ax.legend(fontsize=8, loc="best")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlabel("Verifying accuracy")
        ax.set_ylabel("Average synergy")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.suptitle("Verifying accuracy vs. collaboration synergy\n"
                  "(* = fewer than 10 non-null rows; low-coverage, treat with caution)",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig("fig1_inverted_u.png", dpi=150)
    plt.close(fig)
    print("Wrote fig1_inverted_u.png")


# ============================================================
# Figure 2: Standalone vs verifying accuracy slope plot
# ============================================================

def make_fig2(phase1_rows, phase2_rows, question_domains, all_models):
    standalone_acc = compute_standalone_accuracy(phase1_rows)
    pooled_stats = compute_domain_verifying_stats(phase2_rows, question_domains)

    rows_data = []
    for model in all_models:
        s_acc = standalone_acc.get(model)
        v_acc = pooled_stats.get((model, "pooled"), {}).get("verifying_accuracy")
        if s_acc is None or v_acc is None:
            continue
        rows_data.append((model, s_acc, v_acc))

    # Sort by standalone accuracy for a readable plot
    rows_data.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(8, max(5, len(rows_data) * 0.4)))

    y_positions = np.arange(len(rows_data))
    for i, (model, s_acc, v_acc) in enumerate(rows_data):
        color = "crimson" if v_acc < s_acc else "seagreen"
        ax.plot([s_acc, v_acc], [i, i], color=color, linewidth=1.5, zorder=1, alpha=0.7)
        ax.scatter([s_acc], [i], color="steelblue", s=50, zorder=2, label="Standalone" if i == 0 else None)
        ax.scatter([v_acc], [i], color="darkorange", s=50, zorder=2, label="Verifying" if i == 0 else None)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows_data], fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Standalone vs. verifying accuracy, per model")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig("fig2_standalone_vs_verifying.png", dpi=150)
    plt.close(fig)
    print("Wrote fig2_standalone_vs_verifying.png")


# ============================================================
# Figure 3: Outcome breakdown stacked bar
# ============================================================

def classify_row(row, question):
    if row["proposer_answer"] is None or row["verifier_answer"] is None:
        return None
    is_code = question.domain == "coding" and question.type in CODE_GENERATION_TYPES

    if is_code:
        func_name = _extract_func_name(question.question)
        proposer_score = score_code(row["proposer_answer"], question.test_cases,
                                     expected_func_name=func_name)
        proposer_correct = proposer_score >= 1.0
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


def make_fig3(phase2_rows, questions_by_id, all_models):
    per_verifier_counts = defaultdict(lambda: defaultdict(int))

    for row in phase2_rows:
        question = questions_by_id.get(row["question_id"])
        if question is None:
            continue
        outcome = classify_row(row, question)
        if outcome is None:
            continue
        per_verifier_counts[row["verifier_id"]][outcome] += 1

    outcomes = ["ECHO_CORRECT", "HELPFUL_CORRECTION", "NO_HELP", "HARMFUL_OVERRIDE"]
    colors = {"ECHO_CORRECT": "seagreen", "HELPFUL_CORRECTION": "steelblue",
              "NO_HELP": "lightgray", "HARMFUL_OVERRIDE": "crimson"}

    models_with_data = [m for m in all_models if m in per_verifier_counts]

    def harmful_rate(m):
        counts = per_verifier_counts[m]
        total = sum(counts.values())
        return counts["HARMFUL_OVERRIDE"] / total if total > 0 else 0

    models_with_data.sort(key=harmful_rate, reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(models_with_data))

    for outcome in outcomes:
        vals = []
        for m in models_with_data:
            counts = per_verifier_counts[m]
            total = sum(counts.values())
            vals.append(counts[outcome] / total if total > 0 else 0)
        vals = np.array(vals)
        ax.bar(models_with_data, vals, bottom=bottom, label=outcome, color=colors[outcome])
        bottom += vals

    ax.set_ylabel("Fraction of classifiable rows")
    ax.set_title("Verifier outcome breakdown (sorted by harmful override rate)")
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.25, 1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig.tight_layout()
    fig.savefig("fig3_outcome_breakdown.png", dpi=150)
    plt.close(fig)
    print("Wrote fig3_outcome_breakdown.png")


# ============================================================
# Figure 4: Multi-seed split-half robustness
# ============================================================

def compute_domain_stats_splithalf(phase2_rows, question_domains, seed):
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
        mid = len(shuffled) // 2
        half_a, half_b = shuffled[:mid], shuffled[mid:]

        a_scored = [r for r in half_a if r["collaboration_score"] is not None]
        v_acc = (sum(_numeric(r["collaboration_score"]) for r in a_scored) / len(a_scored)
                  ) if a_scored else None

        b_syn = [r["synergy"] for r in half_b if r["synergy"] is not None]
        avg_syn = sum(b_syn) / len(b_syn) if b_syn else None

        result[key] = {"verifying_accuracy": v_acc, "avg_synergy": avg_syn}
    return result


def make_fig4(phase2_rows, question_domains, all_models, n_seeds=N_SEEDS):
    coefs_by_domain = {"pooled": [], "math": [], "coding": []}

    for domain in ["pooled", "math", "coding"]:
        for seed in range(n_seeds):
            stats_by_key = compute_domain_stats_splithalf(phase2_rows, question_domains, seed)
            xs, ys = [], []
            for model in all_models:
                d = stats_by_key.get((model, domain))
                if d is None or d["verifying_accuracy"] is None or d["avg_synergy"] is None:
                    continue
                xs.append(d["verifying_accuracy"])
                ys.append(d["avg_synergy"])
            if len(xs) < 4:
                continue
            xs, ys = np.array(xs), np.array(ys)
            fit = fit_quadratic(xs, ys)
            coefs_by_domain[domain].append(fit.params[2])

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = [1, 2, 3]
    data = [coefs_by_domain["pooled"], coefs_by_domain["math"], coefs_by_domain["coding"]]
    labels = ["Pooled\n(primary result)", "Math", "Coding"]

    parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_alpha(0.5)

    for pos, vals in zip(positions, data):
        jitter = np.random.normal(0, 0.03, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=25, alpha=0.6,
                   color="black", zorder=3)

    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Quadratic coefficient (split-half)")
    ax.set_title(f"Split-half quadratic coefficient distribution ({n_seeds} random splits)")

    # Add headroom so the "% negative" labels never collide with the title
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + 0.15 * y_range)

    for pos, vals in zip(positions, data):
        sig_frac = np.mean(np.array(vals) < 0)
        ax.text(pos, y_max + 0.05 * y_range, f"{sig_frac:.0%} negative",
                ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig("fig4_splithalf_robustness.png", dpi=150)
    plt.close(fig)
    print("Wrote fig4_splithalf_robustness.png")


if __name__ == "__main__":
    phase1_rows, phase2_rows, questions_by_id, question_domains = load_all_data()
    all_models = sorted(set(row["verifier_id"] for row in phase2_rows))

    make_fig1(phase2_rows, question_domains, all_models)
    make_fig2(phase1_rows, phase2_rows, question_domains, all_models)
    make_fig3(phase2_rows, questions_by_id, all_models)
    make_fig4(phase2_rows, question_domains, all_models)

    print("\nAll 4 figures generated.")
