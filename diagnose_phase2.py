"""
diagnose_phase2.py — Sanity checks and diagnostics for Phase 2 results,
before building any statistical analysis on top of them.

Checks:
  1. Row count matches expected (210 pairs x 100 questions = 21,000)
  2. No duplicate (proposer, verifier, question) keys
  3. Null-propagation integrity: every row with collaboration_score=null
     also has synergy=null, and vice versa (catches the exact bug class
     found during dry-run testing)
  4. Parse failure rate overall and per verifier model
  5. Model-call failures (proposer/verifier infra failures) vs genuine
     parse failures, separated
  6. Synergy value sanity: synergy should be in a plausible range
     (typically -1.0 to 1.0 for non-code; wider but bounded for code
     fractional scores)
  7. Coverage check: every (proposer, verifier) pair has exactly 100 rows

Usage:
    python3 diagnose_phase2.py
"""

import json
from collections import defaultdict

RESULTS_PATH = "phase2_results.jsonl"
EXPECTED_PAIRS = 210
EXPECTED_QUESTIONS = 100
EXPECTED_TOTAL_ROWS = EXPECTED_PAIRS * EXPECTED_QUESTIONS


def load_results(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_row_count(rows: list):
    print(f"[1] Row count: {len(rows)} (expected {EXPECTED_TOTAL_ROWS})")
    if len(rows) != EXPECTED_TOTAL_ROWS:
        print(f"    WARNING: mismatch of {abs(len(rows) - EXPECTED_TOTAL_ROWS)} rows")
    else:
        print("    OK")
    print()


def check_duplicates(rows: list):
    seen = set()
    duplicates = []
    for row in rows:
        key = (row["proposer_id"], row["verifier_id"], row["question_id"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)

    print(f"[2] Duplicate keys: {len(duplicates)}")
    if duplicates:
        print(f"    WARNING: found duplicates, e.g. {duplicates[:5]}")
    else:
        print("    OK — no duplicates")
    print()


def check_null_propagation(rows: list):
    inconsistent = []
    for row in rows:
        collab_is_null = row["collaboration_score"] is None
        synergy_is_null = row["synergy"] is None
        if collab_is_null != synergy_is_null:
            inconsistent.append(row)

    print(f"[3] Null-propagation integrity check: {len(inconsistent)} inconsistent rows")
    if inconsistent:
        print("    WARNING: found rows where collaboration_score and synergy "
              "disagree on null-ness (this is the exact bug class caught "
              "during dry-run testing — investigate immediately):")
        for row in inconsistent[:5]:
            print(f"      {row['proposer_id']} -> {row['verifier_id']} / "
                  f"{row['question_id']}: collab={row['collaboration_score']}, "
                  f"synergy={row['synergy']}")
    else:
        print("    OK — collaboration_score and synergy are null-consistent on every row")
    print()


def check_parse_failures(rows: list):
    total = len(rows)
    genuine_pf = 0
    proposer_infra = 0
    verifier_infra = 0

    per_verifier_pf = defaultdict(lambda: {"total": 0, "pf": 0})

    for row in rows:
        verifier = row["verifier_id"]
        per_verifier_pf[verifier]["total"] += 1

        if not row["parse_failure"]:
            continue

        reason = row.get("reason") or ""
        if reason.startswith("[PROPOSER CALL FAILED]"):
            proposer_infra += 1
        elif reason.startswith("[VERIFIER CALL FAILED]"):
            verifier_infra += 1
        else:
            genuine_pf += 1

        per_verifier_pf[verifier]["pf"] += 1

    total_pf = genuine_pf + proposer_infra + verifier_infra
    print(f"[4] Parse failure breakdown (total rows: {total}):")
    print(f"    Genuine parse failures (verifier responded, bad format): {genuine_pf}")
    print(f"    Proposer infra failures (call failed): {proposer_infra}")
    print(f"    Verifier infra failures (call failed): {verifier_infra}")
    print(f"    Combined failure rate: {total_pf / total:.1%}")
    print()

    print("[5] Parse failure rate by verifier model (top 5 worst):")
    ranked = sorted(
        per_verifier_pf.items(),
        key=lambda kv: -(kv[1]["pf"] / kv[1]["total"]) if kv[1]["total"] > 0 else 0
    )
    for model, stats in ranked[:5]:
        rate = stats["pf"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {model:<22} {stats['pf']:>4}/{stats['total']:<4} ({rate:.1%})")
    print()


def check_synergy_range(rows: list):
    synergy_values = [row["synergy"] for row in rows if row["synergy"] is not None]
    if not synergy_values:
        print("[6] Synergy range check: no non-null synergy values found!")
        return

    min_s = min(synergy_values)
    max_s = max(synergy_values)
    out_of_range = [s for s in synergy_values if s < -1.0 or s > 1.0]

    print(f"[6] Synergy value range: min={min_s:.3f}, max={max_s:.3f} "
          f"(n={len(synergy_values)} non-null)")
    if out_of_range:
        print(f"    WARNING: {len(out_of_range)} synergy values outside [-1.0, 1.0] "
              f"— investigate, this shouldn't happen given scores are in [0,1]")
    else:
        print("    OK — all synergy values within expected [-1.0, 1.0] range")
    print()


def check_pair_coverage(rows: list):
    per_pair = defaultdict(int)
    for row in rows:
        key = (row["proposer_id"], row["verifier_id"])
        per_pair[key] += 1

    print(f"[7] Pair coverage: {len(per_pair)} unique pairs "
          f"(expected {EXPECTED_PAIRS})")
    incomplete = [(pair, count) for pair, count in per_pair.items()
                  if count != EXPECTED_QUESTIONS]
    if incomplete:
        print(f"    WARNING: {len(incomplete)} pairs don't have exactly "
              f"{EXPECTED_QUESTIONS} rows:")
        for pair, count in incomplete[:10]:
            print(f"      {pair[0]} -> {pair[1]}: {count} rows")
    else:
        print(f"    OK — every pair has exactly {EXPECTED_QUESTIONS} rows")
    print()


if __name__ == "__main__":
    rows = load_results(RESULTS_PATH)

    print("=" * 70)
    print("PHASE 2 DIAGNOSTICS")
    print("=" * 70)
    print()

    check_row_count(rows)
    check_duplicates(rows)
    check_null_propagation(rows)
    check_parse_failures(rows)
    check_synergy_range(rows)
    check_pair_coverage(rows)

    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)
