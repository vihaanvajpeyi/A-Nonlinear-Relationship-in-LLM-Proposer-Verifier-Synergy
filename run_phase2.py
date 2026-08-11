"""
run_phase2.py — Phase 2 proposer-verifier collaboration runner.

Runs all 210 ordered (proposer, verifier) pairs from MODEL_ROSTER against
all 100 questions, per protocol_v2.md section 10. For each (proposer,
verifier, question) triple:

  1. Build the proposer prompt (identical to Phase 1 standalone prompt).
  2. Call the proposer model. If this call fails (timeout/connection
     error), log a fully-failed Phase 2 row immediately and skip the
     verifier call entirely — there is nothing meaningful to verify.
  3. Parse and score the proposer's answer.
  4. Build the verifier prompt using the proposer's raw response.
  5. Call the verifier model. If this call fails, log a row with the
     proposer's data present but verifier fields marked as failure.
  6. Parse and score the verifier's answer (this is always the
     collaboration answer, per protocol_v2.md section 8.2).
  7. Look up the verifier's Phase 1 standalone score for this question
     (from phase1_results_reparsed.jsonl) to compute synergy.
  8. Compute derived fields (changed_answer, synergy) and log the row.

Resumable at per-row granularity: before every single (proposer, verifier,
question) triple, the already-logged key set is checked, so a crash or
interruption at any point can be resumed without re-doing or overwriting
any completed work. No retries, per protocol_v2.md section 9.

Iteration order: pair-outer, question-inner. All 100 questions are run for
one (proposer, verifier) pair before moving to the next pair, so partial
results form complete, immediately-analyzable slices rather than being
thinly spread across all 210 pairs.

Usage:
    python3 run_phase2.py
"""

import sys
import time

from benchmark import BenchmarkLoader, Question
from evaluator import (
    score_standalone,
    score_code,
    answers_differ,
    compute_synergy_row,
)
from logger import ResultLogger
from model import ModelWrapper, MODEL_ROSTER, check_all_roster_models
from parser import (
    parse_standalone,
    parse_standalone_code,
    parse_verifier,
    parse_verifier_code,
)
from prompts import build_proposer_prompt, build_verifier_prompt

DATASET_PATH = "question_dataset_v2.json"
PHASE1_RESULTS_PATH = "phase1_results_reparsed.jsonl"
PHASE2_RESULTS_PATH = "phase2_results.jsonl"
PROMPT_VERSION = "v1"

CODE_GENERATION_TYPES = {"code_generation"}

PROGRESS_EVERY = 10  # log a progress line every N completed rows


def _is_code_generation(question: Question) -> bool:
    return question.domain == "coding" and question.type in CODE_GENERATION_TYPES


def _extract_func_name(question: Question):
    import re
    m = re.search(r'`(\w+)\s*\(', question.question)
    return m.group(1) if m else None


def load_phase1_standalone_scores(path: str) -> dict:
    """
    Build a lookup: (model_id, question_id) -> standalone score (bool, float,
    or None for parse failure), from the corrected Phase 1 results file.

    This is used to supply verifier_standalone_correct for synergy
    computation per protocol_v2.md section 8.3 — the verifier's own
    standalone performance on this exact question, measured independently
    in Phase 1, is required as one of the two baselines synergy compares
    the collaboration result against.
    """
    import json

    lookup = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("phase") != 1:
                continue
            key = (row["model_id"], row["question_id"])
            lookup[key] = row["score"]  # bool, float, or None
    return lookup


def build_ordered_pairs(roster: list) -> list:
    """
    All ordered (proposer, verifier) pairs, excluding self-pairing.
    15 models -> 15 x 14 = 210 pairs, per protocol_v2.md section 10.
    """
    return [(p, v) for p in roster for v in roster if p != v]


def run_phase2():
    print("=" * 70)
    print("PHASE 2 — Proposer-Verifier Collaboration")
    print("=" * 70)

    # --- Preflight: verify all roster models are available ---
    print("\nChecking model availability...")
    availability = check_all_roster_models()
    missing = [name for name, ok in availability.items() if not ok]
    if missing:
        print(f"ERROR: {len(missing)} roster model(s) not available in Ollama:")
        for name in missing:
            print(f"  - {name}")
        print("\nPull missing models with `ollama pull <name>` before running.")
        sys.exit(1)
    print(f"All {len(MODEL_ROSTER)} roster models available.\n")

    # --- Load dataset ---
    print(f"Loading dataset from {DATASET_PATH}...")
    loader = BenchmarkLoader(DATASET_PATH)
    questions = loader.load_questions()
    print(f"Loaded {len(questions)} questions "
          f"({len(loader.load_questions(domain='math'))} math, "
          f"{len(loader.load_questions(domain='coding'))} coding).\n")

    # --- Load Phase 1 standalone scores for synergy computation ---
    print(f"Loading Phase 1 standalone scores from {PHASE1_RESULTS_PATH}...")
    standalone_scores = load_phase1_standalone_scores(PHASE1_RESULTS_PATH)
    print(f"Loaded {len(standalone_scores)} Phase 1 (model, question) scores.\n")

    # Sanity check: every (model, question) combination we'll need should
    # exist in the Phase 1 lookup. Missing entries would silently break
    # synergy computation for those rows, so we check up front.
    missing_lookups = 0
    for model_name in MODEL_ROSTER:
        for q in questions:
            if (model_name, q.id) not in standalone_scores:
                missing_lookups += 1
    if missing_lookups > 0:
        print(f"WARNING: {missing_lookups} (model, question) combinations "
              f"are missing from Phase 1 results. Synergy for affected rows "
              f"will use a None standalone baseline (treated as 0.0 in the "
              f"max() comparison, per evaluator.py's documented convention).\n")

    # --- Build ordered pairs ---
    pairs = build_ordered_pairs(MODEL_ROSTER)
    print(f"Ordered pairs: {len(pairs)} "
          f"({len(MODEL_ROSTER)} models x {len(MODEL_ROSTER) - 1})")

    total_combinations = len(pairs) * len(questions)
    print(f"Total (proposer, verifier, question) combinations: {total_combinations}\n")

    # --- Set up logger, check for already-completed work ---
    logger = ResultLogger(PHASE2_RESULTS_PATH)
    already_done = logger.already_logged_keys(phase=2)
    if already_done:
        print(f"Resuming: {len(already_done)} (proposer, verifier, question) "
              f"combinations already logged. These will be skipped.\n")

    remaining = total_combinations - len(already_done)
    print(f"Remaining to run: {remaining}\n")

    # --- Cache one ModelWrapper per model name (avoid re-instantiating) ---
    wrappers = {name: ModelWrapper(name) for name in MODEL_ROSTER}

    # --- Main loop: pair-outer, question-inner ---
    completed = 0
    parse_failures = 0
    start_time = time.time()

    for proposer_id, verifier_id in pairs:
        proposer_wrapper = wrappers[proposer_id]
        verifier_wrapper = wrappers[verifier_id]

        for question in questions:
            key = (proposer_id, verifier_id, question.id)
            if key in already_done:
                continue  # already logged — never re-run or overwrite

            is_code = _is_code_generation(question)
            func_name = _extract_func_name(question) if is_code else None

            # --- Step 1-2: Proposer call ---
            proposer_prompt = build_proposer_prompt(question)
            proposer_response = proposer_wrapper.generate(proposer_prompt)

            if not proposer_response.success:
                # Proposer call itself failed. Per confirmed decision: log a
                # fully-failed row immediately, skip the verifier call
                # entirely (nothing meaningful to verify).
                logger.log_phase2(
                    proposer_id=proposer_id,
                    verifier_id=verifier_id,
                    question_id=question.id,
                    proposer_answer=None,
                    verifier_answer=None,
                    verdict=None,
                    reason=f"[PROPOSER CALL FAILED] {proposer_response.error_message}",
                    changed_answer=None,
                    collaboration_score=None,
                    synergy=None,
                    parse_failure=True,
                    prompt_version=PROMPT_VERSION,
                )
                parse_failures += 1
                completed += 1
                _maybe_print_progress(completed, remaining, parse_failures, start_time,
                                       proposer_id, verifier_id, question.id)
                continue

            proposer_raw = proposer_response.output_text

            # --- Step 3: Parse and score proposer's answer ---
            if is_code:
                proposer_parsed = parse_standalone_code(proposer_raw)
                proposer_score = score_code(
                    proposer_parsed, question.test_cases, expected_func_name=func_name
                )
            else:
                proposer_parsed = parse_standalone(proposer_raw)
                proposer_score = score_standalone(proposer_parsed, question.final_answer)

            # --- Step 4-5: Verifier call ---
            verifier_prompt = build_verifier_prompt(question, proposer_response=proposer_raw)
            verifier_response = verifier_wrapper.generate(verifier_prompt)

            if not verifier_response.success:
                # Verifier call failed. Proposer data is still meaningful and
                # logged; verifier/collaboration fields marked as failure.
                logger.log_phase2(
                    proposer_id=proposer_id,
                    verifier_id=verifier_id,
                    question_id=question.id,
                    proposer_answer=proposer_parsed,
                    verifier_answer=None,
                    verdict=None,
                    reason=f"[VERIFIER CALL FAILED] {verifier_response.error_message}",
                    changed_answer=None,
                    collaboration_score=None,
                    synergy=None,
                    parse_failure=True,
                    prompt_version=PROMPT_VERSION,
                )
                parse_failures += 1
                completed += 1
                _maybe_print_progress(completed, remaining, parse_failures, start_time,
                                       proposer_id, verifier_id, question.id)
                continue

            verifier_raw = verifier_response.output_text

            # --- Step 6: Parse and score verifier's answer ---
            if is_code:
                verifier_parsed_dict = parse_verifier_code(verifier_raw)
                if verifier_parsed_dict is None:
                    verifier_answer = None
                    verdict = None
                    reason = None
                    verifier_score = 0.0  # per protocol 8.1: code parse failure -> 0.0
                    row_parse_failure = True
                else:
                    verifier_answer = verifier_parsed_dict["function"]
                    verdict = verifier_parsed_dict["verdict"]
                    reason = verifier_parsed_dict["reason"]
                    verifier_score = score_code(
                        verifier_answer, question.test_cases, expected_func_name=func_name
                    )
                    row_parse_failure = False
            else:
                verifier_parsed_dict = parse_verifier(verifier_raw)
                if verifier_parsed_dict is None:
                    verifier_answer = None
                    verdict = None
                    reason = None
                    verifier_score = None  # per protocol 8.1: non-code parse failure -> null
                    row_parse_failure = True
                else:
                    verifier_answer = verifier_parsed_dict["answer"]
                    verdict = verifier_parsed_dict["verdict"]
                    reason = verifier_parsed_dict["reason"]
                    verifier_score = score_standalone(verifier_answer, question.final_answer)
                    row_parse_failure = False

            # --- Step 7: Look up verifier's Phase 1 standalone score ---
            verifier_standalone_score = standalone_scores.get(
                (verifier_id, question.id), None
            )

            # --- Step 8: Derived fields and logging ---
            changed = answers_differ(proposer_parsed, verifier_answer)

            synergy_fields = compute_synergy_row(
                proposer_answer=proposer_parsed,
                verifier_answer=verifier_answer,
                verifier_standalone_answer=None,  # not needed; scores passed directly below
                proposer_correct=proposer_score,
                verifier_correct=verifier_score,
                verifier_standalone_correct=verifier_standalone_score,
                changed_answer=changed,
            )

            logger.log_phase2(
                proposer_id=proposer_id,
                verifier_id=verifier_id,
                question_id=question.id,
                proposer_answer=proposer_parsed,
                verifier_answer=verifier_answer,
                verdict=verdict,
                reason=reason,
                changed_answer=synergy_fields["changed_answer"],
                collaboration_score=synergy_fields["collaboration_correct"],
                synergy=synergy_fields["synergy"],
                parse_failure=row_parse_failure,
                prompt_version=PROMPT_VERSION,
            )

            if row_parse_failure:
                parse_failures += 1
            completed += 1
            _maybe_print_progress(completed, remaining, parse_failures, start_time,
                                   proposer_id, verifier_id, question.id)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"Total run this session: {completed}")
    print(f"Parse failures this session: {parse_failures}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Results written to: {PHASE2_RESULTS_PATH}")


def _maybe_print_progress(completed, remaining, parse_failures, start_time,
                           proposer_id, verifier_id, question_id):
    if completed % PROGRESS_EVERY == 0:
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta_seconds = (remaining - completed) / rate if rate > 0 else float("inf")
        eta_hours = eta_seconds / 3600
        print(f"  [{completed}/{remaining}] "
              f"{proposer_id} -> {verifier_id} / {question_id} — "
              f"{rate:.3f} rows/s — "
              f"{parse_failures} parse failures — "
              f"ETA: {eta_hours:.1f}h")


if __name__ == "__main__":
    run_phase2()
