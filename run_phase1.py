"""
run_phase1.py — Phase 1 standalone evaluation runner.

Runs every model in MODEL_ROSTER (model.py) against every question in
question_dataset_v2.json (benchmark.py), building prompts (prompts.py),
parsing responses (parser.py), scoring them (evaluator.py), and logging
every result (logger.py) — per protocol_v2.md.

Resumable: if interrupted and re-run, already-completed (model, question,
role) combinations are skipped rather than re-run or overwritten, per
protocol_v2.md section 11 (no result may be overwritten).

No retries: per protocol_v2.md section 9, a single failed/timed-out call is
logged as a parse failure and never retried.

Usage:
    python3 run_phase1.py
"""

import sys
import time

from benchmark import BenchmarkLoader, Question
from evaluator import score_standalone, score_code
from logger import ResultLogger
from model import ModelWrapper, MODEL_ROSTER, check_all_roster_models
from parser import parse_standalone, parse_standalone_code
from prompts import build_standalone_prompt

DATASET_PATH = "question_dataset_v2.json"
RESULTS_PATH = "phase1_results.jsonl"
PROMPT_VERSION = "v1"
ROLE = "standalone"

CODE_GENERATION_TYPES = {"code_generation"}


def _is_code_generation(question: Question) -> bool:
    return question.domain == "coding" and question.type in CODE_GENERATION_TYPES


def _extract_func_name(question: Question) -> str:
    """
    Best-effort extraction of the expected function name from the question
    text, for use as a hint to grade_code's namespace lookup. Falls back to
    None (grade_code will use AST/regex fallback) if not found.
    """
    import re
    m = re.search(r'`(\w+)\s*\(', question.question)
    if m:
        return m.group(1)
    return None


def run_phase1():
    print("=" * 70)
    print("PHASE 1 — Standalone Evaluation")
    print("=" * 70)

    # --- Preflight: verify all roster models are actually available ---
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

    # --- Set up logger, check for already-completed work ---
    logger = ResultLogger(RESULTS_PATH)
    already_done = logger.already_logged_keys(phase=1)
    if already_done:
        print(f"Resuming: {len(already_done)} (model, question, role) "
              f"combinations already logged. These will be skipped.\n")

    total_combinations = len(MODEL_ROSTER) * len(questions)
    remaining = total_combinations - len(already_done)
    print(f"Total combinations: {total_combinations} "
          f"({len(MODEL_ROSTER)} models x {len(questions)} questions)")
    print(f"Remaining to run: {remaining}\n")

    # --- Main loop ---
    completed = 0
    parse_failures = 0
    start_time = time.time()

    for model_name in MODEL_ROSTER:
        wrapper = ModelWrapper(model_name)

        for question in questions:
            key = (model_name, question.id, ROLE)
            if key in already_done:
                continue  # already logged — never re-run or overwrite

            prompt = build_standalone_prompt(question)
            response = wrapper.generate(prompt)

            if not response.success:
                # Model call itself failed (timeout, connection error, etc.)
                # Log as a parse failure — no retries, per protocol section 9.
                logger.log_phase1(
                    model_id=model_name,
                    question_id=question.id,
                    role=ROLE,
                    raw_response=f"[MODEL CALL FAILED] {response.error_message}",
                    parsed_answer=None,
                    score=None,
                    parse_failure=True,
                    prompt_version=PROMPT_VERSION,
                )
                parse_failures += 1
                completed += 1
                continue

            raw_text = response.output_text

            if _is_code_generation(question):
                parsed = parse_standalone_code(raw_text)
                score = score_code(
                    parsed, question.test_cases,
                    expected_func_name=_extract_func_name(question)
                )
                parse_failure = parsed is None
            else:
                parsed = parse_standalone(raw_text)
                ground_truth = question.final_answer
                score = score_standalone(parsed, ground_truth)
                parse_failure = parsed is None

            logger.log_phase1(
                model_id=model_name,
                question_id=question.id,
                role=ROLE,
                raw_response=raw_text,
                parsed_answer=parsed,
                score=score,
                parse_failure=parse_failure,
                prompt_version=PROMPT_VERSION,
            )

            if parse_failure:
                parse_failures += 1
            completed += 1

            if completed % 25 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed}/{remaining}] "
                      f"{model_name} / {question.id} — "
                      f"{rate:.2f} q/s — "
                      f"{parse_failures} parse failures so far")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Total run this session: {completed}")
    print(f"Parse failures this session: {parse_failures}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Results written to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_phase1()
