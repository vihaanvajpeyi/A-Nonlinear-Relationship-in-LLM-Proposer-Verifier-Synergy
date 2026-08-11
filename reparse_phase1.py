"""
reparse_phase1.py — Re-parse and re-score Phase 1 code-generation rows using
the fixed parser (markdown fence stripping), without re-running any models.

Reads phase1_results.jsonl (original, untouched), re-parses only the
code-generation rows using the corrected parse_standalone_code, re-scores
them with grade_code, and writes a new file: phase1_results_reparsed.jsonl.

The original phase1_results.jsonl is NEVER modified, per the no-overwrite
rule. Non-code rows (math, debugging, output prediction, edge cases) are
copied through unchanged, since the parser bug only affected code
extraction (markdown fence stripping).

Usage:
    python3 reparse_phase1.py
"""

import json

from benchmark import BenchmarkLoader
from evaluator import score_code
from parser import parse_standalone_code

ORIGINAL_RESULTS_PATH = "phase1_results.jsonl"
REPARSED_RESULTS_PATH = "phase1_results_reparsed.jsonl"
DATASET_PATH = "question_dataset_v2.json"

CODE_GENERATION_TYPES = {"code_generation"}


def _extract_func_name(question_text: str):
    import re
    m = re.search(r'`(\w+)\s*\(', question_text)
    return m.group(1) if m else None


def main():
    loader = BenchmarkLoader(DATASET_PATH)
    questions_by_id = {q.id: q for q in loader.load_questions()}

    total_rows = 0
    reparsed_count = 0
    changed_count = 0

    with open(ORIGINAL_RESULTS_PATH, "r") as infile, \
         open(REPARSED_RESULTS_PATH, "w") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_rows += 1

            question = questions_by_id.get(row["question_id"])
            is_code_gen = (
                question is not None
                and question.domain == "coding"
                and question.type in CODE_GENERATION_TYPES
            )

            if not is_code_gen:
                # Not a code-generation row — copy through unchanged.
                outfile.write(json.dumps(row) + "\n")
                continue

            # Re-extract from the raw response using the fixed parser.
            raw_response = row["raw_response"]
            if raw_response.startswith("[MODEL CALL FAILED]"):
                # Infra failure, not a parsing issue — copy through unchanged.
                outfile.write(json.dumps(row) + "\n")
                continue

            old_parsed = row["parsed_answer"]
            old_score = row["score"]

            new_parsed = parse_standalone_code(raw_response)
            new_score = score_code(
                new_parsed,
                question.test_cases,
                expected_func_name=_extract_func_name(question.question),
            )
            new_parse_failure = new_parsed is None

            reparsed_count += 1
            if new_parsed != old_parsed or new_score != old_score:
                changed_count += 1

            new_row = dict(row)  # copy, don't mutate original in place
            new_row["parsed_answer"] = new_parsed
            new_row["score"] = new_score
            new_row["parse_failure"] = new_parse_failure
            new_row["reparsed"] = True  # flag so it's clear this row was corrected

            outfile.write(json.dumps(new_row) + "\n")

    print(f"Total rows processed: {total_rows}")
    print(f"Code-generation rows re-parsed: {reparsed_count}")
    print(f"Rows with changed parsed_answer or score: {changed_count}")
    print(f"Original file untouched: {ORIGINAL_RESULTS_PATH}")
    print(f"Corrected file written: {REPARSED_RESULTS_PATH}")


if __name__ == "__main__":
    main()
