"""
evaluator.py — Scoring engine for the synergy expansion experiment.

Implements scoring rules from protocol_v2.md section 8:
  - 8.1 Standalone accuracy (math/debugging/etc. via normalizer.answers_match;
        code generation via normalizer.grade_code)
  - 8.2 Collaboration outcome (verifier's Final Answer / Corrected Function
        is always the collaboration answer)
  - 8.3 Derived fields (changed_answer, synergy, etc.)
  - 8.4 Synergy definition

Uses normalizer.py's frozen v2.1 comparison logic (`answers_match`,
`grade_code`) rather than reimplementing normalization.

Usage:
    from evaluator import score_standalone, score_code, compute_synergy_row

    # Math / debugging / output prediction / edge cases
    score = score_standalone(parsed_answer, question.final_answer)  # bool or None

    # Code generation
    score = score_code(parsed_function, question.test_cases)  # float 0.0-1.0

    # Full Phase 2 derived-field computation
    row = compute_synergy_row(
        proposer_answer=..., verifier_answer=..., ground_truth=...,
        verifier_standalone_answer=..., is_code=False
    )
"""

from typing import Optional, Union

from normalizer import answers_match, grade_code


def score_standalone(parsed_answer: Optional[str], ground_truth: str) -> Optional[bool]:
    """
    Score a standalone/proposer/verifier answer for math, debugging, output
    prediction, or edge case questions.

    Per protocol_v2.md 8.1: parse failure (parsed_answer is None) -> score
    is null (returned here as None, not False) — must NOT be treated as
    "incorrect" downstream.

    Returns:
        True if correct, False if incorrect, None if parse failure.
    """
    if parsed_answer is None:
        return None  # parse failure -> null, never 0
    return answers_match(parsed_answer, ground_truth)


def score_code(parsed_function: Optional[str], test_cases: list,
               expected_func_name: str = None) -> float:
    """
    Score a code-generation answer against test cases.

    Per protocol_v2.md 8.1: parse failure or runtime error -> score = 0.0
    (NOT null, unlike the non-code case — code questions use a continuous
    0.0-1.0 score where failure is indistinguishable from "passed 0 tests").

    Returns:
        Float in [0.0, 1.0] — fraction of test cases passed.
    """
    if parsed_function is None:
        return 0.0
    return grade_code(parsed_function, test_cases, expected_func_name=expected_func_name)


def compute_synergy_row(
    proposer_answer: Union[bool, float, None],
    verifier_answer: Union[bool, float, None],
    verifier_standalone_answer: Union[bool, float, None],
    proposer_correct: Union[bool, float, None],
    verifier_correct: Union[bool, float, None],
    verifier_standalone_correct: Union[bool, float, None],
    changed_answer: Optional[bool] = None,
) -> dict:
    """
    Compute the Phase 2 derived fields per protocol_v2.md section 8.3-8.4.

    NOTE: this function takes already-scored values (proposer_correct,
    verifier_correct, verifier_standalone_correct — each a bool, float, or
    None for parse failure) rather than raw answers, since the caller must
    first run score_standalone/score_code with the appropriate ground truth
    and test cases before deriving synergy.

    'changed_answer' is optional to pass in directly (raw text comparison
    handled by caller, since it requires the raw parsed answer strings, not
    scores). If omitted, it is left as None in the output.

    Per protocol_v2.md 8.2: the collaboration answer is ALWAYS the
    verifier's Final Answer (or Corrected Function). collaboration_correct
    == verifier_correct.

    Returns a dict matching the Phase 2 logging schema fields:
        changed_answer, collaboration_correct, proposer_correct,
        verifier_correct_standalone, synergy
    """
    collaboration_correct = verifier_correct  # per 8.2 — always the verifier's answer

    # Critical null-propagation rule: if the collaboration outcome itself is
    # unknown (verifier_correct is None -> genuine verifier parse failure on
    # a non-code question), synergy CANNOT be computed at all — there is no
    # collaboration result to compare against anything. Returning a numeric
    # synergy here would fabricate data for a row that protocol_v2.md
    # section 7.3 requires be treated as null, not silently coerced to a
    # specific (and misleadingly precise) number like -1.0.
    #
    # This is distinct from proposer_correct / verifier_standalone_correct
    # being None (missing baseline) — those still allow a defensible,
    # explicitly-documented 0.0 fallback in the max() comparison below,
    # since the collaboration outcome itself is known in that case.
    if collaboration_correct is None:
        return {
            "changed_answer": changed_answer,
            "collaboration_correct": None,
            "proposer_correct": proposer_correct,
            "verifier_correct_standalone": verifier_standalone_correct,
            "synergy": None,
        }

    # Synergy requires numeric comparison. Treat None (parse failure) as 0.0
    # for the purposes of the max() comparison, per standard practice for
    # missing/null BASELINE scores in this kind of aggregate — but flag this
    # explicitly since protocol_v2.md does not specify null-handling for
    # synergy computation itself (only for per-question scoring in 8.1).
    # This fallback only applies to proposer_correct / verifier_standalone_
    # correct, never to collaboration_correct itself (handled above).
    def _as_numeric(v):
        if v is None:
            return 0.0
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        return float(v)

    collab_num = _as_numeric(collaboration_correct)
    proposer_num = _as_numeric(proposer_correct)
    verifier_standalone_num = _as_numeric(verifier_standalone_correct)

    synergy = collab_num - max(proposer_num, verifier_standalone_num)

    return {
        "changed_answer": changed_answer,
        "collaboration_correct": collaboration_correct,
        "proposer_correct": proposer_correct,
        "verifier_correct_standalone": verifier_standalone_correct,
        "synergy": synergy,
    }


def answers_differ(answer_a: Optional[str], answer_b: Optional[str]) -> Optional[bool]:
    """
    Determine whether two raw parsed answer strings differ (for the
    'changed_answer' field). Uses normalizer.answers_match for consistency
    with scoring (so trivially-different-but-equivalent strings, e.g.
    "3.0" vs "3", are NOT counted as "changed").

    Returns None if either answer is a parse failure (None), since
    "did the answer change" is undefined when one side failed to parse.
    """
    if answer_a is None or answer_b is None:
        return None
    return not answers_match(answer_a, answer_b)


if __name__ == "__main__":
    # Smoke test
    print("=== score_standalone ===")
    print("Correct match:", score_standalone("3.75", "3.75"))
    print("Incorrect match:", score_standalone("4", "3.75"))
    print("Parse failure:", score_standalone(None, "3.75"))
    print()

    print("=== score_code ===")
    good_fn = "def running_sum(lst):\n    total = 0\n    result = []\n    for x in lst:\n        total += x\n        result.append(total)\n    return result"
    test_cases = [
        {"input": "[1,2,3,4]", "expected": "[1,3,6,10]"},
        {"input": "[0,0,0]", "expected": "[0,0,0]"},
    ]
    print("Good function score:", score_code(good_fn, test_cases))
    print("Parse failure score:", score_code(None, test_cases))
    print()

    print("=== compute_synergy_row ===")
    # Case: proposer wrong, verifier corrects it -> positive synergy
    row = compute_synergy_row(
        proposer_answer="4", verifier_answer="3.75",
        verifier_standalone_answer="4",
        proposer_correct=False, verifier_correct=True,
        verifier_standalone_correct=False,
        changed_answer=answers_differ("4", "3.75"),
    )
    print("Proposer wrong, verifier corrects:", row)
    print()

    # Case: proposer correct, verifier echoes -> zero synergy
    row = compute_synergy_row(
        proposer_answer="3.75", verifier_answer="3.75",
        verifier_standalone_answer="3.75",
        proposer_correct=True, verifier_correct=True,
        verifier_standalone_correct=True,
        changed_answer=answers_differ("3.75", "3.75"),
    )
    print("Proposer correct, verifier echoes:", row)
