"""
parser.py — Response parser for the synergy expansion experiment.

Extracts structured answers from raw model text responses. All regex logic
is copied verbatim from protocol_v2.md section 7 — do not modify without a
new protocol version.

Usage:
    from parser import parse_standalone, parse_verifier, parse_standalone_code, parse_verifier_code

    answer = parse_standalone(model_response_text)          # math / debugging / etc.
    code   = parse_standalone_code(model_response_text)      # code generation
    result = parse_verifier(model_response_text)             # verifier, non-code
    result = parse_verifier_code(model_response_text)        # verifier, code generation

All parse functions return None on parse failure. Per protocol_v2.md section 7.3:
  - Parse failures are logged with raw response + question ID
  - Scored as null (not 0, not 1)
  - Never retried
"""

import re


def parse_standalone(response: str):
    """
    Parse a standalone (Phase 1) or proposer (Phase 2) response for math,
    debugging, output prediction, or edge case questions.

    Returns the extracted answer string, or None on parse failure
    (missing Final Answer field, or ambiguous answer containing
    "or" / "either" / "depending").
    """
    m = re.search(r'(?i)final answer:\s*(.+)', response)
    if not m:
        return None  # parse failure
    answer = m.group(1).strip()
    if re.search(r'\bor\b|\beither\b|\bdepending\b', answer, re.IGNORECASE):
        return None  # ambiguous answer — treat as parse failure
    return answer


def parse_verifier(response: str):
    """
    Parse a Phase 2 verifier response for math, debugging, output prediction,
    or edge case questions.

    Returns a dict with 'verdict', 'reason', 'answer' keys, or None on parse
    failure (missing or ambiguous Final Answer field).
    """
    verdict = re.search(r'(?i)verdict:\s*(CORRECT|INCORRECT)', response)
    reason = re.search(
        r'(?i)reason:\s*(.+?)(?=\nfinal answer:|\ncorrected function:|$)',
        response, re.DOTALL
    )
    answer = re.search(r'(?i)final answer:\s*(.+)', response)

    if not answer:
        return None  # parse failure

    answer_text = answer.group(1).strip()
    if re.search(r'\bor\b|\beither\b|\bdepending\b', answer_text, re.IGNORECASE):
        return None  # ambiguous — parse failure

    return {
        'verdict': verdict.group(1).upper() if verdict else None,
        'reason': reason.group(1).strip() if reason else None,
        'answer': answer_text,
    }


def parse_standalone_code(response: str):
    """
    Parse a standalone (Phase 1) or proposer (Phase 2) response for
    code-generation questions.

    Strips markdown code fences (```python ... ``` or ``` ... ```) before
    extraction, since many instruction-tuned models wrap code in fences
    despite being told not to. Without this, a trailing ``` gets captured
    as part of the "function" and breaks exec() at scoring time, silently
    producing false 0.0 scores on otherwise-correct code.

    Returns the extracted function definition (from the first 'def' found
    to the end of the response, fences stripped), or None on parse failure.
    """
    # Remove markdown code fences (with or without language tag)
    cleaned = re.sub(r'```(?:python)?\s*\n?', '', response)
    cleaned = cleaned.replace('```', '')

    m = re.search(r'(def\s+\w+\s*\(.*)', cleaned, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def parse_verifier_code(response: str):
    """
    Parse a Phase 2 verifier response for code-generation questions.

    Returns a dict with 'verdict', 'reason', 'function' keys, or None on
    parse failure (missing Corrected Function field).
    """
    verdict = re.search(r'(?i)verdict:\s*(CORRECT|INCORRECT)', response)
    reason = re.search(
        r'(?i)reason:\s*(.+?)(?=\ncorrected function:|$)',
        response, re.DOTALL
    )
    function = re.search(
        r'(?i)corrected function:\s*\n(def\s+\w+.+)',
        response, re.DOTALL
    )

    if not function:
        return None  # parse failure

    return {
        'verdict': verdict.group(1).upper() if verdict else None,
        'reason': reason.group(1).strip() if reason else None,
        'function': function.group(1).strip(),
    }


if __name__ == "__main__":
    # Smoke test with representative example responses

    print("=== parse_standalone (math, clean) ===")
    resp = "First I compute Saturday: 48*2.50+22*3.75=202.50\nSunday: 36*2.50+31*3.75=206.25\nDifference is 3.75\nFinal Answer: 3.75"
    print(repr(parse_standalone(resp)))
    print()

    print("=== parse_standalone (missing Final Answer -> parse failure) ===")
    resp = "The answer is probably around 4 but I'm not fully sure."
    print(repr(parse_standalone(resp)))
    print()

    print("=== parse_standalone (ambiguous -> parse failure) ===")
    resp = "Final Answer: 3.75 or possibly 4 depending on rounding"
    print(repr(parse_standalone(resp)))
    print()

    print("=== parse_verifier (math, clean) ===")
    resp = "Verdict: INCORRECT\nReason: The proposer made an arithmetic error in the Sunday total.\nFinal Answer: 3.75"
    print(parse_verifier(resp))
    print()

    print("=== parse_standalone_code (clean) ===")
    resp = "Here's the function:\n\ndef running_sum(lst):\n    total = 0\n    result = []\n    for x in lst:\n        total += x\n        result.append(total)\n    return result"
    print(repr(parse_standalone_code(resp)))
    print()

    print("=== parse_verifier_code (clean) ===")
    resp = "Verdict: CORRECT\nReason: The function correctly accumulates the running sum.\nCorrected Function:\ndef running_sum(lst):\n    total = 0\n    result = []\n    for x in lst:\n        total += x\n        result.append(total)\n    return result"
    print(parse_verifier_code(resp))
