"""
prompts.py — Prompt builder for the synergy expansion experiment.

Formats questions into the exact frozen prompt templates defined in
protocol_v2.md section 6. Templates are copied verbatim — do not modify
without a new protocol version.

Usage:
    from benchmark import BenchmarkLoader
    from prompts import build_standalone_prompt, build_verifier_prompt

    loader = BenchmarkLoader("question_dataset_v2.json")
    q = loader.get_question_by_id("M001")

    prompt = build_standalone_prompt(q)

    # After getting a proposer response:
    verifier_prompt = build_verifier_prompt(q, proposer_response="...")
"""

from benchmark import Question


# Coding question types that use the "code generation" template (function-only,
# no Final Answer field) vs. the "debugging/output prediction/edge cases"
# template (concise answer + Final Answer field).
CODE_GENERATION_TYPES = {"code_generation"}


def _is_code_generation(question: Question) -> bool:
    """Determine whether a coding question uses the code-generation prompt
    style (raw function, no Final Answer) vs. the debugging/tracing style
    (Final Answer field required)."""
    return question.domain == "coding" and question.type in CODE_GENERATION_TYPES


def build_standalone_prompt(question: Question) -> str:
    """
    Build the Phase 1 standalone eval prompt (also used, unchanged, as the
    Phase 2 proposer prompt per protocol_v2.md section 6.2).
    """
    if question.domain == "math":
        return (
            "Solve the following problem. Show your reasoning step by step.\n\n"
            f"Problem: {question.question}\n\n"
            "At the end of your response, write your answer on its own line "
            "in this exact format:\n"
            "Final Answer: {answer}"
        )

    elif question.domain == "coding":
        if _is_code_generation(question):
            return (
                "Write a Python function to solve the following problem. "
                "Return only the function definition with no extra explanation.\n\n"
                f"Problem: {question.question}"
            )
        else:
            # debugging / output prediction / edge cases
            return (
                "Answer the following question about Python code. Be concise.\n\n"
                f"Question: {question.question}\n\n"
                "At the end of your response, write your answer on its own line "
                "in this exact format:\n"
                "Final Answer: {answer}"
            )

    else:
        raise ValueError(f"Unknown domain: {question.domain}")


def build_proposer_prompt(question: Question) -> str:
    """
    Phase 2 proposer prompt. Identical to the standalone eval prompt per
    protocol_v2.md section 6.2 — the proposer has no knowledge of the
    verifier role.
    """
    return build_standalone_prompt(question)


def build_verifier_prompt(question: Question, proposer_response: str) -> str:
    """
    Build the Phase 2 verifier prompt, given the proposer's raw response.
    """
    if question.domain == "math":
        return (
            "You are reviewing another model's solution to a math problem.\n\n"
            f"Problem: {question.question}\n\n"
            "Proposed solution:\n"
            f"{proposer_response}\n\n"
            "Evaluate whether the proposed solution is correct.\n\n"
            "Respond in exactly this format:\n"
            "Verdict: CORRECT or INCORRECT\n"
            "Reason: {1-3 sentences identifying the key issue, or confirming correctness}\n"
            "Final Answer: {the correct answer}"
        )

    elif question.domain == "coding":
        if _is_code_generation(question):
            return (
                "You are reviewing another model's Python function.\n\n"
                f"Problem: {question.question}\n\n"
                "Proposed function:\n"
                f"{proposer_response}\n\n"
                "Evaluate whether the proposed function is correct for all "
                "inputs, including edge cases.\n\n"
                "Respond in exactly this format:\n"
                "Verdict: CORRECT or INCORRECT\n"
                "Reason: {1-3 sentences identifying the key issue, or confirming correctness}\n"
                "Corrected Function:\n"
                "{raw Python function definition, no markdown fences}"
            )
        else:
            # debugging / output prediction / edge cases
            return (
                "You are reviewing another model's answer to a Python question.\n\n"
                f"Question: {question.question}\n\n"
                "Proposed answer:\n"
                f"{proposer_response}\n\n"
                "Evaluate whether the proposed answer is correct.\n\n"
                "Respond in exactly this format:\n"
                "Verdict: CORRECT or INCORRECT\n"
                "Reason: {1-3 sentences identifying the key issue, or confirming correctness}\n"
                "Final Answer: {the correct answer}"
            )

    else:
        raise ValueError(f"Unknown domain: {question.domain}")


if __name__ == "__main__":
    # Smoke test: build prompts for a few sample questions
    from benchmark import BenchmarkLoader

    loader = BenchmarkLoader("question_dataset_v2.json")

    print("=== Math standalone prompt ===")
    m = loader.load_questions(domain="math")[0]
    print(build_standalone_prompt(m))
    print()

    print("=== Coding standalone prompt (code generation) ===")
    c_gen = next(q for q in loader.load_questions(domain="coding") if q.type == "code_generation")
    print(build_standalone_prompt(c_gen))
    print()

    print("=== Coding standalone prompt (debugging) ===")
    c_debug = next(q for q in loader.load_questions(domain="coding") if q.type == "debugging")
    print(build_standalone_prompt(c_debug))
    print()

    print("=== Verifier prompt (math) ===")
    print(build_verifier_prompt(m, proposer_response="The answer is 42."))
