"""
logger.py — Result logger for the synergy expansion experiment.

Writes result rows to a JSONL file (one JSON object per line), matching the
exact schema from protocol_v2.md section 12. Append-only by design — per
protocol_v2.md section 11 (stopping criteria) and section 9 (retry policy),
no result may ever be overwritten or modified after logging.

Usage:
    from logger import ResultLogger

    logger = ResultLogger("phase1_results.jsonl")

    logger.log_phase1(
        model_id="M10", question_id="M012", role="standalone",
        raw_response="...", parsed_answer="5955.08", score=1.0,
        parse_failure=False, prompt_version="v1"
    )

    logger.log_phase2(
        proposer_id="M03", verifier_id="M11", question_id="M012",
        proposer_answer="5955.08", verifier_answer="5955.08",
        verdict="CORRECT", reason="...", changed_answer=False,
        collaboration_score=1.0, synergy=0.0, parse_failure=False,
        prompt_version="v1"
    )
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ResultLogger:
    """
    Append-only JSONL logger for Phase 1 and Phase 2 results.

    Each call to log_phase1 / log_phase2 appends exactly one line to the
    target file and flushes immediately, so a crash mid-run never loses
    already-logged rows and never corrupts the file.
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        # Create the file (and parent dirs) if it doesn't exist yet.
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            self.output_path.touch()

    def _append_row(self, row: dict) -> None:
        """Append a single JSON row to the output file. Never overwrites."""
        with open(self.output_path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log_phase1(
        self,
        model_id: str,
        question_id: str,
        role: str,
        raw_response: str,
        parsed_answer: Optional[str],
        score: Optional[float],
        parse_failure: bool,
        prompt_version: str = "v1",
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Log a Phase 1 result row per protocol_v2.md section 12.

        'role' is typically "standalone" for Phase 1.
        'score' should be None for parse failures (per section 7.3/8.1),
        a bool for math/debugging/etc., or a float 0.0-1.0 for code.
        """
        row = {
            "phase": 1,
            "model_id": model_id,
            "question_id": question_id,
            "role": role,
            "raw_response": raw_response,
            "parsed_answer": parsed_answer,
            "score": score,
            "parse_failure": parse_failure,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "prompt_version": prompt_version,
        }
        self._append_row(row)
        return row

    def log_phase2(
        self,
        proposer_id: str,
        verifier_id: str,
        question_id: str,
        proposer_answer: Optional[str],
        verifier_answer: Optional[str],
        verdict: Optional[str],
        reason: Optional[str],
        changed_answer: Optional[bool],
        collaboration_score: Optional[float],
        synergy: Optional[float],
        parse_failure: bool,
        prompt_version: str = "v1",
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Log a Phase 2 (collaboration) result row per protocol_v2.md section 12.
        """
        row = {
            "phase": 2,
            "proposer_id": proposer_id,
            "verifier_id": verifier_id,
            "question_id": question_id,
            "proposer_answer": proposer_answer,
            "verifier_answer": verifier_answer,
            "verdict": verdict,
            "reason": reason,
            "changed_answer": changed_answer,
            "collaboration_score": collaboration_score,
            "synergy": synergy,
            "parse_failure": parse_failure,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "prompt_version": prompt_version,
        }
        self._append_row(row)
        return row

    def count_logged_rows(self) -> int:
        """Return the number of rows currently logged (for resumability checks)."""
        if not self.output_path.exists():
            return 0
        with open(self.output_path, "r") as f:
            return sum(1 for _ in f)

    def already_logged_keys(self, phase: int) -> set:
        """
        Return a set of unique keys already logged for the given phase, so a
        run can be resumed without re-doing (and duplicate-logging) completed
        work. This does NOT delete or modify anything — it's read-only,
        purely for the caller to skip already-done combinations.

        Phase 1 key: (model_id, question_id, role)
        Phase 2 key: (proposer_id, verifier_id, question_id)
        """
        keys = set()
        if not self.output_path.exists():
            return keys

        with open(self.output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("phase") != phase:
                    continue
                if phase == 1:
                    keys.add((row["model_id"], row["question_id"], row["role"]))
                elif phase == 2:
                    keys.add((row["proposer_id"], row["verifier_id"], row["question_id"]))
        return keys


if __name__ == "__main__":
    # Smoke test using a throwaway file
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_results.jsonl")
        logger = ResultLogger(test_path)

        print("=== Logging Phase 1 row ===")
        row1 = logger.log_phase1(
            model_id="M10", question_id="M012", role="standalone",
            raw_response="...working...\nFinal Answer: 5955.08",
            parsed_answer="5955.08", score=1.0, parse_failure=False,
        )
        print(row1)
        print()

        print("=== Logging Phase 2 row ===")
        row2 = logger.log_phase2(
            proposer_id="M03", verifier_id="M11", question_id="M012",
            proposer_answer="5955.08", verifier_answer="5955.08",
            verdict="CORRECT", reason="Matches expected computation.",
            changed_answer=False, collaboration_score=1.0, synergy=0.0,
            parse_failure=False,
        )
        print(row2)
        print()

        print("=== Row count ===")
        print(logger.count_logged_rows())
        print()

        print("=== Reading back file contents ===")
        with open(test_path) as f:
            print(f.read())

        print("=== already_logged_keys(phase=1) ===")
        print(logger.already_logged_keys(1))
        print("=== already_logged_keys(phase=2) ===")
        print(logger.already_logged_keys(2))
