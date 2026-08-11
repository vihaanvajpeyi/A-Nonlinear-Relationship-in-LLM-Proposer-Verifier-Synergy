# Methods Note: Code-Generation Parser Correction

**Context:** During Phase 1 standalone evaluation, an initial version of
`parse_standalone_code` (in `parser.py`) extracted code-generation responses
using a regex that captured from the first `def` statement to the end of
the raw model response. Many instruction-tuned models wrap their code
output in markdown fences (```` ```python ... ``` ````) despite the prompt
explicitly requesting a bare function definition. The original parser did
not strip these fences, so the trailing ` ``` ` was included in the
extracted "function," causing `exec()` to fail with a syntax error at
scoring time — even when the underlying function was correct.

**Discovery:** This was identified during post-run analysis when
`deepseek-coder:6.7b` — the only code-specialized model in the roster —
scored unexpectedly low on coding questions (20.5%), lower than several
general-purpose models with no code specialization. Manual inspection of
raw responses confirmed the model was producing correct, well-formed
functions wrapped in markdown fences that the parser was mis-extracting.

**Fix:** `parse_standalone_code` was updated to strip markdown code fences
(with or without a language tag) before extracting the function definition.

**Correction procedure:** Rather than re-querying any models, all
already-collected raw responses for code-generation questions were
re-parsed and re-scored using the corrected parser
(`reparse_phase1.py`). The original `phase1_results.jsonl` file was left
unmodified; corrected results were written to a separate file
(`phase1_results_reparsed.jsonl`), flagged with `"reparsed": true` on
affected rows. Non-code-generation rows (math, debugging, output
prediction, edge cases) were unaffected by this bug and were carried
through unchanged.

**Impact:** 150 code-generation rows were re-scored; 122 (81.3%) had a
changed parsed answer or score. Coding accuracy increased for every model
in the roster, since the bug affected fence-wrapped output universally,
not specific to any one model. `deepseek-coder:6.7b`'s coding accuracy
increased from 20.5% to 41.0%, consistent with genuine code-specialization
strength that had been masked by the extraction bug.

**Reported results:** All coding accuracy figures reported in this paper
use the corrected parser and `phase1_results_reparsed.jsonl` (or Phase 2
runs built on the corrected `parser.py`, applied prospectively from this
point forward). This is disclosed here for methodological transparency,
consistent with standard practice of reporting bug fixes discovered during
analysis rather than treating first-pass pipeline output as ground truth.
