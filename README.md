# A Nonlinear Relationship in LLM Proposer–Verifier Synergy

Code, data, and analysis for the paper *"A Nonlinear Relationship in LLM Proposer-Verifier Synergy"* 

---

## What this project is about

Proposer-verifier pipelines, where one language model answers a question and a second model reviews that answer before it's accepted, are increasingly common in LLM systems: rejection sampling, self-refinement, and agentic workflows all lean on this pattern. The implicit assumption is that a stronger model makes a better reviewer.

This project tests that assumption directly, and largely finds it wanting.

Using 15 small, open-weight language models (0.5B–9B parameters) run entirely on local hardware, every model was paired with every other model as both proposer and verifier: all 210 ordered pairings, across 100 math and coding questions, for 21,000 total collaboration trials. This is a full census of the pairing space.

## Key findings

**1. Verifying is a harder job than solving: for the same model, on the same questions.**
Averaged across the 15 models, standalone problem-solving accuracy was 36.4%, but accuracy when reviewing another model's answer dropped to 22.0% (paired t(14) = -3.20, p = 0.0065). This held for 12 of 15 models and wasn't explained by a simple ceiling effect: models with plenty of standalone headroom still degraded substantially when acting as verifiers.

Every collaboration trial was classified into one of four outcomes: the verifier echoed a correct answer, corrected an incorrect one, failed to correct an incorrect one, or overrode a correct answer with an incorrect one. Harmful overrides (10.4% of trials) slightly outnumbered helpful corrections (10.0%), which alone is enough to push average synergy negative before even accounting for the much larger "no help" category (57.2%).

**2. Verifying accuracy, not standalone accuracy, predicts whether a verifier actually helps, and the relationship isn't a straight line.**
This is the paper's central result. Regressing collaboration synergy on a verifier's *standalone* accuracy produces a weak, non-generalizing fit (R² = 0.030, leave-one-out cross-validation R² = -1.04 — worse than just predicting the mean). But regressing synergy on each verifier's *directly measured verifying* accuracy tells a very different story: a strong inverted-U relationship (quadratic coefficient = -5.93, p = 0.002, R² = 0.75, LOO-CV R² = 0.62), peaking around 27.5% verifying accuracy. Verifiers that are too weak add little value — but so, surprisingly, do verifiers that already solve the task well on their own.

Because verifying accuracy and synergy are computed from overlapping data, this relationship could in principle be a statistical artifact rather than a real effect. That possibility is addressed directly with a split-half validation: verifying accuracy computed from a random half of each model's trials, synergy computed from the disjoint other half, repeated across 20 independent resamples. The inverted-U survives: the coefficient stays negative in all 20 splits and remains statistically significant in 19 of them.

**3. The effect is concentrated in coding questions, not math, which runs against what prior literature would predict.**
Splitting the analysis by domain, the inverted-U is strong and significant for coding questions alone (quadratic coefficient = -5.00, p = 0.042, R² = 0.80) but not significant for math questions (p = 0.061–0.082 depending on which models are included). This is somewhat counterintuitive: prior work generally finds math/logic tasks easier to verify than open-ended ones, which would predict the *opposite* pattern. The paper's working explanation is that code correctness is checkable against concrete unit tests, giving a sharper, less noisy outcome signal than exact-match scoring on multi-step math derivations. This is flagged explicitly as a hypothesis for future work, not a settled mechanism.

## Why this matters practically

Two implications the paper draws out:

- **Standalone benchmark scores are a poor proxy for verifying ability.** If you're designing a proposer-verifier pipeline and picking a verifier based on how well it solves problems on its own, you may be optimizing for the wrong thing entirely.
- **Adding a second model to a pipeline isn't a universal win — it depends on task type.** The benefit shows up clearly for checkable, unit-test-style tasks and is much murkier for open-ended reasoning tasks like math, at least at this model scale.

## Study design at a glance

| | |
|---|---|
| **Models** | 15 open-weight models, 0.5B–9B parameters, run locally via Ollama (Q4_K_M quantization) |
| **Questions** | 100 total: 60 math (GSM8K-style, algebra, number theory, geometry, probability, olympiad-style) and 40 coding (generation, debugging, output prediction, edge cases) |
| **Trials** | 1,500 standalone (Phase 1) + 21,000 collaboration trials across all 210 ordered pairs (Phase 2) |
| **Decoding** | Temperature 0, seed 42, no resampling on failure, fully deterministic and reproducible |
| **Scoring** | Exact-match for math (numeric/symbolic, normalized); unit-test pass rate for code |
| **Core metric** | *Synergy*: how much a proposer-verifier pair outperforms (or underperforms) the better of the two models solving independently |

## Model roster

All 15 models were run locally via [Ollama](https://ollama.com) at Q4_K_M quantization, temperature 0, seed 42:

| Model | Size |
| --- | --- |
| Qwen2.5 | 0.5B |
| Qwen2.5 | 7B |
| TinyLlama | 1.1B |
| Llama 3.2 | 1B |
| Llama 3.2 | 3B |
| SmolLM2 | 1.7B |
| Gemma2 | 2B |
| Gemma2 | 9B |
| Falcon3 | 3B |
| Phi-3-mini | 3.8B |
| Phi-3.5 | 3.8B |
| Yi | 6B |
| DeepSeek-Coder | 6.7B |
| Mistral | 7B |
| Llama 3 | 8B |

Every model served once as proposer and once as verifier against every other model (15 × 14 = 210 ordered pairs), so the roster spans distinct model families rather than family-controlled pairs: this means family-similarity effects (e.g., a model being biased toward answers that "sound like" its own family) can't be isolated, though the overall degradation pattern still holds across this diverse set.

## Statistical validation used

This project leans on several checks specifically designed to rule out the result being a fluke of small sample size or shared-metric dependency, rather than resting on a single headline p-value:

- Paired t-test for the standalone-vs-verifying accuracy gap
- Correlation test to rule out a simple ceiling effect
- Leave-one-out cross-validation on the quadratic regression
- 20-resample split-half validation, computing the predictor and outcome from disjoint halves of the data
- Domain-specific decomposition (math vs. coding) with separate significance testing for each

## Limitations (see the paper's Discussion section for the full treatment)

- Fifteen models is a small sample for fitting a quadratic regression; the LOO-CV and split-half checks mitigate, but don't eliminate, this concern.
- All models are 9B parameters or smaller; whether the same inverted-U pattern holds for larger or reasoning-post-trained models is untested.
- The three weakest models (Falcon3 3B, TinyLlama 1.1B, Llama 3.2 1B) produced few parseable verifying trials, making their individual synergy estimates less reliable — excluding them tightens but doesn't qualitatively change the pooled result.
- The math/coding asymmetry comes from a single 100-question dataset and should be read as a hypothesis for a larger, purpose-built follow-up rather than a settled domain effect.
- The proposed "anchoring" mechanism (verifiers anchoring on a proposer's framing rather than re-deriving judgments from scratch) is offered as an explanation for the standalone-vs-verifying gap but isn't directly tested here.
