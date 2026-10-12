# Ideas: pipeline and register

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

This file is where ideas for quantum machine learning (QML) are proposed, filtered, tested and
recorded. The goal is to find ideas that have a real chance of beating strong classical methods,
and to kill the others quickly and for a written reason.

## The pipeline

```
 1. GENERATE      ChatGPT, DeepSeek, Adaptive AI and Claude each propose ideas
                  (prompt: ideas/PROMPT_GENERATOR.md + ideas/CONTEXT.md)
        │
 2. DEDUPLICATE   one ChatGPT conversation merges duplicates and flags ideas it believes already exist
                  (prompt: ideas/PROMPT_NOVELTY_FILTER.md)
        │
 3. VERIFY        Claude checks every "already exists" / "new" claim against a real arXiv search
                  (scripts/arxiv_search.py) and the field map. Invented references are discarded.
        │
 4. TRIAGE        Claude applies the kill criteria below. Each idea gets REJECTED (with the reason)
                  or SHORTLISTED (with the test that would decide it)
        │
 5. DECIDE        Athmane picks which shortlisted ideas to test
        │
 6. TEST          the idea becomes an experiment in this repository, run against tuned classical
                  baselines with the rules of DESIGN.md. The result goes in RESULTS.md and the
                  register below, whether it wins or loses
```

Why this shape:

- Language models are good at producing many ideas. They are **not** reliable at knowing whether an
  idea already exists, because they can invent references or miss real ones. So step 2 only
  *suggests*, and step 3 checks against arXiv itself.
- An AI "liking" an idea is not evidence. Only step 6 is.

## Idea card (every idea must fill every field)

| Field | Content |
|---|---|
| ID | `I-YYYYMMDD-NN` (assigned when it enters the register) |
| Title | ≤ 12 words |
| Proposed by | ChatGPT / DeepSeek / Adaptive AI / Claude / Athmane |
| Hypothesis | One sentence: what quantum model beats what classical model on what task |
| Mechanism | Why a quantum computer could do better here, and why a classical model could *not* simply copy it |
| Data | Classical data, or quantum data (states or measurements from a quantum system) |
| Milestone | Which FIELD_MAP.md milestone (M1–M9) it would move |
| Minimal test | Dataset, models, classical baselines, metric, and the result that would count as success, all feasible with ≤ 10 qubits simulated on a CPU in ≤ 10 minutes (or a precise theoretical claim) |
| Kill-criteria check | Which of K1–K7 below it might trip, and why it does not |
| Closest prior work | arXiv IDs, **verified** in step 3 (unverified ones are marked NOT VERIFIED) |
| Status | See below |
| Decision note | One line, with the reason |

**Statuses:** `PROPOSED` → `CHECKED` (novelty verified) → `REJECTED` or `SHORTLISTED` →
`TESTING` → `RESULT: WIN` / `RESULT: NO DIFFERENCE` / `RESULT: LOSS` / `RESULT: INCONCLUSIVE`.

## Kill criteria (from FIELD_MAP.md)

An idea is rejected if it trips one of these and offers no convincing way around it:

| # | Criterion | Why (evidence) |
|---|---|---|
| K1 | It claims an advantage on **classical data** from a fixed, problem-agnostic embedding (IQP, angle encoding, hardware-efficient circuit), with no reason why a classical kernel could not imitate it | Power of data, geometric difference (arXiv:2011.01938); our own result g ≈ 1.3–4.9 ≪ √N (RESULTS.md); entanglement did not help (arXiv:2403.07059 and our ablation) |
| K2 | It needs an exponential speed-up from quantum linear algebra (HHL-style) on classical data, without addressing dequantisation and data loading | arXiv:1807.04271, arXiv:1910.06151, doi:10.1038/nphys3272, arXiv:2305.10310 |
| K3 | It is a variational model that is either deep and random (barren plateaus), or trainable for a reason that also makes it classically simulable | arXiv:1803.11173, arXiv:2312.09121 |
| K4 | Its evidence would be a comparison with an untuned or single classical baseline | arXiv:2403.07059 and DESIGN.md §4 |
| K5 | It ignores shot cost, noise or kernel concentration at the scale where it claims to win | arXiv:2208.11060, arXiv:2210.11505 |
| K6 | It cannot be tested: not with ≤ 10 simulated qubits, and not by a precise theoretical statement | Repository rule (DESIGN.md) |
| K7 | It has already been done, and the new version adds nothing testable | Verified arXiv search (step 3) |

**Green flags.** Learning from quantum data or quantum-enhanced sensing (M8). A problem-specific
inductive bias that classical models cannot encode efficiently, such as a group or physical
structure (arXiv:2105.03406, arXiv:2106.03747). Provable separations on natural distributions (M1).
Generative models with classically hard sampling (arXiv:2509.09033).

## How a shortlisted idea becomes an experiment

1. Write its design into DESIGN.md first: data, models, baselines, metric, success rule.
2. Add the model or dataset in `src/qmlrc/` (models.py, data.py, experiments.py) with tests.
3. Add an experiment to `qmlrc.experiments.FULL` so that `python -m qmlrc.run_all` reproduces it.
4. Report the outcome in RESULTS.md and READING.md, update this register, and log it in JOURNAL.md.

## Register

| ID | Title | Proposed by | Milestone | Status | Decision note |
|---|---|---|---|---|---|
| *(empty: the first generation round has not happened yet)* | | | | | |
