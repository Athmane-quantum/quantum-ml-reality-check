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

An idea is rejected if it trips one of these and offers no convincing way around it (K8 was added after round 1):

| # | Criterion | Why (evidence) |
|---|---|---|
| K1 | It claims an advantage on **classical data** from a fixed, problem-agnostic embedding (IQP, angle encoding, hardware-efficient circuit), with no reason why a classical kernel could not imitate it | Power of data, geometric difference (arXiv:2011.01938); our own result g ≈ 1.3–4.9 ≪ √N (RESULTS.md); entanglement did not help (arXiv:2403.07059 and our ablation) |
| K2 | It needs an exponential speed-up from quantum linear algebra (HHL-style) on classical data, without addressing dequantisation and data loading | arXiv:1807.04271, arXiv:1910.06151, doi:10.1038/nphys3272, arXiv:2305.10310 |
| K3 | It is a variational model that is either deep and random (barren plateaus), or trainable for a reason that also makes it classically simulable | arXiv:1803.11173, arXiv:2312.09121 |
| K4 | Its evidence would be a comparison with an untuned or single classical baseline | arXiv:2403.07059 and DESIGN.md §4 |
| K5 | It ignores shot cost, noise or kernel concentration at the scale where it claims to win | arXiv:2208.11060, arXiv:2210.11505 |
| K6 | It cannot be tested: not with ≤ 10 simulated qubits, and not by a precise theoretical statement | Repository rule (DESIGN.md) |
| K7 | It has already been done, and the new version adds nothing testable | Verified arXiv search (step 3) |
| K8 | Its advantage comes from giving the classical learner less access than is physically meaningful. Examples: random examples only, when the quantum example state presupposes an oracle that also answers classical queries; or a black-box classical learner when the quantum model is told the generating process. The test is against the access-matched classical learner | Added after round 1: arXiv:2610.02636 (access matching); [ideas/round-01.md](ideas/round-01.md), themes T4, T5 |

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

Round 1 (2026-10-12 to 2026-10-14): 40 ideas. Full cards in [ideas/round-01-cards.md](ideas/round-01-cards.md); summary and recommendation in [ideas/round-01.md](ideas/round-01.md). "Batch A–D" are the four external assistants, in the order their answers were pasted.

| ID | Title | Proposed by | Milestone | Status | Decision note |
|---|---|---|---|---|---|
| I-20261012-01 | Periodic-neuron learning with quantum statistical queries | batch A | M1 | REJECTED | K6/K7: at d = 4 the concept class is tiny and the theorem is already proven; nothing to measure |
| I-20261012-02 | Single-qubit sensing of weak classical signals | batch A | M8 | REJECTED (in review) | The exponential gap needs fine-tuned signal families; on natural low-dimensional ones a model-aware classical learner pays at most a polynomial price (arXiv:2504.21745) |
| I-20261012-03 | Trainable generative models on hard-to-sample distributions | batch A | M4 | MERGED into 31 | A lookup table defeats its success rule at 6 qubits |
| I-20261012-04 | Symmetry-structured bandwidth for quantum kernels | batch A | M6 | REJECTED | K1/K4: group averaging is equally available to classical kernels; misattributed reference |
| I-20261012-05 | Sample advantage in learning quantum observables | batch A | M1 | REJECTED | The cited separation is computational, not sample-based; at n = 4 a classical learner reproduces the features (K6/K7) |
| I-20261012-06 | Coherent memory for learning states with fewer copies | batch A | M8 | REJECTED | No memory advantage exists for one fixed observable; one reference is the wrong paper |
| I-20261012-07 | Trainable variational circuits that are not classically simulable | batch A | M4 | REJECTED | K6/K7: no architecture named; a 6-qubit check cannot decide simulability |
| I-20261012-08 | Quantum oracle sketching on massive classical data | batch A | M1 | SHORTLISTED | Audit arXiv:2604.07639's natural-data claim against a per-test-point, low-memory classical baseline (recommendation R3) |
| I-20261012-09 | Coherent state classification without tomography | batch B | M8 | MERGED into 34 | The SWAP test is a coarse-graining of Bell sampling; task ill-posed as stated |
| I-20261012-10 | Entangled probes beyond the classical Cramér–Rao bound | batch B | M8 | REJECTED | K7/K5: textbook metrology; its reference does not contain the claim |
| I-20261012-11 | Matchgate Born machine for hard parity distributions | batch B | M4 | REJECTED | K3: matchgate circuits with product inputs are classically simulable |
| I-20261012-12 | SU(2)-symmetric variational models for dynamics | batch B | M4 | REJECTED | K6/K4: exact simulation of a known 6-qubit chain is not a learning test |
| I-20261012-13 | Hidden-shift / discrete-log concept classes | batch B | M1 | REJECTED | K6: brute force or a lookup table solves it at 8 qubits; references do not contain the claim |
| I-20261012-14 | Group-invariant projected quantum kernels | batch B | M6 | MERGED into 37 | Same claim as 04; its fair version is 37 |
| I-20261012-15 | Direct-fidelity-estimation classifier | batch B | M8 | REJECTED | Direct fidelity estimation is itself a measure-first classical learner |
| I-20261012-16 | Variational quantum policy gradient for control | batch B | M4 | REJECTED | K1/K3/K4/K7: no mechanism for fewer interactions; single baseline |
| I-20261012-17 | Coherent-memory prediction of many observables | batch C | M8 | MERGED into 34 | Same mechanism; its strongest baselines and noise arm are kept |
| I-20261012-18 | Single-qubit sensing of sparse temporal signals | batch C | M8 | MERGED into 02 | Same paper and mechanism; misreads what the paper's Fourier coefficients are |
| I-20261012-19 | Ancilla-assisted classification of quantum channels | batch C | M8 | MERGED into 36 | Only its Pauli-channel core is sound |
| I-20261012-20 | Learning observables from labelled dynamics | batch C | M1 | REJECTED | K4/K6/K7 at triage (now also K8): any win comes from knowing U, which a classical learner given U uses too |
| I-20261012-21 | Noise-robust periodic-neuron learning | batch C | M1 | MERGED into 39 | Its noise axis is kept |
| I-20261012-22 | Symmetry-preserving classification of topological phases | batch C | M8 | MERGED into 35 | Same mechanism as 35 |
| I-20261012-23 | Born-machine generalisation beyond moment matching | batch C | M4 | REJECTED | K7: benchmarks exist; arXiv:2608.31117 finds likelihood-trained classical models ahead |
| I-20261012-24 | Optimal entanglement depth for learning dynamics | batch C | M8 | REJECTED | K7/K4: arXiv:2306.03481 already ran this sweep; no classical baseline |
| I-20261012-25 | Few-copy classification with coherent memory | batch D | M8 | REJECTED | K7: exactly solvable and studied; optimal learning error is reached without quantum memory |
| I-20261012-26 | Entangled probe: identity vs depolarizing channel | batch D | M8 | MERGED into 36 | Correct numbers (0.875 vs 0.75) but textbook; kept as a calibration test |
| I-20261012-27 | GHZ vs product probes with tuned estimators | batch D | M8 | REJECTED | K7: textbook; fixed by the quantum Fisher information |
| I-20261012-28 | Symmetry-preserving QCNN for phase classification | batch D | M8 | MERGED into 35 | Duplicate of 35 |
| I-20261012-29 | Stress-testing natural-distribution separations | batch D | M1 | MERGED into 39 | Its 'run the paper unchanged' version is uninformative at ≤ 10 qubits |
| I-20261012-30 | Projected kernels on quantum-state data under shots | batch D | M6 | MERGED into 38 | Its quantum-vs-classical axis is 38's question |
| I-20261012-31 | Learning small hard quantum output distributions | batch D | M4 | SHORTLISTED | Matched-control generative benchmark (killing experiment); apply the review's fixes |
| I-20261012-32 | Syndrome-conditioned logical-state classification | batch D | M9 | REJECTED | K7/K5: the maximum-likelihood lookup decoder is already optimal for Pauli noise |
| I-20261012-33 | Positive control with engineered labels | Claude | M2, M6 | SHORTLISTED | Calibrate the benchmark at a realistic effect size (recommendation R2) |
| I-20261012-34 | Two-copy vs single-copy learning at n ≤ 5 | Claude | M8 | SHORTLISTED | Certified finite-n table plus noise thresholds against the strongest single-copy learners (recommendation R1) |
| I-20261012-35 | QCNN vs classical-shadow learners on phases | Claude | M8 | REJECTED (in review) | Cong et al.'s QCNN output is reproducible from two product bases at ≤ 2× the copies (arXiv:2109.05909); at n = 9 the reviewer's exact check gave 0.99–1.00× for QCNN-type measurements; salvage as an exact ceiling note |
| I-20261012-36 | Entangled-ancilla Pauli-channel learning | Claude | M8 | SHORTLISTED | Recast as an ε-scaling test: Bell sampling ε⁻² vs the best known ancilla-free learners for sparse non-local Lindbladians ε⁻⁴ (arXiv:2607.23044, abstract), a point left open by arXiv:2606.20535 |
| I-20261012-37 | Is symmetry a quantum resource for kernels? | Claude | M6 | SHORTLISTED (not reviewed) | Projection lemma plus a cheap killing experiment against group-averaged classical kernels (with R2); the T6 review covered only idea 33 |
| I-20261012-38 | Geometric difference on quantum-state inputs | Claude | M8 | SHORTLISTED (low priority, not reviewed) | Cheap diagnostic at matched copies; listed among T7's best-version sources, but not in that design or its review |
| I-20261012-39 | Classical baseline for periodic-neuron separations | Claude | M1 | SHORTLISTED | Theorem plus verification code: arXiv:2503.20879's queries are emulable by classical membership queries |
| I-20261012-40 | Heisenberg-limited Hamiltonian learning | Claude | M8 | REJECTED | K7: proven, with matching lower bounds (arXiv:2210.03030, arXiv:2304.07172) |
