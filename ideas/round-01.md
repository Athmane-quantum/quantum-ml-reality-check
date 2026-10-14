# Idea round 1 (2026-10-12 to 2026-10-14)

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

This is the first run of the pipeline in [IDEAS.md](../IDEAS.md). It records what came in, what
was checked, what was killed and why, and what is proposed for testing. Athmane decides what gets
built.

- **This file:** the summary, the shortlist and the recommendation.
- **[round-01-cards.md](round-01-cards.md):** the full idea cards. For each card it gives the
  reference checks, the technical problems, the kill criteria, the verified prior work, the
  decision, and every search that was run. For each theme it adds the sharpened "best version"
  with its adversarial review.
- **[round-01-raw.json](round-01-raw.json):** the raw structured output of every agent.
- **[round-01-checks/](round-01-checks/):** the review agents' own scratch scripts, which back the
  numbers quoted below. They are unreviewed, indicative checks, not repository results.

## Inputs

**40 ideas**, all in the card format of IDEAS.md (`I-20261012-01` … `-40`):

| Source | IDs | Count |
|---|---|---|
| External assistants, batch A (Athmane ran the generator prompt in four assistants; batches are labelled in the order pasted, without attribution) | 01–08 | 8 |
| Batch B | 09–16 | 8 |
| Batch C | 17–24 | 8 |
| Batch D | 25–32 | 8 |
| Claude, written after reading the other 32, aimed at gaps and at fair versions of their mechanisms | 33–40 | 8 |

The ChatGPT deduplication step (`PROMPT_NOVELTY_FILTER.md`) was not run. Claude grouped the 40 ideas
into nine themes and merged duplicates during triage.

## Process

1. **Reference resolution (Claude).** All 23 arXiv IDs cited by the external batches were resolved
   through the arXiv API in one query. All exist. Two are misused:
   - arXiv:2407.07754 is cited as "fundamental limitations of quantum agents"; it is *Random
     unitaries in extremely low depth* (Schuster, Haferkamp & Huang).
   - arXiv:2111.05451 is attributed to "Canatar et al."; it is by Shaydulin & Wild. The Canatar
     et al. bandwidth paper is arXiv:2206.06686.
2. **Triage, one agent per theme.** Each agent:
   - checked that every cited paper says what the idea claims, opening the papers central to the
     mechanism;
   - searched alphaXiv and the arXiv API with several phrasings of each mechanism;
   - checked the physics and mathematics;
   - applied K1–K7;
   - wrote the theme's sharpened best version.
3. **Adversarial review, one skeptic per best version.** Each skeptic was told to refute the
   proposal on four counts:
   - already done;
   - a classical method that would match it;
   - an unfair access model;
   - not feasible or not decisive.

   Most skeptics ran exact computations to check the proposal's own numbers.
4. **Citation audit (Claude).** All 251 distinct arXiv IDs that appear in the agents' outputs were
   resolved again through the arXiv API. All exist, and the titles the agents recorded match.

Total: 17 agents (9 triage, 8 review; one theme had nothing to review), 704 tool calls.

## Outcome

| Status | Count | Ideas |
|---|---|---|
| **SHORTLISTED**, weakened in review (each needs its listed fixes before it is built) | 7 | 08, 31, 33, 34, 36, 37, 39 |
| **SHORTLISTED**, low priority, not part of a reviewed best version | 1 | 38 |
| **REJECTED** at triage | 18 | 01, 04, 05, 06, 07, 10, 11, 12, 13, 15, 16, 20, 23, 24, 25, 27, 32, 40 |
| **REJECTED** in adversarial review (shortlisted at triage, then killed) | 2 | 02, 35 |
| **MERGED** into another idea | 12 | 03, 09, 14, 17, 18, 19, 21, 22, 26, 28, 29, 30 |

The rejected ideas fell into these groups (several fit more than one; details per card):

- **Technically wrong (a sample).**
  - I-11: matchgate circuits with product inputs are efficiently classically simulable, so they
    give no hardness.
  - I-06: for a single fixed observable, quantum memory gives no advantage.
  - I-15: direct fidelity estimation is itself a measure-first classical learner.
  - I-16: parameter-shift gradients cannot reduce environment interactions.
- **Solvable by a lookup table or brute force at ≤ 10 qubits:** 01, 03, 05, 13.
- **Already known, textbook, or adding nothing testable (K7):** 10, 23, 24, 25, 27, 32, 40.
- **Not testable as stated (K6):** 07 (no architecture named; a 6-qubit check cannot decide
  simulability) and 12 (exact simulation of a known 6-qubit chain is not a learning task).
- **A false mechanism (K1/K4):** 04. Symmetry in the embedding is something a group-averaged
  classical kernel can copy (see I-37).

### Reference quality by batch

Verdicts on the references each idea cited:

| Batch | OK | Claim not in the paper | Wrong paper | Misattributed | No reference given |
|---|---|---|---|---|---|
| A | 17 | 3 | 1 | 1 | 0 |
| B | 5 | 7 | 0 | 0 | 1 |
| C | 11 | 2 | 0 | 0 | 0 |
| D | 5 | 3 | 0 | 0 | 2 |
| Claude | 12 | 1 | 0 | 0 | 0 |

No assistant invented an arXiv ID. The common failure is a **real paper cited for a claim it does
not make**. For example, arXiv:2608.13521 (Kannan et al., a single qubit without entanglement) was
cited three times for GHZ-probe sensing, and arXiv:2503.20879 was cited for hidden-shift concept
classes it does not contain. Every one of the 40 minimal tests, Claude's included, had at least one
technical problem as first written.

## What the round taught

These are cross-cutting lessons. Each is backed by the cards and their verified sources.

1. **Many claimed advantages come from the access model.** The usual pattern is that the classical
   side gets less access than is physically meaningful:
   - **Periodic neurons (T4).** The quantum example states of arXiv:2503.20879 presuppose an oracle
     that can also answer classical membership queries. The reviewer checked that every
     statistical query that paper uses can be emulated that way (identities to about 1e-12).
   - **Learning observables (I-20).** The quantum learner is told the evolution U and the classical
     baselines are not.
   - **Oracle sketching (T9).** In arXiv:2604.07639's natural-data numerics, a per-test-point
     quantum procedure is compared with model-building classical baselines.

   This suggests a new kill criterion, **K8**, added to IDEAS.md.
2. **Where separations are real, they are proven theorems already shown on hardware.** Two cases:
   - quantum memory for learning states (T1; arXiv:2112.00778);
   - ancilla-assisted Pauli-channel learning (T8; arXiv:2108.08488, arXiv:2408.03376).

   What is still open at ≤ 10 qubits are the constants and the noise thresholds. In one reviewer's
   exact computation for the arXiv:2112.00778 task, the two-copy advantage over the best
   single-copy learner disappears at local depolarizing noise of about λ ≈ 0.29–0.33 for n = 2–5.
   This number is indicative and would be the subject of experiment R1.
3. **Exponential sensing gaps need fine-tuned signals (T2).** For natural, low-dimensional signal
   families, the gap is bounded by parametric identifiability: a model-aware classical learner gets
   within a constant factor (arXiv:2504.21745).
4. **Symmetry is not a quantum resource for kernels (T6, I-37).** If a kernel is covariant,
   averaging it over the group gives the same gain to the classical and the quantum kernel. A
   one-page projection lemma, still to be written out and checked, formalises this. In a scratch
   check on an S₆-invariant task, the group-averaged RBF kernel scored 0.86 against 0.80 for the
   group-averaged quantum kernel.
5. **Phase recognition at ≤ 10 qubits shows no real copy advantage (T7).** A single-copy entangling
   measurement (QCNN-type) gains at most a constant factor of about 2 over the best product-basis
   learner (arXiv:2109.05909). One reviewer's exact n = 9 computation gave 0.99–1.00×.
6. **A correction to our own text.** We had written that with g ≈ 1.3–4.9 a classical kernel is
   "guaranteed to predict about as well for any labels". That overstates Huang et al.'s bound. The
   bound gives s_C ≤ g² s_Q: the classical error bound is at most g times the quantum one. That
   rules out a *large* advantage for any labels, not a modest one. The wording is fixed in
   READING.md, README.md, FIELD_MAP.md (M2), geometry.py, report.py and ideas/CONTEXT.md.

## The shortlist

Every entry below survived triage. Every one was also **weakened** in adversarial review: the
review lists concrete fixes, all of which are in round-01-cards.md. Idea 38 is also shortlisted, at
low priority. It is a cheap diagnostic that applies `qmlrc.geometry` to quantum-state inputs at
matched copy budgets, and it was not part of a reviewed best version.

| Theme | Best version (ideas) | Kind | Milestone | Rough cost |
|---|---|---|---|---|
| T1 | Two-copy Bell sampling vs the best single-copy learners on the arXiv:2112.00778 task, n = 1–5: certified finite-n table plus noise thresholds (34, with 09 and 17) | replication + new noise thresholds | M8 (M9) | ~350 lines + tests; analytic plus density-matrix simulation; minutes |
| T6 | Positive control: labels engineered from the quantum kernel's geometry, on the fraud inputs (33). Plus the symmetry test against group-averaged classical kernels (37, with 14) | calibration + killing experiment | M2, M6 | ~200 lines; reuses the benchmark; must be split to stay under 10 min per experiment |
| T9 | Audit of the oracle-sketching size claim (arXiv:2604.07639) with a per-test-point, low-memory classical baseline that its authors did not run (08) | killing experiment on a new claim | M1 | ~300 lines + tests; public text datasets |
| T8 | Ancilla-assisted vs ancilla-free Pauli-channel learning, recast as an ε-scaling test (Bell sampling ε⁻² vs the best ancilla-free sparse learners ε⁻⁴), a point left open by arXiv:2606.20535 (36, with 19 and 26) | replication + open scaling question | M8 | ~800 lines |
| T4 | Theorem plus verification code: every statistical query used by arXiv:2503.20879 can be emulated by classical membership queries (39, with 21 and 29) | theory + check | M1 | ~300 lines |
| T3 | Matched-control generative benchmark at 6–10 qubits: does any structure-agnostic Born machine beat tuned lookup tables, chains and complex MPS? (31, with 03) | killing experiment | M4 | ~550 lines, about 12 experiments |

Also salvaged, as cheap notes rather than experiments:

- **T7 "ceiling" computation (~200 lines).** An exact statement that, at ≤ 10 qubits, single-copy
  entangling measurements give at most about a factor 2 over product-basis learners on these phase
  tasks.
- **T2 sensing replication.** Fine-tuned family only. Also a short proposition: natural families
  have a polynomial gap.

## Recommendation for Athmane (pick 1–3)

**R1: quantum memory on quantum data (T1).**
- *Why first:* it is the only shortlisted test where a quantum advantage is *expected to appear*
  under fair rules. It is the positive side of the project, after three sessions of negative
  results on classical data.
- *What is new:* the exact finite-n crossover against the *strongest* classical learners. These are
  derandomised mutually-unbiased-basis cycling (about 1.5× stronger than the baseline first
  proposed) and an exactly optimal adaptive learner at n = 1, 2. The other new part is the noise
  thresholds (memory, gate, readout) that a hardware demonstration would have to beat.
- *A win teaches:* how small a system and how little noise are enough for a real learning
  advantage.
- *A loss teaches:* constants and noise erase the advantage at near-term scale.

**R2: calibrate our own benchmark (T6).**
- *Why:* our negative result is only as strong as the benchmark's power to detect an advantage that
  does exist.
- *What it does:*
  - builds labels where the quantum kernel must win, at a realistic effect size (about +0.02 AUPRC
    with the projected kernel at s = 1, where g ≈ 13);
  - checks that random labels produce no false win;
  - adds the cheap symmetry test (I-37).
- *Power:* by the reviewer's calculation, the minimum detectable effect is about 0.017 AUPRC for
  the kernels.
- *Cost:* the cheapest of the three; it reuses everything.

**R3: test a fresh high-profile claim (T9).**
- *The claim:* Zhao, Zlokapa, Neven, Babbush et al. (arXiv:2604.07639, 2026) report a 4–6
  orders-of-magnitude memory advantage on natural classical data.
- *The gap:* a stronger classical baseline (a per-test-point streaming polynomial estimator, a
  randomised version of the method in their Appendix G) has not been run. The reviewer found that
  simple per-test-point baselines (centroid, Rocchio, kNN) do not close the gap. As first designed,
  though, the verdict depends on accounting conventions, so the fixes (a memory-sample frontier, QOS
  simulated at matched budgets) are required.
- *A loss for the claim* would move FIELD_MAP M1. So would a win.

Claude's suggestion: **R1 then R2**. R1 is the most likely to give the project its first positive,
well-founded result. R2 is cheap and makes every past and future negative result more credible. R3
is the most valuable if Athmane wants to engage with current literature, and also the most
expensive.

## For the field map (next literature watch)

The round surfaced recent papers that bear on milestones but are not yet in FIELD_MAP.md. The
agents read them at the levels recorded in round-01-cards.md. They will be opened and added in the
next literature watch:

| Paper | Milestone | Why |
|---|---|---|
| arXiv:2604.07639 | M1, M7 | Quantum oracle sketching on massive classical data, without QRAM |
| arXiv:2512.10929 | M8, M9 | Noisy quantum learning theory: separations under local depolarizing noise |
| arXiv:2605.21346 | M8 | Evidence of a QML advantage with tens of noisy qubits |
| arXiv:2504.21745 | M8 | Exponential sensing gap tied to fine-tuned signal families |
| arXiv:2610.02636 and arXiv:2610.11360 | M1, M2 | Access matching; geometry-separation controls |
| arXiv:2608.31117 | M4 | Likelihood-trained classical generators beat moment-trained quantum ones |
| arXiv:2606.20535 | M8 | Ancilla-free learning of local Lindbladians is near-optimal |
