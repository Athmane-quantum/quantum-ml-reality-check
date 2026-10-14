# Next session plan (session 5: build the idea Athmane chooses)

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

Written at the end of session 4 (2026-10-14), after the first idea round. A fresh
agent can continue from this file. Read README.md, DESIGN.md, FIELD_MAP.md, RESULTS.md, IDEAS.md and
the last JOURNAL.md entry first. Keep every rule in DESIGN.md: verified sources only, no invented
numbers, tuned baselines, CPU-only, the AI-built disclaimer, and negative results reported.

## 0. State of the repository

- Work is on the branch `claude/new-session-rwyqim`; `main` holds only the initial commit unless
  Athmane merged it. Work where you are assigned and say so in the journal.
- **Initialisation is complete.** It consists of:
  - the field map (9 milestones, 52 verified arXiv sources);
  - the benchmark on credit-card fraud at 8 qubits, with tuned classical baselines, an entanglement
    ablation, finite-shot kernels, geometric difference and kernel concentration;
  - the idea pipeline (IDEAS.md, `ideas/`, `scripts/arxiv_search.py`), run once: round 1,
    40 ideas, in `ideas/round-01.md`.

  RESULTS.md and READING.md give the current numbers and their reading.
- Setup and timings, on a 4-core container:

  ```bash
  python3.13 -m venv .venv && source .venv/bin/activate
  pip install -r requirements-lock.txt && pip install -e . --no-deps
  python -m pytest -q                 # 52 tests, ~8 min
  python scripts/check_sources.py     # must print "0 to look at"
  python -m qmlrc.run_all             # all experiments (see experiments/run_info.json for timings)
  ```

- Grid extensions are finished (DESIGN.md §4). Report remaining edge hits; do not extend again.

## 1. First: Athmane's choice

Round 1 recommends three experiments ([ideas/round-01.md](ideas/round-01.md), "Recommendation").
Build only what Athmane picks. Each one's required fixes from the adversarial review are in
[ideas/round-01-cards.md](ideas/round-01-cards.md), under its theme's "best version". Apply them
in the design before writing code.

- **R1, T1 (ideas 34, 09, 17): two-copy Bell sampling vs single-copy learners, n = 1–5.**
  - New module `src/qmlrc/memory.py`, with tests.
  - Compute N* analytically: exact binomial sums, checked by Monte Carlo and by PennyLane
    `default.mixed`.
  - Classical side: derandomised MUB stabilizer cycling, random global Clifford, an adaptive
    learner, and the rigorous lower bound (eq. D38 of arXiv:2112.00778, plus its noisy extension).
  - Compute the exact adaptive optimum at n = 1, 2 by dynamic programming.
  - Pre-register λ*(n) ≈ 0.29–0.33. Use noise grids fine enough to resolve it and N up to 2^15.
  - The memory, gate and readout noise thresholds are the experimental core.
- **R2, T6 (ideas 33, 37): calibrate the benchmark.**
  - Re-pilot on the real `creditcard_q8` inputs with out-of-sample engineered labels. The strongest
    dose is the projected kernel at s = 1.
  - Use one metric (AUPRC) throughout.
  - Calibrate the false-positive rate of the §6 rule on random labels (about 8.9 % per pair, by the
    reviewer's computation).
  - Use reverse labels that can actually make the tuned quantum SVM lose.
  - Add an oracle arm.
  - Split the runs to keep every experiment under 10 minutes.
  - Then run the I-37 symmetry experiment against group-averaged classical kernels, and write out
    its projection lemma.
- **R3, T9 (idea 08): audit arXiv:2604.07639.**
  - Classical side: per-test-point Chebyshev-polynomial streaming estimator, and the memory-sample
    frontier.
  - QOS simulated at matched sample budgets, with peak-memory accounting on both sides.
  - Non-inferiority margin; include IMDb.

Follow "How a shortlisted idea becomes an experiment" in IDEAS.md: DESIGN.md first, then code and
tests, then `qmlrc.experiments.FULL`, then the reports.

## 2. Literature watch (about 30 minutes)

Same procedure as in JOURNAL.md, session 2: arXiv API queries for submissions since 2026-10-12.
Also open and file the round-1 leads listed at the end of `ideas/round-01.md`, for example
arXiv:2604.07639, arXiv:2512.10929 and arXiv:2504.21745. Open every paper you use, tag it `[F]` or
`[A]`, and run `scripts/check_sources.py --update`. Change a status or percentage only with a
one-line reason and its source.

## 3. Next idea round (when Athmane brings new ideas)

`ideas/CONTEXT.md` now lists what round 1 killed (item 10), and IDEAS.md has kill criterion K8.
Reuse the round-1 workflow: one triage agent per theme, an adversarial reviewer per best version,
then a citation audit of every arXiv ID in the outputs.

## 4. Infrastructure (if time remains)

The field map's green flags point to **quantum data** (milestone 8). A reusable quantum-native
dataset would make those ideas testable. For example, ground states of a transverse-field Ising
chain with ≤ 10 spins, from exact diagonalisation with SciPy, labelled by phase, with classical
baselines on classical-shadow features (Huang et al., arXiv:2106.12627, `[A]`). Design it in
DESIGN.md before coding.

## 5. Statistics and housekeeping

- Use 10 seeds and add a paired bootstrap confidence interval or a Wilcoxon signed-rank test in
  `report.py`.
- Add a natural-prevalence (0.17 %) variant.
- Add a bandwidth sweep of the geometric difference to `run_all`.
- Upgrade PennyLane and JAX together only in a dedicated session (the current pair prints
  deprecation warnings).
