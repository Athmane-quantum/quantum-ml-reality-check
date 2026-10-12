# Next session plan (session 2)

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

Written at the end of session 1 (2026-10-12) so that a fresh agent can continue with no other
context. Read README.md, DESIGN.md, FIELD_MAP.md, RESULTS.md and the last JOURNAL.md entry first.
Keep every rule in DESIGN.md: verified sources only, no invented numbers, tuned baselines, CPU-only,
the AI-built disclaimer, and negative results reported.

## 0. State of the repository

- Work so far is on the branch `claude/new-session-rwyqim`. If Athmane has merged it into `main`, work
  from `main`; otherwise ask, or continue on the branch you are assigned.
- Benchmark v0 is done: credit-card fraud, 8 qubits, 5 seeds. No quantum model beats any classical
  model; all gaps are within noise (RESULTS.md).
- Setup and timings, on a 4-core container:

  ```bash
  python3.13 -m venv .venv && source .venv/bin/activate
  pip install -r requirements-lock.txt && pip install -e . --no-deps
  python -m pytest -q                 # ~3 min
  python scripts/check_sources.py     # ~10 s; must print "0 to look at"
  python -m qmlrc.run_all             # ~8 min with 4 cores; downloads ~70 MB the first time
  ```

- `run_all` rebuilds RESULTS.md and flags `experiments/READING.md` as **STALE** whenever the numbers
  change. After any re-run, rewrite READING.md for the new numbers and update the `results-hash` on
  its first line to the fingerprint printed by `run_all`.

## 1. Literature watch (about 30 minutes)

Look for arXiv papers from the last two weeks that could change a milestone status in FIELD_MAP.md:

- new advantage claims, especially on classical data (milestones 1, 5);
- dequantisations and classical surrogates (milestones 3, 4, 6);
- QML benchmarks against tuned baselines (milestone 5);
- trainability and barren plateaus (milestone 4);
- quantum-data or sensing advantages (milestone 8);
- hardware and error-correction results (milestone 9).

Suggested queries: "quantum machine learning advantage", "dequantization", "quantum kernel",
"barren plateau simulability", "learning from quantum data", "benchmark quantum classical".

For every paper you use: open it (arXiv API plus the PDF pages you rely on), tag it `[F]` or `[A]`,
add it to the Sources table, and run `python scripts/check_sources.py --update`. Change a status or a
percentage only with a written one-line reason and the source. Two papers published after session 1
are worth following up for replications or critiques: arXiv:2509.09033 (generative advantage) and
arXiv:2608.13521 (single-qubit sensing advantage).

## 2. Top item: entanglement ablation (reproduce Bowles et al.'s key test at small scale)

**Why.** Session 1's reading is that tuned quantum kernels pick small bandwidths, where their circuits
are nearly unentangled. Bowles et al. (arXiv:2403.07059, `[F]`) found that removing entanglement
usually does not hurt. If the same holds here, the quantum models' performance owes nothing to
"quantumness". This is the cheapest experiment that tests that reading.

**What to build.**

1. In `src/qmlrc/quantum.py`, add an `entangling: bool = True` argument to `iqp_states` and
   `bloch_vectors` (and to the cached circuit builders). When it is false, pass `pattern=[]` to
   `qml.IQPEmbedding`. Checked in session 1: this gives H–RZ–H–RZ on each qubit with no MultiRZ gates.
2. In `src/qmlrc/vqc.py`, add `entangling: bool = True`. When false, use `pattern=[]` in the
   embedding and replace `StronglyEntanglingLayers` by layers of `qml.Rot` with no CNOTs (same
   parameter count). Readout and training stay identical.
3. In `src/qmlrc/models.py`, add three models with exactly the same grids as their entangled
   counterparts: `QSVM-fidelity-sep`, `QSVM-projected-sep`, `VQC-sep`. Mark their family as `quantum`
   in the description, or add a new family `quantum-separable` and check that `report.py` still
   groups the tables correctly.
4. Tests:
   - for separable states, the fidelity kernel equals the product of single-qubit fidelities (check
     against an independent NumPy computation);
   - the separable states are product states (every single-qubit Bloch vector has length 1);
   - the VQC-sep circuit contains no multi-qubit gates (inspect `qml.specs` or the tape);
   - update `EXPECTED_CONFIGS` in `tests/test_models.py`.
5. Add the models to `quantum_models()` (or a new experiment in `experiments.py`), re-run `run_all`,
   check that the runtime stays near 10 minutes (the VQC-sep adds about 2–3 minutes; reduce its grid
   only if needed and say so), and update READING.md. The key question: is each entangled model better
   than its separable twin under the DESIGN.md §6 rule?
6. Add a sentence to FIELD_MAP.md milestone 5 with the outcome.

**Definition of done.** Tests pass, CI is green, RESULTS.md is regenerated with the new rows,
READING.md is updated and not stale, a JOURNAL.md entry is written, and this file is rewritten.

## 3. Then, if time remains: priorities 2–5

2. **Geometric difference (Huang et al., arXiv:2011.01938 `[F]`, eq. 5).**
   g(K_C ‖ K_Q) = sqrt(‖ sqrt(K_Q) K_C⁻¹ sqrt(K_Q) ‖_∞), with both kernels normalised to trace N and
   a small ridge (λ = 1e-6…1e-3) added before inverting K_C. Compute it on each seed's training set
   between the RBF kernel and each quantum kernel at the CV-selected settings. A small g (close to 1)
   means the classical kernel is guaranteed to be competitive. Add `src/qmlrc/geometry.py`, tests
   (g(K‖K) = 1, and a small case computed by hand), and a table in RESULTS.md.
3. **Finite-shot kernels.**
   - Fidelity kernel: replace each entry k by Binomial(R, k)/R, with diagonal 1 and symmetric.
   - Projected kernel: estimate each Bloch component from R single-qubit shots.
   - Run R ∈ {100, 1 000, 10 000} and report AUPRC against R.
   This connects the benchmark to milestones 6 and 9 (Thanasilp et al.: shot cost).
4. **A quantum-native dataset (milestone 8).** For example: ground states of a transverse-field Ising
   chain with ≤ 10 spins (exact diagonalisation with SciPy), labelled by phase. Compare quantum
   kernels on the states with classical models on classical-shadow features (Huang et al.,
   arXiv:2106.12627, `[A]`). Write the design into DESIGN.md first. This needs care to be a fair
   comparison.
5. **Statistics and realism.**
   - 10 seeds, with a paired bootstrap confidence interval or a Wilcoxon signed-rank test added to
     `report.py`.
   - A natural-prevalence (0.17 %) variant with a larger pool of legitimate transactions.
   - For GradBoost, which still hits grid edges: an equal-budget randomised search, applied the same
     way to the quantum kernels.

## 4. Housekeeping

- Upgrade PennyLane and JAX together only in a dedicated session: the current pair prints JAX
  deprecation warnings. Re-run everything after any upgrade and record the version change.
- Keep `run_all` under about 10 minutes per experiment on 4 cores. If an addition breaks this, split it
  into its own experiment entry in `experiments.py` and say so in the journal.
