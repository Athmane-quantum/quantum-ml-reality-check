# Next session plan (session 3)

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

Written at the end of session 2 (2026-10-12) so that a fresh agent can continue with no other
context. Read README.md, DESIGN.md, FIELD_MAP.md, RESULTS.md and the last two JOURNAL.md entries
first. Keep every rule in DESIGN.md: verified sources only, no invented numbers, tuned baselines,
CPU-only, the AI-built disclaimer, and negative results reported.

## 0. State of the repository

- Work is on the branch `claude/new-session-rwyqim`; `main` holds only the initial commit. If
  Athmane has merged the branch, work from `main`; otherwise continue on the branch you are assigned
  and say so in the journal.
- Results so far (RESULTS.md, fingerprint `d37c043cfce2f0fb`):
  - no quantum model beats any classical model;
  - removing entanglement from the quantum kernels changes nothing;
  - the geometric difference between the tuned quantum kernels and classical kernels is g = 1.3–4.9
    against √N = 37.4;
  - the unentangled VQC control is unstable, so the VQC part of the ablation is inconclusive.
- Setup and timings, on a 4-core container:

  ```bash
  python3.13 -m venv .venv && source .venv/bin/activate
  pip install -r requirements-lock.txt && pip install -e . --no-deps
  python -m pytest -q                 # 38 tests, ~3-4 min
  python scripts/check_sources.py     # must print "0 to look at"
  python -m qmlrc.run_all             # ~13 min (5 experiments, each < 6 min); downloads ~70 MB once
  ```

- After any re-run, `experiments/READING.md` is flagged **STALE** if the numbers changed. Rewrite it
  for the new numbers and put the new fingerprint on its first line.
- Before declaring a result, check that unchanged experiments stayed bit-identical (compare the CSVs,
  ignoring the `seconds` column).

## 1. Literature watch (about 30 minutes)

Same procedure as session 2 (JOURNAL.md): arXiv API queries limited to submissions since 2026-10-12,
on QML advantage, quantum kernels, dequantisation, barren plateaus, QNN vs classical and learning
advantage.

- Open every paper you use and tag it `[F]` or `[A]`.
- Add it to the Sources table and run `scripts/check_sources.py --update`.
- Change a status or percentage only with a one-line reason and its source.
- Follow up on any replication or critique of arXiv:2509.09033 (generative advantage) and
  arXiv:2608.13521 (single-qubit sensing advantage).

## 2. Step 0: owed grid extensions (do these with the next code change; no separate run)

Under the clarified rule (DESIGN.md §4: each end of each axis is extended at most once), these are
owed one extension:

- `SVM_C` in `src/qmlrc/models.py`: add `0.001` and `10000.0`. This affects the RBF-SVM and all four
  QSVMs, which share the axis.
- `VQC_GRID["n_layers"]`: add `1`. This affects both `VQC` and `VQC-sep`.

Update `EXPECTED_CONFIGS` in `tests/test_models.py`:

| Model | Configurations |
|---|---|
| RBF-SVM | 80 |
| QSVM-fidelity(-sep) | 96 |
| QSVM-projected(-sep) | 256 |
| VQC(-sep) | 16 |

Check that each experiment stays under ~10 minutes. Afterwards, report every remaining edge hit as a
flat surface and do not extend again.

## 3. Top item: finite-shot quantum kernels (first step towards hardware realism)

**Why.** Every quantum number so far assumes infinitely many measurements. On hardware each kernel
entry is estimated from R shots, and Thanasilp et al. (arXiv:2208.11060, `[F]`) show that this is
where concentrated kernels fail (milestones 6 and 9). Germain et al. (arXiv:2610.04190, `[A]`)
report on the same fraud data that their kernels did not concentrate severely, so the size of the
effect here is an open question.

**What to build.**

1. `src/qmlrc/shots.py` with two estimators, each taking a NumPy `Generator` so results are seeded:
   - `sample_fidelity_kernel(K, R, rng)`. Each off-diagonal entry is the frequency of the all-zeros
     outcome in R shots of the overlap (Loschmidt-echo) circuit, i.e. Binomial(R, k)/R. Keep the
     matrix symmetric with a diagonal of 1. Use the same estimator for test-versus-train entries.
   - `sample_bloch_vectors(B, R, rng)`. Each Pauli expectation becomes 2·Binomial(R, (1 + ⟨P⟩)/2)/R − 1,
     and the projected kernel is then built from the noisy Bloch vectors.
   - Noisy Gram matrices can be indefinite. Choose one documented fix and test it, for example
     clipping negative eigenvalues of the training Gram matrix to 0.
2. A new experiment `creditcard_q8_shots`: QSVM-fidelity and QSVM-projected for R ∈ {100, 1 000,
   10 000}, re-tuned on the noisy kernels (on hardware one would tune on noisy kernels too), with the
   same rows, splits and folds. Seed the shot noise from (seed, R).
3. Tests:
   - the estimators are unbiased (the mean over many draws is close to the exact value);
   - their variance is close to k(1 − k)/R;
   - the matrices are symmetric with a unit diagonal;
   - results are deterministic given a seed.
4. A RESULTS.md table of AUPRC against R, with the exact-kernel row for reference. Update READING.md,
   FIELD_MAP.md milestones 6 and 9, and the journal.

**Definition of done.** Tests pass, CI is green, the run is reproducible, READING.md is not stale, and
JOURNAL.md and this file are updated.

## 4. Then, if time remains

3. **A clean VQC entanglement control.** Add an entangled VQC with the *same* mean-⟨Zᵢ⟩ readout as
   `VQC-sep`, so that the pair differs only in entanglement. Fix the separable control's flat-output
   problem the same way for both, for example by initialising the output scale `a` at 6 (as in
   Bowles et al.'s σ(6⟨O⟩)), or by standardising the readout before the affine map. Re-run the VQC
   ablation and say clearly in the journal that the control was changed and why.
4. **A quantum-native dataset (milestone 8),** where an advantage is most plausible. For example,
   ground states of a transverse-field Ising chain with ≤ 10 spins (exact diagonalisation with
   SciPy), labelled by phase. Compare quantum kernels on the states with classical models on
   classical-shadow features (Huang et al., arXiv:2106.12627, `[A]`). Write the design into
   DESIGN.md before coding.
5. **Statistics:**
   - 10 seeds;
   - a paired bootstrap confidence interval or a Wilcoxon signed-rank test in `report.py`;
   - a natural-prevalence (0.17 %) variant;
   - a bandwidth sweep of the geometric difference (g against s at fixed seed) inside `run_all`.

## 5. Housekeeping

- Upgrade PennyLane and JAX together only in a dedicated session: the current pair prints JAX
  deprecation warnings. Re-run everything after an upgrade.
- Keep each experiment under about 10 minutes on 4 cores; split experiments rather than exceed it.
