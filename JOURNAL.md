# Journal

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

One entry per session, newest last. Each entry says what was built, what was verified, what failed,
and what should happen next.

---

## 2026-10-12: Session 1, initialisation

**Agent:** Claude (Claude Code on the web). **Branch:** `claude/new-session-rwyqim`. The brief asked
for pushes to `main`, but this session was assigned a feature branch; merging to `main` is left to
Athmane.

### Built

- Package `qmlrc` (`src/qmlrc/`):
  - data download and checks (OpenML 1597 credit-card fraud);
  - subsampling, split and preprocessing shared by every model;
  - PennyLane IQP feature map with fidelity and projected quantum kernels;
  - a variational classifier trained with JAX;
  - four grid-tuned classical baselines;
  - a single-core parallel task runner;
  - `python -m qmlrc.run_all`, which regenerates `RESULTS.md` from scratch.
- Three experiments (`src/qmlrc/experiments.py`):
  - `creditcard_q8`, the main comparison;
  - `creditcard_full_classical`, which shows what the 8-feature limit costs;
  - `kernel_concentration`.
- 26 tests (`tests/`), including an independent NumPy re-implementation of the IQP circuit that
  cross-checks PennyLane's states and Bloch vectors. A GitHub Actions workflow runs `pytest` on every
  push.
- `DESIGN.md`: framework choice (PennyLane + JAX), data, metrics, tuning budgets, and a decision rule
  for "advantage", written while the first run was in progress.
- `FIELD_MAP.md`: nine milestones, five open problems, overall estimate about 15 %.
- `scripts/check_sources.py` with a title snapshot of the 44 cited arXiv IDs.
- `README.md`, `LICENSE` (MIT), `.gitignore`, `pyproject.toml` (exact pins), `requirements-lock.txt`.

### Verified

- **Literature.**
  - All 44 arXiv IDs cited in FIELD_MAP.md resolved through the arXiv API to the expected titles and
    authors.
  - Full text read for 11 sources: Liu et al.; Huang et al. 2021; Tang 2018; McClean et al.; Cerezo
    et al. 2023/25; Bowles et al.; Thanasilp et al.; Shaydulin & Wild; Kübler et al.; Huang et al.
    2025 (generative); Aaronson 2015 (author PDF plus DOI record).
  - The others were read at abstract level and are tagged `[A]`.
  - Corrections to the brief are listed at the top of FIELD_MAP.md: Cerezo et al. is now published
    (2025) and is a perspective, not a theorem; Tang's PCA paper was retitled; Bowles et al. say
    "systematically", not "often"; Liu et al. need fault tolerance.
- **Code.** `pytest`: 26 passed locally; CI green on GitHub for the pushed commits (see the final
  check below).
- **Reproducibility.** The concentration experiment gave identical numbers in runs 1 and 2. A third,
  final run from the committed code reproduced run 2's fingerprint exactly (see the final check
  below).

### Results

**Run 1** (commit `24f7a80`, first grids, archived in `experiments/archive/2026-10-12_run1/`). Test
AUPRC: GradBoost 0.923, MLP 0.921, RBF-SVM 0.918, LogReg 0.908, QSVM-projected 0.913,
QSVM-fidelity 0.911, VQC 0.895.

Several models on *both* sides picked hyperparameters on the edge of their grids:

- GradBoost and MLP chose the strongest regularisation offered on 5 of 5 seeds;
- the quantum kernels and the VQC mostly chose the smallest bandwidth (3–4 of 5 seeds).

**Run 2** (commit `11ea056`, after extending every axis that hit an edge in ≥ 2 of 5 seeds, for both
families). Test AUPRC: RBF-SVM 0.918, MLP 0.918, GradBoost 0.917, LogReg 0.908, QSVM-projected
0.913, QSVM-fidelity 0.911, VQC 0.899.

- The extension moved every model's mean CV AUPRC by at most +0.002, so the remaining edge hits
  (mostly GradBoost) look like a flat CV surface rather than missing range. They are listed
  automatically in RESULTS.md.
- **Conclusion: no quantum model beats any classical model.** Every pair is "no clear difference" under
  the rule; the leaning is classical.
- Kernel concentration: at s = 1 the mean fidelity-kernel value falls from 0.75 (2 qubits) to 0.11
  (10 qubits). Cross-validation never chose s = 1.

### What failed or needed a workaround

- `HistGradientBoostingClassifier` with `class_weight="balanced"` was about 10× slower per fit
  (2.3 s vs 0.2 s) on this data. That axis was dropped for GradBoost and an L2 axis added; this is
  documented in DESIGN.md.
- The first full run took 13.7 min, over the ~10-minute budget. Fixed by running each (seed, model)
  task in a single-threaded worker. XLA kept extra threads despite the flags, so workers are now also
  pinned to one core each (Linux only). The run now takes 7.6 min on 4 cores, and the per-task times
  are single-core.
- Grid edges (see Results): handled by one symmetric extension round. The rule is now part of
  DESIGN.md §4.
- PennyLane 0.45.1 with JAX 0.7.2 prints two JAX `DeprecationWarning`s. They are harmless now but a
  sign the pins should be upgraded together.
- Crossref searches did not return journal versions for Thanasilp et al. and Bowles et al.; both are
  cited by arXiv ID, and Thanasilp's venue is marked NOT VERIFIED.
- Kaggle's licence statement for the dataset was not checked; only OpenML's ("Public") was.

### Next (priority order; details in NEXT_SESSION.md)

1. Entanglement ablation: separable versions of the three quantum models, to test the reading that
   the tuned circuits are nearly unentangled (Bowles et al.'s key test).
2. Geometric difference g(K_classical ‖ K_quantum) (Huang et al. 2021) per seed: is an advantage even
   possible on this data?
3. Finite-shot kernels (100 / 1,000 / 10,000 shots) to add the first piece of hardware realism.
4. A quantum-native dataset (≤ 10-qubit spin-chain phases) to probe milestone 8, where advantage is
   most plausible.
5. Statistics: 10 seeds, paired bootstrap or Wilcoxon tests, a natural-prevalence (0.17 %) variant,
   and randomised search for GradBoost at an equal budget.

### Final check (end of session 1)

- Final full run (`python -m qmlrc.run_all`, 7.6 min, 4 workers) from the clean commit `9aa266c`: fingerprint `6f91668eaad26227`, **identical to run 2**. Every metric column of both result CSVs is byte-identical; only the timing column moved (by less than 0.5 s per task). The committed RESULTS.md comes from this run.
- `pytest`: 26 passed. GitHub Actions `tests` workflow: green on `a09652d` and `9aa266c`.
- `python scripts/check_sources.py`: 44 cited arXiv IDs, 0 to look at.

---

## 2026-10-12: Session 2, entanglement ablation and geometric difference

**Agent:** Claude (Claude Code on the web), same day as session 1, at Athmane's request to continue.
**Branch:** `claude/new-session-rwyqim`. `main` still holds only the initial commit, so merging is
still Athmane's decision.

### Literature watch (arXiv submissions 2026-09-28 → 2026-10-12)

The arXiv API search (six queries on QML advantage, quantum kernels, dequantisation, barren plateaus,
QNNs vs classical, learning advantage) returned 43 papers. Eight were relevant and were read at
abstract level, then added to FIELD_MAP.md:

| Paper | Milestone | What it adds |
|---|---|---|
| arXiv:2610.01141 | M1 | Average-case classical hardness of a physically motivated learning task, under an RSA-factoring assumption. |
| arXiv:2610.11360 | M2 | Large kernel-geometry separation gives an advantage on adversarial tasks, about ten times smaller on natural ones. |
| arXiv:2610.12148 | M5 | QNN forecasting claim against a single BiLSTM baseline, per the abstract. Does not meet our bar. |
| arXiv:2610.00841 | M6 | Classical neural "Fourier surrogates" match data-reuploading QNNs. |
| arXiv:2610.04190 | M6, M9 | Same ULB fraud data, one-class kernels: no severe concentration; Nyström cuts kernel cost. |
| arXiv:2609.40020 | M7 | New quantum-walker qRAM design (theory). |
| arXiv:2609.40105 | M7 | Problem structure can rescue end-to-end speed-ups in optimisation. |
| arXiv:2609.38403 | M8 | Quantum PAC learning: forward-only access to the state-preparation unitary gives no asymptotic advantage. |

**No status or progress estimate changed.** None of these papers turns a contested or open milestone
into a demonstrated one, and the two critical results (M2, M6) agree with the existing statuses.
`scripts/check_sources.py`: 52 cited IDs, 0 to look at.

### Built

- **Entanglement ablation:** `entangling=False` option in `quantum.py` (IQP map with
  `pattern=[]`) and in `vqc.py` (Rot layers without CNOTs, mean-⟨Zᵢ⟩ readout). New models
  `QSVM-fidelity-sep`, `QSVM-projected-sep` and `VQC-sep` run as experiment
  `creditcard_q8_separable`, with identical rows, splits, folds and grids. RESULTS.md gets a pairwise
  ablation table under the §6 rule.
- **Geometric difference:** `src/qmlrc/geometry.py`, Huang et al.'s regularised g (eqs. F19 and
  L8–L11, read in full text this session), minimised over linear and Gaussian kernels and λ with
  g_tra < 0.045. It runs as experiment `geometric_difference` on every seed and tuned kernel.
- `run_all` now records the time of each experiment. The longest takes 5.8 min, the whole run
  13.1 min.
- 12 new tests (38 in total):
  - separable states match an independent NumPy reference and are product states;
  - the separable fidelity kernel is the product of one-qubit kernels;
  - the separable VQC is exactly additive across features while the entangled one is not;
  - the geometric difference gives g(K‖K) = 1 and matches a hand computation;
  - the verdict rule is symmetric.

### Verified

- The refactor left every existing result **bit-identical**: `creditcard_q8`,
  `creditcard_full_classical` and `kernel_concentration` match session 1 except for the timing column.
- `pytest`: 38 passed. CI green on `70c374d`.

### Results (run from commit `70c374d`, fingerprint `d37c043cfce2f0fb`)

- **Kernels without entanglement do as well:**
  - fidelity kernel: 0.912 ± 0.028 without entanglement vs 0.911 ± 0.024 with;
  - projected kernel: 0.917 ± 0.027 without vs 0.913 ± 0.025 with.

  Verdict: no clear difference, which reproduces Bowles et al.'s finding on our real data.
- **VQC pair inconclusive.** `VQC-sep` scores 0.764 ± 0.199, and seed 3 collapses to 0.419 on test
  despite 0.893 in CV. Its outputs are nearly constant (balanced accuracy 0.507). This is a training
  problem of the control we built, combined with its different readout. It is not evidence that
  entanglement helps.
- **Geometric difference:** g = 1.3–4.9 per seed and kernel. Means: fidelity 4.2, projected 2.7,
  fidelity-sep 3.4, projected-sep 1.5. That is against √N = 37.4. The closest classical kernel is
  usually the widest Gaussian. By Huang et al.'s criterion, a classical kernel is guaranteed
  competitive for any labels on these inputs.

### What failed or needs care

- ARPACK did not converge when the top of the spectrum is clustered (the K‖K check). Fixed by
  working with Vᵀ K_Q V (no square root of K_Q), a looser tolerance, a fixed start vector, and a
  dense fallback.
- The separable VQC's training instability (above). It is a design problem in our control, documented
  in DESIGN.md and RESULTS.md.
- Grid edges: the rule is now explicit, with each end of an axis extended at most once. The SVM
  penalty C (both ends) and the VQC depth (lower end) are owed one extension. It was not run this
  session, and it is step 0 of the next session. Expected impact is small: the CV surface has been flat.
- An exploratory check during development (seed 0, bandwidth s = 1, not part of `run_all`) also gave
  g well below √N. It is **not** reported as a result because it was not produced by `run_all`;
  adding a bandwidth sweep of g to `run_all` is on the list.

### Next (details in NEXT_SESSION.md)

1. Step 0: the owed grid extensions (SVM C, VQC depth), bundled with the next code change.
2. Finite-shot kernels (100 / 1,000 / 10,000 shots).
3. A clean VQC entanglement control: the same readout and training for both twins.
4. A quantum-native dataset (milestone 8).
5. Statistics: 10 seeds, paired tests, a natural-prevalence variant.
