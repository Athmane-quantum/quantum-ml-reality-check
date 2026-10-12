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
  final run from the committed code must reproduce run 2's fingerprint `6f91668eaad26227`; its outcome
  is recorded in the last bullet of this entry.

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

- Final full run from commit `FINAL_COMMIT`: fingerprint `FINAL_HASH`. FINAL_STATUS
