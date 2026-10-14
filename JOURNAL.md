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

## 2026-10-12: Session 3, finishing the benchmark and building the idea pipeline (initialisation complete)

**Agent:** Claude (Claude Code on the web), same day as sessions 1–2.
**Branch:** `claude/new-session-rwyqim`. `main` still holds only the initial commit, so merging is
still Athmane's decision.

### Athmane's requests this session

- Athmane asked what this work concretely brings to the quantum-AI project. Answer given: the
  repository is the measuring stick (tuned classical baselines that any quantum idea must beat) and
  the map (which directions are closed and why).
- Athmane asked how to use other assistants (ChatGPT, DeepSeek, Adaptive AI) to generate ideas,
  with ChatGPT removing duplicates and Claude validating. The pipeline below was built for that.
  Caveat recorded in IDEAS.md: a language model cannot reliably tell whether an idea already exists,
  so every novelty claim is checked against arXiv.
- Athmane asked to finish the initialisation and be told when it is done.

### Literature watch

Not repeated: session 2's watch covered arXiv submissions up to today. `scripts/check_sources.py`:
52 cited IDs, 0 to look at.

### Built

- **Idea pipeline** (commit `f5167ed`):
  - IDEAS.md: the six-step pipeline, the idea card, statuses, kill criteria K1–K7 drawn from the
    field map, green flags, and an empty register;
  - `ideas/CONTEXT.md`, `ideas/PROMPT_GENERATOR.md`, `ideas/PROMPT_NOVELTY_FILTER.md`, each with
    instructions in French;
  - `scripts/arxiv_search.py`, a novelty search through the official arXiv API.
- **Owed grid extensions** (commits `9c4e678`, `79c3b6b`, `44983f2`, `e746bba`), under DESIGN.md §4:
  - SVM penalty C gains 0.001 and 10,000;
  - VQC depth gains 1;
  - GradBoost gains five axis ends: the learning rate gains 1 and 0.01, the L2 penalty 100,
    max_leaf_nodes 3 and min_samples_leaf 2. That takes it from 81 to 320 configurations.

  The GradBoost ends had been hit since session 2, in the main table or in the 29-feature run that
  shares its grid. Session 2 listed only C and VQC depth as owed, which left the classical side
  under-tuned by our own rule. The miss was found when the edge list was re-read before closing.
  Extending one axis moved the CV choices of others onto their ends, so this took three runs. Each
  end is extended at most once, so the process is now finished. DESIGN.md §4 now states that hits
  count in every experiment using the same grid.
- **Finite-shot kernels** (`src/qmlrc/shots.py`, experiment `creditcard_q8_shots`):
  - fidelity-kernel entries are sampled as Binomial(R, k)/R;
  - Bloch-vector components are sampled as 2·Binomial(R, (1 + ⟨P⟩)/2)/R − 1;
  - the training Gram matrix is clipped to positive semidefinite;
  - both kernel SVMs are re-tuned at R = 100, 1,000 and 10,000 shots, with the noise seeded per
    (seed, R, train/test, bandwidth).
- **Clean VQC entanglement control.** The session-2 separable VQC failed, because its readout was
  nearly constant at small bandwidth. Three VQC options were added: mean-⟨Zᵢ⟩ readout, a log-scale
  output weight, and a standardised readout. With them, `VQC-mean` (entangled) and `VQC-sep` share
  readout and training and differ only in entanglement. The default VQC is numerically unchanged.
- The ablation experiment is renamed `creditcard_q8_ablation`. Session-2 outputs are archived in
  `experiments/archive/2026-10-12_session2/`.
- 14 new tests (52 in total). They cover shot-sampling unbiasedness and shot counts, PSD clipping,
  the mean-readout requirement for separable VQCs, readout standardisation at tiny bandwidth, and
  the default VQC being unchanged.

### Verified

- Across the four full runs of this session (`9c4e678`, `1df3ec8`, `44983f2`, `e746bba`), every model
  whose grid did not change gave bit-identical metrics. Between the last two runs, nothing changed
  except GradBoost's configuration count: the new learning rate 0.01 was never selected.
- `pytest`: 52 passed, about 7.5 minutes locally. CI green on `e746bba`, the last code commit.
- `scripts/check_sources.py`: 52 cited IDs, 0 to look at.

### Results (run from commit `e746bba`, fingerprint `7abe3ac53d8d7b0f`, 28.0 min in total)

Every experiment ran in under 10 minutes; the longest, the ablation, took 9.5 minutes.

- **Main table** (test AUPRC, 5 seeds):
  - classical: gradient boosting 0.918, MLP 0.918, RBF-SVM 0.917, logistic regression 0.908;
  - quantum: projected-kernel SVM 0.913, fidelity-kernel SVM 0.911, VQC 0.899.

  Every quantum/classical pair is "no clear difference" under §6, leaning classical.
- **GradBoost and selection noise.** The intermediate 144-configuration grid gave GradBoost 0.921
  on test; the final 320-configuration grid gives 0.918, with a higher CV score (0.9075 against
  0.9064). We report the final grid, which the rule fixes, not the better-looking intermediate
  number. The 0.003 gap is selection noise, well below the seed-to-seed spread (±0.028).
- **Entanglement ablation** (entangled vs unentangled):
  - fidelity kernel: 0.911 vs 0.912;
  - projected kernel: 0.913 vs 0.917;
  - VQC with identical standardised readouts (`VQC-mean` vs `VQC-sep`): 0.890 vs 0.902.

  All three pairs are "no clear difference". The separable VQC is now stable (0.902 ± 0.027, against
  0.764 ± 0.199 in session 2).
- **Finite shots** (test AUPRC at 100 / 1,000 / 10,000 shots):

  | Kernel | Exact | 100 | 1,000 | 10,000 |
  |---|---|---|---|---|
  | Fidelity | 0.911 | 0.848 | 0.884 | 0.888 |
  | Projected | 0.913 | 0.898 | 0.893 | 0.917 |

  At 10,000 shots the fidelity kernel is still below exact on all 5 seeds. At 1,000 shots, one
  fitted model's kernels cost 1.8 × 10⁹ circuit executions with the fidelity kernel and 6 × 10⁶ with
  the projected kernel.
- **Geometric difference:** unchanged, g = 1.3–4.9 against √N = 37.4.
- **29-feature reference:** classical models reach 0.925–0.934. The 8-qubit restriction costs
  about 0.01–0.02 AUPRC.

### What failed or needs care

- The GradBoost grid miss (above). From now on the end-of-session check is mechanical: every edge
  hit listed in RESULTS.md is compared with the extension history in DESIGN.md §4.
- Remaining edge hits are at axis ends that have had their one extension, or at a natural limit
  (VQC-sep depth 1). They are reported, not chased.
- The finite-shot experiment models shot noise only. Gate noise and readout errors are not modelled.

### Initialisation complete

The field map, the benchmark with its controls, and the idea pipeline are in place. From session 4
the project moves to idea rounds (NEXT_SESSION.md).

## 2026-10-12 to 2026-10-14: Session 4, first idea round

**Agent:** Claude (Claude Code on the web), continuing the same conversation.
**Branch:** `claude/new-session-rwyqim`. Athmane opened
[PR #1](https://github.com/Athmane-quantum/quantum-ml-reality-check/pull/1) from it into `main`.
Its auto-generated description had wrong counts (26 tests, 44 sources, three experiments) and a
truncated result section, so Claude rewrote it with the final numbers.

### Inputs

- Athmane brought 32 ideas from four external assistants (8 each). They are labelled batches A–D
  in the order pasted; Athmane asked that they not be attributed.
- Claude added 8 of its own (I-33 to I-40), written after reading the 32.
- The ChatGPT deduplication step was not run; Claude grouped the 40 ideas into nine themes instead.
- Athmane also asked that the two session prompts be kept in the repository: they are now in
  `PROMPTS.md`, copied verbatim from the file Athmane uploaded.

### Process

1. The 23 arXiv IDs cited by the external batches were resolved in one arXiv API query. All exist.
   Two are misused: arXiv:2407.07754 cited for the wrong topic, and arXiv:2111.05451 attributed to
   the wrong authors.
2. A workflow ran one triage agent per theme (reference checks, novelty search on alphaXiv and the
   arXiv API, technical checks, K1–K7, a sharpened best version). An adversarial reviewer then
   tried to refute each best version.
   - 17 agents, 704 tool calls, about 3.5 hours.
   - A first launch passed placeholder text instead of the theme list and failed before starting
     any agent. It was fixed by having agents read the cards from a file.
3. Claude resolved all 251 distinct arXiv IDs in the agents' outputs through the arXiv API. All
   exist, and every title the agents recorded matches.
4. Claude wrote [ideas/round-01.md](ideas/round-01.md) (summary, shortlist, recommendation) and
   [ideas/round-01-cards.md](ideas/round-01-cards.md) (the 40 full cards), and filled the
   IDEAS.md register.
   - The agents' raw output is in `ideas/round-01-raw.json`.
   - Their scratch check scripts are in `ideas/round-01-checks/`. They are unreviewed and
     indicative, not repository results.

### Outcome

- **Statuses:** 8 shortlisted, 20 rejected (18 at triage, 2 in review) and 12 merged.
- **Shortlisted:** every shortlisted best version was weakened in review, with concrete fixes listed.
  One, idea 38, is shortlisted at low priority. It was not part of a reviewed best version.
- **Recommendation to Athmane:**
  - R1: two-copy vs single-copy learning, a certified finite-n table plus noise thresholds (M8).
  - R2: calibrate the benchmark with engineered labels, plus the symmetry test (M2, M6).
  - R3: audit the oracle-sketching claim of arXiv:2604.07639 (M1).
- **Reference quality:** no assistant invented an arXiv ID. The common failure was a real paper
  cited for a claim it does not make.
- **New kill criterion K8** (access-model artefacts), added to IDEAS.md.
- **Lessons for future generators** are added to `ideas/CONTEXT.md` (item 10).

### Corrections to our own text

The T6 reviewer pointed out that READING.md, README.md, FIELD_MAP.md (M2), `geometry.py`,
`report.py` and `ideas/CONTEXT.md` overstated Huang et al.'s bound. They said that with
g ≈ 1.3–4.9 a classical kernel "is guaranteed to predict about as well for any labels".

- The bound gives s_C ≤ g² s_Q: the classical error bound is at most g times the quantum one.
- So it rules out a *large* advantage, not a modest one.
- All six places now say this. RESULTS.md was rebuilt with `--report-only`; the fingerprint is
  unchanged (`7abe3ac53d8d7b0f`), since no number changed.

Also during the round, Claude first marked idea 38 as killed in review with the rest of its theme.
The review did not address it, so it was corrected to "shortlisted, low priority" before anything
was committed.

### No status or progress change in FIELD_MAP.md

The round surfaced recent papers relevant to M1, M2, M4 and M8, listed in round-01.md. They will be
opened and added in the next literature watch, not on the strength of the agents' reading alone.

### Next

Athmane chooses among R1–R3 (NEXT_SESSION.md). Nothing is built before that choice.
