# Design

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

This file fixes the rules of the benchmark before results are interpreted. Changing a rule means
writing down why in `JOURNAL.md`.

## 1. What question the benchmark answers

> On the same data, with the same tuning budget, is a simulated quantum model a better learner than
> a well-tuned classical model?

It does **not** test computational quantum advantage. Every quantum model here uses at most 10
qubits and is simulated exactly on a laptop CPU in seconds. Nothing in this repository can show a
quantum speed-up. What it can show is whether the *model classes* proposed in quantum machine
learning (QML) have a useful inductive bias on a real dataset, compared with standard classical
models. A quantum model that loses here would also lose on hardware (where noise and shot noise only
make things worse); a quantum model that wins here would still need much more work before anyone
could call it an advantage (see `FIELD_MAP.md`).

## 2. Framework: PennyLane (not Qiskit)

We use **PennyLane 0.45.1** with its `default.qubit` state-vector simulator, and **JAX** to compile
and differentiate the variational classifier.

Reasons:

1. **Comparability.** The largest independent QML benchmark we rely on (Bowles, Ahmed & Schuld 2024,
   arXiv:2403.07059) is built on PennyLane + JAX + scikit-learn. Using the same stack makes it easy to
   port their models and check our numbers against theirs.
2. **Trainable models on a CPU.** PennyLane's simulators support backpropagation. With `jax.jit`,
   one gradient step of an 8-qubit, 2-layer circuit on a batch of 64 took about 15 ms on this
   machine, against about 115 ms with PennyLane's default autograd interface (measured in session 1).
   Training with a gradient-free optimiser or with finite differences would need many more circuit
   evaluations per step. That would make a *tuned* variational model (6 configurations × 3 folds)
   too slow for the ~10-minute budget.
3. **Standard templates.** `IQPEmbedding` (the ZZ-feature-map family of Havlíček et al.,
   arXiv:1804.11326) and `StronglyEntanglingLayers` are well-known, documented building blocks, so
   readers can check exactly which circuit was run.
4. **One dependency, Apache-2.0 licence, installable with pip.**

Qiskit would be the natural choice for running on IBM hardware later. Kernel values are computed
from exact state vectors (no shot noise); shot noise and hardware noise are on the roadmap
(`NEXT_SESSION.md`).

## 3. Data

**Dataset 1: credit-card fraud** (ULB Machine Learning Group / Worldline). It is downloaded from
OpenML at run time (`data_id=1597`, <https://www.openml.org/d/1597>). We do not redistribute it.

- 284,807 card transactions from two days in September 2013, 492 of them fraudulent (0.172 %).
- Features: V1–V28 (principal components released by the data owners for confidentiality) and
  `Amount`. OpenML's copy has no `Time` column. We apply a fixed `log1p` to `Amount`.
- Licence on OpenML: "Public" (checked 2026-10-12 through the OpenML API).
- Citation requested by the source: A. Dal Pozzolo, O. Caelen, R. A. Johnson, G. Bontempi,
  *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE CIDM 2015.
- `qmlrc.data.load_creditcard` checks the shape (284,807 × 29) and the fraud count (492), and
  scikit-learn checks OpenML's MD5 checksum.

Why this dataset: it is real, public, numeric (no categorical encoding choices to argue about), and
heavily imbalanced, so AUPRC is the right metric rather than accuracy. Fraud detection is also one of
the applications that QML papers name (e.g. Thanasilp et al., arXiv:2208.11060, list it among
proposed uses of quantum kernels).

**Subsampling (per seed).** Each seed draws 2,000 rows without replacement: 200 frauds (out of 492)
and 1,800 legitimate transactions, so **10 % fraud instead of 0.17 %**. Then a stratified 70/30
train/test split (1,400 / 600 rows, 60 frauds in the test set). This keeps every model, including the
kernel methods (whose cost grows with the square of the number of rows), CPU-feasible, and leaves
enough frauds in the test set for AUPRC to mean something. Consequence: our AUPRC values are **not**
comparable to numbers reported on the full, naturally imbalanced dataset.

**Features.** Standardise → PCA to 8 components → standardise again, fitted on the training rows only
(unsupervised, no labels involved). 8 features = 8 qubits. Every model, classical and quantum, gets
exactly these 8 features. A separate reference run gives the classical models all 29 features to show
what the 8-qubit restriction costs; it is labelled as not like-for-like.

## 4. Models and tuning budget

All models are tuned by grid search on the **same 3 stratified CV folds** of the training rows, with
the **same selection metric (AUPRC)**. The test rows are used once, after tuning. The number of
configurations tried is reported in every results table.

| Model | Family | Grid | Configs |
|---|---|---|---|
| LogReg | classical | C ∈ {1e-3 … 1e3} (7) × class_weight ∈ {none, balanced} | 14 |
| RBF-SVM | classical | C ∈ {0.01 … 1000} (6) × γ = f · `scale`, f ∈ {0.1, 0.3, 1, 3, 10} × class_weight (2) | 60 |
| GradBoost | classical | `HistGradientBoostingClassifier`: learning_rate {0.03, 0.1, 0.3} × max_leaf_nodes {7, 31} × min_samples_leaf {5, 20} × l2 {0, 1} | 24 |
| MLP | classical | hidden layers {(16), (64), (32, 32)} × alpha {1e-4, 1e-2, 1} | 9 |
| QSVM-fidelity | quantum | bandwidth s ∈ {0.05, 0.1, 0.2, 0.5, 1} × C (6) × class_weight (2) | 60 |
| QSVM-projected | quantum | s ∈ {0.1, 0.3, 1} × γ-factor {0.1, 1, 10} × C (6) × class_weight (2) | 108 |
| VQC | quantum | s ∈ {0.1, 0.3, 1} × layers {2, 4} | 6 |

Details:

- **Quantum feature map.** `IQPEmbedding(s·x)`: Hadamards, `RZ(s·x_i)` on each qubit, `MultiRZ(s²·x_i·x_j)`
  on every pair, repeated twice (once for the VQC). This is the IQP/ZZ feature map with the bandwidth
  (input-scaling) hyperparameter of Shaydulin & Wild (arXiv:2111.05451). Tuning `s` is the quantum
  analogue of tuning γ for the RBF kernel. Leaving it at 1 would handicap the quantum kernels for no
  reason.
- **QSVM-fidelity.** k(x, x') = |⟨φ(x)|φ(x')⟩|², fed to scikit-learn's `SVC(kernel="precomputed")`.
- **QSVM-projected.** The projected quantum kernel of Huang et al. (arXiv:2011.01938):
  k = exp(−γ Σ_k ‖ρ_k(x) − ρ_k(x')‖²_F), from the one-qubit reduced states of the same feature map.
  Its γ grid is relative to a 'scale'-style default, like the RBF-SVM's.
- **VQC.** `IQPEmbedding(s·x)` (1 repetition) + `StronglyEntanglingLayers`, output
  logit = a·⟨Z₀⟩ + b. Class-weighted binary cross-entropy, Adam (learning rate 0.05), 150 steps of
  batch 64, initial weights N(0, 0.1²). Exact gradients by backpropagation through the simulation.
- **GradBoost** has no `class_weight` axis: in scikit-learn 1.9.1 it made each fit about 10× slower
  on this data (measured during session 1), and reweighting matters little for a ranking metric. It
  gets an L2-regularisation axis instead.
- The quantum models get as many or more configurations than the classical ones (60 vs 60 for the
  SVMs, 108 for the projected kernel). If anything, this favours the quantum side.

## 5. Metrics and reporting

- **AUPRC** (average precision, scikit-learn's `average_precision_score`), the primary metric. A random
  ranking scores the fraud rate of the test set (0.10).
- **Accuracy** (requested; easy to inflate: always answering "not fraud" scores 0.90) and **balanced
  accuracy**, both at the model's default threshold.
- **Wall-clock time** per seed for tuning + final fit + test prediction, on the machine that produced
  RESULTS.md. For quantum models this is *classical simulation time*.
- **5 seeds** (0–4). Each seed redraws the subsample, the split, the CV folds and the model
  initialisation. We report mean ± sample standard deviation and **paired per-seed differences**.

## 6. What "advantage" means here

We say a quantum model **beats** a classical model on a benchmark only if all three hold:

1. **Same data**: same rows, split, CV folds and features (enforced in `qmlrc.data.make_split`).
2. **Same or larger budget for the classical side**: a classical model is never handicapped by a
   smaller tuning grid than the quantum model it is compared with (configuration counts are printed).
3. **Consistent win**: its test AUPRC is higher on at least 4 of 5 seeds *and* the mean paired
   difference is larger than the standard deviation of the paired differences.

The same rule, with the roles swapped, decides when a classical model beats a quantum one. Anything
else is reported as "no clear difference". This rule was written in session 1 while the first full run
was in progress; it treats both sides symmetrically.

Even a quantum "win" under this rule would be a statement about inductive bias at 8 qubits, not a
quantum advantage: the model was simulated classically.

## 7. Known limitations (threats to validity)

- One dataset, 2,000 rows per seed, 600 test rows (60 frauds): AUPRC differences of a few hundredths
  are within seed-to-seed noise.
- Fraud prevalence is artificially raised to 10 %.
- The PCA-to-8 step throws information away for every model (the 29-feature reference run measures
  how much).
- Noiseless, infinite-shot simulation: real hardware would add shot noise and gate noise, which can
  only hurt the quantum models (see FIELD_MAP.md, milestones on kernels and hardware).
- One VQC design (single-qubit readout, fixed optimiser settings). Other designs (data re-uploading,
  trainable input scaling) may do better; see NEXT_SESSION.md.
- Timing is measured on one 4-thread container; numbers move with hardware and load.
- The unsupervised preprocessing (scaler + PCA) is fitted on the whole training set before the CV
  folds are cut. No labels are involved, the test set is never touched, and it is the same for every
  model.
