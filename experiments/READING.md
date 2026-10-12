<!-- results-hash: 6f91668eaad26227 -->
**In one sentence: on this fraud-detection task, no quantum model beat any classical model.** The
classical models are slightly ahead on average, but by less than the seed-to-seed noise.

**What the table says.**

- The best classical models (RBF-SVM and MLP, 0.918 AUPRC; gradient boosting, 0.917) and the best
  quantum models (projected-kernel SVM, 0.913; fidelity-kernel SVM, 0.911) are within 0.01 of each
  other.
- The variational classifier (VQC) is last, at 0.899.
- For scale: a random ranking scores 0.10, so every model finds most of the fraud.
- The standard deviations across seeds (0.023–0.034) are larger than every gap between models. Under
  the decision rule fixed in DESIGN.md §6, every quantum/classical pair comes out as **"no clear
  difference"**. The pattern still points one way: the two quantum kernels are ahead of the RBF-SVM,
  gradient boosting and the MLP on only 2 of 5 seeds, and the VQC on only 1 of 5.
- The only comparison the quantum kernels "win" on 4 of 5 seeds is against logistic regression, by
  0.003–0.005, which is again noise.
- The first run, with smaller hyperparameter grids, gave the same picture (archived in
  `experiments/archive/2026-10-12_run1/`).

**Why the quantum kernels end up looking classical.** Cross-validation always picked a small bandwidth
for the quantum models (s between 0.02 and 0.3; s = 1 was never chosen). In the IQP feature map the
two-qubit (entangling) angles scale as s²·xᵢ·xⱼ, while the one-qubit angles scale as s·xᵢ. At s = 0.05
the entangling angles are therefore about 20 times smaller than the one-qubit ones. So the circuits
the models chose are close to unentangled, smooth feature maps, close to what a classical kernel does.
*This is our reading of the circuit, not something we measured.* The next session tests it by
removing the entangling gates (the "separable" control of Bowles et al., arXiv:2403.07059). It matches
the bandwidth results of Shaydulin & Wild (arXiv:2111.05451): tuned quantum kernels become
competitive with classical ones, not better.

**Accuracy is misleading here.** Logistic regression on seeds 1 and 4, and the RBF-SVM on seed 4,
predict "not fraud" for every test transaction at their default threshold. Their accuracy is then
exactly 0.900 and their balanced accuracy 0.500, even though their AUPRC (ranking quality) is good.
This happens because tuning optimised AUPRC, not the threshold. Read the AUPRC column first.

**What the 8-qubit limit costs.** Given all 29 features instead of 8, the same classical models reach
0.926–0.934 AUPRC. Squeezing the data into 8 qubits costs about 0.01–0.02 AUPRC on its own.

**Cost.** Simulated quantum models are not cheaper either. Per seed, on one CPU core, the VQC took
about 170 s (12 configurations × 3 folds of gradient training), against 0.3 s for logistic regression
and 4 s for the RBF-SVM. On real hardware the fidelity kernel would need one circuit per pair of
points (on the order of N² circuits, each repeated for many shots). The projected kernel needs only N
state preparations, but many measurements per state.

**Kernel concentration, reproduced at small scale.**

- At bandwidth s = 1, the average fidelity-kernel value between two different transactions falls from
  0.75 with 2 qubits to 0.11 with 10 qubits. Its spread falls from 0.30 (4 qubits) to 0.10
  (10 qubits).
- This is the start of the exponential concentration described by Thanasilp et al.
  (arXiv:2208.11060). At 10 qubits it is still far from the fully random value (0.001).
- At s = 0.1 and 0.3 the kernel does not concentrate up to 10 qubits, which is the range the
  benchmark's cross-validation chose.

**What this does and does not show.** It does not show that quantum machine learning cannot help.
It shows that the standard model designs (IQP feature map, fidelity and projected kernels, a hardware-
efficient variational circuit) have no edge over tuned classical models on this real, small, tabular
task at 8 qubits. That agrees with the large benchmark of Bowles et al. (FIELD_MAP.md, milestone 5).
