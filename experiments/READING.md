<!-- results-hash: d37c043cfce2f0fb -->
**In one sentence: on this fraud-detection task, no quantum model beat any classical model, removing
entanglement from the quantum kernels changed nothing, and the quantum kernels are geometrically
close to ordinary wide Gaussian kernels.**

**1. Main comparison (unchanged since session 1; the numbers are bit-identical).**

- Best classical models: RBF-SVM and MLP at 0.918 AUPRC, gradient boosting at 0.917.
- Best quantum models: the projected-kernel SVM at 0.913 and the fidelity-kernel SVM at 0.911. The
  VQC is last at 0.899. A random ranking scores 0.10.
- All gaps are smaller than the seed-to-seed standard deviations (0.023–0.034). Under DESIGN.md §6
  every quantum/classical pair is "no clear difference", with the leaning towards classical: the
  quantum kernels are ahead of the three strongest classical models on only 2 of 5 seeds.

**2. Entanglement ablation (new in session 2).**

- *Kernels.* The two quantum kernels with their entangling gates removed do as well as the originals:
  - fidelity kernel: 0.912 without entanglement vs 0.911 with (paired difference −0.001 ± 0.005);
  - projected kernel: 0.917 without vs 0.913 with (−0.005 ± 0.006). The unentangled version is even
    ahead on 4 of 5 seeds, though within noise.

  This is the test of Bowles et al. (arXiv:2403.07059), and it gives the same answer on our data:
  the performance of these quantum kernels does not come from entanglement. It confirms the
  session-1 reading that the tuned circuits sit at small bandwidths, where the entangling terms
  hardly matter.
- *VQC: inconclusive.* The separable VQC is unstable:
  - test AUPRC 0.764 ± 0.199, with seed 3 collapsing to 0.419 even though its cross-validation
    AUPRC was 0.893;
  - at its default threshold it predicts (almost) a single class on every seed (balanced accuracy
    0.507).

  Its outputs barely vary: it averages eight small ⟨Zᵢ⟩ signals and multiplies them by a trainable
  scale that starts at 1. The twin was also built with a different readout, because a single-qubit
  readout of an unentangled state would see only one feature. So this pair mixes three things:
  entanglement, readout, and an optimisation problem we introduced. **It is not evidence that
  entanglement helps the VQC.** A cleaner control is planned (NEXT_SESSION.md).

**3. Geometric difference (new in session 2).** Following Huang et al. (arXiv:2011.01938, App. L),
we compared each tuned quantum kernel with a linear kernel and nine Gaussian kernels on each seed's
1,400 training rows.

- The geometric difference g ranges from 1.3 to 4.9. By model:
  - fidelity kernel: mean 4.2;
  - projected kernel: mean 2.7;
  - their unentangled twins: 3.4 and 1.5.
- These values are 9 to 25 times smaller than √N = 37.4.
- Huang et al.'s bound s_C ≤ g² s_Q then puts these kernels firmly on the "classical ML predicts
  similar or better" side of their test, for *any* labels on these inputs, not just the fraud labels.
- The closest classical kernel is usually the widest Gaussian we tried (γ = 0.25 / (n·Var)): the
  tuned quantum kernels behave like very smooth Gaussian kernels.
- Entanglement makes the fidelity kernel slightly more different from classical kernels (4.2 vs 3.4),
  but that extra difference buys no accuracy (section 2).
- This explains the main comparison rather than just observing it: with g this small, no relabelling
  of these 1,400 transactions could give these quantum kernels a large advantage.

**4. Accuracy is still misleading here.** Logistic regression on seeds 1 and 4, the RBF-SVM on seed
4, and the separable VQC on every seed predict (almost) a single class at the default threshold. Their
accuracy is then about 0.90 or 0.10 even when their ranking is good. Read the AUPRC column first.

**5. Unchanged from session 1.**

- With all 29 features, the classical models reach 0.926–0.934, so the 8-qubit restriction costs
  about 0.01–0.02 AUPRC.
- Kernel concentration appears only at bandwidth s = 1: the mean fidelity-kernel value falls from
  0.75 at 2 qubits to 0.11 at 10 qubits. Cross-validation never chose s = 1.
- Per seed on one core, the VQC costs about 170 s and the separable VQC about 115 s, against 4 s for
  the RBF-SVM.

**6. Known loose ends.** Some grid axes still hit their edges (listed above). Two of them have not yet
had their one allowed extension: the SVM penalty C (both ends) and the VQC depth (lower end). They
are scheduled for the next run (NEXT_SESSION.md, step 0). The cross-validation surface has been flat
so far (session-1 extensions moved CV AUPRC by ≤ 0.002), so we do not expect them to change the
conclusions, but we will check rather than assume.

**What this does and does not show.** For standard QML designs (IQP feature map, fidelity and
projected kernels, a hardware-efficient VQC) on this real tabular task at 8 qubits:

- they have no edge over tuned classical models;
- their kernels gain nothing from entanglement;
- their geometry leaves no room for an advantage.

This agrees with Bowles et al. and with the "power of data" analysis of Huang et al. It says nothing
about quantum data, other encodings, or larger qubit counts (FIELD_MAP.md, milestones 6 and 8).
