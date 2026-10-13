<!-- results-hash: 7abe3ac53d8d7b0f -->
**In one sentence: on this fraud-detection task, no quantum model beat any classical model.
Removing entanglement changed nothing for any quantum model. The quantum kernels are geometrically
close to ordinary wide Gaussian kernels. Realistic measurement noise only makes the quantum side
worse.**

**1. Main comparison.**

- Best classical models: gradient boosting and MLP 0.918 AUPRC, RBF-SVM 0.917.
- Quantum models: projected-kernel SVM 0.913, fidelity-kernel SVM 0.911, VQC 0.899. A random
  ranking scores 0.10.
- Under DESIGN.md §6 every quantum/classical pair is "no clear difference", with the leaning
  towards classical: the quantum kernels are ahead of the three strongest classical models on only
  2–3 of 5 seeds.
- Session 3 made the last owed grid extensions:
  - SVM penalty C at both ends. Only the RBF-SVM on seed 4 moved (0.939 → 0.933).
  - Gradient boosting at five axis ends (DESIGN.md §4), from 81 to 320 configurations. Its mean CV
    AUPRC rose by 0.001, and its test AUPRC went from 0.917 to 0.918.

  An intermediate grid of 144 configurations gave 0.921 on test, with a lower CV score. The table
  reports the final grid that the rule fixes, not the more favourable intermediate number.
  A selection effect of this size (0.003) is well below the seed-to-seed spread.

**2. Entanglement ablation: entanglement does not help any of the three models.** Each pair below
differs only in entanglement:

| Pair | AUPRC with entanglement | AUPRC without | Paired difference | Verdict |
|---|---|---|---|---|
| Fidelity kernel | 0.911 | 0.912 | −0.001 ± 0.005 | no clear difference |
| Projected kernel | 0.913 | 0.917 | −0.005 ± 0.006 | no clear difference |
| VQC (same standardised mean-⟨Zᵢ⟩ readout) | 0.890 | 0.902 | −0.012 ± 0.021 | no clear difference |

- In all three pairs the unentangled twin is ahead on 3 or 4 of 5 seeds, though within noise.
- The VQC pair is now a clean control. The session-2 version was inconclusive because its separable
  twin had a different readout and trained unreliably. With the same standardised readout for both,
  the separable VQC is stable (0.902 ± 0.027).
- This reproduces on real data the finding of Bowles et al. (arXiv:2403.07059).

**3. Finite-shot kernels: shot noise only hurts, and it hurts the fidelity kernel most.**

| Shots per kernel entry | Fidelity kernel AUPRC | Projected kernel AUPRC |
|---|---|---|
| Exact (infinite shots) | 0.911 | 0.913 |
| 100 | 0.848 (worse than exact) | 0.898 |
| 1,000 | 0.884 | 0.893 |
| 10,000 | 0.888 (worse than exact) | 0.917 |

- The fidelity kernel loses about 0.06 AUPRC at 100 shots and still trails the exact kernel at
  10,000 shots (lower on all 5 seeds).
- *Our reading:* cross-validation keeps choosing small bandwidths (s = 0.02–0.2). There all kernel
  values are close to 1 and the useful differences between them are small. Shot noise of order
  √(k(1 − k)/R) is then of the same size as those differences. Avoiding concentration with a small
  bandwidth (milestone 6) moves the kernel into a regime that is fragile to measurement noise.
- The projected kernel is robust: at 10,000 shots it matches the exact kernel. It is also far
  cheaper to estimate. Building one fitted model's kernels at 1,000 shots takes about 1.8 × 10⁹
  circuit executions for the fidelity kernel (one per pair of points) and 6 × 10⁶ for the projected
  kernel (three measurement bases per point). That is a factor of about 300, before any
  hyperparameter search.
- This is shot noise only. Gate noise and readout errors would come on top.

**4. Geometric difference (unchanged from session 2).**

- g ranges from 1.3 to 4.9 per seed and kernel. Means: fidelity 4.2, projected 2.7, and their
  unentangled twins 3.4 and 1.5. That is against √N = 37.4.
- By Huang et al.'s criterion (arXiv:2011.01938), a classical kernel, usually the widest Gaussian
  tested, is guaranteed to predict about as well for any labels on these inputs.
- This is *why* no quantum kernel can win here.

**5. Other points.**

- Accuracy is misleading on this data. Logistic regression (seeds 1 and 4) and the RBF-SVM (seed 4)
  predict a single class at their default threshold, so read the AUPRC column first.
- With all 29 features, the classical models reach 0.925–0.934, so the 8-qubit limit costs about
  0.01–0.02 AUPRC.
- Kernel concentration appears only at bandwidth s = 1 (mean fidelity-kernel value 0.75 at 2 qubits,
  0.11 at 10 qubits), a bandwidth cross-validation never chose.
- Cost per seed on one core: VQC about 200 s, VQC-mean about 250 s, against 6 s for the RBF-SVM and
  about 90 s for gradient boosting's 320 configurations.
- Remaining grid-edge hits (listed above) are reported, not extended. Each sits at an axis end that
  has had its one extension, or at a natural limit (one VQC layer) (DESIGN.md §4).

**What this does and does not show.** For standard QML designs (IQP feature map, fidelity and
projected kernels, hardware-efficient VQCs) on this real tabular task at 8 qubits:

- they have no edge over tuned classical models;
- entanglement contributes nothing;
- their kernel geometry leaves no room for an advantage;
- measurement noise makes them worse.

It says nothing about quantum data, problem-specific encodings or larger systems. Those are the
directions the idea pipeline (IDEAS.md) should explore.
