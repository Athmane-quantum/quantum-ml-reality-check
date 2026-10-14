# Field map: where quantum machine learning stands

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

**Last updated:** 2026-10-12. Session 2 ran a literature watch for arXiv submissions from 2026-09-28
to 2026-10-12, adding evidence under milestones 1, 2, 5, 6, 7 and 8. Session 3 added this
repository's finite-shot and VQC-ablation results. No status or progress estimate changed. **Question:** what is left to do before quantum computers give
a *useful* advantage in machine learning (ML), meaning better results than the best classical
methods on a task someone cares about?

## How to read this map

The map is a list of **milestones**. Each one is a claim that has to hold for a useful QML advantage
to exist.

- **Status** of the claim:
  - `DEMONSTRATED`: proven, or shown in experiment, at least in some setting.
  - `CONTESTED`: serious evidence on both sides.
  - `OPEN`: not shown either way.
  - `REFUTED-IN-GENERAL`: shown false in general, with narrow exceptions.
- **Progress (0–100 %)**: our estimate of how far the field is from having this milestone in a form
  that would support a *useful* advantage. 0 means nothing has been done. 100 means it is done for a
  practically relevant task on real hardware. These numbers are a **judgement, not a measurement**.
  Each comes with a one-sentence reason, so you can disagree with the reason rather than the number.
- **Sources.** Every factual claim cites a primary source by arXiv ID or DOI. Each source was opened in
  the session shown, at the level shown:
  - `[F]`: full text read (the relevant pages).
  - `[A]`: abstract and bibliographic record read, from the arXiv API or the publisher's DOI record.
    Claims tagged `[A]` stay at the level of detail of the abstract.
  - `NOT VERIFIED`: could not be opened; the claim must not be relied on.

  The full list is in [Sources](#sources). `scripts/check_sources.py` re-checks every arXiv ID
  against the arXiv API.

## Corrections to the brief

The session-1 brief named several anchor papers. All were opened and exist as described, with these
corrections and clarifications:

| Brief said | What the sources show |
|---|---|
| "absence of barren plateaus ⇔ classical simulability (Cerezo et al. ~2023)" | Preprint Dec 2023 (arXiv:2312.09121), published in *Nat. Commun.* **16**, 7907 (**2025**). The v1 title had a subtitle ("Or, why we need to rethink variational quantum computing") that the final version drops. It is a **perspective backed by case-by-case evidence, not a theorem**. The authors state that they do not prove every barren-plateau-free model is simulable, and they build contrived counterexamples themselves. |
| "Tang 2018 and follow-ups" | Correct as preprint years. The recommendation-system paper appeared at STOC 2019 (doi:10.1145/3313276.3316310). The PCA/clustering follow-up (arXiv:1811.00414) was **retitled**: v1 "Quantum-inspired classical algorithms for principal component analysis and supervised clustering" became "Quantum principal component analysis only achieves an exponential speedup because of its state preparation assumptions" (*PRL* **127**, 060503, 2021). |
| "large-scale benchmarking showing classical baselines **often** win (Bowles, Ahmed, Schuld 2024)" | The paper says something stronger: out-of-the-box classical models **systematically** outperform the 12 quantum models across 160 datasets. It also gives the limits: small scale (circuits up to 18 qubits), mostly synthetic data. |
| "provable advantage on contrived/cryptographic tasks (Liu, Arunachalam, Temme 2021)" | Correct. One precision: the quantum learner needs a **fault-tolerant** quantum computer, because its feature map runs Shor's discrete-log algorithm. |
| "barren plateaus (McClean et al. 2018)" | Correct. One precision: the theorem covers parameterised circuits random enough to approximate unitary 2-designs, not every circuit. |

---

## Milestone 1: A provable learning advantage on classical data

> *There is a learning problem with classical inputs that a quantum learner solves efficiently and no
> efficient classical learner can.*

**Status: DEMONSTRATED** (in theory, on engineered tasks). **Progress: 55 %.** Separations are proven
under standard assumptions and tolerate shot noise, and newer ones use less artificial distributions.
But every one is still an engineered task, most need fault-tolerant hardware, and none has been run
end to end.

- Liu, Arunachalam & Temme build a classification task from the discrete logarithm. Assuming discrete
  log is classically hard, no efficient classical learner beats 1/2 + 1/poly(n) test accuracy. An SVM
  whose kernel is estimated on a fault-tolerant quantum computer learns it, even with finite-shot
  kernel estimates. arXiv:2010.02174, *Nat. Phys.* (2021), doi:10.1038/s41567-021-01287-z `[F]`
- Gyurik & Dunjko point out that known classical-data separations are cryptography-inspired, and
  rely on the hardness of *evaluating* the target function rather than *identifying* it. They give new
  separations. arXiv:2306.16028 `[A]`
- Molteni, Gyurik & Dunjko prove advantages for a physically motivated task: learning quantum
  observables from classical (measured-out) data. arXiv:2405.02027 `[A]`
- Lewis, Gilboa & McClean prove an exponential advantage in the quantum statistical-query model for
  learning "periodic neurons" under a family of non-uniform input distributions. arXiv:2503.20879 `[A]`
- Jerbi et al. show that some advantage survives when the quantum computer is used **only during
  training** and the deployed model is classical ("shadow models"), under complexity assumptions.
  arXiv:2306.00061, *Nat. Commun.* **15**, 5676 (2024) `[A]`
- *New (session 2).* Hashimoto & Kawachi prove average-case classical hardness for "Hamiltonian
  function learning", a physically motivated task: predicting Tr[f(H)ρ] from classical descriptions of
  H and ρ. The proof assumes that factoring random RSA moduli is hard, so the separation is still
  cryptographic underneath. arXiv:2610.01141 (Oct 2026 preprint) `[A]`

## Milestone 2: Quantum-generated data alone gives an advantage ("power of data")

> *If labels come from a quantum process that is hard to simulate, a quantum model will predict them
> better than classical ML trained on the same data.*

**Status: REFUTED-IN-GENERAL.** **Progress: 25 %.** We now have a test (the geometric difference) for
whether an advantage is even possible on a dataset, and engineered datasets that pass it in
simulation. No natural dataset is known to pass it.

- Huang et al. show that classical ML **with data** can predict functions that are classically hard to
  compute. They give prediction-error bounds and the "geometric difference" g between kernels. The
  classical model's error bound is at most g times the quantum model's, so a small g rules out a
  large quantum advantage for any labels (a modest one is not excluded). The fidelity quantum
  kernel tends to the identity matrix as the effective dimension grows, so it generalises poorly. They introduce
  *projected* quantum kernels. On **engineered** datasets the projected kernel beats the best tuned
  classical model by more than 20 % in accuracy (simulations up to 30 qubits), and this advantage
  shrinks as the training set grows. arXiv:2011.01938, *Nat. Commun.* **12**, 2631 (2021) `[F]`
- Huang, Kueng & Preskill: for predictions that are accurate **on average** over inputs, classical ML
  needs about as many runs of a quantum experiment as the best quantum ML. Exponential separations
  appear only for accuracy on **all** inputs. arXiv:2101.02464, *PRL* **126**, 190505 (2021) `[A]`
- Huang et al.: classical ML trained on measurement data efficiently predicts ground-state properties
  of gapped Hamiltonians in finite dimensions. Even "quantum" problems can be learned classically
  from data. arXiv:2106.12627, *Science* **377** (2022) `[A]`
- *New (session 2).* The same geometric test applied to photonic boson-sampling kernels: a large
  geometric separation from classical kernels gives an accuracy advantage of +0.22 ± 0.05 on
  adversarially constructed tasks, "an order of magnitude smaller on natural tasks". Kar & Babu H,
  arXiv:2610.11360 (Oct 2026 preprint) `[A]`
- **This repository (session 2).** On the fraud data, the geometric difference between each of our
  four tuned quantum kernels (entangled and unentangled) and classical kernels is g = 1.3–4.9 on
  every seed, against √N = 37.4. That is far inside the "classical ML predicts similar or better"
  region of Huang et al.'s test, for *any* labels on these inputs. The closest classical kernel is
  usually a wide Gaussian (RESULTS.md).

## Milestone 3: Exponential speed-ups from quantum linear algebra survive fair comparison

> *HHL-style QML algorithms (recommendation systems, PCA, regression) are exponentially faster than
> the best classical algorithms given comparable access to the data.*

**Status: REFUTED-IN-GENERAL** (for low-rank problems under QRAM-style input assumptions).
**Progress: 10 %.** Dequantisation showed that these exponential speed-ups came from the input
model. What is left is polynomial speed-ups or other regimes, which still need QRAM and fault
tolerance.

- Aaronson lists four caveats that each can kill HHL's exponential speed-up: loading the input vector
  quickly (QRAM), applying e^(−iAt), the condition number, and reading out the answer. *Nat. Phys.*
  **11**, 291 (2015), doi:10.1038/nphys3272 `[F]` (author's PDF; DOI record checked)
- Tang gives a classical algorithm that, with ℓ²-norm sampling access to the matrix (the classical
  analogue of the quantum algorithm's data structure), samples recommendations only polynomially
  slower than Kerenidis–Prakash. The polynomial exponents are very large. arXiv:1807.04271, STOC 2019,
  doi:10.1145/3313276.3316310 `[F]`
- Tang does the same for quantum PCA and nearest-centroid clustering. arXiv:1811.00414, *PRL* **127**,
  060503 (2021) `[A]`
- Chia et al. give a general classical framework for the quantum singular value transformation under
  sampling assumptions. It gives "compelling evidence" that, in the QRAM input model, quantum SVT
  yields no exponential speed-up. arXiv:1910.06151, STOC 2020 `[A]`
- Arrazola et al. find quantum-inspired algorithms do better in practice than their bounds suggest,
  but degrade as rank and condition number grow. Dequantisation is a statement about complexity, not
  a practical replacement. arXiv:1905.10415, *Quantum* **4**, 307 (2020) `[A]`

## Milestone 4: Variational models that are trainable *and* not classically simulable

> *Parameterised quantum circuits can be trained at scale (no barren plateaus) while computing
> something a classical computer cannot.*

**Status: CONTESTED.** **Progress: 20 %.** For the standard loss-based models, the known ways to
avoid barren plateaus also seem to make the model classically simulable. A 2025 generative-model
result claims both properties together, but for target distributions that are themselves quantum
circuits.

- McClean et al.: for random parameterised circuits approximating 2-designs, gradients have zero mean
  and variance exponentially small in the number of qubits ("barren plateaus"). Numerics go up to 24
  qubits. arXiv:1803.11173, *Nat. Commun.* **9**, 4812 (2018) `[F]`
- Global cost functions cause barren plateaus even in shallow circuits (Cerezo et al., arXiv:2001.00550,
  *Nat. Commun.* **12**, 1791 (2021) `[A]`). So does hardware noise when depth grows with qubit count
  (Wang et al., arXiv:2007.14384, *Nat. Commun.* **12**, 6961 (2021) `[A]`). Review: Larocca et al.,
  arXiv:2405.00781, *Nat. Rev. Phys.* **7**, 174 (2025) `[A]`.
- Cerezo et al. collect case-by-case evidence that models with **provable** absence of barren plateaus
  live in polynomially small subspaces. Their losses are then classically simulable, sometimes after a
  quantum "data acquisition" phase. The authors list caveats: it is not a theorem; contrived
  counterexamples exist; warm starts; polynomial advantages. arXiv:2312.09121, *Nat. Commun.* **16**,
  7907 (2025) `[F]`
- Quantum convolutional neural networks turn out to be effectively classically simulable (Pauli
  shadows, up to 1,024 qubits) on the "locally easy" datasets they are usually benchmarked on.
  Bermejo et al., arXiv:2408.12739, *PRX Quantum* **7**, 020304 (2026) `[A]`
- **Counterpoint.** Huang et al. (Google Quantum AI) introduce generative models with these
  properties: training is efficient (in fact classical), there are no barren plateaus, and sampling is
  classically hard unless the polynomial hierarchy collapses. They run them on a 68-qubit
  superconducting processor, equivalent to 816 "shallow" qubits. The target distributions are
  generated by quantum circuits with chosen parameters. The classical baselines are transformers and
  XGBoost. Preprint, not yet peer reviewed. arXiv:2509.09033 `[F]`
- Generative models whose outputs anti-concentrate are untrainable on average, even those with a
  quantum advantage. Herbst et al., arXiv:2512.24801 `[A]`

## Milestone 5: Quantum models beat tuned classical baselines on standard classical datasets

> *On ordinary classical data (tabular, images), a quantum model outperforms well-tuned classical
> models under a fair protocol.*

**Status: REFUTED-IN-GENERAL** (at the scales anyone can test today; we know of no fair-protocol
counterexample). **Progress: 5 %.** Every large, careful comparison we found has the classical
baselines matching or beating the quantum models.

- Bowles, Ahmed & Schuld test 12 popular QML models on 160 datasets from 6 data-generating
  procedures, with hyperparameter search for every model. Out-of-the-box classical models
  **systematically** outperform the quantum ones, and removing entanglement usually does not hurt.
  Their literature review: of 55 arXiv papers with "outperform" in the abstract, about 40 % claimed a
  quantum model beat a classical one, and 3 papers (4 %) reported that it did not. arXiv:2403.07059
  `[F]`
- With a tuned bandwidth, quantum kernels stop degrading with qubit count and become "competitive
  with the best classical methods", not better. Shaydulin & Wild, arXiv:2111.05451, *PRA* **106**,
  042407 (2022) `[F]`
- 2026 studies point the same way. Hybrid quantum-classical networks on medical data match their
  classical counterparts at best and are usually worse (Freinberger & Moser, arXiv:2601.04732 `[A]`).
  Seven quantum/classical model pairs: the quantum models do not surpass the classical baselines
  (Yu et al., arXiv:2607.01197 `[A]`). Network-intrusion detection: tuned random forests and XGBoost
  match or beat quantum models on all four datasets; two narrow effects survive multiple-testing
  correction (Gillani et al., arXiv:2608.18155 `[A]`).
- *New (session 2), a claim that does not meet the bar of this map.* 1,368 QNN configurations for
  GBP/USD exchange-rate forecasting; the best is reported to outperform "a classical BiLSTM baseline".
  According to the abstract, that is the only classical comparison. Runs on IQM hardware show gate
  errors and decoherence as a significant barrier. Waller et al., arXiv:2610.12148 (Oct 2026
  preprint) `[A]`
- **This repository** (credit-card fraud, 8 qubits, 5 seeds, session 1): no quantum model beats any
  tuned classical model. The best classical models reach 0.918 AUPRC and the best quantum model 0.913,
  a gap inside the seed-to-seed noise, and the variational classifier is last at 0.899. See
  [RESULTS.md](RESULTS.md).
- **This repository, sessions 2–3.** Removing entanglement leaves every quantum model's AUPRC
  unchanged, within noise:
  - fidelity kernel: 0.912 without vs 0.911 with;
  - projected kernel: 0.917 vs 0.913;
  - VQC with the same standardised readout on both sides: 0.902 vs 0.890.

  This reproduces Bowles et al.'s "entanglement does not help" finding on real data.

## Milestone 6: Quantum kernels that generalise, stay estimable, and are not classically easy

> *Some quantum kernel for classical data (a) does not concentrate exponentially, (b) generalises,
> and (c) cannot be efficiently computed or approximated classically, on a relevant dataset.*

**Status: OPEN.** **Progress: 20 %.** Each property can be had alone. Getting all three at once on a
useful dataset has not been shown.

- Quantum kernel values can concentrate exponentially in the number of qubits. With polynomially many
  shots, the trained model's predictions then no longer depend on the input. Four causes:
  expressive embeddings; the global measurement inherent to the fidelity kernel (even for unentangled
  embeddings); entanglement (which hurts *projected* kernels); and noise. Thanasilp et al.,
  arXiv:2208.11060 `[F]` (journal version NOT VERIFIED this session)
- If the embedding is too expressive, fidelity-kernel ridge regression provably cannot learn anything
  from polynomially many samples. An advantage needs a bias towards a small class of functions that
  are classically hard to compute, and such "biased" kernels may need exponentially many
  measurements. The authors conclude there is "no indication" QML will substantially improve
  supervised learning on classical datasets. Kübler, Buchholz & Schölkopf, arXiv:2106.03747, NeurIPS
  2021 `[F]`
- Tuning the bandwidth (input scaling) removes the concentration and restores generalisation
  (Shaydulin & Wild, arXiv:2111.05451 `[F]`; Canatar et al., arXiv:2206.06686 `[A]`). *Our note,
  from the circuit definition:* in the IQP map the pair terms scale as s² and the single-qubit terms
  as s. A small bandwidth s therefore pushes the embedding towards a product (unentangled) state,
  which is exactly the regime where classical approximation becomes easier.
- Random Fourier features classically approximate many variational/kernel models. Sweke et al. give
  necessary and sufficient conditions for when this works (arXiv:2309.11647, *Quantum* **9**, 1640
  (2025) `[A]`). See also Landman et al., arXiv:2210.13200 `[A]`.
- A structured route: covariant quantum kernels for data with group structure, run on 27 qubits of a
  superconducting processor. Glick et al., arXiv:2105.03406, *Nat. Phys.* **20**, 479 (2024) `[A]`
- *New (session 2).* A classical neural network that learns coefficients on the same finite Fourier
  support as data-reuploading QNNs ("Neural Fourier Surrogates") is competitive with classical
  baselines and comparable to the QNNs on tabular benchmarks. Knitter, Mei & Kim, arXiv:2610.00841
  (Sep 2026 preprint) `[A]`
- *New (session 2), same dataset as ours.* One-class quantum-kernel anomaly detection on the ULB
  credit-card fraud data finds no severe exponential concentration over the qubit counts tested.
  Nyström approximation with five landmarks keeps 94–97 % of the exact-kernel average precision.
  Germain, Goldweber & Rodriguez, arXiv:2610.04190 (Oct 2026 preprint) `[A]`
- **This repository:** on the fraud data at bandwidth s = 1, the mean off-diagonal fidelity-kernel
  value falls from 0.75 (2 qubits) to 0.11 (10 qubits): the onset of concentration. At s ≤ 0.3 there
  is none up to 10 qubits, and cross-validation always chose s ≤ 0.3 (RESULTS.md).
- **This repository, session 3: finite shots.**
  - The fidelity kernel loses about 0.06 AUPRC at 100 shots per entry, and still trails the exact
    kernel on all 5 seeds at 10,000 shots (0.888 vs 0.911).
  - The projected kernel matches its exact version at 10,000 shots.
  - *Our reading:* the small bandwidths that avoid concentration make every fidelity-kernel value
    close to 1, so shot noise swamps the small differences between them. Escaping concentration by
    bandwidth trades one fragility for another. This is consistent with Thanasilp et al.'s warning
    on shot cost.

## Milestone 7: Loading classical data into quantum states (QRAM)

> *Large classical datasets can be loaded into quantum states fast enough for exponential-speed-up
> algorithms.*

**Status: OPEN** (no scalable QRAM exists, and there are strong arguments against cheap passive
QRAM). **Progress: 10 %.** Proposals exist, but cheap, scalable, error-tolerant QRAM has not been
built, and its cost may cancel the speed-ups it is meant to enable.

- If preparing the input state takes n^c steps, HHL's exponential speed-up "vanishes in the very first
  step". QRAM must be *passive*: if every stored number needs its own processor, the problem is
  solvable classically in parallel. Aaronson, doi:10.1038/nphys3272 `[F]`
- The bucket-brigade QRAM architecture needs O(log N) switches per memory call. Giovannetti, Lloyd &
  Maccone, arXiv:0708.1879, *PRL* **100**, 160501 (2008) `[A]`
- With *active* QRAM, the same hardware could run a massively parallel classical algorithm just as
  fast, and most asymptotic advantages of quantum linear algebra disappear. Every non-circuit
  (passive) QRAM proposal examined falls short somewhere. Cheap, scalable passive QRAM is "unlikely
  with existing proposals", though not provably impossible. Jaques & Rattew, arXiv:2305.10310,
  *Quantum* **9**, 1922 (2025) `[A]`
- Even with superposition queries, preparing the state of an arbitrary length-n vector takes Ω(√n)
  time (a consequence of the quantum search lower bound). Tang, arXiv:1807.04271 `[F]`
- *New (session 2).* A qRAM design based on quantum walkers uses a number of walkers linear in the
  address and message sizes, with O(n + m) circuit depth in its short-range variant. It is a
  theoretical proposal, not hardware. De Riso, Catalano & Lloyd, arXiv:2609.40020 (Sep 2026
  preprint) `[A]`
- *New (session 2).* Problem structure can make state preparation cheap enough for a speed-up to
  survive end-to-end costing: for facility location, structure-aware encoding restores a quadratic
  speed-up that structure-agnostic encoding loses to dequantisation. This is optimisation, not ML.
  Jiang, Zhang & Xiang, arXiv:2609.40105 (Sep 2026 preprint) `[A]`

## Milestone 8: An advantage when learning from quantum data or quantum-enhanced experiments

> *When the data comes from quantum systems and the learner can process it coherently, it needs
> exponentially fewer experiments than any measure-first classical learner, on a task of scientific
> value.*

**Status: DEMONSTRATED** (proofs and experiments for specific tasks; scientific usefulness still
OPEN). **Progress: 45 %.** This is the most credible direction: the separations are proven
information-theoretically, not from complexity conjectures, and have been observed on hardware. The
remaining gap is turning them into a measurement scientists need, with real sensors feeding quantum
memory.

- Without quantum memory, shadow tomography of M observables on an n-qubit state needs
  Ω(min(M, 2ⁿ)) copies, and there are exponential separations between learners with and without
  quantum memory. Chen, Cotler, Huang & Li, arXiv:2111.05881 (FOCS 2021) `[A]`
- Predicting all Pauli expectation values of an n-qubit state takes 2^Ω(n) copies for classical ML and
  O(n) for quantum ML. Huang, Kueng & Preskill, arXiv:2101.02464 `[A]`
- Experiment: with up to 40 superconducting qubits and 1,300 gates, quantum-enhanced learning needs
  exponentially fewer experiments than conventional learning, for predicting properties, quantum PCA
  and learning dynamics. Huang et al., arXiv:2112.00778, *Science* **376**, 1182 (2022),
  doi:10.1126/science.abn7293 `[A]`
- Simulations of current hardware, with its noise, suggest a clear separation already at 30–40 noisy
  qubits. The bottleneck becomes data acquisition: a measure-first strategy would need months to
  years of measurements to match. Danaci et al., arXiv:2605.21346 (2026, preprint; simulation, not
  experiment) `[A]`
- Coupling a single controllable qubit to a conventional sensor exponentially reduces the
  measurements needed to learn *classical* signals, with a 10⁷-fold reduction measured on a
  superconducting cavity–qubit device. Kannan et al., arXiv:2608.13521 (Aug 2026 preprint, not yet
  peer reviewed) `[A]`
- *New (session 2), a caveat on access models.* In quantum PAC learning, forward-only access to the
  unitary that prepares the data state gives no asymptotic advantage over classical data or state
  copies; access to its inverse is essential for the known improvement. Isogai, Yoshida & Murao,
  arXiv:2609.38403 (Sep 2026 preprint) `[A]`

## Milestone 9: Hardware that runs QML circuits at useful scale (noise, shots, error mitigation)

> *Quantum hardware can run QML circuits large enough to matter, with errors controlled well enough
> that the results beat classical simulation.*

**Status: OPEN.** **Progress: 20 %.** Error correction is now below threshold on small codes, but
useful QML needs many logical qubits. Meanwhile the cost of error mitigation grows exponentially with
depth, and classical simulation keeps catching up with "beyond-classical" experiments.

- NISQ devices are limited by gate noise. Preskill, arXiv:1801.00862, *Quantum* **2**, 79 (2018) `[A]`
- Error mitigation has a sampling overhead that grows exponentially with circuit depth for layered
  circuits under local depolarising noise (Takagi et al., arXiv:2109.04457, *npj QI* **8**, 114 (2022)
  `[A]`). In the worst case it needs superpolynomially many samples even at shallow depth (Quek et
  al., arXiv:2210.11505, *Nat. Phys.* **20**, 1648 (2024) `[A]`).
- Noise causes barren plateaus when depth grows linearly with qubit count. Wang et al.,
  arXiv:2007.14384 `[A]`
- Shot cost: an exponentially concentrated kernel needs exponentially many shots to estimate.
  Thanasilp et al., arXiv:2208.11060 `[F]`
- **This repository, session 3.** At 1,000 shots per entry, estimating the kernels of a single
  fitted fidelity-kernel model on 1,400 + 600 points takes about 1.8 × 10⁹ circuit executions; a
  projected-kernel model takes 6 × 10⁶. That is before any hyperparameter search, for an
  8-qubit model that a laptop simulates exactly in seconds.
- A 127-qubit "utility" experiment (*Nature* **618**, 500 (2023)) was simulated classically, more
  accurately than the quantum processor, with tensor networks and belief propagation. Tindall et al.,
  arXiv:2306.14887, *PRX Quantum* **5**, 010308 (2024) `[A]`. A 2026 review covers this ongoing
  competition. Kshetrimayum et al., arXiv:2603.18825 `[A]`
- Surface-code memories now run below threshold. Logical error falls by a factor of 2.14 each time the
  code distance grows by two, reaching 0.143 % error per cycle with 101 qubits at distance 7. Google
  Quantum AI, arXiv:2408.13687, *Nature* **638**, 920 (2025) `[A]`

---

## What remains to be done

The five most important open problems, in our judgement:

1. **A learning separation on a *natural* classical-data task that survives tuned classical
   baselines and classical surrogates.** Known separations are cryptographic or engineered
   (arXiv:2306.16028, arXiv:2010.02174). Benchmarks on real data favour classical models
   (arXiv:2403.07059). Data can lift classical models (arXiv:2011.01938). Milestones 1, 2, 5.
2. **Embeddings that avoid concentration without becoming classically easy.** We need kernels and
   circuits with a problem-specific inductive bias that classical models cannot copy
   (arXiv:2208.11060, arXiv:2106.03747, arXiv:2309.11647, arXiv:2105.03406). Milestone 6.
3. **Trainable-but-not-simulable models beyond quantum-generated targets.** The generative result
   (arXiv:2509.09033) needs independent replication and targets that come from the world rather than
   from a chosen quantum circuit. The simulability argument (arXiv:2312.09121) needs either a proof
   or a counterexample of practical interest. Milestone 4.
4. **Data access.** Either a QRAM whose cost does not cancel the speed-up, or problems where the data
   is generated inside the quantum computer (arXiv:2305.10310, doi:10.1038/nphys3272,
   arXiv:1807.04271). Milestone 7.
5. **An end-to-end, scientifically useful quantum-data learning advantage on hardware.** This means
   real sensors or quantum simulators feeding quantum memory, with errors under control
   (arXiv:2112.00778, arXiv:2605.21346, arXiv:2608.13521, arXiv:2408.13687, arXiv:2210.11505).
   Milestones 8, 9.

## Overall estimate

**About 15 % of the way to a useful quantum advantage in machine learning.**

We split the question in two, because the answer is very different for each half:

- **Classical data** (tables, images, text, the fraud data in this repository): **about 5 %.**
  Proven separations exist only for engineered tasks (M1). The "hard quantum process" and
  linear-algebra routes are refuted in general (M2, M3). Kernels and variational models run into
  concentration, barren plateaus and dequantisation (M4, M6). Data loading is unsolved (M7).
  The careful benchmarks we reviewed, ours included, find classical models ahead or level (M5).
- **Quantum data and quantum-enhanced experiments: about 40 %.** Separations are
  information-theoretic and have been seen on hardware (M8). What is missing is a task scientists
  need, at scale, with reliable hardware (M9).

The overall 15 % weights the quantum-data route more heavily, because it is where real progress has
been made, while noting that most of the public narrative is about classical data. These numbers are
the agent's judgement, reviewed by Athmane Mekhalfia, and will move as the weekly literature watch
adds evidence.

---

## Sources

Opened on 2026-10-12 in session 1 unless marked session 2. `[F]` = full text read, `[A]` = abstract + record read.

| Source | Venue (as listed by arXiv/DOI record) | Level | Used in |
|---|---|---|---|
| Liu, Arunachalam, Temme, arXiv:2010.02174 | *Nat. Phys.* 2021, doi:10.1038/s41567-021-01287-z | F | M1 |
| Gyurik, Dunjko, arXiv:2306.16028 | preprint | A | M1 |
| Molteni, Gyurik, Dunjko, arXiv:2405.02027 | preprint | A | M1 |
| Lewis, Gilboa, McClean, arXiv:2503.20879 | arXiv lists *Nat. Commun.* 17, 1341 | A | M1 |
| Jerbi et al., arXiv:2306.00061 | *Nat. Commun.* 15, 5676 (2024) | A | M1 |
| Huang et al., arXiv:2011.01938 | *Nat. Commun.* 12, 2631 (2021) | F | M2, M5 |
| Huang, Kueng, Preskill, arXiv:2101.02464 | *PRL* 126, 190505 (2021) | A | M2, M8 |
| Huang et al., arXiv:2106.12627 | *Science* 377 (2022) | A | M2 |
| Aaronson, "Read the fine print" | *Nat. Phys.* 11, 291 (2015), doi:10.1038/nphys3272 | F | M3, M7 |
| Tang, arXiv:1807.04271 | STOC 2019, doi:10.1145/3313276.3316310 | F | M3, M7 |
| Tang, arXiv:1811.00414 | *PRL* 127, 060503 (2021) | A | M3 |
| Chia et al., arXiv:1910.06151 | STOC 2020, doi:10.1145/3357713.3384314 | A | M3 |
| Arrazola et al., arXiv:1905.10415 | *Quantum* 4, 307 (2020) | A | M3 |
| McClean et al., arXiv:1803.11173 | *Nat. Commun.* 9, 4812 (2018) | F | M4 |
| Cerezo et al., arXiv:2001.00550 | *Nat. Commun.* 12, 1791 (2021) | A | M4 |
| Wang et al., arXiv:2007.14384 | *Nat. Commun.* 12, 6961 (2021) | A | M4, M9 |
| Larocca et al., arXiv:2405.00781 | *Nat. Rev. Phys.* 7, 174 (2025) | A | M4 |
| Cerezo et al., arXiv:2312.09121 | *Nat. Commun.* 16, 7907 (2025) | F | M4 |
| Bermejo et al., arXiv:2408.12739 | *PRX Quantum* 7, 020304 (2026) | A | M4 |
| Huang et al., arXiv:2509.09033 | preprint (2025) | F | M4 |
| Herbst, Brandić, Pérez-Salinas, arXiv:2512.24801 | preprint (2025) | A | M4 |
| Bowles, Ahmed, Schuld, arXiv:2403.07059 | preprint (2024) | F | M5 |
| Shaydulin, Wild, arXiv:2111.05451 | *PRA* 106, 042407 (2022) | F | M5, M6 |
| Freinberger, Moser, arXiv:2601.04732 | preprint (2026) | A | M5 |
| Yu et al., arXiv:2607.01197 | preprint (2026) | A | M5 |
| Gillani et al., arXiv:2608.18155 | preprint (2026) | A | M5 |
| Thanasilp et al., arXiv:2208.11060 | journal version NOT VERIFIED | F | M6, M9 |
| Kübler, Buchholz, Schölkopf, arXiv:2106.03747 | NeurIPS 2021 | F | M6 |
| Canatar et al., arXiv:2206.06686 | preprint | A | M6 |
| Sweke et al., arXiv:2309.11647 | *Quantum* 9, 1640 (2025) | A | M6 |
| Landman et al., arXiv:2210.13200 | preprint | A | M6 |
| Glick et al., arXiv:2105.03406 | *Nat. Phys.* 20, 479 (2024) | A | M6 |
| Giovannetti, Lloyd, Maccone, arXiv:0708.1879 | *PRL* 100, 160501 (2008) | A | M7 |
| Jaques, Rattew, arXiv:2305.10310 | *Quantum* 9, 1922 (2025) | A | M7 |
| Chen, Cotler, Huang, Li, arXiv:2111.05881 | FOCS 2021 | A | M8 |
| Huang et al., arXiv:2112.00778 | *Science* 376, 1182 (2022), doi:10.1126/science.abn7293 | A | M8 |
| Danaci et al., arXiv:2605.21346 | preprint (2026) | A | M8 |
| Kannan et al., arXiv:2608.13521 | preprint (2026) | A | M8 |
| Preskill, arXiv:1801.00862 | *Quantum* 2, 79 (2018) | A | M9 |
| Takagi et al., arXiv:2109.04457 | *npj QI* 8, 114 (2022) | A | M9 |
| Quek et al., arXiv:2210.11505 | *Nat. Phys.* 20, 1648 (2024) | A | M9 |
| Tindall et al., arXiv:2306.14887 | *PRX Quantum* 5, 010308 (2024) | A | M9 |
| Kshetrimayum et al., arXiv:2603.18825 | preprint (2026) | A | M9 |
| Google Quantum AI, arXiv:2408.13687 | *Nature* 638, 920 (2025) | A | M9 |
| Hashimoto, Kawachi, arXiv:2610.01141 | preprint (2026), opened in session 2 | A | M1 |
| Kar, Babu H, arXiv:2610.11360 | preprint (2026), opened in session 2 | A | M2 |
| Waller, Liang, Makris, arXiv:2610.12148 | preprint (2026), opened in session 2 | A | M5 |
| Knitter, Mei, Kim, arXiv:2610.00841 | preprint (2026), opened in session 2 | A | M6 |
| Germain, Goldweber, Rodriguez, arXiv:2610.04190 | preprint (2026), opened in session 2 | A | M6 |
| De Riso, Catalano, Lloyd, arXiv:2609.40020 | preprint (2026), opened in session 2 | A | M7 |
| Jiang, Zhang, Xiang, arXiv:2609.40105 | preprint (2026), opened in session 2 | A | M7 |
| Isogai, Yoshida, Murao, arXiv:2609.38403 | preprint (2026), opened in session 2 | A | M8 |

Also used in DESIGN.md: Havlíček et al., arXiv:1804.11326, *Nature* 567, 209 (2019) `[A]`; Bowles et
al. and Shaydulin & Wild as above.
