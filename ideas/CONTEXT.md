# Context to paste before the generator prompt

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

> **Mode d'emploi :** colle ce texte en premier dans une nouvelle conversation (ChatGPT, DeepSeek,
> Adaptive AI), puis colle `PROMPT_GENERATOR.md`. Mets ce fichier à jour quand FIELD_MAP.md change.

---

CONTEXT (state of quantum machine learning as of October 2026, from a verified literature map; every
claim below comes from a paper with its arXiv ID)

What is established:

1. On ordinary CLASSICAL data, careful benchmarks find tuned classical models ahead of or level with
   quantum models. Bowles, Ahmed & Schuld (arXiv:2403.07059) tested 12 QML models on 160 datasets;
   removing entanglement usually did not hurt.
2. Our own benchmark: credit-card fraud data, 8 qubits, exact simulation. No quantum model beats tuned
   classical models (RBF-SVM, gradient boosting, MLP), and unentangled versions of the quantum
   kernels score the same. The geometric difference between the quantum kernels and classical
   Gaussian kernels is g ≈ 1.3–4.9, while an advantage needs g of order √N ≈ 37. By Huang et al.'s
   criterion (arXiv:2011.01938), a classical kernel is guaranteed competitive on these inputs.
3. "Power of data" (arXiv:2011.01938): classical ML trained on data can predict functions that are
   hard to compute, so hard quantum circuits do not automatically give a learning advantage.
4. Exponential speed-ups of quantum linear-algebra ML (recommendation systems, PCA) were dequantised
   by classical sampling algorithms (arXiv:1807.04271, arXiv:1910.06151). Loading classical data
   into quantum states (QRAM) is unsolved (arXiv:2305.10310).
5. Variational circuits: random deep circuits have barren plateaus (arXiv:1803.11173). Known ways to
   avoid them tend to make the model classically simulable (arXiv:2312.09121), with caveats.
6. Quantum kernels concentrate exponentially when embeddings are expressive or noisy, so they need
   exponentially many shots (arXiv:2208.11060). Tuning the bandwidth avoids this but makes them
   behave like classical kernels (arXiv:2111.05451).
7. Provable advantages on classical data exist, but on engineered or cryptographic tasks
   (arXiv:2010.02174, arXiv:2306.16028). Newer ones use more natural distributions (arXiv:2503.20879,
   arXiv:2405.02027).
8. The most credible direction: learning from QUANTUM data, where the learner processes quantum
   states coherently (quantum memory). Exponential advantages are proven and observed on hardware
   (arXiv:2112.00778, 40 qubits). Others: quantum-enhanced sensing of classical signals
   (arXiv:2608.13521, preprint) and generative models with classically hard sampling
   (arXiv:2509.09033, preprint, 68 qubits).
9. Hardware: error mitigation has exponential costs (arXiv:2210.11505). Error correction is now below
   threshold on small codes (arXiv:2408.13687).

Our testing constraints: ideas must be testable with ≤ 10 qubits in exact classical simulation, on a
laptop CPU, in ≤ 10 minutes per experiment, against TUNED classical baselines. Alternatively, they
must make a precise theoretical claim that can be checked.
