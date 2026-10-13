# Prompts — dépôt quantum-ml-reality-check

Deux prompts.

- **Le premier sert une seule fois.** C'est la session d'initialisation, à lancer avec les crédits cloud.
- **Le second sert chaque vendredi,** avec tes crédits normaux.

Colle-les tels quels dans une session Claude Code sur le web, attachée au dépôt `Athmane-quantum/quantum-ml-reality-check` avec le droit d'écriture.

---

## PROMPT 1 — Initialisation (une seule fois)

```
You are initializing the public GitHub repository Athmane-quantum/quantum-ml-reality-check.
Owner and director: Athmane Mekhalfia (18, self-taught in quantum computing, gap year). You are an AI agent doing the build; he directs and must be able to explain every result. Work autonomously, commit often with clear messages, and push to main. Use as much of this session as needed to finish the whole initialization.

MISSION
Build an honest, living answer to: "Where does quantum machine learning (QML) actually stand against strong classical baselines, and what is left to do before a real, useful quantum advantage in ML?"
The repo has 3 parts: (1) a field map, (2) a reproducible benchmark, (3) a session journal.

NON-NEGOTIABLE RULES
1. Every factual claim in the field map cites a primary source (arXiv ID or DOI) that you actually opened in this session. If you cannot open it, write "NOT VERIFIED" next to it. Never invent a paper, an author, a number or a result.
2. Never fabricate benchmark numbers. Every number in a results table must come from code in this repo that you ran, with the seed and command written down.
3. Classical baselines must be strong and tuned (e.g., SVM with RBF kernel and grid search, gradient-boosted trees, small MLP). An untuned baseline invalidates the comparison.
4. Everything must run on a laptop CPU in under ~10 minutes per experiment: small qubit counts (≤ 10), simulators only (Qiskit / qiskit-machine-learning or PennyLane, pick ONE and justify it in DESIGN.md), small or subsampled public datasets.
5. README states clearly at the top: "Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia." Same line in every report.
6. Report negative results as clearly as positive ones. If classical wins, say so.

DELIVERABLES OF THIS SESSION
A. Repository skeleton
   - README.md (what this is, the AI-built disclaimer, how to run, current headline results, links to the 3 parts)
   - LICENSE (MIT), .gitignore, pyproject.toml (pinned versions), src/qmlrc/ package, tests/, a GitHub Actions workflow running pytest on push
   - DESIGN.md: framework choice, dataset choice, metrics, what "advantage" means here (same data, same budget, fair tuning)
B. FIELD_MAP.md — the state of the field, organized as milestones toward "useful quantum advantage in ML". For each milestone: status (DEMONSTRATED / CONTESTED / OPEN / REFUTED-IN-GENERAL), a progress estimate 0–100 % with one sentence explaining why, and 2–5 verified sources. Cover at least:
   - provable advantage on contrived/cryptographic learning tasks (e.g., Liu, Arunachalam, Temme 2021)
   - "power of data" and when classical ML catches up (Huang et al. 2021)
   - dequantization of quantum-inspired linear algebra (Tang 2018 and follow-ups)
   - trainability: barren plateaus (McClean et al. 2018) and the link between absence of barren plateaus and classical simulability (Cerezo et al. ~2023)
   - large-scale benchmarking showing classical baselines often win (Bowles, Ahmed, Schuld 2024, arXiv:2403.07059)
   - quantum kernels: exponential concentration, and when they help
   - data loading / QRAM bottleneck
   - learning from quantum data (quantum-native datasets) as the most credible advantage direction
   - hardware reality: noise, shot cost, error mitigation
   Verify every one of these yourself; fix any detail above that turns out wrong and note the correction.
   End with "What remains to be done" — the 5 most important open problems, each tied to sources, and an overall honest progress estimate with its reasoning.
C. Benchmark v0 (src/qmlrc + experiments/)
   - Dataset 1: a small public credit-card fraud style dataset or a standard tabular dataset (document source and license); subsample to keep it CPU-feasible.
   - Models: fidelity quantum kernel + SVM (ZZ-feature-map style), projected quantum kernel if feasible, one variational classifier; versus tuned RBF-SVM, gradient boosting, logistic regression.
   - Metrics: AUPRC for imbalanced data plus accuracy; 5 seeds; mean ± std; wall-clock time.
   - A single script `python -m qmlrc.run_all` that reproduces RESULTS.md from scratch.
   - RESULTS.md with the table, the exact commands, seeds, versions, and a plain-language reading of the result.
D. JOURNAL.md — first entry: date, what was built, what was verified, what failed, what the next session should do (a prioritized TODO list of 5 items).
E. NEXT_SESSION.md — the exact plan for the next weekly session, so a fresh agent can continue without any other context.

QUALITY BAR BEFORE YOU FINISH
- pytest passes; CI is green.
- run_all reproduces RESULTS.md.
- Every source in FIELD_MAP.md has been opened; unverifiable ones are marked.
- Re-read README as a skeptical admissions reviewer: nothing overstated.
Finish with a short summary (in French) of what was done, the headline result, and the top 3 items for next week.
```

---

## PROMPT 2 — Session hebdomadaire (chaque vendredi)

```
Continue the repository Athmane-quantum/quantum-ml-reality-check. You are an AI agent; Athmane Mekhalfia directs. First read README.md, DESIGN.md, FIELD_MAP.md, RESULTS.md, JOURNAL.md and NEXT_SESSION.md. Keep all the rules in README/DESIGN (verified sources only, no fabricated numbers, strong tuned classical baselines, CPU-feasible, AI-built disclaimer, negative results reported honestly).

THIS SESSION
1. Literature watch: search arXiv for QML papers from the last 2 weeks that change a milestone in FIELD_MAP.md (new advantage claims, dequantizations, benchmarks, trainability results). Open each one you use. Update statuses and progress estimates only with a written justification and source.
2. Do the top item of NEXT_SESSION.md (typically: reproduce one paper's experiment at small scale, add one dataset, add one model, or add one classical baseline), with tests.
3. Re-run run_all if code changed; update RESULTS.md only from real runs.
4. Append a JOURNAL.md entry (date, done, verified, failed, next) and rewrite NEXT_SESSION.md with the 5 next priorities.
5. pytest green, commit with clear messages, push to main.
If something cannot be verified or finished, say so in the journal rather than guessing.
End with a short summary in French: what changed, the most important new fact about the field, and what is next.
```
