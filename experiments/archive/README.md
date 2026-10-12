# Archived runs

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

Raw results of earlier full runs, kept so that changes to the protocol stay visible. Current results
live in `experiments/<experiment>/results.csv` and `RESULTS.md`.

| Folder | Code | What changed afterwards |
|---|---|---|
| `2026-10-12_run1/` | commit `24f7a80` | Several models, classical and quantum, picked hyperparameters on the edge of their grids. Every axis whose CV-selected value sat on an edge in at least 2 of 5 seeds was extended by one value (JOURNAL.md, 2026-10-12). Tasks also moved to single-threaded worker processes. |
| `2026-10-12_session2/` | commit `70c374d` | Session 3 extended the SVM penalty C (0.001 and 10,000) and the VQC depth (1), the last grid extensions owed under DESIGN.md §4. It replaced the `VQC`/`VQC-sep` pairing, which differed in readout and whose separable control trained unreliably, with `VQC-mean`/`VQC-sep` (same standardised readout). The ablation experiment was renamed `creditcard_q8_ablation`, and the finite-shot experiment was added. |
