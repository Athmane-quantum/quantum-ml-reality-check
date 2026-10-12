# Archived runs

Raw results of earlier full runs, kept so that changes to the protocol stay visible. Current results
live in `experiments/<experiment>/results.csv` and `RESULTS.md`.

| Folder | Code | What changed afterwards |
|---|---|---|
| `2026-10-12_run1/` | commit `24f7a80` | Several models, classical and quantum, picked hyperparameters on the edge of their grids. Every axis whose CV-selected value sat on an edge in at least 2 of 5 seeds was extended by one value (JOURNAL.md, 2026-10-12). Tasks also moved to single-threaded worker processes. |
