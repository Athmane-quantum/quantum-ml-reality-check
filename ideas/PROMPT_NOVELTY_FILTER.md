# Prompt: merge and pre-filter ideas

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

> **Mode d'emploi :** nouvelle conversation ChatGPT → colle `CONTEXT.md`, puis tout le texte sous la
> ligne, puis toutes les idées reçues des générateurs (avec le nom de l'IA qui les a proposées).
> Donne ensuite la sortie complète à Claude, qui vérifie sur arXiv avant toute décision.

---

Below are research ideas proposed by several AI assistants (each labelled with its source). Your job
is a FIRST, conservative filter. You do not make final decisions; a second reviewer will check every
claim against a real arXiv search.

Do the following:

1. **Merge duplicates.** Ideas that are essentially the same (same mechanism, same test) become one
   entry that lists all its sources.
2. **Flag likely prior work.** For each idea, say whether you believe it has already been studied.
   Give arXiv IDs only if you are confident they are correct; otherwise write "suspected, reference
   unknown". Never invent an arXiv ID or a title.
3. **Flag conflicts with the context.** Say whether the idea runs into a known obstacle from the
   CONTEXT (power of data, dequantisation, barren plateaus, kernel concentration, data loading, weak
   baselines) and whether the idea explains how it gets around it.
4. **Check testability.** Is the minimal test feasible with ≤ 10 qubits in simulation on a laptop in
   ≤ 10 minutes against tuned classical baselines, or is it a precise theoretical claim?
5. **Do not delete anything.** Classify every idea as KEEP, DOUBTFUL or LIKELY-REJECT, with a one-line
   reason. The second reviewer may overturn your classification.

Output: one table with columns ID (number them 1, 2, …) | Title | Sources | Prior work (IDs or
"suspected, reference unknown" or "none known") | Known obstacle hit? | Testable? | Class | Reason.
After the table, give the full original text of every idea, unchanged, under its ID.
