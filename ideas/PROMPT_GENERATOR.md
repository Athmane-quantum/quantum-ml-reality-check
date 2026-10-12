# Prompt: generate ideas

*Built by AI agents (Claude), directed and reviewed by Athmane Mekhalfia.*

> **Mode d'emploi :** nouvelle conversation → colle `CONTEXT.md`, puis tout le texte sous la ligne.
> Fais-le séparément dans ChatGPT, DeepSeek et Adaptive AI. Récupère les réponses telles quelles et
> donne-les à `PROMPT_NOVELTY_FILTER.md` (ChatGPT, nouvelle conversation), puis à Claude.

---

You are helping a research project that asks where quantum machine learning can genuinely beat strong
classical methods. Using the CONTEXT above, propose **8 research ideas** that have a real chance of
showing a quantum advantage in machine learning, or of settling an open question about one.

Rules:

1. Do NOT propose ideas that the context already rules out unless you explain precisely how your idea
   escapes the problem. That covers, for example, a fixed generic quantum feature map on classical
   tabular data, HHL-style speed-ups on classical data, deep random variational circuits, and
   comparisons against weak classical baselines.
2. Prefer directions with a mechanism a classical model cannot copy: quantum data or sensing, problem
   structure (symmetries, physics), provable separations on natural distributions, or classically hard
   sampling.
3. Every idea must have a minimal test that fits ≤ 10 qubits of exact simulation on a laptop CPU in
   ≤ 10 minutes, compared against tuned classical baselines. Alternatively, it must be a precise
   theoretical statement that could be proven or refuted.
4. Do NOT invent references. If you cite a paper, give its arXiv ID only if you are confident it is
   correct; otherwise write "reference unknown". Wrong references will be checked and the idea
   penalised.
5. Be honest about weaknesses. An idea with a clearly stated risk is more useful than an oversold one.

Output format: for EACH idea, fill exactly these fields.

- Title (≤ 12 words):
- Hypothesis (one sentence: which quantum model beats which classical model on which task):
- Mechanism (why quantum could win, and why a classical model could not simply imitate it):
- Data (classical, or quantum: states or measurements from a quantum system):
- Minimal test (dataset, quantum model, classical baselines, metric, and the result that counts as
  success):
- Main risk (the most likely reason it fails, with reference to the context):
- Closest prior work you know (arXiv IDs or "reference unknown"):
