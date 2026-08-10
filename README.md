# Blackbox OS: A Reliable, Sandboxed Execution Runtime & Modular Orchestrator for LLM Agents at Scale

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21413144.svg)](https://doi.org/10.5281/zenodo.21413144)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Blackbox OS** is a graph-based agentic operating system designed for quantitative research, data science, and algorithmic trading. Rather than relying on monolithic context windows—which suffer from severe routing degradation and execution collapse as the tool library scales—Blackbox OS orchestrates specialized agents across modular sub-graphs, executing calculations via a containerized **Sandbox Delegation Pattern** and protecting pipelines with **Dynamic Validation Guardrails**.

---

## 🔬 Key Scientific & Empirical Discoveries

### 1. The Skill Phase Transition (Routing Collapse)
* **Monolithic Routing Collapse:** Injecting all available tool schemas into a single LLM prompt suffers from non-linear accuracy collapse beyond $N \sim 60$ tools due to semantic conflation.
* **Geometric Origin:** Semantic packing analysis demonstrates that as catalog size $N$ increases ($15 \to 1000$), the mean nearest-neighbor cosine similarity ($\bar{S}_{\text{NN}}$) between target tools and distractors rises from $0.3500$ to $0.5100$. When similarity crosses the critical **$0.50\text{--}0.53$ threshold**, self-attention heads fail, causing routing accuracy to drop sharply (e.g., GPT-OSS 20B falling to $46.7\%$).
* **SOP Sub-Graph Bounding:** Bounding active contexts to $K \le 15$ tools per node via Process Templates (SOPs) keeps local cosine similarity bounded below $\bar{S}_{\text{NN}} \le 0.3559$, eliminating semantic attention saturation.

### 2. The Two-Stage Execution Collapse
Even when correctly routed, LLMs fail to execute tools reliably under context pressure due to two distinct failure modes:
* **Schema Collapse:** Bare prompts fail to adhere to JSON structures in up to $73\%$ of trials, omitting required keys or returning malformed outputs.
* **Arithmetic Collapse:** When schemas are enforced (via structured outputs), LLMs frequently fail at floating-point calculations and statistical aggregation (mental math hallucinations).
* **Mitigation:** Enforcing expert **Process Templates (SOPs)** completely eliminates schema collapse, while delegating calculations to a **Python Sandbox** bypasses the mental math bottleneck, lifting E2E success from $25\%$ to $>90\%$.

### 3. Robustness Boundaries & Fracture Points
* **Noise Gradient ($L_0\text{--}L_5$):** Agents remain robust to typos, fluff, and prompt reordering ($L_0\text{--}L_4$ success at $97\text{--}100\%$). However, they fracture under adversarial prompt injections ($L_5$), collapsing to $0\%$ success. Adding a **Script-Integrity Guardrail** restores success to $100\%$.
* **Query Variation Matrix ($2 \times 2$):** High phrasing novelty or high semantic ambiguity alone do not degrade routing. However, their combination causes routing accuracy to drop to $70\%$ as the agent is pulled toward semantically adjacent tools.

---

## 🛠️ Core Agentic Architectures

### 1. The Sandbox Delegation Pattern & Self-Healing Loop
Rather than outputting direct text predictions, agents generate self-contained Python scripts executed in an isolated runtime environment. If execution fails (e.g., `KeyError`, `ZeroDivisionError`), the traceback is returned to the agent in a corrective feedback loop for automated repair.
