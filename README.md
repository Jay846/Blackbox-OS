# Blackbox OS: A Self-Healing, Sandboxed Execution Runtime for Autonomous Quant Agents

Blackbox OS is a graph-based agentic operating system designed for quantitative research, data science, and algorithmic trading. Rather than relying on static prompt reasoning or monolithic context windows—which suffer from severe routing degradation as the tool library scales—Blackbox OS orchestrates specialized agents across modular sub-graphs, executing calculations via a containerized **Sandbox Delegation Pattern** with **Self-Healing Remediation**.

---

## Key Architectures

### 1. The Sandbox Delegation Pattern
LLMs are notoriously weak at performing mental arithmetic, scaling matrices, or auditing data frames directly inside prompt contexts. Blackbox OS resolves this by delegating execution to a sandboxed Python runtime. The agent writes, tests, and refines executable scripts that parse real datasets (CSV/Parquet) and produce schema-validated JSON outputs.

### 2. Self-Healing Code Remediation
If a generated script encounters a runtime error (e.g. `KeyError`, `ZeroDivisionError`, or pandas degrees-of-freedom mismatches), the exception stack trace is fed back into a correction loop. The agent automatically rewrites and re-runs the script, achieving up to 100% execution accuracy.

### 3. Hierarchical Graph Routing (LangGraph)
As the number of tools ($N$) grows, self-attention matrices experience interference, causing routing accuracy to drop. Blackbox OS structures the tool library into isolated sub-graphs (Data Scientist, Quant Researcher, Quant Trader) where each node's local choice set is strictly bounded ($K \le 100$), preserving perfect routing resolution.

```
                      [ Root Gateway ]
                             │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   [ Data Scientist ]  [ Quant Researcher ]  [ Quant Trader ]
     (77 skills)         (100 skills)          (100 skills)
```

---

## Empirical Performance Matrix

We evaluated **OpenAI GPT-4o-mini**, **DeepSeek V4 Flash**, and **Google Gemini 3.1 Flash Lite** on a 3-node sequential pipeline (`lookahead_bias_audit -> standard_scaler_apply -> kelly_position_size`) across library sizes $N=60$ and $N=500$:

| Model | Library Size ($N$) | Prompt Math E2E Success | Sandboxed Code E2E Success | Avg Sandbox Node Time |
| :--- | :---: | :---: | :---: | :---: |
| **OpenAI GPT-4o-mini** | 60 | 0.0% | **100.0%** | 13.70s |
| | 500 | 0.0% | **100.0%** | 15.27s |
| **DeepSeek V4 Flash** | 60 | 0.0% | **100.0%** | **9.09s** |
| | 500 | 0.0% | **100.0%** | **9.50s** |
| **Gemini 3.1 Flash Lite** | 60 | 0.0% | **100.0%** | 9.59s |
| | 500 | 0.0% | **100.0%** | 9.86s |

---

## Scientific Discovery: The Skill Phase Transition

By analyzing the semantic vector space density using a local `all-MiniLM-L6-v2` transformer, we mapped the maximum cosine similarity ($S_{\max}$) between target tools and distractors as a function of library size $N$:

![Semantic Packing Density & Max Similarity vs Routing Accuracy](images/semantic_density_vs_accuracy.png)

### The Metadata Paradox
Adding rich metadata (disambiguators and examples) increases the **global vector similarity** because of shared syntax and structures. However, it **improves routing accuracy** (raising accuracy by **19.6%** at $N=1000$) because the model's self-attention mechanism utilizes the additional context to resolve fine-grained ambiguities.

---

## Directory Structure

```text
├── README.md                      # Project documentation
├── images/                        # Visualizations and plots
├── skill_experiment/              # Similarity analysis and vector data
│   ├── fillers_v4.json            # 8,605 distractor skills
│   └── targets_v4.json            # 58 target skills
└── blackbox_os/
    ├── state/                     
    │   ├── shared_state.py        # Global blackboard state
    │   └── validation_manager.py  # Schema-validation & code execution
    └── roles/
        └── data_scientist/
            └── workflows/         # Core LangGraph experiment scripts
                ├── run_production_experiment.py
                ├── run_pipeline_experiment.py
                ├── run_branching_experiment.py
                └── run_validation_sweep.py
```

---

## Reproduction Guide

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/Jay846/Blackbox-OS.git
   cd Blackbox-OS
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib sentence-transformers pandas scikit-learn pytest
   ```

### Running Experiments

1. **Run the Math vs. Sandbox Production Sweep:**
   ```bash
   python3 blackbox_os/roles/data_scientist/workflows/run_production_experiment.py --model deepseek-chat --provider deepseek
   ```

2. **Run the Multi-Node LangGraph Pipeline Sweep:**
   ```bash
   python3 blackbox_os/roles/data_scientist/workflows/run_pipeline_experiment.py --model deepseek-chat --provider deepseek
   ```

3. **Run the Dynamic Branching Orchestration Sweep:**
   ```bash
   python3 blackbox_os/roles/data_scientist/workflows/run_branching_experiment.py
   ```

4. **Verify Similarity and Phase Transition Curves:**
   ```bash
   python3 skill_experiment/characterize_phase_transition.py
   ```
