# Blackbox OS: Modular Multi-Agent Routing System

This repository implements a modular, graph-based routing architecture designed to scale tool integration horizontally without suffering from the attention degradation known as the **"Skill Phase Transition"**.

## Architecture Overview

Instead of forcing a single, monolithic LLM context to evaluate all 277 skills, Blackbox OS divides tools among isolated sub-graphs and routes queries through a hierarchical gateway:

```
                      [ Root Gateway ]
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  [ Data Scientist ]  [ Quant Researcher ]  [ Quant Trader ]
    (77 skills)         (100 skills)          (100 skills)
```

### 1. Data Scientist Sub-Graphs
Data Scientist tasks are divided into 4 sequential workflows (SOPs):
*   **SOP 1: Data Preparation & Ingestion** (Imputation, Scaling, Imbalanced Data handling)
*   **SOP 2: Feature Selection & Modeling** (Feature selection, Regularization, Ensembling)
*   **SOP 3: Optimization & Evaluation** (Hyperparameter tuning, Cross-Validation, Metrics)
*   **SOP 4: Deployment & Audit** (Leakage checks, Explainability, Drift monitoring)

---

## Directory Structure

```text
blackbox_os/
├── README.md                 # Architecture overview & documentation
├── state/
│   └── shared_state.py       # Global blackboard state
├── router/
│   ├── root_router.py        # High-level classifier (DS / QR / QT)
│   └── gateway.py            # Pre-routing heuristic gateway
└── roles/
    ├── data_scientist/
    │   ├── README.md         # DS workflow docs
    │   ├── skills/           # Skill schema definitions (77 total)
    │   └── workflows/        # LangGraph SOP execution files
    ├── quant_researcher/     # QR skills (100 total)
    └── quant_trader/         # QT skills (100 total)
```

## Setup & Running
1. Open this directory (`/Users/jaysalvi11/.gemini/antigravity/scratch/blackbox_os`) as your active workspace in your IDE.
2. Follow the detailed workflow configuration files located in the role subdirectories.
