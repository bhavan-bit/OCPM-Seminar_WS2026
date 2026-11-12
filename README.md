# Do Graph Neural Networks Really Help?
### A Feature Richness and Subgraph Entanglement Ablation on Object-Centric Process Data

**Chair of Process and Data Science (PADS)**  
**RWTH Aachen University**  
**Dataset:** BPI Challenge 2017 (`BPIC17` Loan Application OCEL)  
**Target Variable:** Remaining Time ($P3$, standardized/normalized)

---

## 📌 Overview & Research Questions

Graph Neural Networks (GNNs) have gained significant popularity in object-centric predictive process monitoring. However, a fundamental empirical question remains:
> *Do GNNs outperform tabular baselines because of message passing over graph topology, or simply because object-centric graphs capture richer feature interactions that could be explicitly engineered into tabular models?*

This repository investigates this question via two complementary experimental designs:
1. **Design #1 (Feature Tier Ablation):** Slices feature information into **Tier 1** (Node-Local), **Tier 2** (Paper Baseline from Adams et al., 2022), and **Tier 3** (Full OCEL Graph Context / Manual Message Passing), evaluating across Linear models (OLS, Ridge), Gradient Boosted Trees (XGBoost, CatBoost), Sequence models (LSTM), and Graph Neural Networks (GCN).
2. **Design #2 (Subgraph Entanglement Stratification):** Stratifies the test distribution into topological complexity regimes (**Regime 1: Single-Path**, **Regime 2: 2-Lane Fork**, **Regime 3: Multi-Instance Entangled**) across prefix horizons $k \in \{4, 8, 12\}$.

---

## 🏛 System Architecture: "Extract Once, Benchmark Fast"

Computing graph metrics and object interactions on `BPIC17` is computationally intensive. We use a **2-Stage Decoupled Pipeline**:

```
[Stage 1: One-Time Extraction & Caching]
   BPIC17 OCEL (.jsonocel / .csv)
        │
        ├──► Extract Features via Standard OCPA Library
        ├──► Compute Full Tier 3 Feature Superset
        ├──► Extract Prefix Subgraphs (k ∈ {4, 8, 12})
        └──► Compute Topological Complexity Metadata (Δdeg, β1, O3)
        │
        ▼
   Cache: `data/processed/bpic17_tier3_master.parquet` + `subgraphs_k*.pt`

[Stage 2: Model & Stratification Benchmarking]
   Master Cached Artifacts
        │
        ├──► Design #1: Train models on column slices [Tier 1 | Tier 2 | Tier 3]
        └──► Design #2: Post-hoc join with entanglement metadata for stratified MAE
```

> **Invariant:** Feature extraction is executed **once**. Tiers 1 and 2 are strict column subsets of Tier 3 and are filtered at training time.

---

## 📂 Repository Structure

```text
.
├── data/
│   ├── raw/                        # Original BPIC17 OCEL log (.jsonocel / .csv)
│   ├── splits.json                 # Fixed deterministic train/val/test execution IDs
│   └── processed/                  # Cached parquet features and PyG subgraph tensors
├── src/
│   ├── __init__.py
│   ├── extract_features.py         # One-time master Tier 3 feature extractor
│   ├── feature_selection.py        # SHAP-based Recursive Feature Elimination (SHAP-RFE)
│   ├── extract_entanglement.py     # Topological complexity & regime assigner
│   ├── train_tabular.py            # OLS, RidgeCV, XGBoost, CatBoost (MAE optimization)
│   ├── train_seq_graph.py          # PyG GCN & PyTorch LSTM models
│   ├── evaluate_and_shap.py        # Stratified evaluation tables & TreeSHAP
│   └── sample_aoe_sessions.py      # OCEL 2.0 SQLite CTE session sampling & event extraction
├── results/
│   ├── predictions/                # Per-event test predictions (y_true, y_pred)
│   └── figures/                    # Publication-grade figures for report and slides
├── environment.yml                 # Conda environment specification (Python 3.9 + pip)
├── requirements.txt                # Pinned Python package versions for reproducibility
├── ruff.toml                       # Linter & formatter configuration (targets Python 3.9)
├── GEMINI.md                       # Workspace agent & developer guidelines
├── AGENTS.md                       # Symlink to workspace guidelines
├── LICENSE                         # License definition
└── README.md                       # Project documentation
```

---

## ⚙️ Environment Setup

Because the original `ocpa` codebase dates from 2022, modern `pandas >= 2.0` and `networkx >= 3.0` introduce breaking API incompatibilities (e.g. `df.append()`, `nx.from_scipy_sparse_matrix`). All dependencies are strictly pinned in [`requirements.txt`](file:///home/nmartin/Documents/Seminar/Project/OCPM-Seminar_WS2026/requirements.txt) and [`environment.yml`](file:///home/nmartin/Documents/Seminar/Project/OCPM-Seminar_WS2026/environment.yml).

### Option A: One-Step Conda Setup (Recommended)
```bash
conda env create -f environment.yml
conda activate ocpm_seminar
```

### Option B: Manual Conda + Pip Setup
```bash
conda create -n ocpm_seminar python=3.9 -y
conda activate ocpm_seminar
pip install -r requirements.txt
```

---

## 🔬 Experimental Workflow

### 1. Feature Extraction (Tier 3 Master via OCPA)
```bash
# Debug run (fast test on first 500 executions)
python -m src.extract_features --raw data/raw/bpic17.jsonocel --debug 500

# Full dataset extraction
python -m src.extract_features --raw data/raw/bpic17.jsonocel
```

### 2. Feature Selection (SHAP-RFE Pipeline)
```bash
# Run collinearity pruning and validation SHAP recursive feature elimination
python -m src.feature_selection
```

### 3. Model Training
```bash
# Tabular models (XGBoost optimizing MAE on Tier 3)
python -m src.train_tabular --model xgboost --tier 3

# Graph Neural Network (GCN on Tier 2 with horizon k=4)
python -m src.train_seq_graph --model gcn --tier 2 --k 4
```

### 4. Stratified Evaluation & Explainability
```bash
# Stratified MAE across Regimes 1, 2, and 3
python -m src.evaluate_and_shap --eval

# TreeSHAP feature importance analysis
python -m src.evaluate_and_shap --shap
```

### 5. OCEL 2.0 Dataset Subsampling (Age of Empires II)
Extract related events for randomly sampled session objects from an OCEL 2.0 SQLite database using a single CTE SQL query:
```bash
# Sample 5 sessions and preview events
python -m src.sample_aoe_sessions --n-sessions 5

# Sample and export to Parquet or CSV
python -m src.sample_aoe_sessions --n-sessions 5 --output data/processed/sampled_sessions.parquet
```

---

## 📊 Entanglement Regimes

Test instances are stratified into three structural regimes based on prefix subgraph topology:
* **Regime 1 (Single-Path):** Branching/Join excess $\Delta_{\text{deg}} = 0$.
* **Regime 2 (2-Lane Fork):** $\Delta_{\text{deg}} > 0$ and Offer multiplicity $O3_{\text{offer}} \le 1$.
* **Regime 3 (Multi-Instance Entangled):** $\Delta_{\text{deg}} > 0$ and Offer multiplicity $O3_{\text{offer}} \ge 2$.

---

## 📖 References & Attribution
- Adams, J. N., Park, G. Y., & van der Aalst, W. M. P. (2022). *Object-Centric Event Log Feature Extraction for Predictive Process Monitoring*.
- BPI Challenge 2017 Dataset: [4TU.ResearchData](https://doi.org/10.4121/uuid:5f3067eb-f100-45da-abb1-0ae576bc338b)
- RWTH Aachen University, Chair of Process and Data Science (PADS).
