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
        ├──► Patched ocpa P5 Synchronization Time
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
│   ├── patch_ocpa_p5.py            # P5 Synchronization Time boundary condition patch
│   ├── extract_features.py         # One-time master Tier 3 feature extractor
│   ├── extract_entanglement.py     # Topological complexity & regime assigner
│   ├── train_tabular.py            # OLS, RidgeCV, XGBoost, CatBoost (MAE optimization)
│   ├── train_seq_graph.py          # PyG GCN & PyTorch LSTM models
│   └── evaluate_and_shap.py        # Stratified evaluation tables & TreeSHAP
├── results/
│   ├── predictions/                # Per-event test predictions (y_true, y_pred)
│   └── figures/                    # Publication-grade figures for report and slides
├── GEMINI.md                       # Workspace agent & developer guidelines
├── AGENTS.md                       # Symlink to workspace guidelines
├── LICENSE                         # License definition
└── README.md                       # Project documentation
```

---

## ⚙️ Environment Setup

Because the original `ocpa` codebase dates from 2022, modern `pandas >= 2.0` and `networkx >= 3.0` introduce breaking API incompatibilities (e.g. `df.append()`, `nx.from_scipy_sparse_matrix`).

Create an environment with pinned dependencies:

```bash
conda create -n ocpm_seminar python=3.9 -y
conda activate ocpm_seminar

# Pinned dependencies
pip install "pandas<2.0.0" "numpy<1.24.0" "networkx<3.0" scikit-learn xgboost catboost shap
pip install torch torchvision torchaudio torch-geometric

# Install ocpa
git clone https://github.com/ocpm/ocpa
pip install -e ./ocpa
```

---

## 🔬 Experimental Workflow

### 1. Synchronization Time ($P5$) Patch
The original `ocpa` implementation conflated single-predecessor nodes ($\text{deg}^-(v) \le 1$) with Waiting Time ($P10$). Our patch enforces the mathematical boundary condition:
$$\text{deg}^-(v) \le 1 \implies P5 = 0.0$$
Apply automatically via `from src.patch_ocpa_p5 import apply_p5_patch`.

### 2. Feature Extraction (Tier 3 Master)
```bash
# Debug run (fast test on first 500 executions)
python -m src.extract_features --raw data/raw/bpic17.jsonocel --debug 500

# Full dataset extraction
python -m src.extract_features --raw data/raw/bpic17.jsonocel
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
