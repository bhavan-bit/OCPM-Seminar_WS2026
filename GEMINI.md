# Workspace Rules: OCPM-Seminar_WS2026

## Project Overview
**Title:** *Do Graph Neural Networks Really Help? A Feature Richness and Subgraph Entanglement Ablation on Object-Centric Process Data*  
**Chair:** Process and Data Science (PADS), RWTH Aachen University  
**Dataset:** BPI Challenge 2017 (`BPIC17` Loan Application OCEL)  
**Target Variable:** Remaining Time ($P3$, standardized/normalized)

---

## 1. Core Architectural Invariants & Rules

### Rule 1: Extract Once, Benchmark Fast (Decoupled Pipeline)
- **Never re-extract features across tiers**: Tier 1 (Node-Local) and Tier 2 (Paper Baseline) are strict column subsets of Tier 3 (Full OCEL Graph Context).
- Extract Tier 3 **once** and save to disk (`data/processed/bpic17_tier3_master.parquet` and `data/processed/subgraphs_k{4,8,12}.pt`).
- During training, filter feature columns via Pandas column masks / PyTorch tensors.

### Rule 2: Legacy `ocpa` Compatibility & Environment Invariants
- `ocpa` codebase dates from 2022. It is **incompatible** with modern `pandas >= 2.0` (e.g., deprecated `df.append()`) and `networkx >= 3.0` (e.g., removed `from_scipy_sparse_matrix`).
- Pinned dependency constraints:
  - Python 3.9
  - `pandas < 2.0.0`
  - `numpy < 1.24.0`
  - `networkx < 3.0`
  - PyTorch & PyTorch Geometric (PyG), scikit-learn, xgboost, catboost, shap
- When adding or running scripts, always verify compatibility against these pinned libraries.

### Rule 3: The $P5$ Synchronization Time Bug Fix
- In `ocpa/algo/predictive_monitoring/event_based_features/extraction_functions.py`, ensure the mathematical definition of Synchronization Time ($P5$) is strictly respected:
  - When an event has $\le 1$ predecessor ($\text{deg}^-(v) \le 1$), return `0.0`.
  - Never conflate $P5$ with Waiting Time ($P10$: time since last predecessor).
  - For $> 1$ predecessors: $\max(\text{timestamps}) - \min(\text{timestamps})$.

### Rule 4: Deterministic Splits & Fair Benchmarking
- Always use a fixed `0.7 / 0.3` train/test split with `0.2` validation split.
- Process execution IDs (`p_id`) for `train_idx`, `val_idx`, and `test_idx` must be exported and loaded from `data/splits.json`.
- All tabular, sequence, and graph models must evaluate on the exact same test instances.
- **Target & Feature Normalization**: Standardize target remaining time ($P3$) using parameters ($z$-score) fitted **only** on the training split. Never evaluate test MAE on denormalized scales if comparing against the paper's normalized baseline (~0.7598 mean MAE).

### Rule 5: Missing Value & NaN Handling
- Initial events in an execution have no predecessors ($\text{deg}^-(v) = 0$).
- Standardize all missing/uncomputable feature values (e.g., $P10$ Waiting Time, $D1$ Previous Requested Amount) by imputing `0.0` across all models before feature scaling.

---

## 2. Experimental Designs & Methodological Standards

### Design #1: Feature Tier Ablation ($3 \text{ Tiers} \times 5 \text{ Models}$)
- **Tier 1 (Node-Local):** Current Activity ($C5$), Current Requested Amount ($D3$), Service Time ($P9$), Elapsed Time ($P2$).
- **Tier 2 (Paper Baseline):** Preceding Activities ($C2$), Avg Prev Requested Amount ($D1$), Elapsed Time ($P2$), Prev Offer Count ($O3$).
- **Tier 3 (Full OCEL Graph Context):** Superset of Tiers 1 & 2 + Patched Sync Time ($P5$), Waiting Time ($P10$), Pooling Time ($P7$), Prev Object Count ($O2$), Current Object Count ($O5$), Cumulative Activity Counts ($C3$).
- **Models & Objectives:**
  - OLS Linear Regression: Baseline replication.
  - Regularized Linear: `RidgeCV` / `LassoCV` tuned on validation split.
  - GBDTs: XGBoost (`objective="reg:absoluteerror"`) and CatBoost (`loss_function="MAE"`) with `early_stopping_rounds=50` on validation split. Split criteria must directly optimize MAE.
  - Sequence & Graph: 2-layer LSTM and 2-layer GCN with L1 Loss, Adam optimizer, checkpointed on validation MAE.
- Log both **Test MAE** and **Wall-Clock Training Time (seconds)** for every run.

### Design #2: Subgraph Entanglement Stratification
- Induced prefix subgraphs of size $k \in \{4, 8, 12\}$.
- Metrics computed per target event $e_t$:
  1. Branching & Join Excess: $\Delta_{\text{deg}} = \sum \max(0, \text{deg}^- - 1) + \sum \max(0, \text{deg}^+ - 1)$
  2. Undirected 1st Betti Number: $\beta_1 = \max(0, |E| - |V| + c)$
  3. Offer Multiplicity: $O3_{\text{offer}}$
- Stratification Regimes:
  - **Regime 1 (Single-Path):** $\Delta_{\text{deg}} = 0$
  - **Regime 2 (2-Lane Fork):** $\Delta_{\text{deg}} > 0$ and $O3_{\text{offer}} \le 1$
  - **Regime 3 (Multi-Instance Entangled):** $\Delta_{\text{deg}} > 0$ and $O3_{\text{offer}} \ge 2$
- Evaluate per-regime Test MAE for each model.

### Explainability (TreeSHAP)
- Run `shap.TreeExplainer` on Tier 3 XGBoost.
- Evaluate feature importance across Regimes 1, 2, and 3 to test the manual message-passing hypothesis.

---

## 3. Codebase Organization & Coding Standards

```text
ocpm-seminar/
├── data/
│   ├── raw/                        # Original BPIC17 OCEL log (.jsonocel / .csv)
│   ├── splits.json                 # Fixed deterministic train/val/test IDs
│   └── processed/
│       ├── bpic17_tier3_master.parquet
│       ├── entanglement_metadata.parquet
│       └── subgraphs_k{4,8,12}.pt
├── src/
│   ├── patch_ocpa_p5.py            # P5 synchronization patch
│   ├── extract_features.py         # One-time master Tier 3 extractor
│   ├── extract_entanglement.py     # Topological & regime computation
│   ├── train_tabular.py            # OLS, Ridge, XGBoost, CatBoost
│   ├── train_seq_graph.py          # PyG GCN & PyTorch LSTM
│   └── evaluate_and_shap.py        # Stratified MAE tables + TreeSHAP
├── results/
│   ├── predictions/                # Test prediction tables (.parquet / .csv)
│   └── figures/                    # Publication-ready plots for report/slides
```

### Prototyping Rule (Fast Iteration)
- When developing or debugging new extraction or training routines, always test on a small slice (e.g. first 500 process executions) to ensure execution finishes in $< 2$ minutes before running on the full BPIC17 dataset.

