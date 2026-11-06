### Summary of Changes
- Briefly describe the contribution (feature tier, model, evaluation metric).

### Related Issue / Milestone
- Resolves #<!-- issue number --> (e.g. Week 1, Week 2, Week 3, Week 4 milestone)

### Pre-Merge Invariant Checklist
- [ ] **Legacy Compatibility**: Compatible with `pandas < 2.0.0`, `numpy < 1.24.0`, and `networkx < 3.0`.
- [ ] **Deterministic Splits**: Uses fixed execution IDs from `data/splits.json` (no random splits).
- [ ] **Metric Scale Consistency**: Remaining time ($P3$) evaluated on the paper's normalized scale.
- [ ] **No Artifact Leaks**: No `.parquet`, `.pt`, or raw `.jsonocel`/`.csv` files staged in git.
- [ ] **Fast Prototyping Tested**: Verified on `--debug 500` slice before running full BPIC17.
- [ ] **Code Quality**: Formatted and linted cleanly (`ruff` / `black`).
