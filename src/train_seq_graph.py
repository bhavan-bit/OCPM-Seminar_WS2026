"""Sequential and Graph Neural Network models (PyG GCN & PyTorch LSTM).

Evaluates across:
- Feature Tiers 1, 2, 3
- Subgraph Horizons k in {4, 8, 12}
- Trained with L1Loss (MAE), Adam optimizer, checkpointed on validation MAE.
"""

import argparse


def train_seq_graph(model_type: str, tier: int, k: int, graphs_pt: str, splits_json: str):
    """Trains an LSTM or GCN on prefix subgraphs."""
    print(f"Training {model_type} on Tier {tier} features with horizon k={k}...")
    # TODO: PyTorch / PyG training loop with validation checkpointing
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sequence/Graph Models")
    parser.add_argument("--model", choices=["lstm", "gcn"], default="gcn")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--k", type=int, choices=[4, 8, 12], default=4)
    parser.add_argument("--graphs", default="data/processed/subgraphs_k4.pt")
    parser.add_argument("--splits", default="data/splits.json")
    args = parser.parse_args()

    train_seq_graph(args.model, args.tier, args.k, args.graphs, args.splits)

