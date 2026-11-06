"""Topological feature extraction & Subgraph Entanglement Stratification.

Computes metrics on induced prefix subgraphs of size k in {4, 8, 12}:
1. Branching & Join Excess: delta_deg = sum(max(0, deg^- - 1)) + sum(max(0, deg^+ - 1))
2. Undirected 1st Betti Number: betti_1 = max(0, |E| - |V| + c)
3. Historical Offer Multiplicity: O3_offer

Stratifies test instances into:
- Regime 1: Single-Path Window (delta_deg == 0)
- Regime 2: 2-Lane Type Fork (delta_deg > 0 and o3_offers <= 1)
- Regime 3: Multi-Instance Entangled (delta_deg > 0 and o3_offers >= 2)
"""

import networkx as nx
import pandas as pd


def compute_entanglement_row(subgraph: nx.DiGraph, o3_offer_count: int, k: int) -> dict:
    """Computes topological complexity and assigns an entanglement regime.

    Args:
        subgraph: Induced prefix DiGraph of size k.
        o3_offer_count: Cumulative offer count seen up to target event.
        k: Subgraph horizon size.

    Returns:
        dict: Entanglement metrics and assigned regime string.
    """
    n_nodes = subgraph.number_of_nodes()
    n_edges = subgraph.number_of_edges()

    in_excess = sum(max(0, d - 1) for _, d in subgraph.in_degree())
    out_excess = sum(max(0, d - 1) for _, d in subgraph.out_degree())
    delta_deg = in_excess + out_excess

    n_comp = nx.number_weakly_connected_components(subgraph) if n_nodes > 0 else 0
    betti_1 = max(0, n_edges - n_nodes + n_comp)

    # Regime classification aligned with BPIC17 Object Lanes
    if delta_deg == 0:
        regime = "Regime 1: Single-Path Window (No Fork/Join)"
    elif o3_offer_count <= 1:
        regime = "Regime 2: 2-Lane Type Fork (1 App + 1 Offer)"
    else:
        regime = "Regime 3: Multi-Instance Entangled (1 App + 2+ Offers)"

    return {
        "k": k,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "delta_deg": delta_deg,
        "betti_1": betti_1,
        "o3_offers": o3_offer_count,
        "regime": regime,
    }

