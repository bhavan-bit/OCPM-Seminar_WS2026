"""One-time master feature extraction pipeline (Tier 3).

Rule 1 (Extract Once, Benchmark Fast):
Tier 1 (Node-Local) and Tier 2 (Paper Baseline) are strict column subsets of Tier 3.
Extract Tier 3 once, impute 0.0 for initial boundary NaNs, and persist to:
- data/processed/bpic17_tier3_master.parquet
- data/processed/subgraphs_k{4,8,12}.pt
"""

import argparse
from src.patch_ocpa_p5 import apply_p5_patch

# Feature Tier Definitions
TIER1_FEATURES = [
    # C5: Current Activity
    # D3: Current Requested Amount
    # P9: Service Time
    # P2: Elapsed Time
]

TIER2_FEATURES = [
    # C2: Preceding Activities
    # D1: Avg Prev Requested Amount
    # P2: Elapsed Time
    # O3: Prev Offer Count
]

TIER3_FEATURES = [
    # All of Tier 1 & Tier 2
    # P5: Patched Sync Time
    # P10: Waiting Time
    # P7: Pooling Time
    # O2: Prev Object Count
    # O5: Current Object Count
    # C3: Cumulative Activity Counts
]


def extract_master_dataset(raw_path: str, output_parquet: str, debug_sample: int = None):
    """Extracts the Tier 3 master dataset from BPIC17 OCEL.

    Args:
        raw_path: Path to raw BPIC17 log (.jsonocel or .csv).
        output_parquet: Destination parquet path.
        debug_sample: Optional limit on number of process executions for fast prototyping.
    """
    apply_p5_patch()
    print(f"Starting feature extraction from {raw_path}...")
    if debug_sample:
        print(f"Debug mode: Slicing first {debug_sample} process executions.")
    # TODO: Implement ocpa feature factory extraction & graph caching


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Master Tier 3 Features")
    parser.add_argument("--raw", default="data/raw/bpic17.jsonocel", help="Path to raw BPIC17 log")
    parser.add_argument("--output", default="data/processed/bpic17_tier3_master.parquet", help="Output path")
    parser.add_argument("--debug", type=int, default=None, help="Debug sample limit (e.g. 500)")
    args = parser.parse_args()

    extract_master_dataset(args.raw, args.output, args.debug)

