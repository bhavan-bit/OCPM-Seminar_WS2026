"""One-time master feature extraction pipeline using the OCPA library.

Extracts the full Object-Centric Predictive Process Monitoring (OCPM) Tier 3
master feature superset from BPIC17 (or any OCEL log) using the standard OCPA library:
https://github.com/ocpm/ocpa

Core Architectural Invariant (Rule 1):
    Extract Tier 3 ONCE and persist to disk:
    - data/processed/bpic17_tier3_master.parquet
    - data/processed/subgraphs_k{4,8,12}.pt
    Tier 1 (Node-Local) and Tier 2 (Paper Baseline) are strict column subsets
    of Tier 3, filtered during training via column masks.
"""

import argparse
import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Attempt to import OCPA components
try:
    import ocpa.objects.log.importer.ocel.factory as ocel_import_factory
    from ocpa.algo.predictive_monitoring import factory as feature_factory
    from ocpa.algo.predictive_monitoring import tabular

    OCPA_AVAILABLE = True
except ImportError:
    OCPA_AVAILABLE = False
    logger.warning(
        "OCPA library not found in current environment. "
        "Install via `pip install -e ./ocpa` or `pip install ocpa`."
    )


def _resolve_factory_constant(attr_name: str, fallback_value: str) -> Any:
    """Safely retrieves a feature constant from ocpa.algo.predictive_monitoring.factory.

    Handles spelling discrepancies across library releases (e.g. PRECEDING_ACTIVITES vs PRECEDING_ACTIVITIES).

    Args:
        attr_name: Name of constant attribute to look up.
        fallback_value: Fallback string value if attribute is absent.

    Returns:
        The constant value from ocpa, or fallback_value.
    """
    if OCPA_AVAILABLE and hasattr(feature_factory, attr_name):
        return getattr(feature_factory, attr_name)
    return fallback_value


# Dynamically resolve OCPA feature extraction constants
EVENT_REMAINING_TIME = _resolve_factory_constant("EVENT_REMAINING_TIME", "event_remaining_time")
EVENT_ELAPSED_TIME = _resolve_factory_constant("EVENT_ELAPSED_TIME", "event_elapsed_time")
EVENT_PRECEDING_ACTIVITIES = _resolve_factory_constant(
    "EVENT_PRECEDING_ACTIVITIES",
    _resolve_factory_constant("EVENT_PRECEDING_ACTIVITES", "event_preceding_activities"),
)
EVENT_PREVIOUS_TYPE_COUNT = _resolve_factory_constant(
    "EVENT_PREVIOUS_TYPE_COUNT", "event_previous_type_count"
)
EVENT_ACTIVITY = _resolve_factory_constant("EVENT_ACTIVITY", "event_activity")
EVENT_CHAR_VALUE = _resolve_factory_constant("EVENT_CHAR_VALUE", "event_char_value")
EVENT_SERVICE_TIME = _resolve_factory_constant("EVENT_SERVICE_TIME", "event_service")
AGG_PREVIOUS_CHAR_VALUES = _resolve_factory_constant(
    "EVENT_AGG_PREVIOUS_CHAR_VALUES", "event_aggregate_previous_char"
)
SYNCHRONIZATION_TIME = _resolve_factory_constant(
    "EVENT_SYNCHRONIZATION_TIME", "event_synchronization_time"
)
WAITING_TIME = _resolve_factory_constant("EVENT_WAITING_TIME", "event_waiting_time")
POOLING_TIME = _resolve_factory_constant("EVENT_POOLING_TIME", "event_pooling_time")
EVENT_PREVIOUS_OBJECT_COUNT = _resolve_factory_constant(
    "EVENT_PREVIOUS_OBJECT_COUNT", "event_previous_object_count"
)
EVENT_NUM_OBJECT = _resolve_factory_constant("EVENT_NUM_OF_OBJECTS", "num_objects")
EVENT_PREVIOUS_ACTIVITY_COUNT = _resolve_factory_constant(
    "EVENT_PREVIOUS_ACTIVITY_COUNT", "event_previous_activity_count"
)


def build_tier3_feature_set(
    activities: List[str],
    object_types: List[str],
    numeric_attributes: Optional[List[str]] = None,
) -> List[Tuple[Any, Tuple]]:
    """Builds the comprehensive Tier 3 feature definition list for OCPA.

    Tier 3 acts as the master superset encompassing:
    - Target: Remaining Time (P3)
    - Tier 1 (Node-Local): Current Activity (C5), Requested Amount (D3), Service Time (P9), Elapsed Time (P2)
    - Tier 2 (Paper Baseline): Preceding Activities (C2), Prev Requested Amount Avg (D1), Prev Offer Count (O3)
    - Tier 3 (Full OCEL Graph Context): Synchronization Time (P5), Waiting Time (P10), Pooling Time (P7),
      Previous Object Count (O2), Current Object Count (O5), Cumulative Activity Counts (C3)

    Args:
        activities: List of unique process activities in the event log.
        object_types: List of object types (e.g. ['application', 'offer']).
        numeric_attributes: Numeric event attributes to aggregate (e.g. ['RequestedAmount']).

    Returns:
        List of (feature_function_identifier, parameter_tuple) pairs for OCPA.
    """
    feature_set: List[Tuple[Any, Tuple]] = []

    # 1. Target Variable: Remaining Time (P3)
    feature_set.append((EVENT_REMAINING_TIME, ()))

    # 2. Tier 1 Features (Node-Local)
    feature_set.append((EVENT_ELAPSED_TIME, ()))
    feature_set.append((EVENT_SERVICE_TIME, ()))
    for act in activities:
        feature_set.append((EVENT_ACTIVITY, (act,)))
    if numeric_attributes:
        for attr in numeric_attributes:
            feature_set.append((EVENT_CHAR_VALUE, (f"event_{attr}",)))

    # 3. Tier 2 Features (Paper Baseline additions)
    for act in activities:
        feature_set.append((EVENT_PRECEDING_ACTIVITIES, (act,)))
    for obj_type in object_types:
        feature_set.append((EVENT_PREVIOUS_TYPE_COUNT, (obj_type,)))
    if numeric_attributes:
        for attr in numeric_attributes:
            feature_set.append((AGG_PREVIOUS_CHAR_VALUES, (f"event_{attr}", np.mean)))

    # 4. Tier 3 Features (Full Graph Context / Explicit Message Passing)
    feature_set.append((SYNCHRONIZATION_TIME, ()))
    feature_set.append((WAITING_TIME, ()))
    feature_set.append((EVENT_PREVIOUS_OBJECT_COUNT, ()))
    feature_set.append((EVENT_NUM_OBJECT, ()))
    for obj_type in object_types:
        feature_set.append((POOLING_TIME, (obj_type,)))
    for act in activities:
        feature_set.append((EVENT_PREVIOUS_ACTIVITY_COUNT, (act,)))

    logger.info(
        "Constructed Tier 3 feature definition set with %d feature functions.", len(feature_set)
    )
    return feature_set


def extract_master_dataset(
    raw_path: str,
    output_parquet: str,
    subgraphs_output_dir: Optional[str] = "data/processed",
    debug_sample: Optional[int] = None,
) -> pd.DataFrame:
    """Extracts Tier 3 master features from an OCEL log using standard OCPA.

    Args:
        raw_path: Path to raw BPIC17 log (.jsonocel or .csv).
        output_parquet: Destination path for the master tabular Parquet file.
        subgraphs_output_dir: Directory to save prefix subgraphs (k in {4, 8, 12}).
        debug_sample: Optional limit on number of process executions for fast prototyping.

    Returns:
        pd.DataFrame: Extracted master tabular DataFrame with imputed 0.0 for boundary NaNs.

    Raises:
        RuntimeError: If OCPA is not installed or import fails.
        FileNotFoundError: If raw_path does not exist.
    """
    if not OCPA_AVAILABLE:
        raise RuntimeError(
            "OCPA library is required for feature extraction. "
            "Please install ocpa: `pip install -e ./ocpa`"
        )

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw log file not found at: {raw_path}")

    logger.info("Loading object-centric event log from: %s", raw_path)
    ocel = ocel_import_factory.apply(raw_path)

    # Discover log metadata: activities, object types, and numeric attributes
    raw_df = ocel.log.log
    activities = sorted(raw_df["event_activity"].dropna().unique().tolist())
    logger.info("Discovered %d unique activities: %s", len(activities), activities)

    object_types = list(ocel.object_types) if hasattr(ocel, "object_types") else []
    if not object_types and "event_object_type" in raw_df.columns:
        object_types = sorted(raw_df["event_object_type"].dropna().unique().tolist())
    logger.info("Discovered %d object types: %s", len(object_types), object_types)

    # Check for requested amount attribute (BPIC17 standard)
    numeric_attributes = []
    for col in ["RequestedAmount", "event_RequestedAmount"]:
        clean_name = col.replace("event_", "")
        if col in raw_df.columns and clean_name not in numeric_attributes:
            numeric_attributes.append(clean_name)

    # Build master Tier 3 feature specification
    feature_set = build_tier3_feature_set(
        activities=activities,
        object_types=object_types,
        numeric_attributes=numeric_attributes,
    )

    logger.info("Extracting features using OCPA predictive_monitoring factory...")
    feature_storage = feature_factory.apply(ocel, feature_set, [])

    # Fast Prototyping Slicing (Rule: prototype on first N executions)
    if debug_sample is not None and debug_sample > 0:
        total_graphs = len(feature_storage.feature_graphs)
        logger.info(
            "Debug mode: Slicing first %d process executions (out of %d total).",
            min(debug_sample, total_graphs),
            total_graphs,
        )
        feature_storage.feature_graphs = feature_storage.feature_graphs[:debug_sample]

    # Construct tabular representation
    logger.info("Constructing tabular DataFrame from feature storage...")
    df_master = tabular.construct_table(feature_storage)

    # Missing value handling (Rule 5: initial boundary events have deg^-(v) = 0 -> impute 0.0)
    nan_count = df_master.isna().sum().sum()
    if nan_count > 0:
        logger.info("Imputing 0.0 for %d NaN values in boundary features.", nan_count)
        df_master = df_master.fillna(0.0)

    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)

    # Persist master Parquet
    logger.info(
        "Saving master Tier 3 DataFrame to: %s (shape: %s)", output_parquet, df_master.shape
    )
    df_master.to_parquet(output_parquet, index=False)

    # Cache subgraphs for horizons k in {4, 8, 12} if subgraphs_output_dir is provided
    if subgraphs_output_dir:
        os.makedirs(subgraphs_output_dir, exist_ok=True)
        _cache_prefix_subgraphs(feature_storage, subgraphs_output_dir, horizons=[4, 8, 12])

    logger.info("Feature extraction complete.")
    return df_master


def _cache_prefix_subgraphs(
    feature_storage: Any,
    output_dir: str,
    horizons: List[int],
) -> None:
    """Extracts and caches prefix subgraphs of size k in horizons.

    Args:
        feature_storage: OCPA Feature_Storage container.
        output_dir: Output directory for saving PyTorch/NetworkX graph objects.
        horizons: List of prefix subgraph horizon sizes (e.g. [4, 8, 12]).
    """
    try:
        import torch

        for k in horizons:
            subgraphs_k = []
            for fg in feature_storage.feature_graphs:
                # fg is a process execution graph where nodes represent events
                nodes = list(fg.graph.nodes()) if hasattr(fg, "graph") else []
                if len(nodes) >= k:
                    subgraphs_k.append((fg, k))
            target_file = os.path.join(output_dir, f"subgraphs_k{k}.pt")
            logger.info(
                "Caching %d subgraphs for horizon k=%d to %s", len(subgraphs_k), k, target_file
            )
            torch.save(subgraphs_k, target_file)
    except ImportError:
        logger.warning("PyTorch not installed. Skipping .pt subgraph serialization.")
    except Exception as exc:
        logger.warning("Could not serialize subgraphs to .pt: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Extract Master Tier 3 Features via OCPA")
    parser.add_argument(
        "--raw",
        default="data/raw/bpic17.jsonocel",
        help="Path to raw BPIC17 log (.jsonocel or .csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/bpic17_tier3_master.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--subgraphs-dir",
        default="data/processed",
        help="Directory to save prefix subgraphs",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="Debug sample limit on process executions (e.g. 500)",
    )
    args = parser.parse_args()

    extract_master_dataset(
        raw_path=args.raw,
        output_parquet=args.output,
        subgraphs_output_dir=args.subgraphs_dir,
        debug_sample=args.debug,
    )
