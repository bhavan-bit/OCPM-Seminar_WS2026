"""Sample session objects and extract their events from an OCEL 2.0 SQLite database.

Executes a single SQL query with Common Table Expressions (CTEs) to randomly sample
5 distinct session objects from the object_Session table and extract all associated events
via the OCEL 2.0 mapping table (event_object).
"""

import argparse
import os
import sqlite3
from typing import Optional

import pandas as pd

DEFAULT_DB_CANDIDATES = [
    "age_of_empires_ocel2_10_match_filter.sqlite",
    os.path.join("data", "raw", "age_of_empires_ocel2_10_match_filter.sqlite"),
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "raw",
        "age_of_empires_ocel2_10_match_filter.sqlite",
    ),
]


def resolve_database_path(db_path: Optional[str] = None) -> str:
    """Resolves the database file path, falling back to standard project paths."""
    if db_path and os.path.exists(db_path):
        return db_path

    for candidate in DEFAULT_DB_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    return db_path or DEFAULT_DB_CANDIDATES[0]


def sample_session_events(
    db_path: Optional[str] = None,
    n_sessions: int = 5,
) -> pd.DataFrame:
    """Randomly samples n_sessions distinct session objects from an OCEL 2.0 SQLite database

    and extracts all of their related events using a single CTE SQL query.

    Args:
        db_path: Path to the SQLite database file. If None, resolves standard locations.
        n_sessions: Number of distinct session objects to sample (default: 5).

    Returns:
        pd.DataFrame containing all columns from the event table plus session_id,
        sorted by session_id and ocel_time.
    """
    resolved_path = resolve_database_path(db_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"SQLite database not found at '{resolved_path}'. "
            "Please check the path to age_of_empires_ocel2_10_match_filter.sqlite."
        )

    query = """
    WITH SampledSessions AS (
        SELECT DISTINCT ocel_id AS session_id, ocel_time
        FROM object_Session
        ORDER BY RANDOM()
        LIMIT :n_sessions
    )
    SELECT 
        e.*,
        s.session_id
    FROM SampledSessions s
    JOIN event_object eo 
        ON s.session_id = eo.ocel_object_id
    JOIN event e 
        ON eo.ocel_event_id = e.ocel_id
    ORDER BY session_id, ocel_time;
    """

    with sqlite3.connect(resolved_path) as conn:
        df = pd.read_sql_query(query, conn, params={"n_sessions": n_sessions})

    return df


def main() -> None:
    """CLI entrypoint for sampling OCEL 2.0 session events."""
    parser = argparse.ArgumentParser(
        description="Sample session objects and extract related events from an OCEL 2.0 SQLite database."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the OCEL 2.0 SQLite database (default: age_of_empires_ocel2_10_match_filter.sqlite)",
    )
    parser.add_argument(
        "--n-sessions",
        type=int,
        default=5,
        help="Number of session objects to sample (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save output (e.g. data/processed/sampled_events.parquet or .csv)",
    )

    args = parser.parse_args()

    print(f"Connecting to database and sampling {args.n_sessions} sessions...")
    df = sample_session_events(db_path=args.db_path, n_sessions=args.n_sessions)

    sampled_sessions = df["session_id"].nunique()
    total_events = len(df)
    print(f"Successfully sampled {sampled_sessions} sessions with {total_events} related events.")
    print("\nPreview of extracted events:")
    print(df.head(10))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if args.output.endswith(".parquet"):
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"\nSaved extracted events to {args.output}")


if __name__ == "__main__":
    main()

