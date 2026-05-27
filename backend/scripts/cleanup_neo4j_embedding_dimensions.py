#!/usr/bin/env python3
"""
Cleanup Neo4j Graphiti/Vector-index embedding properties with wrong dimensions.

This script is designed to fix errors like:
  vector.similarity.cosine(): Invalid input ... vectors do not have the same number of dimensions.

What it does:
1. Scans all node properties whose key ends with a suffix (default: "_embedding").
2. Computes the dimension for each vector (via Cypher `size(v)`).
3. Picks the most common dimension as the expected one.
4. With `--execute`, removes properties whose vector dimension != expected dimension.
   (By default it does NOT delete nodes.)

Usage:
  cd backend && uv run python scripts/cleanup_neo4j_embedding_dimensions.py
  cd backend && uv run python scripts/cleanup_neo4j_embedding_dimensions.py --execute
  cd backend && uv run python scripts/cleanup_neo4j_embedding_dimensions.py --expected-dim 1024 --execute
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


def _get_config() -> Any:
    # Import after sys.path tweaking.
    from app.config import Config

    return Config


@dataclass(frozen=True)
class Neo4jCreds:
    uri: str
    user: str
    password: str
    database: str


def _build_creds(config: Any) -> Neo4jCreds:
    return Neo4jCreds(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER or "neo4j",
        password=config.NEO4J_PASSWORD or "",
        database=config.NEO4J_DATABASE,
    )


def _scan_dimension_counts(
    driver: Any,
    *,
    suffix: str,
    limit_dims: int = 20,
) -> list[tuple[int, int]]:
    cypher = """
    MATCH (n)
    UNWIND keys(n) AS k
    WITH n, k, n[k] AS v
    WHERE k ENDS WITH $suffix
      AND v IS NOT NULL
    WITH size(v) AS dim, count(*) AS cnt
    RETURN dim, cnt
    ORDER BY cnt DESC
    LIMIT $limit_dims
    """
    with driver.session() as session:
        rows = session.run(
            cypher,
            suffix=suffix,
            limit_dims=limit_dims,
        )
        out: list[tuple[int, int]] = []
        for r in rows:
            dim = r.get("dim")
            cnt = r.get("cnt")
            if dim is None:
                continue
            out.append((int(dim), int(cnt)))
        return out


def _count_outliers(
    driver: Any,
    *,
    suffix: str,
    expected_dim: int,
    limit: int | None = None,
) -> int:
    cypher = """
    MATCH (n)
    UNWIND keys(n) AS k
    WITH n, k, n[k] AS v
    WHERE k ENDS WITH $suffix
      AND v IS NOT NULL
    WITH size(v) AS dim
    WHERE dim <> $expected_dim
    RETURN count(*) AS cnt
    """
    if limit is not None:
        # When limiting, we're counting the number of scanned outlier properties in this run.
        cypher = """
        MATCH (n)
        UNWIND keys(n) AS k
        WITH n, k, n[k] AS v
        WHERE k ENDS WITH $suffix
          AND v IS NOT NULL
        WITH size(v) AS dim
        WHERE dim <> $expected_dim
        LIMIT $limit
        RETURN count(*) AS cnt
        """
    with driver.session() as session:
        row = session.run(
            cypher,
            suffix=suffix,
            expected_dim=expected_dim,
            limit=limit,
        ).single()
        return int((row or {}).get("cnt", 0))


def _execute_remove_property_outliers(
    driver: Any,
    *,
    suffix: str,
    expected_dim: int,
    limit: int,
) -> int:
    cypher = """
    MATCH (n)
    UNWIND keys(n) AS k
    WITH n, k, n[k] AS v
    WHERE k ENDS WITH $suffix
      AND v IS NOT NULL
    WITH n, k, v, size(v) AS dim
    WHERE dim <> $expected_dim
    LIMIT $limit
    REMOVE n[k]
    RETURN count(*) AS removed
    """
    with driver.session() as session:
        row = session.run(
            cypher,
            suffix=suffix,
            expected_dim=expected_dim,
            limit=limit,
        ).single()
        return int((row or {}).get("removed", 0))


def _resolve_expected_dim(args: argparse.Namespace, dim_counts: list[tuple[int, int]]) -> int:
    if args.expected_dim is not None:
        return int(args.expected_dim)
    if not dim_counts:
        raise RuntimeError(
            "Could not detect any vector dimensions from the DB. "
            "Check the embedding property suffix and whether the DB contains vectors."
        )
    # Pick the most common dimension as expected.
    expected_dim, _expected_cnt = dim_counts[0]
    return int(expected_dim)


def _expected_dim_from_config(config: Any) -> int | None:
    """
    Best-effort expected dimension from current runtime configuration.
    Falls back to DB mode if we can't infer it safely.
    """
    embedder_choice = (getattr(config, "GRAPHITI_EMBEDDER", "") or "").strip().lower()

    # Gemini/Vertex embedder uses an explicit embedding_dim in our config.
    if (
        (getattr(config, "LLM_PROVIDER", "") or "").strip().lower() == "vertex"
        or bool(getattr(config, "LLM_USE_VERTEX_AI", False))
        or embedder_choice in {"vertex", "gemini"}
    ):
        return int(getattr(config, "GRAPHITI_GEMINI_EMBEDDING_DIM", 1024))

    # Local embedder: avoid downloading model; use same heuristic as local_embedder.py.
    if embedder_choice == "local":
        model_name = str(getattr(config, "GRAPHITI_LOCAL_EMBEDDING_MODEL", "") or "")
        return 384 if "bge-small" in model_name else 1024

    # OpenAI-compatible embedder: map common embedding models.
    # graphiti_client.py defaults to text-embedding-3-small.
    import os

    if embedder_choice == "azure" or (getattr(config, "LLM_PROVIDER", "") or "").strip().lower() == "azure":
        embedding_model = os.environ.get("GRAPHITI_AZURE_EMBEDDING_DEPLOYMENT") or "text-embedding-3-small"
    else:
        embedding_model = os.environ.get("GRAPHITI_EMBEDDING_MODEL") or "text-embedding-3-small"

    embedding_model = str(embedding_model).lower()
    if "large" in embedding_model:
        return 3072
    # Default to "small" dimension.
    return 1536


def main() -> int:
    config = _get_config()
    creds = _build_creds(config)

    ap = argparse.ArgumentParser(description="Cleanup Neo4j embedding dimensions mismatch.")
    ap.add_argument(
        "--suffix",
        default="_embedding",
        help="Property key suffix for embedding vectors (default: _embedding).",
    )
    ap.add_argument(
        "--expected-dim",
        type=int,
        default=None,
        help="Expected embedding dimension. If omitted, uses the most common dimension in DB.",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete vector properties. Default is dry-run reporting only.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=50_000,
        help="Max number of outlier embedding properties to remove in this run (default: 50000).",
    )
    ap.add_argument(
        "--preview-limit",
        type=int,
        default=10,
        help="How many top dimensions to print (default: 10).",
    )
    args = ap.parse_args()

    if not creds.password:
        print("WARNING: NEO4J_PASSWORD is empty. Check your .env / env vars.", file=sys.stderr)

    print(f"Connecting to Neo4j: {creds.uri} (db={creds.database})")
    driver = GraphDatabase.driver(creds.uri, auth=(creds.user, creds.password))

    try:
        dim_counts = _scan_dimension_counts(
            driver,
            suffix=args.suffix,
            limit_dims=max(1, args.preview_limit),
        )
        if not dim_counts:
            print(f"No properties ending with {args.suffix!r} were found.")
            return 0

        # Pretty print dimension histogram.
        hist = ", ".join([f"{d}x: {c}" for d, c in dim_counts[: args.preview_limit]])
        print(f"Top embedding dimensions by frequency: {hist}")

        config_expected_dim = _expected_dim_from_config(config) if args.expected_dim is None else None
        expected_dim = config_expected_dim if config_expected_dim is not None else _resolve_expected_dim(args, dim_counts)
        print(f"Expected dimension: {expected_dim}")

        outliers = _count_outliers(
            driver,
            suffix=args.suffix,
            expected_dim=expected_dim,
        )
        print(f"Outlier embedding properties (dim != {expected_dim}): {outliers}")

        if not args.execute:
            print("Dry-run mode: not removing anything. Re-run with --execute to apply changes.")
            return 0

        removed = _execute_remove_property_outliers(
            driver,
            suffix=args.suffix,
            expected_dim=expected_dim,
            limit=max(1, int(args.limit)),
        )
        print(f"Removed embedding properties: {removed}")
        if removed < outliers:
            print(
                "Note: removal was limited by --limit. Re-run the script to continue cleaning remaining outliers."
            )

        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())

