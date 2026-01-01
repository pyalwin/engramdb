#!/usr/bin/env python3
"""
Ingest CUAD contracts into EngramDB.

Creates a persistent DuckDB database with all CUAD contracts,
ready for querying and exploration.

Usage:
    uv run python scripts/ingest_cuad.py
    uv run python scripts/ingest_cuad.py --max-contracts 50
    uv run python scripts/ingest_cuad.py --embedding-backend openai
"""

import argparse
import sys
import time
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from engramdb import EngramDB
from datasets.cuad_loader import CUADLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest CUAD contracts into EngramDB"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/engram.duckdb"),
        help="Path to DuckDB database file (default: data/engram.duckdb)"
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=None,
        help="Maximum number of contracts to ingest (default: all)"
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["mock", "openai", "local"],
        default="mock",
        help="Embedding backend to use (default: mock)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (faster, but no vector search)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing database"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CUAD Ingestion into EngramDB")
    print("=" * 60)
    print(f"Database: {args.db_path}")
    print(f"Embedding backend: {args.embedding_backend}")
    print(f"Max contracts: {args.max_contracts or 'all'}")
    print()

    # Check if database exists
    if args.db_path.exists() and not args.force:
        print(f"Database already exists at {args.db_path}")
        print("Use --force to overwrite, or specify a different --db-path")
        sys.exit(1)

    # Remove existing database if force
    if args.db_path.exists() and args.force:
        print(f"Removing existing database...")
        args.db_path.unlink()

    # Ensure parent directory exists
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    # Load CUAD contracts
    print("Loading CUAD dataset...")
    loader = CUADLoader()
    contracts = loader.load(max_contracts=args.max_contracts)

    print(f"Loaded {len(contracts)} contracts")
    print()

    # Create EngramDB
    print("Initializing EngramDB...")
    db = EngramDB(
        db_path=args.db_path,
        embedding_backend=args.embedding_backend
    )

    # Ingest contracts
    total_engrams = 0
    total_synapses = 0
    start_time = time.time()

    for i, contract in enumerate(contracts):
        contract_start = time.time()

        print(f"[{i+1}/{len(contracts)}] {contract.title[:60]}...")

        try:
            result = db.ingest(
                contract.text,
                doc_id=contract.id,
                generate_embeddings=not args.skip_embeddings
            )

            total_engrams += result.num_engrams
            total_synapses += result.num_synapses

            elapsed = time.time() - contract_start
            print(f"  -> {result.num_engrams} engrams, {result.num_synapses} synapses ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

    elapsed_total = time.time() - start_time

    # Print summary
    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Contracts ingested: {len(contracts)}")
    print(f"Total engrams: {total_engrams}")
    print(f"Total synapses: {total_synapses}")
    print(f"Total time: {elapsed_total:.1f}s")
    print(f"Database size: {args.db_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print(f"Database saved to: {args.db_path}")
    print()
    print("To explore the database:")
    print(f"  duckdb {args.db_path} -ui")
    print()
    print("To query from Python:")
    print(f"  from engramdb import EngramDB")
    print(f"  db = EngramDB(db_path='{args.db_path}')")
    print(f"  result = db.query('What are the termination conditions?')")

    db.close()


if __name__ == "__main__":
    main()
