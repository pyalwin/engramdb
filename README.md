# EngramDB

Schema-aware hybrid retrieval for multi-hop legal reasoning.

EngramDB combines vector retrieval with graph traversal over document-native structure (sections, definitions, cross-references). It is built for contract-style documents where relevant evidence is often spread across multiple sections.

## Features

- Rule-based ingestion pipeline (no LLM needed for structure extraction)
- DuckDB storage for engrams (nodes), synapses (edges), and embeddings
- Hybrid retrieval:
  - Vector search to find anchor nodes
  - Graph expansion to collect connected context
- Pluggable embedding backends: `mock`, `openai`, `local`

## Requirements

- Python `>=3.12`
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`

## Install

```bash
# Core dependencies
uv sync

# Development tools (pytest, ruff, mypy)
uv sync --extra dev

# Local embedding backend (sentence-transformers)
uv sync --extra local

# Benchmark dependencies
uv sync --extra benchmark
```

`pip` alternative:

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[local]"
pip install -e ".[benchmark]"
```

## Quick Start

```python
from engramdb import EngramDB

contract_text = """
MUTUAL NON-DISCLOSURE AGREEMENT

1.1 Definitions
"Confidential Information" means non-public information.

4.2 Termination
Either party may terminate this Agreement with 30 days notice.
"""

with EngramDB(db_path="data/example.duckdb", embedding_backend="mock") as db:
    ingest = db.ingest(contract_text, doc_id="nda_001")
    print(f"Engrams: {ingest.num_engrams}, Synapses: {ingest.num_synapses}")

    result = db.query(
        "Can either party terminate this agreement?",
        top_k_anchors=3,
        max_hops=2,
        max_context_items=10,
    )

    print(db.get_context_string(result, include_metadata=True))
    print(db.stats())
```

## Embedding Backends

- `mock`: deterministic pseudo-embeddings for tests and local development
- `openai`: uses OpenAI embeddings (`OPENAI_API_KEY` required)
- `local`: uses sentence-transformers (`uv sync --extra local`)

Example:

```python
db = EngramDB(embedding_backend="mock")
db = EngramDB(embedding_backend="openai")
db = EngramDB(embedding_backend="local")
```

## Ingestion Pipeline

1. Parse sections and heading hierarchy (`ingestion/parser.py`)
2. Extract defined terms (`ingestion/definitions.py`)
3. Extract and resolve references (`ingestion/references.py`)
4. Create engrams and synapses
5. Optionally generate embeddings and store in DuckDB

## Benchmark Workflow (CUAD)

1. Download/process CUAD contracts:
```bash
uv run python benchmarks/datasets/cuad_loader.py
```

2. Generate multi-hop QA dataset:
```bash
uv run python benchmarks/datasets/multihop_generator.py
```

3. Run hybrid vs vector-only benchmark:
```bash
export OPENAI_API_KEY=your_key_here
uv run python benchmarks/evaluation/benchmark.py
```

Helpful ingestion script for a persistent EngramDB file:

```bash
uv run python scripts/ingest_cuad.py --embedding-backend mock --max-contracts 50 --force
```

Artifacts are written under `data/cuad/` (for example `benchmark_results.json`).

## Development

Run tests:

```bash
uv run --extra dev pytest -q
```

Lint/type-check:

```bash
uv run --extra dev ruff check .
uv run --extra dev mypy src
```

## Project Layout

```text
src/engramdb/
  core/         # Engram and Synapse models
  embeddings/   # Embedder backends and factory
  ingestion/    # Section parsing, definition extraction, reference linking
  retrieval/    # Hybrid and vector-only retrieval
  storage/      # DuckDB persistence and graph/vector queries
  db.py         # Main user-facing API

benchmarks/
  datasets/     # CUAD loader and multi-hop QA generator
  evaluation/   # Benchmark runner and metrics
  baselines/    # Naive RAG baseline scaffold

scripts/        # Utility scripts (e.g., CUAD ingestion)
tests/          # Unit and integration tests
```

## Status

Alpha (`0.1.0`).

- Core ingestion and retrieval are implemented.
- Baseline implementation in `benchmarks/baselines/naive_rag.py` is currently a scaffold.

## License

MIT
