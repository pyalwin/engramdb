# EngramDB

Schema-aware hybrid retrieval for multi-hop legal reasoning.

EngramDB combines vector retrieval with graph traversal over document-native structure (sections, definitions, cross-references). It is built for contract-style documents where relevant evidence is often spread across multiple sections.

## The Core Problem

Standard RAG (vector-only) retrieves document chunks by semantic similarity. This works when the answer lives in a single chunk, but legal contracts often scatter related information across multiple sections.

Example:

- A Termination clause says "Either party may terminate for Cause"
- "Cause" is defined earlier in Definitions
- Exceptions to termination live in a separate General Provisions subsection

If you ask "Under what conditions can we terminate?", vector search may return the Termination clause but miss the definition and exceptions because they do not share enough vocabulary with the query. The LLM then answers from partial context.

## What EngramDB Does Differently

EngramDB treats each document section as a node (Engram) in a graph, connected by edges (Synapses) extracted from document structure:

- Section hierarchy: `Article I -> Section 1.1 -> Section 1.1.1`
- Defined terms: `"Cause"` defined in one section and used in others
- Cross-references: `See Section 4.2`, `per Article III`

Retrieval runs in two phases:

1. Vector search finds relevant anchor nodes
2. Graph walk expands from anchors across structural edges (`1-3` hops)

This gathers context that is structurally related to the answer, not only semantically similar.

## Why This Is Interesting (Research)

The structure extraction is entirely rule-based (regex patterns for headings, definitions, and cross-references). There are no LLM calls during ingestion.

Hypothesis: document-native structure in contracts is a free, reliable signal that vector similarity misses, especially for multi-hop reasoning.

## Benchmark Results

Evaluated on 183 multi-hop questions across 35 CUAD contracts (SEC filings). Each question requires retrieving 2-3 structurally linked sections to answer correctly.

| Metric | Hybrid | Vector-only |
|--------|--------|-------------|
| **Overall recall** | 92.8% | 68.2% |
| **2-hop recall** | 97.6% | 66.8% |
| **3-hop recall** | 86.5% | 70.0% |

By question type:

| Type | Hybrid | Vector | Improvement |
|------|--------|--------|-------------|
| Cross-reference | 99.4% | 65.4% | +34.0pp |
| Termination chain | 100% | 78.1% | +21.9pp |
| Definition usage | 80.5% | 70.3% | +10.2pp |

Cross-references and termination chains benefit most — these are cases where the answer spans sections connected by `REFERENCES` or `DEFINES` edges that share little vocabulary with the query. Definition usage shows a smaller gain because defined terms like "Agreement" appear broadly across contracts, making structural signal less distinctive.

### Failure Analysis

Of 183 questions, 13 still fail (hybrid recall < 1.0). Two failure modes:

**Traversed but ranked out (majority):** The graph discovers the required section but scoring drops it outside the top-K window. This happens when dense graph neighborhoods produce 50-80+ traversed candidates competing for ~7 non-anchor slots. Mitigated by edge-type-aware reservation (REFERENCES/DEFINES edges get priority over PARENT_OF) and semantic re-ranking within each edge-type tier.

**Not reachable from anchors (minority):** Queries referencing sections by number (e.g., "Section 6 references Section 3") get no relevant anchors because vector search can't match section numbers. Addressed by injecting section-number anchors via metadata lookup when the query mentions `Section N`.

### Scoring Details

The retrieval pipeline scores each candidate as a weighted blend:

```
score = 0.5 * semantic_similarity + 0.5 * structural_score
```

Structural score decays by hop distance (`0.75` per hop) and edge type weight (`REFERENCES: 1.0`, `DEFINES: 0.9`, `PARENT_OF: 0.55`). Anchors get `structural_score = 1.0`.

Reserved traversal slots (default 4) ensure graph-discovered nodes aren't completely displaced by high-similarity anchors. Within reserved slots, candidates are sorted by edge-type tier first, then by semantic similarity — preventing PARENT_OF nodes with decent similarity from consuming slots over more relevant REFERENCES nodes.

Everything runs locally in a single DuckDB file with no external vector database and no GPU requirement for the database engine.

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
