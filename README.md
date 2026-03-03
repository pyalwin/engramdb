# EngramDB

Schema-Aware Hybrid Retrieval for Multi-Hop Legal Reasoning

EngramDB is a hybrid vector + graph memory engine that extracts document-native structure (headings, cross-references, defined terms) and uses it to improve retrieval on multi-hop reasoning tasks. It targets legal contracts where standard vector RAG fails to connect facts spread across distant sections.

## The Problem

Vector-only RAG retrieves by semantic similarity. This works for single-fact lookups but breaks down when answering a question requires connecting information across multiple document sections:

- **"Can we terminate if they breach confidentiality?"** requires linking the Termination clause to the Confidentiality clause via cross-references
- **"What does 'Material Adverse Change' mean in the context of termination?"** requires finding a definition, then tracing where it's used
- **"Are there exceptions to the non-compete?"** requires traversing parent-child section relationships

Standard RAG retrieves the most similar chunks independently. EngramDB retrieves anchor chunks *and then walks the document's own reference graph* to gather connected context.

## How It Works

Every ingested document section becomes an **Engram** (node) with three layers:

1. **Content** — raw text
2. **Vector** — semantic embedding for fuzzy search
3. **Graph** — explicit relationships (**Synapses**) extracted from document structure

Retrieval uses **hybrid traversal**:

```
Query → Embed → Vector search (find anchors) → Graph walk (expand context) → Rank → Return
```

The graph edges come from the document itself — no LLM extraction needed:

| Edge Type | Source |
|-----------|--------|
| `PARENT_OF` / `CHILD_OF` | Section hierarchy (Article I → Section 1.1 → Section 1.1.1) |
| `DEFINES` | Definition sections ("Confidential Information" means...) |
| `REFERENCES` | Cross-references (See Section 4.2, per Article III) |

## Quick Start

```bash
# Install
pip install -e .

# Or with uv
uv sync
```

### Ingest a contract

```python
from engramdb import EngramDB

db = EngramDB(db_path="my_contracts.duckdb", embedding_backend="mock")

with open("contract.md") as f:
    text = f.read()

result = db.ingest(text, document_id="nda-001", title="Acme NDA")
print(f"Ingested {result.engram_count} sections, {result.synapse_count} relationships")
```

### Query with hybrid retrieval

```python
db = EngramDB(db_path="my_contracts.duckdb")

result = db.query("Can we terminate if they breach confidentiality?", top_k=10, hops=2)

for engram in result.engrams:
    meta = engram.metadata
    print(f"[{engram.engram_type.value}] {meta.get('section_number', '')} {meta.get('title', '')}")
    print(f"  {engram.content[:120]}...")
    print()
```

### Embedding backends

```python
# Mock (deterministic hashing, no API calls — good for testing)
db = EngramDB(db_path="test.duckdb", embedding_backend="mock")

# OpenAI (text-embedding-3-small, 1536 dims)
db = EngramDB(db_path="prod.duckdb", embedding_backend="openai")

# Local sentence-transformers (requires `pip install -e ".[local]"`)
db = EngramDB(db_path="local.duckdb", embedding_backend="local")
```

## Ingestion Pipeline

The ingestion pipeline is entirely rule-based (regex, no LLM calls):

1. **Section Parser** — detects heading patterns (`ARTICLE I`, `Section 1.2.3`, `DEFINITIONS`, etc.) and builds a hierarchy tree
2. **Definition Extractor** — finds defined terms (`"X" means...`, `"X" shall mean...`) and links them to their usage sites
3. **Reference Linker** — resolves cross-references (`Section 4.2`, `Article III`, `Exhibit A`) to actual section nodes

Each step creates Engrams (nodes) and Synapses (edges) stored in DuckDB.

## Project Structure

```
src/engramdb/
  core/           # Engram and Synapse data models
  storage/        # DuckDB backend (CRUD, vector search, graph traversal)
  embeddings/     # OpenAI, local, and mock embedding providers
  ingestion/      # Parser, definition extractor, reference linker
  retrieval/      # Hybrid vector+graph retrieval
  db.py           # Main API

benchmarks/
  baselines/      # Naive RAG baseline
  datasets/       # CUAD loader, multi-hop QA generator
  evaluation/     # Metrics and benchmark runner

tests/            # Unit and integration tests
scripts/          # CUAD ingestion script
data/             # DuckDB file, CUAD dataset, synthetic contracts
```

## Evaluation

EngramDB is evaluated against the [CUAD dataset](https://www.atticusprojectai.org/cuad) (510 commercial legal contracts, 13,000+ expert QA annotations, 41 clause categories).

The benchmark measures multi-hop reasoning improvement over naive vector RAG:

| Hops | Baseline (vector-only) | Target (EngramDB) |
|------|----------------------|-------------------|
| 1-hop | ~80% | >= 80% (no regression) |
| 2-hop | ~30% | >= 60% |
| 3-hop | ~15% | >= 50% |

### Run the CUAD ingestion

```bash
# Mock embeddings (fast, no API key needed)
uv run python scripts/ingest_cuad.py --embedding-backend mock

# OpenAI embeddings
export OPENAI_API_KEY=sk-...
uv run python scripts/ingest_cuad.py --embedding-backend openai --max-contracts 50
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **Storage**: DuckDB (embedded, single-file, vector search via cosine similarity)
- **Language**: Python 3.12+
- **Embeddings**: OpenAI `text-embedding-3-small` (1536d) or local `intfloat/e5-base-v2` (768d)
- **No external services required** — everything runs on a laptop

## Status

Alpha (v0.1.0). Core ingestion and hybrid retrieval are functional. Benchmark evaluation is in progress.

## License

MIT
