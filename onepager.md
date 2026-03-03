# EngramDB: Schema-Aware Hybrid Retrieval for Multi-Hop Legal Reasoning

**Status:** Working prototype with benchmark results. Seeking co-author for arXiv submission.

---

## The Problem

Standard RAG retrieves document chunks by semantic similarity. This works when the answer lives in a single chunk, but legal contracts scatter related information across sections connected by cross-references, defined terms, and hierarchy.

**Concrete example:** A query asks *"Under what conditions can we terminate?"*

- The **Termination clause** says *"Either party may terminate for Cause"*
- **"Cause"** is defined three pages earlier in Definitions
- **Exceptions** to termination live in a separate General Provisions subsection

Vector search returns the Termination clause but misses the definition and exceptions — they share no vocabulary with the query. The LLM answers from partial context.

## The Insight

Structured legal documents **self-declare their schema** through headings, defined terms (`"X" means...`), and cross-references (`See Section 3.2`). This is a free, reliable signal that vector similarity ignores. We can extract it with deterministic rules — no LLM calls needed during ingestion.

## What We Built

**EngramDB** treats each document section as a node in a graph, connected by edges extracted from document structure:

1. **Ingest:** Rule-based parsing extracts section hierarchy, defined terms, and cross-references. Each section becomes a node (Engram); each structural link becomes an edge (Synapse).
2. **Retrieve:** Vector search finds anchor nodes. Graph traversal expands 1-3 hops across structural edges. A blended scoring function ranks candidates by `0.5 * semantic_similarity + 0.5 * structural_score`, where structural score decays by hop distance and edge type weight.
3. **Key mechanism:** Edge-type-aware traversal reservation ensures graph-discovered nodes (especially via REFERENCES and DEFINES edges) aren't displaced by high-similarity anchors.

Everything runs locally in a single DuckDB file. Python, ~2,400 lines of core code, MIT licensed.

## Results

Evaluated on **183 multi-hop questions** across **35 CUAD contracts** (SEC filings). Each question requires retrieving 2-3 structurally linked sections.

| Metric | Hybrid (ours) | Vector-only | Improvement |
|--------|:---:|:---:|:---:|
| Overall recall | **92.8%** | 68.2% | +24.6pp |
| 2-hop recall | **97.6%** | 66.8% | +30.8pp |
| 3-hop recall | **86.5%** | 70.0% | +16.5pp |

Largest gains on cross-reference questions (+34pp) where structural edges connect sections with no vocabulary overlap.

## Related Work & Positioning

| Approach | How We Differ |
|----------|---------------|
| Microsoft GraphRAG (2024) | They use LLM extraction (expensive, non-deterministic). We use rule-based extraction (free, reproducible). |
| Domain-Partitioned Hybrid RAG for Legal (arXiv:2602.23371) | Neo4j-based, reports 70% accuracy. Different architecture and extraction method. |
| Parent-Document / Sentence-Window RAG | Single-edge hierarchy only. We exploit cross-references, definitions, and hierarchy jointly. |

## What's Needed to Publish

The core system and results are done. To strengthen for a venue submission beyond arXiv:

- [ ] Add baselines: Parent-Document RAG, BM25+reranker, and ideally GraphRAG
- [ ] End-to-end answer accuracy evaluation (not just retrieval recall)
- [ ] Ablation study: vector-only vs graph-only vs hybrid
- [ ] Statistical significance testing
- [ ] Paper writing (Introduction, Related Work, Method, Experiments, Analysis)

**Target venues:** arXiv preprint (immediate), then EMNLP NLLP Workshop, SIGIR, or CIKM.

## Why Collaborate

- Working codebase with real results — this is not a proposal, it's a prototype with empirical evidence
- Clean research story: document structure is a free lunch for multi-hop retrieval
- Open-source (MIT), reproducible on CUAD (public dataset)
- Flexible on authorship order and contribution split

**Repo:** https://github.com/pyalwin/engramdb

---

*Arun — March 2026*
