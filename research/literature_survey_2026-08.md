# EngramDB Pivot — Literature & Landscape Survey (August 2026)

Survey of prior art for the EngramDB pivot: a smart retrieval layer that, given a
natural-language query and a token budget, assembles the most relevant LLM context
across heterogeneous sources (live SQL, JSON stores, vector-indexed documents,
files/blobs), building a query-time identity/topology graph instead of an
ingest-time knowledge graph.

> Sourcing caveat: direct arXiv fetches were blocked by the survey environment's
> network policy, so paper summaries are close paraphrases synthesized from search
> results (arXiv IDs cross-confirmed across multiple independent results), plus one
> direct fetch of the Microsoft Research LazyGraphRAG blog post. Re-verify primary
> sources before formal citation.

## Executive Summary

The single biggest threat to the novelty claim is **LazyGraphRAG** (Microsoft
Research, blog only, no arXiv paper, Nov 2024) — the closest prior art to "defer
graph-building to query time," but it operates *only over a static text corpus*
(NLP noun-phrase extraction at index time, LLM claim-extraction deferred to query
time via best-first/breadth-first/iterative-deepening search over communities). It
has **no notion of live SQL/JSON sources, no re-fetch-at-query-time semantics, and
no identity-vs-state distinction**. Closer prior art on "query-time graph
instantiation over structured stores" is **SAG (arXiv:2606.15971, Zleap AI,
2026)**, which builds query-time dynamic hyperedges via SQL joins on shared entity
keys extracted from chunks — mechanically very close to the "deterministic edges
via shared identifiers" commitment, but still corpus-derived (chunks→entities), not
multi-source-live. Also highly relevant: **HippoRAG (2405.14831)**, **LightRAG
(2410.05779)**, **KAG (2409.13731)**, and the broader "no pre-built graph" line
(**LogicRAG / "You Don't Need Pre-built Graphs for RAG," 2508.06105**) which argues
for per-query reasoning-structure extraction instead of corpus-wide graphs —
conceptually parallel but text-only and not persistence/caching-oriented.

On agent memory, **Zep/Graphiti (2501.13956)** is architecturally close on
bi-temporal edges and incremental graph growth but is a conversational-memory
system, not a multi-source-database retrieval layer, and it does store some
volatile facts in the graph itself (a philosophical difference from the
"graph = topology cache only" rule). On context assembly, three 2026 papers are
directly on-point to the knapsack/packing commitment: **"What Survives Into
Context" (2607.00725)**, **"Structure and Diversity Aware Context Bubble
Construction" (2601.10681)**, and **"Replace, Don't Expand"/SEAL-RAG
(2512.10787)** — all cast context assembly as a budgeted/submodular optimization
problem; none integrate cross-source entity resolution or live re-fetch.

Commercially, **Glean** (knowledge graph + RAG over enterprise connectors) and the
emerging "context layer" funding wave (Contextual AI, Modus, Interloom,
Pebblous/Jedify) show the market thesis is real and funded — execution/positioning
risk more than technical-novelty risk.

Overall: the *combination* — lazy query-time graph limited to identity/topology,
live re-fetch of state, deterministic-then-LLM-fallback edge tiering, cross-source
entity resolution, and budget-aware packing with provenance, all as an embedded
library — does not appear to exist as a single system; each piece has partial
precedent.

---

## Per-Area Findings

### 1. GraphRAG and its evolution

- **GraphRAG** — Darren Edge, Ha Trinh, et al. (Microsoft Research),
  **arXiv:2404.16130**, "From Local to Global: A Graph RAG Approach to
  Query-Focused Summarization." Builds an LLM-extracted entity/relationship graph
  over the *entire* corpus at index time, partitions it into a community hierarchy
  (Leiden), and pre-generates community summaries used for global "sensemaking"
  queries. **Relation**: exactly the ingest-time batch construction EngramDB
  rejects — the baseline the "no ingest-time KG" commitment argues against.
- **LazyGraphRAG** — Edge, Trinh, Larson (Microsoft Research), **Microsoft Research
  blog only, Nov 25 2024 — no arXiv paper exists** (verified; DataStax's
  `graph-retriever` library has an open-source re-implementation). Defers *all LLM
  calls* to query time: indexing is cheap NLP (noun-phrase extraction +
  co-occurrence graph + graph statistics for hierarchical communities, "0.1% of
  GraphRAG's indexing cost"); query time does best-first search (chunk embeddings
  ranked by similarity), breadth-first search (LLM sentence-relevance scoring of
  top-k chunks), and iterative deepening into sub-communities, then LLM
  claim-extraction and ranking. **Crucially: text-corpus only, no SQL/live-database
  dimension, no persistent cross-query edge cache, no state/identity separation** —
  it re-derives relevance every query rather than accumulating a durable topology
  graph. **Relation**: nearest prior art for "lazy," but it (a) never touches
  structured/live sources, (b) doesn't persist/accumulate a graph across queries,
  and (c) has no deterministic-edge-first design (concept graph is entirely
  NLP/co-occurrence based, no schema introspection). The single most important
  comparison to characterize precisely.
- **HippoRAG** — Gutierrez, Shu, Gu, Yasunaga, Su, **arXiv:2405.14831** (NeurIPS
  2024). LLM-built open-IE graph + Personalized PageRank; 10–30x cheaper than
  iterative retrieval (IRCoT). Still full-corpus batch graph construction.
  **Relation**: PPR is a traversal/ranking algorithm worth reusing for subgraph
  expansion; same ingest-time paradigm we reject.
- **LightRAG** — **arXiv:2410.05779** (EMNLP 2025 findings). Dual-level (entity +
  relationship) retrieval; incremental-update-friendly graph index. **Relation**:
  incremental rather than batch updates, but graph still built from document
  extraction, not query-time probing of live sources.
- **KAG** — Ant Group/OpenSPG, **arXiv:2409.13731**. Logical-form-guided reasoning
  over a schema-constrained KG; addresses OpenIE noise via domain schema
  constraints. **Relation**: schema-constrained extraction parallels our
  "schema introspection first, LLM fallback second" tiering; graph still pre-built.
- **GRAG** — **arXiv:2405.16506** (NAACL 2025 findings). Retrieves and prunes k-hop
  textual ego-subgraphs (divide-and-conquer), feeds text-view + graph-view to the
  LLM. **Relation**: subgraph-retrieval-and-pruning technique worth borrowing for
  the "materialize only the relevant subgraph" step.
- Surveys: **"A Survey of Graph Retrieval-Augmented Generation for Customized
  LLMs"** (Zhang, Chen, et al., **arXiv:2501.13958**); **"When to use Graphs in
  RAG: A Comprehensive Analysis"** (Xiang et al., **arXiv:2506.05690**, ICLR'26,
  introduces GraphRAG-Bench) — central finding: GraphRAG "frequently underperforms
  vanilla RAG" on many real tasks; graph value is scenario-dependent (multi-hop /
  relational queries). Directly useful caution for positioning.

### 2. Query-time / on-demand / incremental graph construction

- **SAG: SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges** —
  Wu, Li, Liang, Chen, Liang, Mo, Li (Zleap AI), **arXiv:2606.15971**. Converts
  each chunk into one "event" + indexing entities at index time; **at query time
  uses SQL join queries on shared entity keys to dynamically link events into local
  hyperedges** — query-time-instantiated local structure, not a persistent global
  graph. Best-of-9 Recall@K on HotpotQA/2WikiMultiHop/MuSiQue (80% Recall@5 on
  MuSiQue). **Relation**: mechanically the closest match to our deterministic-edge
  design — but homogeneous document-derived event tables, no persistent cross-query
  cache, no live re-fetch of volatile state. Cite explicitly, differentiate
  carefully, benchmark against.
- **LogicRAG / "You Don't Need Pre-built Graphs for RAG"** — Chen, Zhou, Yuan,
  Zhang, Cui, Chen, Xiao, Cao, Huang (PolyU), **arXiv:2508.06105** (AAAI).
  Dynamically extracts a small per-question reasoning structure at inference time.
  **Relation**: shares the "no ingest-time graph" philosophy; graph is throwaway
  per-query (no persistent topology cache) and text-only.
- **"Towards Practical GraphRAG"** — **arXiv:2507.03226**. Dependency parsing
  instead of LLM extraction for KG construction (≈94% of LLM quality at a fraction
  of cost) + RRF-fused hybrid retrieval. **Relation**: supports "deterministic/
  cheap edges first" tiering; still index-time and text-only.
- Lower-confidence finds: **RAGA** (agentic KG CRUD loop), **ROGRAG** (incremental
  DB construction for dynamic KG expansion), **TagRAG** (tag-guided hierarchical KG
  for incremental updates) — incremental but still batch/corpus-oriented
  maintenance, not query-triggered probing of live heterogeneous stores.
- **Query-time Entity Resolution** — **arXiv:1111.0045** (2011, classic
  Whang & Garcia-Molina-era) and **FastER: On-Demand Entity Resolution in Property
  Graphs** — **arXiv:2504.01557** (2025, ISWC). Deferring *resolution* rather than
  graph construction; evidence the "lazy" pattern has deep DB-community roots
  predating the RAG framing by over a decade.

### 3. Retrieval over heterogeneous / structured+unstructured sources

- **HybridQA** — Chen et al., **arXiv:2004.07347**; **TAT-QA** (finance table+text
  QA); **S3HQA** (**arXiv:2305.11725**) as a representative three-stage descendant.
  **Relation**: eval benchmarks for table+text fusion; static aligned pairs, not
  live SQL + JSON + files.
- **HybridRAG** — **arXiv:2408.04948** (ACM ICAIF 2024). VectorRAG + GraphRAG for
  financial earnings-call QA; combination beats either alone. **Relation**:
  validates combining graph structure with vector retrieval; no live databases.
- **Federated RAG line**: **FeB4RAG** (**arXiv:2402.11891**, SIGIR 2024,
  federated-search benchmark for RAG), **HyFedRAG** (**arXiv:2509.06444**,
  privacy-preserving, heterogeneous modalities, edge-cloud split), **RAGRoute**
  (**arXiv:2502.19280**, lightweight neural router selecting data sources at query
  time), **Federated RAG for Multi-Product QA** (**arXiv:2501.14998**).
  **Relation**: RAGRoute's query-time source selection is adjacent to our
  "probe sources with query entities" step; none build a cross-source
  identity/topology graph — they route/merge results, no relationship caching.
- **Schema-First Retrieval: Embedding Catalogs for Natural Language Analytics** —
  Agrawal & Indukuri, **arXiv:2606.28387** (2026). Embeds *catalog metadata*
  (tables, columns, metrics, relationships, query history) rather than warehouse
  rows; vector search + lineage expansion + reranking + access-control gating
  before SQL generation; strong results on CRUSH4SQL/SEDE/BIRD. **Relation**:
  directly relevant template for the SQL-side schema-introspection mechanism.
- **KG-RAG4SM** — **arXiv:2501.08686**. External KGs to assist schema matching
  across databases. **Relation**: adjacent, different problem (schemas, not
  records/entities at query time).
- **LlamaIndex `SQLAutoVectorQueryEngine` / RouterQueryEngine**, **LangChain SQL +
  vector routing** (docs, no paper) — LLM-selector routing between a SQL tool and a
  vector tool; no graph, no cross-source join/ER, no persistent cache, no packing.
  **Relation**: the naive baseline EngramDB must visibly outperform.
- **"Scalable and Explainable Enterprise Knowledge Discovery Using Graph-Centric
  Hybrid Retrieval"** — Rao, Srivastava, Sharma, Shrivastava (Persistent Systems),
  **arXiv:2510.10942**. Unified KG from Jira/Git/Confluence/wikis; 80% relevance
  improvement over vanilla GPT-RAG. **Relation**: enterprise multi-source
  precedent, but batch-built from semi-structured artifacts, not live SQL.

### 4. Agent memory systems with graphs

- **Zep / Graphiti** — **arXiv:2501.13956**, "Zep: A Temporal Knowledge Graph
  Architecture for Agent Memory." Bi-temporal edges (event-occurred vs. ingested
  time, validity intervals), three-tier memory (episodic → semantic entities/facts
  → community summaries); 18.5% accuracy gain / 90% latency reduction vs.
  baselines; beats MemGPT on DMR and LongMemEval. **Relation**: closest
  agent-memory analog; borrow bi-temporal edge validity for staleness handling —
  but Graphiti's graph *stores fact content in edges*, the opposite of our
  "no state in the graph" rule; built for conversational memory, not federated
  live-database retrieval.
- **MemGPT / Letta** — Packer, Wooders, Lin, Fang, Patil, Gonzalez,
  **arXiv:2310.08560**. OS-inspired virtual-context paging between context window
  and external storage. No graph. **Relation**: relevant to token-budget framing
  (context as paged scarce resource); no topology/ER dimension.
- **Mem0** — **arXiv:2504.19413** (ECAI 2025). Extract→consolidate→
  ADD/UPDATE/DELETE/NOOP pipeline over conversational memory; "Mem0g" graph
  variant. 91% lower p95 latency, >90% token savings vs. full-context baselines.
  **Relation**: its reconciliation loop is a useful analog for maintaining/
  invalidating a persistent edge cache.
- **Cognee** — open-source (topoteretes/cognee), no arXiv paper found. Vector +
  graph + relational (consolidated onto Postgres); "remember/recall/improve/forget"
  API; graph + ontology, continuously updated. **Relation**: closest OSS *product*
  analog to graph-as-cache accumulated over time, but ingests everything
  proactively — not lazy/query-triggered, no live-source re-fetch.

### 5. Context assembly / context engineering

- **"A Survey of Context Engineering for Large Language Models"** — Mei et al.,
  **arXiv:2507.13334** (July 2025). First systematic survey; frames context
  engineering as superset of prompt engineering (Context Retrieval/Generation,
  Processing, Management). **Relation**: vocabulary/taxonomy for positioning
  EngramDB; citation trail for "context window as the product."
- **"What Survives Into Context: A Diagnostic for Budget-Constrained Multi-Hop
  RAG"** — Ananto Nayan Bala, **arXiv:2607.00725** (2026). Retrieval recall is the
  wrong metric under a fixed reader budget; introduces **"answer-in-context"**
  (does the gold answer survive as a contiguous span in the *packed* context) —
  much better predictor of answer F1 (r=0.39–0.55 vs ~0.31 for recall). Casts
  packing as **budgeted monotone submodular maximization** (relevance + coverage +
  representativeness + diversity); +5.1 F1 on HotpotQA at a 160-token budget.
  **Relation**: the formal framework for our packer; adopt the framing and the
  diagnostic.
- **"Structure and Diversity Aware Context Bubble Construction for Enterprise
  Retrieval Augmented Systems"** — Amir Khurshid et al., **arXiv:2601.10681**
  (Jan 2026). "Context bubbles": coherent, citable span bundles under strict token
  budget; multi-granular spans (sections/rows), task-conditioned structural priors,
  constrained selection balancing relevance/coverage/redundancy; targets
  auditability. **Relation**: extremely close to our packing + provenance
  commitment; strongest single paper to benchmark against for the packing layer.
  Document-structure-aware but not multi-source-with-live-SQL.
- **"Replace, Don't Expand" (SEAL-RAG)** — Lahmy & Yozevitch, **arXiv:2512.10787**
  (Dec 2025). Context window as scarce fixed-k resource; Search→Extract→Assess→Loop
  with entity-anchored gap detection, swapping distractors for gap-closing
  evidence. **Relation**: their "gap specification" is a useful pattern for our
  explainability trace.
- **LLMLingua family** — **arXiv:2310.05736** (LLMLingua, EMNLP 2023),
  **arXiv:2310.06839** (LongLLMLingua), **arXiv:2403.12968** (LLMLingua-2).
  Perplexity/classifier-based token pruning, up to 20x compression. **Relation**:
  orthogonal compression stage — could compress within blocks after the packer
  selects them; not a substitute for retrieval/selection.

### 6. Entity resolution with LLMs across sources

- **Query-time Entity Resolution** (2011), **arXiv:1111.0045** — two-stage "expand
  and resolve" strategy for resolving entities only as needed for a specific query.
  **Relation**: direct intellectual ancestor of lazy cross-source ER; cite for
  lineage. EngramDB's contribution = applying it to LLM-context assembly across
  heterogeneous modern stores.
- **FastER: On-Demand Entity Resolution in Property Graphs** — **arXiv:2504.01557**
  (2025, ISWC). ER-on-demand via Graph Differential Dependencies; resolves only
  query-relevant entities. **Relation**: closest modern match to lazy ER; note its
  warning that on-demand work gets redone for popular queries unless cached.
- **LLM entity matching**: "Match, Compare, or Select? An Investigation of LLMs for
  Entity Matching" (**arXiv:2405.16884**), "OpenSanctions Pairs: Large-Scale Entity
  Matching with LLMs" (**arXiv:2603.11051**), "Unlocking the Power of LLMs for
  Multi-Table Entity Matching" (**arXiv:2604.21238**), "KcMF: Knowledge-compliant
  Framework for Schema and Entity Matching with Fine-tuning-free LLMs"
  (**arXiv:2410.12480**). **Relation**: active, fairly mature line; benchmark our
  tier-2 fallback against these; consider classification framing (KcMF-style) over
  full generative reasoning for query-time cost.

### 7. Commercial / open-source landscape

- **LlamaIndex / LangChain**: RouterQueryEngine, SQLAutoVectorQueryEngine,
  SQLRouterQueryEngine — LLM-selector routing, no persistent cross-source graph,
  no ER, no budget-optimal packing.
- **WrenAI, Vanna.ai, MindsDB** — text-to-SQL / "GenBI" tools. Query-generation
  tools, not context-assembly layers; none combine SQL with document/vector
  retrieval into a single assembled context with provenance.
- **Glean** — enterprise search with proprietary Knowledge Graph (100+ connectors),
  subject-predicate-object triples over people/docs/tools; markets a permissioned
  "system of context" for enterprise AI. **Relation**: most direct commercial
  positioning overlap — but graph is built continuously/proactively from connector
  ingestion (always-on indexing, not query-time lazy construction) and doesn't
  treat live transactional SQL as a first-class retrieval target. Prepare a
  "how is this different from Glean" comparison.
- **Contextual AI** — "RAG 2.0," end-to-end-trained retriever+generator (no paper
  found). Different axis of innovation (model training) than ours (architecture).
- **Zep, Cognee** — covered above; adjacent "memory/context layer" competitive set.
- **"Context layer" funding wave (2025–2026, press only)**: Modus ("Context
  Warehouse," $10M seed, Insight Partners), Interloom ("context graphs," $16.5M
  seed, DN Capital), Pebblous/Jedify ($24M Series A). Market validation +
  competitive pressure; no technical papers found.

---

## Design Learnings

1. **Cite and precisely differentiate from LazyGraphRAG.** One-slide answer:
   LazyGraphRAG defers *LLM claim extraction* over a *static text corpus*; EngramDB
   defers *graph construction itself* over *live, heterogeneous, stateful sources*,
   re-fetching volatile state at query time rather than ever summarizing/caching it.
2. **Adopt the "answer-in-context" diagnostic** (arXiv:2607.00725) as the primary
   internal eval signal for the packer — not recall@k.
3. **Frame the packer explicitly as budgeted submodular maximization** (relevance +
   coverage + representativeness + diversity), matching 2607.00725 and 2601.10681 —
   well-studied framing with greedy near-optimality guarantees.
4. **Borrow bi-temporal edge validity from Graphiti/Zep** for staleness: timestamp
   both "when the relationship held" and "when we last verified it" on cached
   edges — while still refusing to store fact *values* in the graph.
5. **Read SAG (2606.15971) closely before finalizing architecture**; benchmark
   against its Recall@K results on HotpotQA/2WikiMultiHop/MuSiQue; articulate why
   persistent cross-query accumulation + true multi-source design goes further.
6. **Known failure mode of lazy approaches**: on-demand resolution gets repeatedly
   redone for popular queries unless cached (FastER, query-time-ER literature). Our
   persistent cache is the fix — but benchmark cache-hit rate / amortized cost over
   a realistic workload, since this is where naive lazy loses to batch.
7. **Graphs often don't help** (arXiv:2506.05690): don't force graph traversal for
   every query; graph is one signal among several; the planner may skip it.
   GraphRAG-Bench can double as part of our eval suite to show *when* the graph
   layer adds value.
8. **Schema-First Retrieval (2606.28387)**'s catalog-embedding + lineage-expansion
   + access-control-gating pipeline is a strong template for the SQL-side schema
   introspection — adapt rather than reinvent.

---

## Novelty Assessment

**Likely genuinely novel (as a combination):**

- Query-time graph construction restricted to *identity/topology only* (never
  volatile state), with mandatory live re-fetch of state at answer time. This
  precise state/identity split appears in no paper found (LazyGraphRAG is text-only
  and doesn't distinguish; Graphiti/Zep stores fact content in edges — the opposite
  choice). **Make this the headline architectural claim.**
- A single system spanning live SQL + JSON + vector-indexed docs + files/blobs with
  cross-source deterministic edge discovery (schema introspection, key-shape
  statistics) + LLM-fallback tier-2 edges + persistent cross-query edge caching +
  budget-aware submodular packing + explainability trace. Each piece exists in
  isolation (SAG, Schema-First Retrieval, 2601.10681/2607.00725, Graphiti,
  FastER); the union does not.
- Embedded-library posture (pip install, DuckDB/SQLite files) combined with this
  graph-caching architecture. Graph/agent-memory systems found are
  services/platforms; embedded text-to-SQL tools have no graph/ER layer.

**Should be softened:**

- "No ingest-time knowledge-graph construction" as headline → soften to "no
  *LLM-based* ingest-time extraction" / "no corpus-wide batch construction"
  (LazyGraphRAG, LogicRAG, SAG also avoid full ingest-time LLM graph-building).
  The distinguishing move is *live multi-source + state/identity split + persistent
  accumulation*, not laziness per se.
- "Lazy" as a term — already branded (LazyGraphRAG) and diluted; prefer something
  like "query-triggered topology cache," crediting LazyGraphRAG for the pattern.
- Claims about entity resolution being unsolved — LLM entity matching and
  on-demand ER are mature-ish; our contribution is cheap application across
  heterogeneous live sources at interactive latency.

**Should be sharpened:**

- The state/identity separation (only IDs/keys/edges in the graph; never
  amounts/statuses/content) — most defensible and unprecedented piece found.
- Deterministic-edges-first / LLM-fallback-with-confidence-and-evidence tiering
  applied *across the SQL/JSON/vector/file boundary* — sharpen with the concrete
  worked example (PO number appearing in Postgres + CRM JSON + PDF).

---

## Benchmark / Eval Pointers

- **Multi-hop QA over text**: HotpotQA, 2WikiMultiHop, MuSiQue — used by SAG
  (2606.15971), HippoRAG, "What Survives Into Context" (2607.00725).
- **GraphRAG-specific**: GraphRAG-Bench (arXiv:2506.05690, ICLR'26) — shows when
  the graph layer earns its keep.
- **Table+text hybrid QA**: HybridQA (arXiv:2004.07347), TAT-QA.
- **Text-to-SQL / schema retrieval**: CRUSH4SQL, SEDE, BIRD (used by
  Schema-First Retrieval, 2606.28387).
- **Long-horizon / conversational memory**: LongMemEval (ICLR 2025), LME-V2
  (arXiv:2605.12493), DMR (per Zep, arXiv:2501.13956) — if extended to
  agent-memory use cases.
- **Federated retrieval**: FeB4RAG (arXiv:2402.11891) — adaptable to a
  SQL+doc+JSON federated setting.
- **Budget-constrained packing**: the answer-in-context protocol of
  arXiv:2607.00725 (fixed token budgets, e.g. 160 tokens on HotpotQA with a 3B
  reader); citability/coherence metrics of arXiv:2601.10681 for the
  provenance/explainability side.
- **Entity matching**: benchmarks used by OpenSanctions Pairs (2603.11051) and
  multi-table entity matching (2604.21238).
