# EngramDB Design Document

**Status**: Draft v1 — founding document for the pivot
**Date**: August 2026
**Supersedes**: the schema-aware legal-contract retrieval prototype (preserved in git
history; see `research/literature_survey_2026-08.md` for the prior-art survey that
informed this design)

---

## 1. Vision

**EngramDB is a declarative context query layer for LLM systems.** A calling
system states an information need and a resource budget; EngramDB's planner
decides how to satisfy it across heterogeneous live sources — SQL databases, JSON
stores, vector-indexed documents, files/blobs — and returns a single packed,
provenance-tagged, budget-fitted context window plus a machine-readable trace of
how it was built.

The governing analogy is SQL itself. SQL won by separating intent from execution:
the caller declares *what*, the optimizer decides *how*, using statistics it
maintains about the data. EngramDB applies the same separation to LLM context:

| Relational world | EngramDB |
|---|---|
| SQL statement | `context(query, budget, constraints)` |
| Query planner / optimizer | The planner: source routing, probe strategy, edge tiering |
| Table statistics & indexes | Topology cache + entity-resolution map |
| Execution engine | Federated source executors |
| Result set | Packed, provenance-tagged context window |
| `EXPLAIN` | The trace (first-class, machine-readable) |
| Prepared statements | Plan cache for recurring query shapes |

**The planner is the product.** Sources, embeddings, and vector search are
commodity. The engineered decision layer — what to probe, which edges to follow,
what to spend, what to pack — is the moat, exactly as the optimizer is the crown
jewel of every serious database.

### 1.1 Target caller

The caller is a **system, not an end user**: an agent, a support bot, a pipeline —
software that needs a reliable, cost-effective, repeatable way to obtain context.
This drives requirements human-facing tools don't have:

- **Determinism where possible**: same query + same data ⇒ same context.
- **Structured inputs**: callers may already hold entity IDs, source scopes, and
  freshness requirements — the API accepts them and skips inference.
- **Typed failure semantics**: partial results, budget infeasibility, and low
  confidence are data the caller can branch on, not prose.
- **Cost envelopes**: budget covers tokens *and* LLM calls *and* latency; the
  planner degrades gracefully inside the envelope.
- **Variance as a quality metric**: a predictable planner beats a brilliant
  erratic one for this buyer.

The status quo this replaces: an agent doing retrieval through a tool-calling
loop *is* a planner — one made of chain-of-thought, non-deterministic,
unauditable, and billed per token on every query. EngramDB moves planning out of
the LLM loop into an engineered layer that is deterministic on the fast path and
spends LLM calls only at genuinely ambiguous edges.

### 1.2 Posture

**Embedded library, not a server.** `pip install engramdb`; state is a single
local file next to the application. Consequences:

- Data never leaves the host environment. Probes run against the caller's own
  sources over the caller's own network with the caller's own credentials. The
  only external call is to whatever LLM the host already uses.
- "Live" is real: we can read the row at query time because we sit next to the
  database. (A SaaS platform cannot offer this without becoming a proxy in the
  customer's critical path.)
- Unit of deployment is the *application*, not the enterprise. Every deployment's
  topology cache is isolated and scoped to its own sources.

---

## 2. Core architectural commitments

These five commitments define the system. Everything in later sections derives
from them.

### C1. The graph stores identity and topology — never state

The graph knows *what exists*, *what it's called everywhere*, and *what it's
connected to*. It never knows *what's true about it right now*.

- **Identity facts** (stored): IDs, keys, aliases, types, addresses of rows /
  chunks / objects, and the edges between them. Effectively immutable — an
  invoice doesn't change which PO it belongs to.
- **State facts** (never stored): statuses, amounts, text content — anything that
  answers the caller's actual question. Volatile. Always fetched from the source
  at query time.

A cache hit skips *rediscovering the map*; the drive to the actual data happens
fresh on every query. This is the headline architectural claim: per the
literature survey, no published system makes this split (LazyGraphRAG is
text-only and doesn't distinguish; Zep/Graphiti stores fact content in edges —
the opposite choice).

Corollary: the graph is tiny (references, keys, edges — megabytes even for large
corpora), which is what makes the embedded single-file posture viable.

### C2. Graph construction is query-triggered, and the graph accretes

No corpus-wide batch construction. Entities extracted from the *query* aim
probes; results carry their own connective tissue (keys, FKs, references); we
materialize only that neighborhood; resolved edges persist. The global graph is
the accumulated union of every subgraph any query has caused to exist — its shape
mirrors the query distribution, not the corpus. Never-queried data stays
unmaterialized forever, costing nothing.

(Terminology note: we deliberately avoid branding this "lazy" — LazyGraphRAG
owns that word for a different mechanism. Ours is a **query-triggered topology
cache**.)

### C3. Deterministic edges first; inference is a scored fallback

- **Tier 1 — deterministic**: read directly off sources. Foreign keys, shared
  identifiers (UUIDs, emails, `PO-88231`-style tokens), JSON nesting,
  field-name ↔ column-name matches, file containment, exact key co-occurrence
  across sources. Free, exact, never wrong. The overwhelming majority of edges.
- **Tier 2 — inferred**: fuzzy links adjudicated by an LLM or heuristic ("does
  this SOW paragraph pertain to that invoice?"). Costly, probabilistic. Stored
  with a confidence score and the evidence that justified them. Never mixed
  indistinguishably with tier 1.

The planner exhausts tier 1 before spending on tier 2, and the cost envelope caps
tier-2 spend per query.

### C4. Genericity comes from universal invariants, not domain parsers

The system must work on any domain with zero domain code, via a three-rung
ladder:

- **Rung 1 — zero config**: SQL `information_schema` introspection, key-shape
  detection (syntactic classes + cardinality/uniqueness statistics), cross-source
  key co-occurrence, query-side entity extraction.
- **Rung 2 — hints, not parsers**: ~10 lines of config ("`accounts.name` is an
  entity name"; "match tickets to accounts via `external_ref`").
- **Rung 3 — optional domain packs**: pluggable extractors for structure-poor
  corpora (the old contract parser could return as `engramdb-pack-legal`). The
  core never requires rung 3.

Honest boundary: rung 1 depends on the data having a key spine. Operational
business data almost always does; a pile of unlinked essays does not — that
territory is conceded to batch GraphRAG (see §11).

### C5. The product boundary is the assembled context window

Not search hits. The deliverable is a packed context: selected under an explicit
token budget by a submodular objective, formatted per source type, tagged with
provenance, and accompanied by a trace explaining why each block was included.
Success is measured by what *survives into* the packed context (the
"answer-in-context" diagnostic, arXiv:2607.00725), not recall@k on the retrieved
set.

---

## 3. Public API

Python-first. The Python call *is* the query language for now; a declarative
textual surface may crystallize later from observed usage. Do not invent a
grammar in v0.1.

```python
import engramdb

# Open (or create) the local state file: topology cache, plan cache, catalog,
# and document index live here. One file, like SQLite.
eng = engramdb.open("./app_context.engram")

# Attach sources. Attachment introspects and catalogs; it does NOT bulk-ingest.
eng.attach_sql("crm", "postgresql://ro_user@db:5432/crm", read_only=True)
eng.attach_files("contracts", "/data/contracts")   # builds/updates chunk index
eng.attach_json("tickets", "./zendesk_export/")    # v0.1: via SQL JSON columns

ctx = eng.context(
    "Why did Acme Corp's invoice 4312 fail reconciliation?",
    budget=engramdb.TokenBudget(6000),
    constraints=engramdb.Constraints(          # everything optional
        known_entities=[engramdb.EntityRef(source="crm", table="invoices", pk=4312)],
        sources=["crm", "contracts"],           # scope; default = all attached
        freshness="live",                       # "live" | "cached_ok" | max_age_s=N
        cost=engramdb.CostEnvelope(max_llm_calls=2, max_latency_ms=2000),
        deterministic=True,                     # forbid nondeterministic steps
    ),
)

ctx.status        # OK | PARTIAL | BUDGET_INFEASIBLE | LOW_CONFIDENCE
ctx.blocks        # list[ContextBlock] — ranked, budget-fitted
ctx.to_prompt()   # formatted string for the LLM
ctx.trace         # PlanTrace: machine-readable JSON of every phase
ctx.cost          # CostReport: sql_queries, probes, llm_calls, tokens, latency
```

### 3.1 ContextBlock

```python
@dataclass
class ContextBlock:
    content: str                 # fetched LIVE at query time (C1)
    tokens: int
    score: float                 # final utility contribution
    provenance: Provenance       # source id, address, fetched_at
    path: list[EdgeStep]         # how the planner reached this block
    confidence: float            # 1.0 for tier-1-reached deterministic blocks
```

`path` is the edge chain from a probe hit to this block (e.g. *query entity
"invoice 4312" → SQL PK probe → FK edge to purchase_orders → key-match edge
(PO-88231) to contracts/acme_sow.pdf §3.2*). It is both the explainability story
and the audit trail.

### 3.2 Failure semantics

- `PARTIAL`: one or more sources unreachable or timed out; per-source flags in
  the trace; blocks from healthy sources still returned.
- `BUDGET_INFEASIBLE`: the minimum viable context (at least one probe hit,
  formatted) exceeds the token budget.
- `LOW_CONFIDENCE`: assembly relied on tier-2 edges below a configured
  threshold, or no probe anchored deterministically. The caller decides whether
  to proceed, widen the budget, or fall back.
- Hard errors (bad connection string, catalog mismatch) raise typed exceptions.
  Everything else is a status, not an exception — systems branch on statuses.

---

## 4. Pipeline: the nine phases

Every `context()` call runs this pipeline. Phases P1, P5 are the only ones that
may spend LLM calls; both are skippable (P1 via `known_entities`, P5 via the
cost envelope or `deterministic=True`).

```
P0 Plan lookup      → recognized query shape? replay cached plan skeleton
P1 Need analysis    → entities, identifiers, intent type
P2 Routing          → which sources, via catalog match
P3 Probe            → deterministic lookups against routed sources
P4 Expand           → materialize the neighborhood subgraph (tier-1 + cache)
P5 Adjudicate       → optional tier-2 edge confirmation (budgeted)
P6 Fetch state      → dereference every selected node ref, live
P7 Assemble         → submodular packing under budget; format; tag provenance
P8 Learn            → write back edges, aliases, plan stats
```

### P0 — Plan lookup

Production callers ask the same *shapes* of question forever ("context for
ticket {id}"). We fingerprint the query template: literals and identifiers are
slotted out (`invoice 4312` → `invoice {num}`), the remainder is normalized and
hashed together with the constraint signature. A hit yields a **plan skeleton**:
which sources were routed, which probe strategies hit, which edge types mattered,
typical fan-out. The pipeline still runs, but skips exploration — it replays the
skeleton with new slot values. This is the prepared-statement equivalent, and it
is where per-query cost collapses for exactly the callers we target.

### P1 — Need analysis

Extract from the query:

- **Identifiers**, by syntactic class: UUIDs, emails, URLs, `[A-Z]{2,6}-\d+`
  tokens, quoted strings, numbers adjacent to a catalog noun ("invoice 4312").
  Pure regex/heuristics — no LLM.
- **Entity names**: candidate proper nouns. v0.1: capitalization + catalog-value
  heuristics; optionally one small LLM/NER call when heuristics are weak and the
  envelope allows.
- **Intent type**: `lookup` (specific entity), `relational` (entity + connected
  context — the sweet spot), `semantic` (no anchors, conceptual), `mixed`.
  Determines how much of P3/P4 vs. plain vector search runs. Per
  arXiv:2506.05690, graph machinery frequently loses to plain retrieval outside
  relational queries — the planner must be free to skip the graph entirely for
  `semantic` intents.

`known_entities` in constraints bypasses extraction for those entities entirely.

### P2 — Routing

Match extracted needs against the **catalog** (§6.3): embedded schema metadata —
table names, column names, types, key statistics, sample-value sketches, source
descriptions, plus file-index metadata. Following Schema-First Retrieval
(arXiv:2606.28387): embed *catalog metadata*, not rows. Routing outputs, per
need: candidate sources with expected probe strategies, e.g. *"4312" + noun
"invoice" → crm.invoices.invoice_number (unique, int-like)*.

### P3 — Probe

Execute deterministic lookups:

- **SQL**: point queries on PK/unique/indexed columns matched in P2.
  Parameterized, read-only, `LIMIT`-guarded.
- **Key index**: exact-match lookups of extracted identifiers against the
  inverted key index (§6.4) — this is what finds `PO-88231` inside a PDF chunk
  or a JSON field without knowing the domain.
- **Vector**: semantic search over the chunk index for `semantic`/`mixed`
  intents, and as fallback when no probe anchors.

Probe hits become the **anchor set**: node refs with known addresses.

### P4 — Expand

Materialize the neighborhood around anchors, bounded (default: 2 hops,
fan-out cap per node, global node cap):

1. **Topology-cache hits first**: known edges from prior queries are followed
   for free.
2. **Tier-1 discovery at the frontier**: follow FKs of fetched rows (both
   directions, via catalog FK map); look up each new node's key values in the
   key index for cross-source co-occurrence; JSON nesting and file containment.
3. Every traversal records `(hop, edge_type, tier)` per discovered node — this
   feeds scoring in P7 (structural signal decays with hop distance; edge types
   are weighted, references/key-matches above containment).

The expansion is a *candidate* set — typically larger than what survives
packing.

### P5 — Adjudicate (tier 2, optional)

For candidate links with weak signals (name similarity, co-mention without a
shared strong key), a bounded number of adjudication calls — classification
framing, not generative ("Is chunk X about entity Y? yes/no + confidence +
evidence span"), following the fine-tuning-free classification results in the
LLM entity-matching literature (arXiv:2410.12480, 2405.16884). Results at or
above the storage threshold become tier-2 edges with confidence + evidence.
Skipped entirely when `deterministic=True` or the envelope is exhausted —
the pipeline must produce a useful (if smaller) result without P5.

### P6 — Fetch state

Dereference every candidate node ref **live**: SQL `SELECT` by address for rows;
file read (with mtime/hash check against the chunk index — re-chunk on drift)
for chunks; JSON pointer fetch for objects. Nothing is served from the graph
(C1). Fetches are batched per source and run concurrently across sources.
Failures here produce `PARTIAL`, never stale substitutes.

### P7 — Assemble

Budgeted submodular maximization, per arXiv:2607.00725 / 2601.10681:

```
maximize  f(S) = Σ_b rel(b)  +  λ_cov · coverage(S)  −  λ_red · redundancy(S)
subject to Σ_b tokens(b) ≤ budget
```

- `rel(b)`: blend of semantic similarity to the query, structural score from P4
  (edge-type weight × hop decay; anchors = 1.0), and tier confidence.
- `coverage(S)`: rewards spanning distinct entities/sub-needs from P1.
- `redundancy(S)`: penalizes near-duplicate content (embedding similarity
  between selected blocks).
- Greedy selection (near-optimal for monotone submodular objectives), with
  marginal-gain-per-token as the selection criterion.

Formatting is per source type: SQL rows render as compact `column: value` lines
with table context; chunks as quoted spans with location; every block carries
its provenance tag and path. The trace records every candidate *considered and
rejected*, with the reason (budget, redundancy, score) — the `EXPLAIN` output.

### P8 — Learn

Write-backs, transactional, after the response is formed:

- New tier-1 edges and identity nodes/aliases discovered in P4.
- Tier-2 edges that cleared threshold in P5 (with evidence).
- `last_verified` refresh on every cache edge that was re-confirmed en route.
- Plan-cache upsert: skeleton, hit statistics, cost actuals.
- Counters feeding the planner's own statistics (per-source latency, probe hit
  rates) — the equivalent of a DB's table statistics.

---

## 5. Graph model

### 5.1 Node kinds

| Kind | Address (examples) | Created by |
|---|---|---|
| `record` | `{source, table, pk}` | SQL probe / FK expansion |
| `chunk` | `{source, path, chunk_id, span}` | file index probe / key match |
| `json_obj` | `{source, doc, json_pointer}` | JSON probe |
| `file` | `{source, path}` | containment edges |
| `entity` | — (canonical identity) | entity resolution (§5.3) |

Nodes are **references**: address + kind + labels + timestamps. No content
columns exist in the schema — C1 is enforced structurally, not by convention.

### 5.2 Edge types

| Type | Tier | Discovered via |
|---|---|---|
| `fk` | 1 | schema introspection + row fetch |
| `key_match` | 1 | inverted key index co-occurrence (exact) |
| `containment` | 1 | file/dir structure, JSON nesting |
| `field_match` | 1 | column-name ↔ field-name correspondence |
| `same_as` | 1 or 2 | entity resolution (strong key = tier 1; fuzzy = tier 2) |
| `inferred` | 2 | LLM adjudication (stores confidence + evidence span) |

Every edge carries **bi-temporal validity** (borrowed from Zep/Graphiti,
arXiv:2501.13956, without their fact-content storage): `valid_from` (when the
relationship held in the world, when knowable) and `last_verified` (when we last
confirmed it against sources). `last_verified` drives re-probe policy (§7).

### 5.3 Identity nodes and entity resolution

When nodes in ≥2 sources share a strong key (same email, UUID, exact normalized
identifier), an `entity` node is created and `same_as` edges (tier 1) attach the
appearances. Weak matches (name similarity across sources) go through P5 and
attach as tier-2 `same_as` with confidence. Aliases accumulate on the entity
node ("Acme Corp" / "Acme Corporation" / "Acme") — this lazily-built,
query-driven alias map is the cross-source Rosetta stone, and one of the most
durable assets in the cache. Prior art acknowledged: on-demand ER has DB-era
roots (arXiv:1111.0045, 2011; FastER, arXiv:2504.01557) — our contribution is
its application inside an LLM context layer across heterogeneous live sources.

---

## 6. Storage layout

One embedded database file (v0.1: SQLite via stdlib, with DuckDB considered for
the analytics-heavy catalog; decision at implementation — the schema below is
engine-neutral). Everything EngramDB persists lives here:

### 6.1 Topology tables

```sql
nodes(node_id PK, kind, source_id, address JSON, labels JSON,
      first_seen, last_verified)
edges(edge_id PK, src FK, dst FK, edge_type, tier, key_evidence,
      confidence, valid_from, last_verified, created_by_query)
entities(entity_id PK, canonical_name, created_at)
aliases(entity_id FK, alias, source_id, strength)
```

### 6.2 Key index (inverted)

```sql
key_postings(key_norm, key_class, node_id FK, location JSON)
-- key_norm: normalized literal ("po-88231"); key_class: uuid|email|id_token|...
```

Populated two ways: (a) file/JSON indexing extracts identifier-shaped tokens
from content at attach/refresh time — cheap regex, no LLM; (b) SQL key values
encountered during probes/expansion are posted opportunistically (we do NOT
bulk-scan SQL tables at attach time; postings accumulate with use, consistent
with C2). A cross-source `key_match` edge is two postings with the same
`key_norm` in different sources.

### 6.3 Catalog

```sql
catalog_sources(source_id PK, kind, dsn_ref, attached_at, schema_hash)
catalog_units(source_id FK, unit,          -- table/column/file-collection
              meta JSON,                    -- types, key stats, fk targets
              embedding BLOB)               -- for P2 routing
```

Built at `attach_*` time from introspection (SQL) or a walk (files). Refreshed
when `schema_hash` drifts. Row *content* is never cataloged — only metadata and
small statistical sketches (cardinality, uniqueness ratio, value shape).

### 6.4 Document chunk index

```sql
chunks(chunk_id PK, source_id, path, span, content_hash, mtime, embedding BLOB)
```

The one place where derived data from content is persisted — embeddings and
hashes, which are *index* data over slow-changing sources, not cached state:
chunk text itself is still re-read from the file at P6, and `mtime`/hash drift
triggers re-chunking. Live SQL content is never embedded or indexed in v0.1 —
only its schema/catalog is. Vector search backend v0.1: brute-force over
in-file embeddings (numpy), with `sqlite-vec`/HNSW as the upgrade path — at
embedded scale (10⁴–10⁵ chunks) brute force is fine and keeps dependencies at
zero.

### 6.5 Plan cache

```sql
plans(template_hash PK, skeleton JSON, hits, last_used,
      cost_actuals JSON, catalog_hash)
```

Invalidated when `catalog_hash` changes (schema migrations).

---

## 7. Caching and invalidation semantics

The rules, stated once:

1. **State is never cached.** P6 always dereferences live. No exceptions, no
   "while we're at it" content caching — this is the failure mode that would
   silently rebuild the drifting second-source-of-truth we reject.
2. **Topology is cached indefinitely, verified opportunistically.** Every time a
   cached edge is traversed and its endpoints both dereference successfully,
   `last_verified` refreshes for free. An edge whose endpoint dereference fails
   (row deleted, file gone) is tombstoned immediately.
3. **Staleness policy is per source class**: cached edges older than a TTL
   (default: generous for FK/containment edges — they are near-immutable;
   shorter for tier-2 edges) trigger background re-verification on hit, not
   blocking the query.
4. **Neighborhood completeness** (new documents arriving that mention an
   already-cached key): on cache-hit expansions, the planner re-probes the key
   index for the anchor's strong keys — postings lookups are cheap — so new
   arrivals join the neighborhood on the next query that visits it.
5. **Known failure mode, owned**: the on-demand-ER literature warns that lazy
   work gets redone for popular queries unless cached (FastER). The topology +
   plan caches are the remedy; the eval plan (§10) therefore must measure
   cache-hit rate and cost amortization curves, because that is where naive
   lazy systems lose to batch systems.

---

## 8. Cost model and determinism

Every response carries a `CostReport`:

```python
@dataclass
class CostReport:
    sql_queries: int; key_probes: int; vector_searches: int
    llm_calls: int; llm_tokens: int
    latency_ms: dict[str, int]        # per phase
    cache: dict[str, float]           # topology hit rate, plan hit
```

The `CostEnvelope` is enforced by degradation, not failure: exhaust tier-2
budget → skip P5; latency budget tight → cached edges only in P4, skip
re-verification; `deterministic=True` → no LLM anywhere in the pipeline (P1
heuristics only, P5 off). A query layer called 10⁴×/day must have a fully
deterministic, LLM-free fast path — LLM steps are escape hatches, not the spine.

---

## 9. Security posture

- All SQL access is read-only by contract: `attach_sql` requires (and verifies
  where the engine allows) a read-only role; all queries are parameterized; no
  DDL/DML ever issued.
- EngramDB inherits the host's credentials — it does not manage permissions.
  This is stated honestly: an agent holding a read-everything DB user will pack
  context an end user perhaps shouldn't see. Mitigations in v0.1: per-source
  scoping in constraints, and the provenance trace showing exactly what was
  touched (auditability). Row-level permission awareness is future work and a
  known gap vs. platform products (Glean's genuinely hard-won asset).
- The state file contains identifiers, keys, aliases, and embeddings — treat it
  with the same sensitivity as an application database file.

---

## 10. Evaluation plan

Two axes: **context quality** and **systems behavior**. Baselines: naive vector
RAG; LlamaIndex `SQLAutoVectorQueryEngine` (the routing status quo); an
agent-with-tools loop (the planning status quo).

Quality:

- **Answer-in-context at fixed budgets** (primary metric, per arXiv:2607.00725):
  does the gold fact survive as a span in the *packed* context?
- **Live-state correctness** (the eval batch systems fail by construction):
  mutate the database between queries; measure whether packed context reflects
  post-mutation state. Target: 100% by architecture.
- Multi-hop retrieval comparability: MuSiQue / HotpotQA / 2WikiMultiHop (shared
  with SAG, HippoRAG); HybridQA / TAT-QA for table+text fusion; GraphRAG-Bench
  (arXiv:2506.05690) to show *when* the graph layer earns its keep — and to
  verify the planner correctly skips it when it doesn't.

Systems behavior:

- **Amortization curve**: cost and latency per query vs. query count over a
  realistic workload (repeated shapes, overlapping neighborhoods) — the
  cache-hit story, plotted, including cold-start first-query latency.
- **Variance**: repeatability of block selection across runs (deterministic
  mode: must be exact); tail latency; cost-per-query distribution.
- **Entity-resolution precision/recall** on cross-source matches, against the
  LLM entity-matching baselines (arXiv:2405.16884 line).

---

## 11. Prior art and differentiation

Full survey: `research/literature_survey_2026-08.md`. The one-slide versions:

- **vs. GraphRAG** (arXiv:2404.16130): they batch-build an LLM-extracted graph
  over the whole corpus at ingest; we build nothing until a query demands it,
  and never store extracted content.
- **vs. LazyGraphRAG** (MSR blog, Nov 2024): they defer *LLM claim extraction*
  over a *static text corpus*, re-deriving relevance per query; we defer *graph
  construction itself* over *live heterogeneous sources*, accumulate a
  persistent topology cache, and re-fetch volatile state. No live SQL, no
  cross-query cache, no identity/state split on their side.
- **vs. SAG** (arXiv:2606.15971 — closest mechanical neighbor): query-time
  SQL-join hyperedges on shared entity keys, but over homogeneous
  document-derived event tables; no persistent cross-query accumulation, no
  live multi-source federation, no live re-fetch. Benchmark head-to-head on
  MuSiQue/HotpotQA.
- **vs. Zep/Graphiti** (arXiv:2501.13956): we borrow bi-temporal edge validity;
  they store fact content in edges (opposite of C1) and target conversational
  memory, not federated data retrieval.
- **vs. Glean**: they copy the enterprise into their platform on crawl schedules
  and sell search to employees; we live inside the application, read state live,
  and sell context to systems. Different trust boundary, different buyer,
  different freshness guarantee.
- **vs. LlamaIndex/LangChain routers**: LLM-selector routing between a SQL tool
  and a vector tool — no topology cache, no cross-source entity resolution, no
  plan cache, no budget-optimal packing, no trace. This is the naive baseline.

Conceded territory (deliberately): corpus-wide synthesis questions ("themes
across all our contracts") belong to batch GraphRAG's community summaries; pure
prose corpora without a key spine give rung-1 nothing to grip. A
GraphRAG-style summarization index could later become one more executor behind
the planner — the architecture doesn't preclude it.

---

## 12. v0.1 scope

Prove the thesis end-to-end, deliberately narrow:

**In**: Python ≥3.11. Sources: SQLite + Postgres (`attach_sql`), directory of
text/markdown files (`attach_files`). Embedders: mock / OpenAI / local
(sentence-transformers) behind the existing-style ABC. Planner: heuristic-only
P1 (regex identifier extraction + catalog matching; no LLM required); P5
stubbed behind its interface (tier-2 edges designed-in, not implemented).
Tier-1 edges: `fk`, `key_match`, `containment`. Topology cache + key index +
catalog + plan cache in one SQLite file. Greedy submodular packer. Full trace
and cost report. `deterministic=True` fully honored.

**Out (designed for, not built)**: JSON stores as first-class (v0.1 route:
SQL JSON columns), blob/OCR extraction, LLM-assisted P1/P5, background
re-verification workers, row-level permissions, HNSW indexing, concurrency
beyond per-source parallel fetch.

**The proving demo**: a seeded SQLite database + a directory of documents and
a question whose correct answer requires a row, an FK-linked row, and a
key-matched document chunk — answered by one `context()` call, correctly
re-answered after a `UPDATE` to the database (live-state), and cheaper on the
second run (cache). Side-by-side with naive RAG and a LlamaIndex router failing
or going stale on the same script.

**Milestones**:

- **M0**: repo reset (tag legacy, clean tree), package scaffold, state-file
  schema, `attach_sql` introspection + catalog.
- **M1**: P1–P3 (need analysis, routing, probes) + key index over files.
- **M2**: P4 + topology cache (expansion, edge persistence, verification).
- **M3**: P6–P7 (live fetch, packer, trace, cost report).
- **M4**: P0/P8 (plan cache, learning), proving demo, eval harness with
  answer-in-context + live-state + amortization measurements.

---

## 13. Open questions

1. **SQLite vs. DuckDB for the state file** — SQLite: zero-dep, battle-tested
   concurrent readers; DuckDB: better analytics for catalog/stats, native
   Postgres attach (could double as the SQL executor). Decide at M0.
2. **Chunking policy for files** — structural (headings) vs. fixed-window; how
   much of rung-3 lives in the default file adapter.
3. **Plan-skeleton granularity** — how much execution detail to replay vs.
   re-derive; when a skeleton should be evicted for underperforming.
4. **Score calibration across sources** — a SQL point hit and a 0.83-cosine
   chunk need comparable `rel(b)`; initial approach is rank-based normalization
   per executor, to be validated in M4 evals.
5. **Naming** — package stays `engramdb`. Inclination on record: the "DB"
   suffix may eventually yield to a query-flavored identity (EngramQ or
   similar) as the planner-not-database positioning hardens. Decision
   deliberately deferred; not load-bearing now.
