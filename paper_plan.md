# Paper Plan: EngramDB

## Target
**Title:** Schema-Aware Hybrid Retrieval for Multi-Hop Legal Reasoning
**Venue:** arXiv preprint (immediate), then EMNLP NLLP Workshop / SIGIR / CIKM
**Format:** 8-page conference paper + appendix

---

## Phase 1: Strengthen the Evaluation (Weeks 1-2)

The current benchmark measures **retrieval recall** (do we get the right sections?) but not **end-to-end answer quality**. The paper needs both.

### 1a. Fix evaluation gaps

| Gap | What to do | Why it matters |
|-----|-----------|----------------|
| No end-to-end accuracy | Add LLM-as-judge evaluation: feed retrieved context to GPT-4o-mini, compare answer to gold answer using both exact match and LLM semantic equivalence | Reviewers will ask "does better retrieval actually help the LLM answer correctly?" |
| No statistical significance | Add bootstrap confidence intervals (95%) on all main metrics. We have 183 questions — enough for meaningful CIs | Without this, the 24.6pp improvement could be dismissed as noise |
| Questions are template-generated | Audit a random sample of 30 questions manually. Fix or discard any that are trivially answerable or have wrong gold answers | Protects against inflated numbers from easy/broken questions |
| Only one baseline | Add **Parent-Document RAG** (retrieve chunk, expand to parent section) and **BM25 + reranker** as baselines — these are what practitioners actually use | Vector-only is a strawman. Reviewers expect at least 3 baselines |

### 1b. Add ablation study

Run the benchmark with these configurations to isolate what matters:

| Configuration | Purpose |
|--------------|---------|
| Vector-only (current baseline) | Lower bound |
| Graph-only (traverse from keyword-matched anchors, no embeddings) | Shows graph contribution in isolation |
| Hybrid without traversal reservation | Shows value of the reservation mechanism |
| Hybrid without cross-reference edges | Shows value of cross-ref extraction |
| Hybrid without definition edges | Shows value of definition extraction |
| Full hybrid (current system) | Upper bound |

### 1c. Per-contract variance analysis

- Report per-contract recall (not just overall mean) to show consistency
- Identify and discuss failure cases: contracts where hybrid doesn't help (why?)
- Already have `tune_hybrid.py` failure buckets — formalize this into the paper

**Deliverable:** Updated `benchmark.py` with new baselines, ablations, and end-to-end evaluation. New results JSON.

---

## Phase 2: Additional Experiments (Week 3)

### 2a. Scalability analysis

- Measure ingestion time and retrieval latency as document length grows (use contracts of varying size from CUAD)
- Plot: sections per document vs. retrieval time (hybrid vs. vector-only)
- Show that graph overhead is minimal (DuckDB is fast)

### 2b. Sensitivity analysis

- Use existing `tune_hybrid.py` grid search results
- Plot performance vs. key parameters: `semantic_weight`, `hop_decay`, `min_traversed_items`
- Show the system isn't brittle — works across a range of hyperparameters

### 2c. Edge type contribution analysis

- Breakdown of recall improvement by edge type: which edges (REFERENCES, DEFINES, PARENT_OF) contribute most to the improvement?
- This is the "what structure matters?" question that makes the paper insightful, not just a system paper

**Deliverable:** Figures and tables for scalability, sensitivity, and edge contribution analysis.

---

## Phase 3: Write the Paper (Weeks 4-5)

### Paper structure

#### 1. Introduction (1 page)
- **Hook:** "RAG fails when the answer requires connecting information across structurally linked but semantically distant document sections"
- Concrete legal contract example (the Termination → Cause → Definitions chain from `research_plan.md`)
- Key insight: structured documents declare their own schema through headings, defined terms, and cross-references
- Contribution bullets:
  1. Schema-aware ingestion pipeline that extracts document structure with deterministic rules (no LLM calls)
  2. Hybrid retrieval with edge-type-aware traversal reservation that outperforms vector-only by +24.6pp recall on multi-hop legal QA
  3. Open-source benchmark of 183 multi-hop questions across 35 CUAD contracts

#### 2. Related Work (1 page)
- **RAG improvements:** Sentence-window, Parent-Document, RAPTOR, Self-RAG — all modify chunking or generation, none exploit document-declared structure
- **Graph-augmented retrieval:** Microsoft GraphRAG (LLM-extracted entities — expensive, non-deterministic), KG-RAG, G-Retriever — contrast with our rule-based extraction
- **Legal NLP:** CUAD, LegalBench, domain-partitioned RAG (arXiv:2602.23371 using Neo4j) — position as complementary
- **Key differentiation:** We don't extract entities or build a knowledge graph from scratch. We parse the document's own structural declarations.

#### 3. Method (2 pages)

**3.1 Problem Formulation**
- Define multi-hop retrieval: query Q requires retrieving sections {S₁, S₂, ..., Sₖ} where Sᵢ and Sᵢ₊₁ are connected by structural edges but may share no vocabulary with Q
- Formal notation for the graph: G = (V, E) where V = engrams, E = synapses with typed edges

**3.2 Schema-Aware Ingestion**
- Section hierarchy extraction (regex-based heading parser)
- Definition extraction (`"X" means...` patterns → DEFINES edges)
- Cross-reference linking (`Section N.M`, `as defined herein` → REFERENCES edges)
- Complexity analysis: O(n) per document, no API calls
- Include the ingestion pipeline diagram from `research_plan.md`

**3.3 Hybrid Retrieval**
- Step 1: Vector search → top-K anchors
- Step 2: Typed graph traversal (bidirectional BFS, N hops, edge-type weights)
- Step 3: Blended scoring: `score = α · semantic_sim + (1-α) · structural_score`
  - structural_score = edge_weight × hop_decay^(hop-1) for traversed nodes
  - structural_score = 1.0 for anchors
  - structural_score = 0.0 for backfill nodes
- Step 4: Traversal reservation — guarantee min_traversed_items non-anchor slots, prioritizing high-value edges (REFERENCES/DEFINES ≥ 0.9)
- This is the core algorithmic contribution — explain *why* reservation is needed (large anchor budgets can wash out graph signal)

**3.4 Implementation**
- DuckDB (embedded, single-file), Python, ~2,400 LOC
- Embedding: OpenAI text-embedding-3-small (for reproducibility, report dimensionality)

#### 4. Experimental Setup (1 page)

**4.1 Dataset**
- CUAD: 35 contracts from SEC filings (public, CC-BY-4.0)
- Multi-hop question generation: rule-based from extracted graph structure (describe the `multihop_generator.py` approach)
- 183 questions: definition_usage, cross_reference, termination_chain types
- Hop distribution: 2-hop and 3-hop questions
- Discuss limitations of generated questions (address in limitations section too)

**4.2 Baselines**
- Vector-only (same embeddings, top-K=15)
- Parent-Document RAG
- BM25 + cross-encoder reranker
- (Optional: GraphRAG if time permits, but acknowledge as future work if not)

**4.3 Metrics**
- Retrieval Recall@K (primary)
- Hop Coverage (what fraction of the reasoning chain is retrieved)
- End-to-end Answer Accuracy (LLM-as-judge)
- Latency (ms per query)

**4.4 Hyperparameters**
- Report the grid search from `tune_hybrid.py`
- Final config: top_k_anchors=8, max_hops=2, max_context_items=15, min_traversed_items=4

#### 5. Results (1.5 pages)

**Table 1: Main results** — Retrieval recall and answer accuracy across all methods, broken down by hop count

**Table 2: By question type** — definition_usage, cross_reference, termination_chain

**Table 3: Ablation study** — Which components contribute (see Phase 1b)

**Figure 1: Edge type contribution** — Bar chart of recall improvement attributable to each edge type

**Figure 2: Sensitivity analysis** — Line plots of recall vs. semantic_weight, hop_decay, min_traversed_items

**Figure 3: Per-contract variance** — Box plot showing consistency across contracts

#### 6. Analysis (0.5 pages)
- Why does hybrid help most on cross-reference questions? (Vocabulary gap is largest)
- When does hybrid hurt? (Discussion of failure cases from `analyze_hybrid_failures.py`)
- Traversal reservation: show cases where it saves nodes that would be ranked out

#### 7. Limitations and Future Work (0.5 pages)
- Questions are template-generated from graph structure (circular bias risk — acknowledge and mitigate with manual audit subset)
- Single-document only (multi-document future work)
- English legal contracts only (format-dependent rules)
- No PDF/OCR handling (assumes clean text input)
- Rule-based extraction may miss implicit relationships
- Future: adaptive hop depth, LLM-augmented edge extraction for edge cases, multi-document reasoning

#### 8. Conclusion (0.25 pages)
- Structured documents provide free structural signals that dramatically improve multi-hop retrieval
- Lightweight, deterministic extraction beats the status quo with no LLM ingestion cost

#### References (~40 citations)

#### Appendix
- Full hyperparameter grid search results
- Sample questions and retrieved contexts (qualitative examples)
- Complete list of extraction rules (regex patterns)

---

## Phase 4: Figures and Tables (Week 5, parallel with writing)

| Figure/Table | Source data | Tool |
|-------------|-----------|------|
| Table 1: Main results | benchmark_results.json | LaTeX |
| Table 2: By question type | benchmark_results.json | LaTeX |
| Table 3: Ablation | New ablation runs | LaTeX |
| Figure 1: Architecture diagram | Manual | TikZ or draw.io |
| Figure 2: Edge type contribution | New analysis | matplotlib → PDF |
| Figure 3: Sensitivity | tune_hybrid.py output | matplotlib → PDF |
| Figure 4: Per-contract variance | New analysis | matplotlib → PDF |
| Figure 5: Qualitative example | Manual selection | LaTeX listing |

---

## Phase 5: Review and Submit (Week 6)

### Internal review checklist
- [ ] All claims backed by numbers in the results section
- [ ] No overclaiming (we improve *retrieval*, acknowledge limits of generated questions)
- [ ] Reproducibility: all hyperparameters reported, code will be open-sourced
- [ ] Figures are readable in grayscale (accessibility)
- [ ] Related work is fair to competitors (especially GraphRAG)
- [ ] Limitations section is honest

### Submission
1. Post to arXiv (cs.IR or cs.CL)
2. Simultaneously prepare camera-ready for target venue
3. Open-source the code + benchmark dataset on GitHub

---

## Division of Work (Suggested)

| Task | Lead | Support |
|------|------|---------|
| Additional baselines (Parent-Doc, BM25) | Co-author | You |
| End-to-end evaluation (LLM-as-judge) | Co-author | You |
| Statistical analysis (CIs, significance) | Co-author | You |
| Ablation study | You | Co-author |
| Writing: Introduction, Method | You | Co-author reviews |
| Writing: Related Work | Co-author | You review |
| Writing: Experiments, Results | Split | Split |
| Writing: Analysis, Limitations | You | Co-author reviews |
| Figures and tables | Split | Split |
| Final review pass | Both | Both |

---

## Timeline Summary

| Week | Milestone |
|------|-----------|
| 1 | New baselines implemented, end-to-end eval working |
| 2 | Ablation study complete, statistical analysis done |
| 3 | Sensitivity/scalability/edge analysis experiments |
| 4 | Paper draft sections 1-4 |
| 5 | Paper draft sections 5-8 + figures |
| 6 | Internal review, revisions, arXiv submission |

---

## Critical Risks

| Risk | Mitigation |
|------|-----------|
| End-to-end accuracy doesn't improve proportionally to retrieval recall | Frame paper as retrieval contribution; report both metrics honestly |
| Template-generated questions inflate results | Manual audit of 30 questions; create 20 manually-written questions as validation set |
| Co-author can't commit time | Paper is structured so it can ship as single-author arXiv preprint with reduced baselines |
| GraphRAG comparison requested by reviewers | Acknowledge in Related Work; note different cost model (LLM extraction vs. rule-based); leave as future work if not feasible |
