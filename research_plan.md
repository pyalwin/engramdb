# EngramDB: Research Plan

## Schema-Aware Hybrid Retrieval for Multi-Hop Legal Reasoning

---

## 1. Problem Statement

Large Language Models (LLMs) augmented with Retrieval-Augmented Generation (RAG) have become the standard for knowledge-intensive tasks. However, current vector-based RAG systems fail on **multi-hop reasoning**—queries requiring the synthesis of multiple, non-adjacent pieces of information.

**Example Failure:**
- *Clause A:* "Confidential Information is defined in Section 5.2"
- *Clause B (Section 5.2):* "Confidential Information excludes publicly available data"
- *Clause C:* "Breach of confidentiality allows immediate termination"
- *Query:* "Can we terminate if the other party shares our public press releases?"

Vector RAG retrieves chunks by semantic similarity, often missing the chain: Definition → Exclusion → Consequence. The LLM hallucinates or fails to answer.

**Core Hypothesis:**
Structured documents (legal contracts) contain explicit relational information (cross-references, defined terms, section hierarchy) that can be extracted and leveraged for superior multi-hop retrieval.

---

## 2. Proposed Solution: EngramDB

EngramDB is a hybrid retrieval engine that combines:

1. **Vector Layer:** Semantic embeddings for fuzzy matching
2. **Graph Layer:** Explicit relationships extracted from document structure
3. **Schema-Aware Ingestion:** Rule-based extraction of document-native schema

### 2.1 Key Innovation

Unlike generic graph+vector systems (Neo4j, Weaviate), EngramDB does not require users to define schemas or relationships. The **document declares its own schema** through:

- Section headings and hierarchy
- Defined terms (`"X" means...`)
- Cross-references (`Section 3.2`, `as defined herein`)
- Party definitions and obligations

We extract this structure using deterministic, rule-based parsing—avoiding LLM extraction errors.

### 2.2 Retrieval Mechanism: Hybrid Traversal

1. **Anchor:** Vector similarity identifies the most relevant entry node
2. **Traverse:** Graph edges are followed (2-hop) to gather connected context
3. **Aggregate:** Anchor node + traversed nodes form the retrieval context
4. **Generate:** LLM receives enriched context for answer generation

---

## 3. Scope and Constraints

### In Scope
- Legal contracts (NDAs, MSAs, employment agreements, license agreements)
- Clean/structured text input (Markdown, plaintext with clear formatting)
- English language documents
- Multi-hop reasoning evaluation

### Out of Scope (for initial research)
- PDF/OCR parsing (assumes preprocessing is done)
- Non-English documents
- Real-time/streaming ingestion
- Multi-document reasoning (focus on single-document first)

---

## 4. Technical Architecture

### 4.1 Stack
- **Database:** DuckDB (embedded, supports vectors via VSS extension)
- **Language:** Python
- **Embeddings:** OpenAI text-embedding-3-small or open-source alternative (e5-base)
- **LLM for QA:** GPT-4 or Claude (consistent across baseline and EngramDB)

### 4.2 Data Model

```
Engram (Node)
├── id: UUID
├── content: TEXT (raw clause/section text)
├── embedding: FLOAT[1536] (vector)
├── type: ENUM (section, definition, clause, party, date)
├── metadata: JSON (section_number, parent_section, source_location)
└── timestamp: DATETIME

Synapse (Edge)
├── source_id: UUID
├── target_id: UUID
├── type: ENUM (REFERENCES, DEFINES, PARENT_OF, RELATED_TO)
└── metadata: JSON (confidence, extraction_method)
```

### 4.3 Ingestion Pipeline

```
Raw Contract Text
       │
       ▼
┌──────────────────┐
│ Section Parser   │ ──→ Identify headings, hierarchy
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Definition       │ ──→ Extract "X" means patterns
│ Extractor        │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Cross-Reference  │ ──→ Find Section X.Y, Article N references
│ Linker           │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Embedding        │ ──→ Generate vectors for each node
│ Generator        │
└──────────────────┘
       │
       ▼
     EngramDB
```

### 4.4 Query Pipeline

```
User Query
       │
       ▼
┌──────────────────┐
│ Query Embedding  │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Vector Search    │ ──→ Top-K anchor nodes (K=3)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Graph Traversal  │ ──→ 2-hop expansion from anchors
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Context Assembly │ ──→ Deduplicate, order by relevance
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ LLM Generation   │ ──→ Answer with citations
└──────────────────┘
```

---

## 5. Benchmark Design

### 5.1 Dataset

| Source | Count | Purpose |
|--------|-------|---------|
| Synthetic contracts (template-generated) | 40-50 | Controlled ground truth, systematic complexity variation |
| Real contracts (CUAD subset) | 10-15 | Generalization validation |

**Synthetic Contract Generation:**
- 5-10 contract templates (NDA, MSA, Employment, SaaS, License)
- Variable substitution for parties, dates, terms, values
- Deliberately structured multi-hop dependencies

**CUAD Annotation:**
- Select contracts with rich cross-referencing
- Manually create 5-10 multi-hop questions per contract
- Annotate reasoning chain (Clause A → Clause B → Answer)

### 5.2 Query Types

| Type | Hops | Example | Expected RAG Performance |
|------|------|---------|--------------------------|
| Single-hop | 1 | "What is the termination notice period?" | Good (~80%) |
| Two-hop | 2 | "Can we assign this contract to a subsidiary?" | Poor (~30%) |
| Three-hop | 3 | "If we breach confidentiality of Proprietary Data, can they terminate without notice?" | Very Poor (~15%) |

### 5.3 Baselines

1. **Naive RAG:** Chunk document (512 tokens) → Embed → Vector search → Top-K to LLM
2. **Sentence-Window RAG:** Retrieve sentence + surrounding context
3. **Parent-Document RAG:** Retrieve chunk, expand to parent section

All baselines use identical embedding model and LLM for fair comparison.

### 5.4 Metrics

| Metric | Definition |
|--------|------------|
| **Answer Accuracy** | Exact match or semantic equivalence with ground truth |
| **Retrieval Recall** | % of required clauses present in retrieved context |
| **Faithfulness** | Does the answer cite only retrieved content? (no hallucination) |
| **Hop Success Rate** | Accuracy broken down by reasoning chain length |

---

## 6. Evaluation Protocol

### 6.1 Procedure

1. Ingest contract into both EngramDB and baseline systems
2. Run identical query set against both systems
3. Record retrieved context and generated answer
4. Compare against ground truth
5. Compute metrics

### 6.2 Statistical Rigor

- Report mean and standard deviation across contracts
- Paired t-test for significance (p < 0.05)
- Ablation study: Vector-only vs Graph-only vs Hybrid

### 6.3 Success Criteria

| Query Type | Baseline Expected | EngramDB Target |
|------------|-------------------|-----------------|
| Single-hop | ~80% | ≥80% (no regression) |
| Two-hop | ~30% | ≥60% |
| Three-hop | ~15% | ≥50% |

**Primary claim:** 2x improvement on multi-hop reasoning tasks.

---

## 7. Expected Contributions

1. **Empirical Evidence:** First rigorous benchmark demonstrating graph+vector superiority on multi-hop legal reasoning

2. **Schema-Aware Ingestion:** Novel approach using document-native structure rather than LLM extraction

3. **Reproducible Benchmark:** Open-source dataset of synthetic contracts with multi-hop QA pairs

4. **Practical Architecture:** Embedded, lightweight implementation (DuckDB) suitable for production

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Rule-based extraction fails on varied formats | Medium | High | Start with well-structured contracts; expand rules iteratively |
| Graph traversal returns too much noise | Medium | Medium | Tune hop depth; add relevance filtering |
| Baseline performs better than expected | Low | High | Ensure baseline is properly optimized; use multiple baselines |
| Synthetic contracts don't reflect real-world complexity | Medium | Medium | Validate on CUAD subset; report both results |
| DuckDB vector performance issues | Low | Low | Fall back to dedicated vector store if needed |

---

## 9. Project Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up DuckDB with vector extension
- [ ] Implement basic data model (Engram, Synapse)
- [ ] Build section parser for contracts
- [ ] Build definition extractor
- [ ] Build cross-reference linker

### Phase 2: Retrieval (Weeks 3-4)
- [ ] Implement vector search
- [ ] Implement graph traversal
- [ ] Build hybrid retrieval pipeline
- [ ] Integrate with LLM for answer generation

### Phase 3: Benchmark (Weeks 5-6)
- [ ] Create contract templates
- [ ] Generate synthetic dataset (40-50 contracts)
- [ ] Create multi-hop question set
- [ ] Annotate CUAD subset (10-15 contracts)

### Phase 4: Evaluation (Weeks 7-8)
- [ ] Implement baseline RAG systems
- [ ] Run experiments
- [ ] Compute metrics
- [ ] Statistical analysis

### Phase 5: Paper (Weeks 9-10)
- [ ] Write paper draft
- [ ] Create figures and tables
- [ ] Internal review and revision
- [ ] Submit

---

## 10. Target Venues

| Venue | Type | Deadline (typical) | Fit |
|-------|------|-------------------|-----|
| EMNLP | Conference | June | Strong (NLP + retrieval) |
| ACL | Conference | January | Strong |
| NAACL | Conference | December | Strong |
| SIGIR | Conference | January | Good (IR focus) |
| CIKM | Conference | May | Good |
| arXiv | Preprint | Anytime | For early visibility |

---

## 11. Open Questions

1. **Embedding model choice:** Open-source (reproducibility) vs proprietary (performance)?
2. **Hop depth:** Fixed 2-hop or adaptive based on query complexity?
3. **Edge weighting:** Should some relationships be stronger than others?
4. **Chunking strategy:** How to segment contracts into nodes optimally?

---

## 12. References

- CUAD Dataset: https://github.com/TheAtticusProject/cuad
- DuckDB VSS: https://duckdb.org/docs/extensions/vss
- RAG Survey: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- GraphRAG: Microsoft Research, 2024

---

*Document Version: 1.0*
*Last Updated: December 2024*
