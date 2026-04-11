"""
Parent-Document RAG Baseline: Section-aware chunking with parent expansion.

Parses document into sections (reusing EngramDB's section parser), embeds
each section, and on retrieval expands matched chunks to include their
parent section. Uses only PARENT_OF hierarchy — no cross-references or
definition edges.

This baseline demonstrates the value of multi-edge-type graph traversal
by showing what single-edge hierarchy alone achieves.
"""

from engramdb.core.engram import Engram, EngramType
from engramdb.core.synapse import Synapse, SynapseType
from engramdb.embeddings.embedder import Embedder, create_embedder
from engramdb.storage.duckdb import DuckDBStorage
from engramdb.ingestion.parser import SectionParser
from engramdb.retrieval.hybrid import RetrievalResult


class ParentDocumentRAG:
    """
    Parent-Document RAG: retrieve chunk, expand to parent section.

    Pipeline:
    1. Parse document into sections (structural chunking)
    2. Embed each section
    3. Query: vector search for top-K, then expand each to parent
    4. Deduplicate and return
    """

    def __init__(
        self,
        embedding_backend: str = "mock",
        embedder: Embedder | None = None,
    ):
        self.storage = DuckDBStorage(None)
        self.storage.connect()
        self.embedder = embedder or create_embedder(backend=embedding_backend)
        self.parser = SectionParser()
        # child_id -> parent_id mapping
        self._parent_map: dict[str, str] = {}

    def ingest(self, document: str, doc_id: str) -> int:
        """Parse into sections, embed, and store. Returns engram count."""
        sections = self.parser.parse(document)

        flat_sections = []
        for s in sections:
            flat_sections.extend(s.flatten())

        engrams = []
        section_id_map: dict[str, str] = {}

        for section in flat_sections:
            heading_parts = []
            if section.number:
                heading_parts.append(f"Section {section.number}")
            if section.title:
                heading_parts.append(section.title)
            heading = " - ".join(heading_parts).strip()
            body = section.content.strip()
            content = f"{heading}\n\n{body}" if heading and body else heading or body

            engram = Engram(
                content=content,
                engram_type=EngramType.SECTION,
                metadata={
                    "document_id": doc_id,
                    "section_number": section.number,
                    "title": section.title,
                    "level": section.level,
                },
            )
            engrams.append(engram)
            if section.number:
                section_id_map[section.number] = engram.id
            if section.title:
                section_id_map[section.title] = engram.id

        # Build parent map from hierarchy
        for section in sections:
            self._build_parent_map(section, section_id_map)

        # Embed
        texts = [e.content for e in engrams]
        embeddings = self.embedder.embed_batch(texts)
        for engram, embedding in zip(engrams, embeddings):
            engram.embedding = embedding

        self.storage.insert_engrams_batch(engrams)
        return len(engrams)

    def _build_parent_map(self, section, section_id_map: dict[str, str]):
        """Recursively build child->parent mapping."""
        parent_id = section_id_map.get(section.number) or section_id_map.get(section.title)
        for child in section.children:
            child_id = section_id_map.get(child.number) or section_id_map.get(child.title)
            if parent_id and child_id:
                self._parent_map[child_id] = parent_id
            self._build_parent_map(child, section_id_map)

    def query(self, question: str, top_k: int = 5, max_context_items: int = 10) -> RetrievalResult:
        """
        Retrieve sections by vector search, then expand to parent sections.
        """
        query_embedding = self.embedder.embed(question)
        results = self.storage.search_similar(query_embedding, top_k=top_k)

        seen_ids: set[str] = set()
        final_engrams: list[Engram] = []
        scores: dict[str, float] = {}

        for engram, score in results:
            if len(final_engrams) >= max_context_items:
                break

            # Add the matched section
            if engram.id not in seen_ids:
                final_engrams.append(engram)
                scores[engram.id] = score
                seen_ids.add(engram.id)

            # Expand to parent
            parent_id = self._parent_map.get(engram.id)
            if parent_id and parent_id not in seen_ids:
                parent = self.storage.get_engram(parent_id)
                if parent:
                    final_engrams.append(parent)
                    scores[parent.id] = score * 0.8  # Discount parent slightly
                    seen_ids.add(parent.id)

        final_engrams = final_engrams[:max_context_items]

        return RetrievalResult(
            engrams=final_engrams,
            anchor_ids=[e.id for e, _ in results],
            traversed_ids=[eid for eid in seen_ids if eid not in {e.id for e, _ in results}],
            scores=scores,
        )

    def close(self):
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
