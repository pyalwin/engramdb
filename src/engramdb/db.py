"""
EngramDB: Main database class that ties everything together.

This is the primary user-facing API for ingesting documents,
building the knowledge graph, and performing hybrid retrieval.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .core.engram import Engram, EngramType
from .core.synapse import Synapse, SynapseType
from .storage.duckdb import DuckDBStorage
from .embeddings.embedder import Embedder, create_embedder
from .ingestion.parser import SectionParser, Section
from .ingestion.definitions import DefinitionExtractor, Definition
from .ingestion.references import ReferenceLinker, ReferenceEdge
from .retrieval.hybrid import HybridRetriever, RetrievalResult


@dataclass
class IngestionResult:
    """Result from document ingestion."""
    document_id: str
    num_engrams: int
    num_synapses: int
    sections: list[Section]
    definitions: list[Definition]
    edges: list[ReferenceEdge]


class EngramDB:
    """
    EngramDB: Schema-aware hybrid retrieval for legal documents.

    Example usage:
        # Initialize
        db = EngramDB()  # In-memory
        # or
        db = EngramDB(db_path="./my_contracts.db")

        # Ingest a contract
        result = db.ingest(contract_text, doc_id="contract_001")

        # Query with hybrid retrieval
        result = db.query("Can we terminate if they breach confidentiality?")
        print(result.engrams)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedder: Optional[Embedder] = None,
        embedding_backend: str = "mock"
    ):
        """
        Initialize EngramDB.

        Args:
            db_path: Path to database file. None for in-memory.
            embedder: Custom embedder instance. If None, creates one.
            embedding_backend: "openai", "local", or "mock" (if no embedder provided)
        """
        self.db_path = Path(db_path) if db_path else None

        # Initialize storage
        self.storage = DuckDBStorage(self.db_path)
        self.storage.connect()

        # Initialize embedder
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = create_embedder(backend=embedding_backend)

        # Initialize ingestion components
        self.parser = SectionParser()
        self.definition_extractor = DefinitionExtractor()
        self.reference_linker = ReferenceLinker()

        # Initialize retriever
        self.retriever = HybridRetriever(self.storage, self.embedder)

    def ingest(
        self,
        text: str,
        doc_id: str,
        generate_embeddings: bool = True
    ) -> IngestionResult:
        """
        Ingest a document into EngramDB.

        This will:
        1. Parse document structure (sections)
        2. Extract defined terms
        3. Extract cross-references and create edges
        4. Generate embeddings (optional)
        5. Store everything in the database

        Args:
            text: Document text
            doc_id: Unique document identifier
            generate_embeddings: Whether to generate embeddings

        Returns:
            IngestionResult with statistics
        """
        # Step 1: Parse structure
        sections = self.parser.parse(text)
        flat_sections = []
        for s in sections:
            flat_sections.extend(s.flatten())

        # Step 2: Extract definitions
        definitions, term_usages = self.definition_extractor.extract_with_usages(text)

        # Step 3: Extract references and create edges
        references, ref_edges = self.reference_linker.extract_and_link(text, sections)

        # Step 4: Create engrams for sections
        engrams = []
        section_id_map = {}  # section_number -> engram_id
        section_key_quality = {}  # section key -> body length quality

        for section in flat_sections:
            section_content = self._build_section_content(section)
            body_quality = len(section.content.strip())
            engram = Engram(
                content=section_content,
                engram_type=EngramType.SECTION,
                metadata={
                    "document_id": doc_id,
                    "section_number": section.number,
                    "title": section.title,
                    "level": section.level,
                }
            )
            engrams.append(engram)

            # Map section identifiers to engram IDs
            if section.number:
                if body_quality >= section_key_quality.get(section.number, -1):
                    section_id_map[section.number] = engram.id
                    section_key_quality[section.number] = body_quality
            if section.title:
                if body_quality >= section_key_quality.get(section.title, -1):
                    section_id_map[section.title] = engram.id
                    section_key_quality[section.title] = body_quality

        # Step 5: Create engrams for definitions
        definition_id_map = {}
        for defn in definitions:
            engram = Engram(
                content=f'"{defn.term}" means {defn.definition}',
                engram_type=EngramType.DEFINITION,
                metadata={
                    "document_id": doc_id,
                    "term": defn.term,
                }
            )
            engrams.append(engram)
            definition_id_map[defn.term] = engram.id
            section_id_map[defn.term] = engram.id

        # Step 6: Generate embeddings
        if generate_embeddings:
            texts = [e.content for e in engrams]
            embeddings = self.embedder.embed_batch(texts)
            for engram, embedding in zip(engrams, embeddings):
                engram.embedding = embedding

        # Step 7: Store engrams
        self.storage.insert_engrams_batch(engrams)

        # Step 8: Create synapses from reference edges
        synapses = []
        for edge in ref_edges:
            source_id = section_id_map.get(edge.source_section)
            target_id = section_id_map.get(edge.target_section)

            if source_id and target_id and source_id != target_id:
                synapse = Synapse(
                    source_id=source_id,
                    target_id=target_id,
                    synapse_type=SynapseType.REFERENCES,
                    metadata={
                        "reference_text": edge.reference_text,
                        "reference_type": edge.reference_type,
                    }
                )
                synapses.append(synapse)

        # Step 9: Create parent-child synapses for section hierarchy
        for section in sections:
            self._create_hierarchy_synapses(section, section_id_map, synapses)

        # Step 9.5: Connect definitions to sections where terms are used.
        # This gives definition-usage queries real graph paths instead of isolated definition nodes.
        seen_definition_links = set()
        for term, usages in term_usages.items():
            definition_id = definition_id_map.get(term)
            if not definition_id:
                continue

            for usage in usages:
                containing_section = self._find_section_for_position(
                    usage.position,
                    flat_sections
                )
                if containing_section is None:
                    continue

                section_key = containing_section.number or containing_section.title
                if not section_key:
                    continue

                section_id = section_id_map.get(section_key)
                if not section_id or section_id == definition_id:
                    continue

                edge_key = (definition_id, section_id)
                if edge_key in seen_definition_links:
                    continue

                synapses.append(Synapse(
                    source_id=definition_id,
                    target_id=section_id,
                    synapse_type=SynapseType.DEFINES,
                    metadata={"term": term}
                ))
                seen_definition_links.add(edge_key)

        # Step 10: Store synapses
        if synapses:
            self.storage.insert_synapses_batch(synapses)

        return IngestionResult(
            document_id=doc_id,
            num_engrams=len(engrams),
            num_synapses=len(synapses),
            sections=flat_sections,
            definitions=definitions,
            edges=ref_edges
        )

    def _create_hierarchy_synapses(
        self,
        section: Section,
        section_id_map: dict,
        synapses: list
    ):
        """Create parent-child synapses for section hierarchy."""
        parent_id = section_id_map.get(section.number) or section_id_map.get(section.title)

        for child in section.children:
            child_id = section_id_map.get(child.number) or section_id_map.get(child.title)

            if parent_id and child_id:
                synapses.append(Synapse(
                    source_id=parent_id,
                    target_id=child_id,
                    synapse_type=SynapseType.PARENT_OF
                ))

            # Recurse
            self._create_hierarchy_synapses(child, section_id_map, synapses)

    def _build_section_content(self, section: Section) -> str:
        """Build section text with heading context to avoid empty vectors."""
        heading_parts = []
        if section.number:
            heading_parts.append(f"Section {section.number}")
        if section.title:
            heading_parts.append(section.title)

        heading = " - ".join(heading_parts).strip()
        body = section.content.strip()

        if heading and body:
            return f"{heading}\n\n{body}"
        if heading:
            return heading
        return body

    def _find_section_for_position(
        self,
        position: int,
        sections: list[Section]
    ) -> Optional[Section]:
        """Find the parsed section containing a character position."""
        for section in sections:
            if section.start_pos <= position < section.end_pos:
                return section
        return None

    def query(
        self,
        query: str,
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10,
        min_traversed_items: int = 0
    ) -> RetrievalResult:
        """
        Query the database using hybrid retrieval.

        Args:
            query: Natural language query
            top_k_anchors: Number of vector search anchors
            max_hops: Graph traversal depth
            max_context_items: Maximum engrams to return
            min_traversed_items: Minimum non-anchor traversed items to keep in context

        Returns:
            RetrievalResult with retrieved engrams
        """
        return self.retriever.retrieve(
            query=query,
            top_k_anchors=top_k_anchors,
            max_hops=max_hops,
            max_context_items=max_context_items,
            min_traversed_items=min_traversed_items
        )

    def query_graph_only(
        self,
        query: str,
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10
    ) -> RetrievalResult:
        """
        Query using graph-only retrieval (ablation baseline).

        Uses vector search to find anchors, then ranks purely by
        structural score without semantic blending.
        """
        return self.retriever.retrieve_graph_only(
            query=query,
            top_k_anchors=top_k_anchors,
            max_hops=max_hops,
            max_context_items=max_context_items
        )

    def query_vector_only(
        self,
        query: str,
        top_k: int = 10
    ) -> RetrievalResult:
        """
        Query using vector-only retrieval (baseline).

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            RetrievalResult with retrieved engrams
        """
        return self.retriever.retrieve_vector_only(query=query, top_k=top_k)

    def get_context_string(
        self,
        result: RetrievalResult,
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieval result as context string for LLM.

        Args:
            result: RetrievalResult from query()
            include_metadata: Include section numbers, types, etc.

        Returns:
            Formatted context string
        """
        return self.retriever.format_context(result, include_metadata)

    def stats(self) -> dict:
        """Return database statistics."""
        return {
            "num_engrams": self.storage.count_engrams(),
            "num_synapses": self.storage.count_synapses(),
            "db_path": str(self.db_path) if self.db_path else ":memory:",
        }

    def close(self):
        """Close the database connection."""
        self.storage.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
