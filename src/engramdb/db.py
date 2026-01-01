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

        for section in flat_sections:
            engram = Engram(
                content=section.content,
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
                section_id_map[section.number] = engram.id
            if section.title:
                section_id_map[section.title] = engram.id

        # Step 5: Create engrams for definitions
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

    def query(
        self,
        query: str,
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10
    ) -> RetrievalResult:
        """
        Query the database using hybrid retrieval.

        Args:
            query: Natural language query
            top_k_anchors: Number of vector search anchors
            max_hops: Graph traversal depth
            max_context_items: Maximum engrams to return

        Returns:
            RetrievalResult with retrieved engrams
        """
        return self.retriever.retrieve(
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
