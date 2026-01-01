"""
DuckDB storage backend for EngramDB.

Handles persistence of Engrams and Synapses, vector similarity search,
and graph traversal queries.
"""

from pathlib import Path
from typing import Optional
import json

try:
    import duckdb
except ImportError:
    duckdb = None

from ..core.engram import Engram, EngramType
from ..core.synapse import Synapse, SynapseType


class DuckDBStorage:
    """
    DuckDB-based storage for EngramDB.

    Provides:
    - Engram storage with vector embeddings
    - Synapse (edge) storage
    - Vector similarity search
    - Graph traversal
    """

    # Schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize DuckDB storage.

        Args:
            db_path: Path to database file. If None, uses in-memory database.
        """
        if duckdb is None:
            raise ImportError("duckdb is required. Install with: pip install duckdb")

        self.db_path = db_path
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> "DuckDBStorage":
        """
        Establish database connection and ensure schema exists.

        Returns self for method chaining.
        """
        if self._connection is not None:
            return self

        if self.db_path:
            self._connection = duckdb.connect(str(self.db_path))
        else:
            self._connection = duckdb.connect(":memory:")

        self._ensure_schema()
        return self

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        # Engrams table
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS engrams (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                engram_type VARCHAR NOT NULL,
                embedding DOUBLE[],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Synapses table (using sequence for auto-increment)
        self._connection.execute("""
            CREATE SEQUENCE IF NOT EXISTS synapse_id_seq START 1
        """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS synapses (
                id INTEGER PRIMARY KEY DEFAULT nextval('synapse_id_seq'),
                source_id VARCHAR NOT NULL,
                target_id VARCHAR NOT NULL,
                synapse_type VARCHAR NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster graph traversal
        self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_synapses_source ON synapses(source_id)
        """)
        self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_synapses_target ON synapses(target_id)
        """)

        # Schema version tracking
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)

    def _ensure_connected(self) -> None:
        """Ensure we have an active connection."""
        if self._connection is None:
            raise RuntimeError("Not connected. Call connect() first.")

    # === Engram Operations ===

    def insert_engram(self, engram: Engram) -> None:
        """Insert an engram into storage."""
        self._ensure_connected()

        self._connection.execute("""
            INSERT INTO engrams (id, content, engram_type, embedding, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            engram.id,
            engram.content,
            engram.engram_type.value,
            engram.embedding,
            json.dumps(engram.metadata),
            engram.created_at
        ])

    def insert_engrams_batch(self, engrams: list[Engram]) -> None:
        """Insert multiple engrams efficiently."""
        self._ensure_connected()

        data = [
            (e.id, e.content, e.engram_type.value, e.embedding, json.dumps(e.metadata), e.created_at)
            for e in engrams
        ]

        self._connection.executemany("""
            INSERT INTO engrams (id, content, engram_type, embedding, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)

    def get_engram(self, engram_id: str) -> Optional[Engram]:
        """Retrieve an engram by ID."""
        self._ensure_connected()

        result = self._connection.execute("""
            SELECT id, content, engram_type, embedding, metadata, created_at
            FROM engrams WHERE id = ?
        """, [engram_id]).fetchone()

        if result is None:
            return None

        return self._row_to_engram(result)

    def get_engrams(self, engram_ids: list[str]) -> list[Engram]:
        """Retrieve multiple engrams by ID."""
        self._ensure_connected()

        if not engram_ids:
            return []

        placeholders = ",".join(["?"] * len(engram_ids))
        results = self._connection.execute(f"""
            SELECT id, content, engram_type, embedding, metadata, created_at
            FROM engrams WHERE id IN ({placeholders})
        """, engram_ids).fetchall()

        return [self._row_to_engram(row) for row in results]

    def get_all_engrams(self) -> list[Engram]:
        """Retrieve all engrams."""
        self._ensure_connected()

        results = self._connection.execute("""
            SELECT id, content, engram_type, embedding, metadata, created_at
            FROM engrams
        """).fetchall()

        return [self._row_to_engram(row) for row in results]

    def update_engram_embedding(self, engram_id: str, embedding: list[float]) -> None:
        """Update the embedding for an engram."""
        self._ensure_connected()

        self._connection.execute("""
            UPDATE engrams SET embedding = ? WHERE id = ?
        """, [embedding, engram_id])

    def delete_engram(self, engram_id: str) -> None:
        """Delete an engram and its associated synapses."""
        self._ensure_connected()

        # Delete synapses first
        self._connection.execute("""
            DELETE FROM synapses WHERE source_id = ? OR target_id = ?
        """, [engram_id, engram_id])

        # Delete engram
        self._connection.execute("""
            DELETE FROM engrams WHERE id = ?
        """, [engram_id])

    def _row_to_engram(self, row: tuple) -> Engram:
        """Convert a database row to an Engram object."""
        id_, content, engram_type, embedding, metadata, created_at = row

        return Engram(
            id=id_,
            content=content,
            engram_type=EngramType(engram_type),
            embedding=list(embedding) if embedding else None,
            metadata=json.loads(metadata) if metadata else {},
            created_at=created_at
        )

    # === Synapse Operations ===

    def insert_synapse(self, synapse: Synapse) -> None:
        """Insert a synapse into storage."""
        self._ensure_connected()

        self._connection.execute("""
            INSERT INTO synapses (source_id, target_id, synapse_type, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, [
            synapse.source_id,
            synapse.target_id,
            synapse.synapse_type.value,
            json.dumps(synapse.metadata),
            synapse.created_at
        ])

    def insert_synapses_batch(self, synapses: list[Synapse]) -> None:
        """Insert multiple synapses efficiently."""
        self._ensure_connected()

        data = [
            (s.source_id, s.target_id, s.synapse_type.value, json.dumps(s.metadata), s.created_at)
            for s in synapses
        ]

        self._connection.executemany("""
            INSERT INTO synapses (source_id, target_id, synapse_type, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, data)

    def get_synapses_from(self, source_id: str) -> list[Synapse]:
        """Get all synapses originating from a node."""
        self._ensure_connected()

        results = self._connection.execute("""
            SELECT source_id, target_id, synapse_type, metadata, created_at
            FROM synapses WHERE source_id = ?
        """, [source_id]).fetchall()

        return [self._row_to_synapse(row) for row in results]

    def get_synapses_to(self, target_id: str) -> list[Synapse]:
        """Get all synapses pointing to a node."""
        self._ensure_connected()

        results = self._connection.execute("""
            SELECT source_id, target_id, synapse_type, metadata, created_at
            FROM synapses WHERE target_id = ?
        """, [target_id]).fetchall()

        return [self._row_to_synapse(row) for row in results]

    def _row_to_synapse(self, row: tuple) -> Synapse:
        """Convert a database row to a Synapse object."""
        source_id, target_id, synapse_type, metadata, created_at = row

        return Synapse(
            source_id=source_id,
            target_id=target_id,
            synapse_type=SynapseType(synapse_type),
            metadata=json.loads(metadata) if metadata else {},
            created_at=created_at
        )

    # === Vector Search ===

    def search_similar(
        self,
        embedding: list[float],
        top_k: int = 5,
        engram_types: Optional[list[EngramType]] = None
    ) -> list[tuple[Engram, float]]:
        """
        Find engrams with similar embeddings using cosine similarity.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            engram_types: Filter by engram types (optional)

        Returns:
            List of (engram, similarity_score) tuples, sorted by similarity descending
        """
        self._ensure_connected()

        # Build type filter
        type_filter = ""
        params = [embedding, embedding, top_k]

        if engram_types:
            type_values = [t.value for t in engram_types]
            placeholders = ",".join(["?"] * len(type_values))
            type_filter = f"AND engram_type IN ({placeholders})"
            params = [embedding, embedding] + type_values + [top_k]

        # Cosine similarity using DuckDB's list functions
        # cosine_sim = dot(a, b) / (norm(a) * norm(b))
        results = self._connection.execute(f"""
            WITH query_norm AS (
                SELECT sqrt(list_sum(list_transform(?, x -> x * x))) AS norm
            )
            SELECT
                e.id, e.content, e.engram_type, e.embedding, e.metadata, e.created_at,
                list_sum(list_transform(
                    list_zip(e.embedding, ?),
                    x -> x[1] * x[2]
                )) / (
                    sqrt(list_sum(list_transform(e.embedding, x -> x * x))) *
                    (SELECT norm FROM query_norm)
                ) AS similarity
            FROM engrams e
            WHERE e.embedding IS NOT NULL
            {type_filter}
            ORDER BY similarity DESC
            LIMIT ?
        """, params).fetchall()

        return [
            (self._row_to_engram(row[:6]), row[6])
            for row in results
        ]

    # === Graph Traversal ===

    def get_connected(
        self,
        engram_id: str,
        hops: int = 2,
        direction: str = "both"
    ) -> list[str]:
        """
        Get all engram IDs connected within N hops.

        Args:
            engram_id: Starting engram ID
            hops: Maximum traversal depth
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of connected engram IDs (including the starting node)
        """
        self._ensure_connected()

        visited = {engram_id}
        frontier = {engram_id}

        for _ in range(hops):
            next_frontier = set()

            for node_id in frontier:
                # Get outgoing connections
                if direction in ("outgoing", "both"):
                    outgoing = self._connection.execute("""
                        SELECT target_id FROM synapses WHERE source_id = ?
                    """, [node_id]).fetchall()
                    next_frontier.update(row[0] for row in outgoing)

                # Get incoming connections
                if direction in ("incoming", "both"):
                    incoming = self._connection.execute("""
                        SELECT source_id FROM synapses WHERE target_id = ?
                    """, [node_id]).fetchall()
                    next_frontier.update(row[0] for row in incoming)

            # Only explore unvisited nodes
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier

            if not frontier:
                break

        return list(visited)

    def get_subgraph(
        self,
        engram_ids: list[str]
    ) -> tuple[list[Engram], list[Synapse]]:
        """
        Extract subgraph containing specified engrams and their connections.

        Args:
            engram_ids: Engram IDs to include

        Returns:
            Tuple of (engrams, synapses) in the subgraph
        """
        self._ensure_connected()

        if not engram_ids:
            return [], []

        # Get engrams
        engrams = self.get_engrams(engram_ids)

        # Get synapses between these engrams
        placeholders = ",".join(["?"] * len(engram_ids))
        results = self._connection.execute(f"""
            SELECT source_id, target_id, synapse_type, metadata, created_at
            FROM synapses
            WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
        """, engram_ids + engram_ids).fetchall()

        synapses = [self._row_to_synapse(row) for row in results]

        return engrams, synapses

    # === Utility ===

    def count_engrams(self) -> int:
        """Return total number of engrams."""
        self._ensure_connected()
        return self._connection.execute("SELECT COUNT(*) FROM engrams").fetchone()[0]

    def count_synapses(self) -> int:
        """Return total number of synapses."""
        self._ensure_connected()
        return self._connection.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]

    def clear(self) -> None:
        """Clear all data (for testing)."""
        self._ensure_connected()
        self._connection.execute("DELETE FROM synapses")
        self._connection.execute("DELETE FROM engrams")

    def __enter__(self) -> "DuckDBStorage":
        """Context manager entry."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
