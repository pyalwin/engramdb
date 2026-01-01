"""
Synapse: The edge/relationship between Engrams in EngramDB.

A Synapse represents an explicit relationship extracted from document structure,
such as cross-references, definitions, or hierarchical containment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SynapseType(Enum):
    """Types of relationships between engrams."""
    REFERENCES = "references"      # Cross-reference (e.g., "See Section 3.2")
    DEFINES = "defines"            # Definition relationship
    PARENT_OF = "parent_of"        # Hierarchical containment
    CHILD_OF = "child_of"          # Inverse of parent_of
    RELATED_TO = "related_to"      # General association
    SUPERSEDES = "supersedes"      # Temporal replacement (for updates)


@dataclass
class Synapse:
    """
    An edge in the EngramDB knowledge graph.

    Attributes:
        source_id: ID of the source engram
        target_id: ID of the target engram
        synapse_type: Classification of this relationship
        metadata: Additional data (confidence, extraction_method, etc.)
        created_at: Timestamp of creation
    """
    source_id: str
    target_id: str
    synapse_type: SynapseType
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return f"Synapse({self.source_id[:8]}... --[{self.synapse_type.value}]--> {self.target_id[:8]}...)"
