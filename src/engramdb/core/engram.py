"""
Engram: The fundamental node/memory unit in EngramDB.

An Engram represents a discrete piece of information extracted from a document,
such as a clause, definition, section, or entity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class EngramType(Enum):
    """Types of engrams that can be extracted from documents."""
    SECTION = "section"
    DEFINITION = "definition"
    CLAUSE = "clause"
    PARTY = "party"
    DATE = "date"
    REFERENCE = "reference"


@dataclass
class Engram:
    """
    A node in the EngramDB knowledge graph.

    Attributes:
        id: Unique identifier
        content: Raw text content
        embedding: Vector embedding (None until computed)
        engram_type: Classification of this engram
        metadata: Additional structured data (section_number, source_location, etc.)
        created_at: Timestamp of creation
    """
    content: str
    engram_type: EngramType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Engram(type={self.engram_type.value}, content='{preview}')"
