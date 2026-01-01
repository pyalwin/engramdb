"""
EngramDB: Schema-Aware Hybrid Retrieval for Multi-Hop Legal Reasoning

A hybrid vector + graph memory engine that extracts and leverages
document-native structure for superior multi-hop retrieval.
"""

__version__ = "0.1.0"

from .db import EngramDB, IngestionResult
from .core.engram import Engram, EngramType
from .core.synapse import Synapse, SynapseType
from .retrieval.hybrid import RetrievalResult, RetrievalTrace
from .embeddings.embedder import create_embedder

__all__ = [
    "EngramDB",
    "IngestionResult",
    "Engram",
    "EngramType",
    "Synapse",
    "SynapseType",
    "RetrievalResult",
    "RetrievalTrace",
    "create_embedder",
]
