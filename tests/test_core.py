"""Tests for core data models."""

import pytest
from engramdb.core import Engram, Synapse
from engramdb.core.engram import EngramType
from engramdb.core.synapse import SynapseType


class TestEngram:
    """Tests for Engram model."""

    def test_create_engram(self):
        """Test basic engram creation."""
        engram = Engram(
            content="This is a test clause.",
            engram_type=EngramType.CLAUSE
        )
        assert engram.content == "This is a test clause."
        assert engram.engram_type == EngramType.CLAUSE
        assert engram.id is not None
        assert engram.embedding is None

    def test_engram_with_metadata(self):
        """Test engram with metadata."""
        engram = Engram(
            content="Section 1.1 content",
            engram_type=EngramType.SECTION,
            metadata={"section_number": "1.1", "title": "Definitions"}
        )
        assert engram.metadata["section_number"] == "1.1"


class TestSynapse:
    """Tests for Synapse model."""

    def test_create_synapse(self):
        """Test basic synapse creation."""
        synapse = Synapse(
            source_id="abc123",
            target_id="def456",
            synapse_type=SynapseType.REFERENCES
        )
        assert synapse.source_id == "abc123"
        assert synapse.target_id == "def456"
        assert synapse.synapse_type == SynapseType.REFERENCES
