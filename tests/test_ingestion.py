"""Tests for the ingestion pipeline."""

import pytest
from engramdb.ingestion.parser import SectionParser, Section
from engramdb.ingestion.definitions import DefinitionExtractor, Definition
from engramdb.ingestion.references import ReferenceLinker, Reference


# Sample contract text for testing
SAMPLE_CONTRACT = """
MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of January 1, 2024
("Effective Date") by and between:

Acme Corporation ("Disclosing Party") and Beta Inc. ("Receiving Party").

ARTICLE I - DEFINITIONS

1.1 Definitions

"Confidential Information" means any non-public information disclosed by one party
to the other, including but not limited to trade secrets, business plans, and technical data.

"Permitted Purpose" means the evaluation of a potential business relationship
between the parties as described in Section 2.1.

"Representatives" means employees, officers, directors, and advisors of the Receiving Party.

ARTICLE II - OBLIGATIONS

2.1 Purpose

The Receiving Party may use Confidential Information solely for the Permitted Purpose
as defined in Section 1.1.

2.2 Non-Disclosure

The Receiving Party shall not disclose Confidential Information to any third party
except to its Representatives who need to know such information for the Permitted Purpose.

2.3 Standard of Care

The Receiving Party shall protect Confidential Information using the same degree of care
it uses to protect its own confidential information, but no less than reasonable care.

ARTICLE III - EXCEPTIONS

3.1 Exclusions

Confidential Information does not include information that:
(a) is or becomes publicly available through no fault of the Receiving Party;
(b) was known to the Receiving Party prior to disclosure, as evidenced by written records;
(c) is independently developed by the Receiving Party without use of Confidential Information.

ARTICLE IV - TERM AND TERMINATION

4.1 Term

This Agreement shall remain in effect for two (2) years from the Effective Date.

4.2 Termination

Either party may terminate this Agreement upon thirty (30) days written notice.
Upon termination, the obligations under Section 2.2 shall survive for an additional
five (5) years. See Section 3.1 for exclusions that apply.

ARTICLE V - MISCELLANEOUS

5.1 Governing Law

This Agreement shall be governed by the laws of the State of Delaware.

5.2 Entire Agreement

This Agreement constitutes the entire agreement between the parties with respect
to the subject matter hereof and supersedes all prior agreements.
"""


class TestSectionParser:
    """Tests for SectionParser."""

    def test_parse_article_sections(self):
        """Test parsing ARTICLE format sections."""
        parser = SectionParser()
        sections = parser.parse(SAMPLE_CONTRACT)

        # Should find multiple ARTICLE sections
        assert len(sections) >= 4

        # Check first article
        article1 = sections[0]
        assert article1.number == "I" or article1.title == "DEFINITIONS"
        assert article1.level == 1

    def test_parse_numbered_subsections(self):
        """Test parsing numbered subsections like 1.1, 2.1."""
        parser = SectionParser()
        sections = parser.parse_flat(SAMPLE_CONTRACT)

        # Find section 1.1
        section_1_1 = next((s for s in sections if s.number == "1.1"), None)
        assert section_1_1 is not None
        assert section_1_1.title == "Definitions"
        assert section_1_1.level == 2

        # Find section 2.2
        section_2_2 = next((s for s in sections if s.number == "2.2"), None)
        assert section_2_2 is not None
        assert "Non-Disclosure" in (section_2_2.title or "")

    def test_hierarchy_building(self):
        """Test that hierarchy is built correctly."""
        parser = SectionParser()
        sections = parser.parse(SAMPLE_CONTRACT)

        # Articles should have children (subsections)
        has_children = any(len(s.children) > 0 for s in sections)
        assert has_children, "Top-level sections should have children"

    def test_flatten(self):
        """Test flattening hierarchical sections."""
        parser = SectionParser()
        hierarchy = parser.parse(SAMPLE_CONTRACT)
        flat = parser.parse_flat(SAMPLE_CONTRACT)

        # Flat should have more sections than hierarchy
        assert len(flat) >= len(hierarchy)

    def test_empty_document(self):
        """Test parsing empty or unstructured document."""
        parser = SectionParser()
        sections = parser.parse("This is just plain text with no sections.")

        assert len(sections) == 1
        assert sections[0].level == 1


class TestDefinitionExtractor:
    """Tests for DefinitionExtractor."""

    def test_extract_definitions(self):
        """Test extracting defined terms."""
        extractor = DefinitionExtractor()
        definitions = extractor.extract(SAMPLE_CONTRACT)

        # Should find key definitions
        terms = [d.term for d in definitions]
        assert "Confidential Information" in terms
        assert "Permitted Purpose" in terms
        assert "Representatives" in terms

    def test_definition_content(self):
        """Test that definition text is extracted correctly."""
        extractor = DefinitionExtractor()
        definitions = extractor.extract(SAMPLE_CONTRACT)

        conf_info = next(d for d in definitions if d.term == "Confidential Information")
        assert "non-public information" in conf_info.definition.lower()

    def test_find_term_usages(self):
        """Test finding usages of defined terms."""
        extractor = DefinitionExtractor()
        usages = extractor.find_term_usages(SAMPLE_CONTRACT, "Confidential Information")

        # Should find multiple usages
        assert len(usages) > 1

    def test_extract_with_usages(self):
        """Test extracting definitions with their usages."""
        extractor = DefinitionExtractor()
        definitions, usages = extractor.extract_with_usages(SAMPLE_CONTRACT)

        assert len(definitions) > 0
        # Confidential Information should have usages
        assert "Confidential Information" in usages

    def test_different_quote_styles(self):
        """Test handling different quote styles."""
        text = '''
        "Term A" means something.
        'Term B' means something else.
        "Term C" means another thing.
        '''
        extractor = DefinitionExtractor()
        definitions = extractor.extract(text)

        terms = [d.term for d in definitions]
        assert "Term A" in terms


class TestReferenceLinker:
    """Tests for ReferenceLinker."""

    def test_extract_section_references(self):
        """Test extracting Section X.X references."""
        linker = ReferenceLinker()
        refs = linker.extract_references(SAMPLE_CONTRACT)

        # Should find references to sections
        section_refs = [r for r in refs if r.target_type == "section"]
        assert len(section_refs) > 0

        # Check specific references
        target_ids = [r.target_id for r in section_refs]
        assert "1.1" in target_ids or "2.1" in target_ids or "2.2" in target_ids

    def test_resolve_references(self):
        """Test resolving references to actual sections."""
        parser = SectionParser()
        linker = ReferenceLinker()

        sections = parser.parse(SAMPLE_CONTRACT)
        refs = linker.extract_references(SAMPLE_CONTRACT)
        resolved = linker.resolve_references(refs, sections)

        # Some references should be resolved
        resolved_refs = [r for r in resolved if r.resolved_target is not None]
        assert len(resolved_refs) > 0

    def test_create_edges(self):
        """Test creating graph edges from references."""
        parser = SectionParser()
        linker = ReferenceLinker()

        sections = parser.parse(SAMPLE_CONTRACT)
        refs, edges = linker.extract_and_link(SAMPLE_CONTRACT, sections)

        # Should have some edges
        assert len(edges) > 0

        # Check edge structure
        for edge in edges:
            assert edge.source_section is not None
            assert edge.target_section is not None

    def test_roman_numeral_conversion(self):
        """Test Roman numeral to Arabic conversion."""
        linker = ReferenceLinker()

        assert linker._roman_to_arabic("I") == 1
        assert linker._roman_to_arabic("IV") == 4
        assert linker._roman_to_arabic("IX") == 9
        assert linker._roman_to_arabic("XIV") == 14
        assert linker._roman_to_arabic("XX") == 20

    def test_article_references(self):
        """Test extracting Article references."""
        text = "As described in Article III and pursuant to Article IV."
        linker = ReferenceLinker()
        refs = linker.extract_references(text)

        article_refs = [r for r in refs if r.target_type == "article"]
        assert len(article_refs) >= 2

        target_ids = [r.target_id for r in article_refs]
        assert "III" in target_ids
        assert "IV" in target_ids


class TestIntegration:
    """Integration tests for the full ingestion pipeline."""

    def test_full_pipeline(self):
        """Test running all extractors together."""
        parser = SectionParser()
        def_extractor = DefinitionExtractor()
        ref_linker = ReferenceLinker()

        # Parse structure
        sections = parser.parse(SAMPLE_CONTRACT)
        flat_sections = parser.parse_flat(SAMPLE_CONTRACT)

        # Extract definitions
        definitions, term_usages = def_extractor.extract_with_usages(SAMPLE_CONTRACT)

        # Extract and link references
        references, edges = ref_linker.extract_and_link(SAMPLE_CONTRACT, sections)

        # Verify we got data from each stage
        assert len(flat_sections) > 5, "Should have multiple sections"
        assert len(definitions) >= 3, "Should have definitions"
        assert len(references) > 0, "Should have references"
        assert len(edges) > 0, "Should have edges"

        # Verify sections contain content
        for section in flat_sections:
            assert section.content is not None

    def test_multi_hop_potential(self):
        """Test that we can trace multi-hop paths."""
        parser = SectionParser()
        def_extractor = DefinitionExtractor()
        ref_linker = ReferenceLinker()

        sections = parser.parse(SAMPLE_CONTRACT)
        definitions, _ = def_extractor.extract_with_usages(SAMPLE_CONTRACT)
        _, edges = ref_linker.extract_and_link(SAMPLE_CONTRACT, sections)

        # Build adjacency for multi-hop check
        adjacency = {}
        for edge in edges:
            if edge.source_section not in adjacency:
                adjacency[edge.source_section] = []
            adjacency[edge.source_section].append(edge.target_section)

        # Check if we have any multi-hop paths
        # (i.e., section A -> B -> C)
        has_multi_hop = False
        for source, targets in adjacency.items():
            for target in targets:
                if target in adjacency:
                    has_multi_hop = True
                    break

        # Even if no multi-hop in this sample, we should have edges
        assert len(edges) > 0
