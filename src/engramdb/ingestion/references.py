"""
Reference Linker: Extract and resolve cross-references within documents.

Legal contracts contain explicit cross-references like:
- "as defined in Section 3.2"
- "pursuant to Article IV"
- "subject to the terms of Exhibit A"
"""

from dataclasses import dataclass
from typing import Optional
import re

from .parser import Section


@dataclass
class Reference:
    """An extracted cross-reference."""
    source_position: int     # Character position of the reference
    reference_text: str      # The reference text (e.g., "Section 3.2")
    target_type: str         # "section", "article", "exhibit", "schedule", "paragraph"
    target_id: str           # The identifier (e.g., "3.2", "IV", "A")
    resolved_target: Optional[str] = None  # Resolved section ID if found
    context: str = ""        # Surrounding text for context

    def __repr__(self) -> str:
        resolved = f" -> {self.resolved_target}" if self.resolved_target else ""
        return f"Reference({self.reference_text}{resolved})"


@dataclass
class ReferenceEdge:
    """A resolved edge between two document elements."""
    source_section: str      # Section number/ID containing the reference
    target_section: str      # Section number/ID being referenced
    reference_text: str      # Original reference text
    reference_type: str      # Type of reference


class ReferenceLinker:
    """
    Extracts and resolves cross-references in legal documents.

    Patterns recognized:
    - Section X.Y.Z
    - Article X (Roman or Arabic)
    - Exhibit A
    - Schedule 1
    - Paragraph X
    - Clause X
    - herein, hereof, thereof, hereby
    """

    # Roman numeral conversion
    ROMAN_VALUES = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
    }

    # Reference patterns: (regex, type, id_group)
    PATTERNS = [
        # Section references
        (r'[Ss]ections?\s+(\d+(?:\.\d+)*)', 'section', 1),
        (r'[Ss]ections?\s+(\d+(?:\.\d+)*)\s+(?:and|or|through|to)\s+(\d+(?:\.\d+)*)', 'section_range', (1, 2)),
        (r'§\s*(\d+(?:\.\d+)*)', 'section', 1),

        # Article references (Roman or Arabic)
        (r'[Aa]rticles?\s+([IVXLC]+)', 'article', 1),
        (r'[Aa]rticles?\s+(\d+)', 'article', 1),

        # Exhibit/Schedule references
        (r'[Ee]xhibits?\s+([A-Z](?:-\d+)?|\d+)', 'exhibit', 1),
        (r'[Ss]chedules?\s+([A-Z]|\d+)', 'schedule', 1),
        (r'[Aa]ppendix\s+([A-Z]|\d+)', 'appendix', 1),
        (r'[Aa]nnex\s+([A-Z]|\d+)', 'annex', 1),

        # Paragraph/Clause references
        (r'[Pp]aragraphs?\s+(\d+(?:\.\d+)*)', 'paragraph', 1),
        (r'[Cc]lauses?\s+(\d+(?:\.\d+)*)', 'clause', 1),
        (r'[Ss]ubsections?\s+(\d+(?:\.\d+)*)', 'subsection', 1),

        # Defined term references
        (r'[Aa]s\s+defined\s+(?:in\s+)?[Ss]ection\s+(\d+(?:\.\d+)*)', 'definition_ref', 1),
        (r'[Ss]ee\s+[Ss]ection\s+(\d+(?:\.\d+)*)', 'section', 1),
    ]

    # Contextual references (need special handling)
    CONTEXTUAL_PATTERNS = [
        (r'\bherein\b', 'contextual', 'this_agreement'),
        (r'\bhereof\b', 'contextual', 'this_agreement'),
        (r'\bhereto\b', 'contextual', 'this_agreement'),
        (r'\bhereby\b', 'contextual', 'this_agreement'),
        (r'\bthereof\b', 'contextual', 'antecedent'),
        (r'\btherein\b', 'contextual', 'antecedent'),
        (r'\bthereto\b', 'contextual', 'antecedent'),
        (r'\bthis\s+[Ss]ection\b', 'contextual', 'current_section'),
        (r'\bthis\s+[Aa]rticle\b', 'contextual', 'current_article'),
    ]

    def __init__(self):
        """Compile reference patterns."""
        self._patterns = [
            (re.compile(pattern, re.IGNORECASE), ref_type, id_group)
            for pattern, ref_type, id_group in self.PATTERNS
        ]
        self._contextual_patterns = [
            (re.compile(pattern, re.IGNORECASE), ref_type, meaning)
            for pattern, ref_type, meaning in self.CONTEXTUAL_PATTERNS
        ]

    def extract_references(self, text: str) -> list[Reference]:
        """
        Extract all cross-references from text.

        Args:
            text: Document text

        Returns:
            List of extracted references
        """
        references = []
        seen_positions = set()

        for pattern, ref_type, id_group in self._patterns:
            for match in pattern.finditer(text):
                if match.start() in seen_positions:
                    continue

                # Handle range references
                if ref_type == 'section_range':
                    start_id, end_id = id_group
                    # Add both as separate references
                    for gid in [start_id, end_id]:
                        target_id = match.group(gid) if isinstance(gid, int) else gid
                        references.append(self._create_reference(
                            match, text, 'section', target_id
                        ))
                else:
                    target_id = match.group(id_group) if isinstance(id_group, int) else id_group
                    references.append(self._create_reference(
                        match, text, ref_type, target_id
                    ))

                seen_positions.add(match.start())

        # Sort by position
        references.sort(key=lambda r: r.source_position)

        return references

    def _create_reference(
        self,
        match: re.Match,
        text: str,
        ref_type: str,
        target_id: str
    ) -> Reference:
        """Create a Reference object from a regex match."""
        # Extract context (50 chars before and after)
        ctx_start = max(0, match.start() - 50)
        ctx_end = min(len(text), match.end() + 50)
        context = text[ctx_start:ctx_end]

        return Reference(
            source_position=match.start(),
            reference_text=match.group(0),
            target_type=ref_type,
            target_id=target_id,
            context=context
        )

    def resolve_references(
        self,
        references: list[Reference],
        sections: list[Section]
    ) -> list[Reference]:
        """
        Resolve references to actual section IDs.

        Args:
            references: Extracted references
            sections: Parsed sections from document

        Returns:
            References with resolved_target populated where possible
        """
        # Build a lookup map of section numbers
        section_map = self._build_section_map(sections)

        for ref in references:
            resolved = self._resolve_single(ref, section_map)
            if resolved:
                ref.resolved_target = resolved

        return references

    def _build_section_map(self, sections: list[Section]) -> dict[str, Section]:
        """Build a map from section identifiers to Section objects."""
        section_map = {}

        def add_section(section: Section):
            if section.number:
                # Add exact number
                section_map[section.number] = section
                # Add normalized versions
                section_map[section.number.lower()] = section
                # Handle roman numerals
                if self._is_roman(section.number):
                    arabic = str(self._roman_to_arabic(section.number))
                    section_map[arabic] = section

            if section.title:
                section_map[section.title.lower()] = section

            for child in section.children:
                add_section(child)

        for section in sections:
            add_section(section)

        return section_map

    def _resolve_single(
        self,
        ref: Reference,
        section_map: dict[str, Section]
    ) -> Optional[str]:
        """Resolve a single reference to a section ID."""
        target_id = ref.target_id

        # Try exact match first
        if target_id in section_map:
            return section_map[target_id].number or section_map[target_id].title

        # Try lowercase
        if target_id.lower() in section_map:
            s = section_map[target_id.lower()]
            return s.number or s.title

        # For articles with Roman numerals, try Arabic
        if ref.target_type == 'article' and self._is_roman(target_id):
            arabic = str(self._roman_to_arabic(target_id))
            if arabic in section_map:
                return section_map[arabic].number

        # Try partial matching for nested sections
        # e.g., "3.2" might match "3.2.1" if "3.2" doesn't exist
        for key in section_map:
            if key.startswith(target_id + '.'):
                return target_id  # Return the requested ID even if we matched a child

        return None

    def _is_roman(self, s: str) -> bool:
        """Check if string is a valid Roman numeral."""
        return bool(re.match(r'^[IVXLCDM]+$', s.upper()))

    def _roman_to_arabic(self, s: str) -> int:
        """Convert Roman numeral to Arabic number."""
        s = s.upper()
        total = 0
        prev = 0

        for char in reversed(s):
            value = self.ROMAN_VALUES.get(char, 0)
            if value < prev:
                total -= value
            else:
                total += value
            prev = value

        return total

    def create_edges(
        self,
        references: list[Reference],
        sections: list[Section]
    ) -> list[ReferenceEdge]:
        """
        Create graph edges from references.

        Args:
            references: Resolved references
            sections: Parsed sections

        Returns:
            List of edges connecting sections
        """
        edges = []

        # Find which section each reference is in
        flat_sections = []
        for s in sections:
            flat_sections.extend(s.flatten())

        for ref in references:
            if not ref.resolved_target:
                continue

            # Find the source section containing this reference
            source_section = self._find_containing_section(
                ref.source_position, flat_sections
            )

            if source_section and (source_section.number or source_section.title):
                source_id = source_section.number or source_section.title
                edges.append(ReferenceEdge(
                    source_section=source_id,
                    target_section=ref.resolved_target,
                    reference_text=ref.reference_text,
                    reference_type=ref.target_type
                ))

        return edges

    def _find_containing_section(
        self,
        position: int,
        sections: list[Section]
    ) -> Optional[Section]:
        """Find which section contains a given position."""
        for section in sections:
            if section.start_pos <= position < section.end_pos:
                return section
        return None

    def extract_and_link(
        self,
        text: str,
        sections: list[Section]
    ) -> tuple[list[Reference], list[ReferenceEdge]]:
        """
        Extract references and create edges in one pass.

        Args:
            text: Document text
            sections: Parsed sections

        Returns:
            Tuple of (references, edges)
        """
        references = self.extract_references(text)
        references = self.resolve_references(references, sections)
        edges = self.create_edges(references, sections)

        return references, edges
