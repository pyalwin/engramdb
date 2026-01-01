"""
Section Parser: Extract document structure (headings, sections, hierarchy).

Identifies section boundaries and builds the hierarchical structure of a document.
"""

from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class Section:
    """A parsed section from a document."""
    number: Optional[str]  # e.g., "1.2.3" or "III"
    title: Optional[str]   # e.g., "Definitions"
    content: str
    level: int             # Nesting depth (1 = top level)
    children: list["Section"] = field(default_factory=list)
    start_pos: int = 0     # Character position in original text
    end_pos: int = 0       # End position in original text

    def __repr__(self) -> str:
        num = self.number or ""
        title = self.title or ""
        label = f"{num} {title}".strip()
        return f"Section(level={self.level}, '{label}', children={len(self.children)})"

    def flatten(self) -> list["Section"]:
        """Return this section and all descendants as a flat list."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


@dataclass
class HeadingMatch:
    """A detected heading in the document."""
    start_pos: int
    end_pos: int
    full_match: str
    number: Optional[str]
    title: Optional[str]
    level: int
    pattern_type: str  # 'numbered', 'article', 'named'


class SectionParser:
    """
    Extracts hierarchical section structure from contract text.

    Handles common patterns:
    - Numbered sections: "1.", "1.1", "1.1.1"
    - Article format: "Article I", "Article II"
    - Named sections: "DEFINITIONS", "CONFIDENTIALITY"
    """

    # Roman numeral pattern
    ROMAN = r'(?:I{1,3}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)'

    # Patterns for different heading types (order matters - more specific first)
    PATTERNS = [
        # "ARTICLE I - Title" or "ARTICLE 1 - Title"
        (
            rf'^[ \t]*ARTICLE\s+({ROMAN}|\d+)[\s.:\-—]+([A-Z][A-Za-z\s]+?)[ \t]*$',
            'article',
            lambda m: (m.group(1), m.group(2).strip(), 1)
        ),
        # "ARTICLE I" alone
        (
            rf'^[ \t]*ARTICLE\s+({ROMAN}|\d+)[ \t]*$',
            'article',
            lambda m: (m.group(1), None, 1)
        ),
        # "Section 1.2.3" or "Section 1.2.3 - Title"
        (
            r'^[ \t]*[Ss]ection\s+(\d+(?:\.\d+)*)[\s.:\-—]*([A-Za-z][A-Za-z\s]*)?[ \t]*$',
            'numbered',
            lambda m: (m.group(1), m.group(2).strip() if m.group(2) else None,
                      m.group(1).count('.') + 1)
        ),
        # "1.2.3 Title" or "1.2.3. Title"
        (
            r'^[ \t]*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z\s]+?)[ \t]*$',
            'numbered',
            lambda m: (m.group(1), m.group(2).strip(), m.group(1).count('.') + 1)
        ),
        # "1.2.3" alone on a line
        (
            r'^[ \t]*(\d+(?:\.\d+)*)\.?[ \t]*$',
            'numbered',
            lambda m: (m.group(1), None, m.group(1).count('.') + 1)
        ),
        # ALL CAPS TITLE (e.g., "DEFINITIONS", "CONFIDENTIALITY")
        # Must be 2+ words or specific known section names
        (
            r'^[ \t]*([A-Z]{2,}(?:\s+[A-Z]{2,})+)[ \t]*$',
            'named',
            lambda m: (None, m.group(1).strip(), 1)
        ),
        # Single word ALL CAPS that are common section names
        (
            r'^[ \t]*(DEFINITIONS|RECITALS|PREAMBLE|CONFIDENTIALITY|TERMINATION|'
            r'INDEMNIFICATION|WARRANTIES|REPRESENTATIONS|MISCELLANEOUS|'
            r'NOTICES|ASSIGNMENT|AMENDMENTS|SEVERABILITY|COUNTERPARTS|'
            r'GOVERNING LAW|DISPUTE RESOLUTION|LIMITATION OF LIABILITY|'
            r'INTELLECTUAL PROPERTY|FORCE MAJEURE|ENTIRE AGREEMENT)[ \t]*$',
            'named',
            lambda m: (None, m.group(1).strip(), 1)
        ),
    ]

    def __init__(self):
        """Compile regex patterns for section detection."""
        self._compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), ptype, extractor)
            for pattern, ptype, extractor in self.PATTERNS
        ]

    def parse(self, text: str) -> list[Section]:
        """
        Parse document text into hierarchical sections.

        Args:
            text: Raw document text

        Returns:
            List of top-level sections (with nested children)
        """
        # Detect all headings
        headings = self._detect_headings(text)

        if not headings:
            # No structure detected - return entire document as single section
            return [Section(
                number=None,
                title=None,
                content=text.strip(),
                level=1,
                start_pos=0,
                end_pos=len(text)
            )]

        # Extract content for each heading
        sections = self._extract_content(text, headings)

        # Build hierarchy
        return self._build_hierarchy(sections)

    def _detect_headings(self, text: str) -> list[HeadingMatch]:
        """Detect heading positions, text, and level."""
        headings = []
        seen_positions = set()  # Avoid duplicate matches

        for pattern, ptype, extractor in self._compiled_patterns:
            for match in pattern.finditer(text):
                start = match.start()

                # Skip if we already have a heading at this position
                if start in seen_positions:
                    continue

                number, title, level = extractor(match)

                headings.append(HeadingMatch(
                    start_pos=start,
                    end_pos=match.end(),
                    full_match=match.group(0).strip(),
                    number=number,
                    title=title,
                    level=level,
                    pattern_type=ptype
                ))
                seen_positions.add(start)

        # Sort by position in document
        headings.sort(key=lambda h: h.start_pos)

        return headings

    def _extract_content(self, text: str, headings: list[HeadingMatch]) -> list[Section]:
        """Extract content between headings."""
        sections = []

        for i, heading in enumerate(headings):
            # Content starts after the heading line
            content_start = heading.end_pos

            # Content ends at the next heading or end of document
            if i + 1 < len(headings):
                content_end = headings[i + 1].start_pos
            else:
                content_end = len(text)

            content = text[content_start:content_end].strip()

            sections.append(Section(
                number=heading.number,
                title=heading.title,
                content=content,
                level=heading.level,
                start_pos=heading.start_pos,
                end_pos=content_end
            ))

        return sections

    def _build_hierarchy(self, sections: list[Section]) -> list[Section]:
        """Build nested section structure from flat section list."""
        if not sections:
            return []

        # Use a stack-based approach to build hierarchy
        root_sections = []
        stack: list[Section] = []

        for section in sections:
            # Pop sections from stack that are at same or higher level
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                # This section is a child of the top of stack
                stack[-1].children.append(section)
            else:
                # This is a root-level section
                root_sections.append(section)

            # Push current section onto stack
            stack.append(section)

        return root_sections

    def parse_flat(self, text: str) -> list[Section]:
        """
        Parse document and return flat list of all sections.

        Convenience method that flattens the hierarchy.
        """
        hierarchy = self.parse(text)
        result = []
        for section in hierarchy:
            result.extend(section.flatten())
        return result
