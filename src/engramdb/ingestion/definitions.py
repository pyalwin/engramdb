"""
Definition Extractor: Identify and extract defined terms from contracts.

Legal contracts define terms explicitly using patterns like:
- "Confidential Information" means...
- "Effective Date" shall mean...
- As used herein, "Party" refers to...
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class Definition:
    """An extracted defined term."""
    term: str              # The defined term (e.g., "Confidential Information")
    definition: str        # The definition text
    start_pos: int         # Start position in document
    end_pos: int           # End position in document
    source_section: Optional[str] = None  # Section number where found

    def __repr__(self) -> str:
        preview = self.definition[:50] + "..." if len(self.definition) > 50 else self.definition
        return f"Definition('{self.term}' = '{preview}')"


@dataclass
class TermUsage:
    """A usage of a defined term in the document."""
    term: str
    position: int
    context: str  # Surrounding text


class DefinitionExtractor:
    """
    Extracts defined terms from legal documents.

    Patterns recognized:
    - "Term" means ...
    - "Term" shall mean ...
    - "Term" refers to ...
    - "Term" is defined as ...
    - "Term" has the meaning ...
    """

    # Pattern components
    QUOTE_PATTERNS = [
        r'"([^"]+)"',      # Double quotes
        r'"([^"]+)"',      # Smart double quotes
        r"'([^']+)'",      # Single quotes
        r"'([^']+)'",      # Smart single quotes
    ]

    DEFINITION_VERBS = [
        r'means?',
        r'shall\s+mean',
        r'refers?\s+to',
        r'is\s+defined\s+as',
        r'has\s+the\s+meaning',
        r'shall\s+have\s+the\s+meaning',
        r'includes?',
        r'shall\s+include',
    ]

    def __init__(self):
        """Compile patterns for definition extraction."""
        self._definition_patterns = self._compile_patterns()
        self._term_cache: dict[str, re.Pattern] = {}

    def _compile_patterns(self) -> list[re.Pattern]:
        """Build regex patterns for definition detection."""
        patterns = []

        for quote_pat in self.QUOTE_PATTERNS:
            for verb in self.DEFINITION_VERBS:
                # "Term" means ...
                pattern = rf'{quote_pat}\s+{verb}\s+'
                patterns.append(re.compile(pattern, re.IGNORECASE))

                # As used herein, "Term" means ...
                pattern = rf'[Aa]s\s+used\s+(?:herein|in\s+this\s+[Aa]greement),?\s+{quote_pat}\s+{verb}\s+'
                patterns.append(re.compile(pattern, re.IGNORECASE))

        return patterns

    def extract(self, text: str) -> list[Definition]:
        """
        Extract all defined terms from document text.

        Args:
            text: Document text (or section text)

        Returns:
            List of extracted definitions
        """
        definitions = []
        seen_terms = set()  # Avoid duplicates

        for pattern in self._definition_patterns:
            for match in pattern.finditer(text):
                term = match.group(1).strip()

                # Skip if we've already found this term
                term_lower = term.lower()
                if term_lower in seen_terms:
                    continue

                # Extract the definition text (until end of sentence/paragraph)
                def_start = match.end()
                def_text = self._extract_definition_text(text, def_start)

                if def_text:
                    definitions.append(Definition(
                        term=term,
                        definition=def_text,
                        start_pos=match.start(),
                        end_pos=def_start + len(def_text)
                    ))
                    seen_terms.add(term_lower)

        # Sort by position
        definitions.sort(key=lambda d: d.start_pos)

        return definitions

    def _extract_definition_text(self, text: str, start: int) -> str:
        """
        Extract definition text starting from a position.

        Definitions typically end at:
        - A period followed by a new defined term
        - A semicolon (in definition lists)
        - Double newline (paragraph break)
        - End of document
        """
        # Look ahead up to 2000 characters for definition end
        search_text = text[start:start + 2000]

        # Find potential end points
        end_markers = []

        # Period followed by quote (next definition)
        for m in re.finditer(r'\.\s*[""\']', search_text):
            end_markers.append(m.start() + 1)

        # Semicolon (common in definition lists)
        for m in re.finditer(r';\s*(?=[""\'])', search_text):
            end_markers.append(m.start() + 1)

        # Double newline
        for m in re.finditer(r'\n\s*\n', search_text):
            end_markers.append(m.start())

        # Period followed by section number
        for m in re.finditer(r'\.\s*\n\s*\d+\.', search_text):
            end_markers.append(m.start() + 1)

        # Simple sentence end (period + space + capital) - but not abbreviations
        for m in re.finditer(r'(?<![A-Z])\.(?:\s+[A-Z]|\s*$)', search_text):
            # Check it's not an abbreviation (e.g., "Inc.", "Corp.")
            before = search_text[max(0, m.start()-5):m.start()]
            if not re.search(r'\b(?:Inc|Corp|Ltd|Mr|Mrs|Dr|Co|No|vs)\s*$', before):
                end_markers.append(m.start() + 1)

        if end_markers:
            # Use the earliest reasonable end point (at least 20 chars)
            valid_ends = [e for e in end_markers if e >= 20]
            if valid_ends:
                end = min(valid_ends)
                return search_text[:end].strip()

        # If no clear end, take up to first period
        period_match = re.search(r'\.', search_text)
        if period_match:
            return search_text[:period_match.end()].strip()

        # Fall back to first 500 chars
        return search_text[:500].strip()

    def find_term_usages(self, text: str, term: str) -> list[TermUsage]:
        """
        Find all positions where a defined term is used.

        Args:
            text: Document text
            term: The defined term to search for

        Returns:
            List of term usages with positions and context
        """
        usages = []

        # Get or compile pattern for this term
        if term not in self._term_cache:
            # Match the term as a whole word, case-insensitive
            escaped = re.escape(term)
            self._term_cache[term] = re.compile(
                rf'\b{escaped}\b',
                re.IGNORECASE
            )

        pattern = self._term_cache[term]

        for match in pattern.finditer(text):
            # Extract surrounding context (50 chars each side)
            ctx_start = max(0, match.start() - 50)
            ctx_end = min(len(text), match.end() + 50)
            context = text[ctx_start:ctx_end]

            usages.append(TermUsage(
                term=term,
                position=match.start(),
                context=context
            ))

        return usages

    def extract_with_usages(self, text: str) -> tuple[list[Definition], dict[str, list[TermUsage]]]:
        """
        Extract definitions and find all their usages in one pass.

        Returns:
            Tuple of (definitions, {term: [usages]})
        """
        definitions = self.extract(text)

        usages = {}
        for defn in definitions:
            term_usages = self.find_term_usages(text, defn.term)
            # Exclude the definition itself
            term_usages = [u for u in term_usages if u.position != defn.start_pos]
            if term_usages:
                usages[defn.term] = term_usages

        return definitions, usages
