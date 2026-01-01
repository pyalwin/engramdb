"""
Multi-Hop QA Generator

Generates multi-hop questions from CUAD contracts using EngramDB's
graph structure. These questions require connecting information from
multiple sections/clauses to answer correctly.

Multi-hop patterns:
1. Definition → Usage: "What happens if X (defined in Section A) is violated?"
2. Cross-reference chain: "Section A references B, B references C - what's the outcome?"
3. Conditional reasoning: "If condition X in Section A, what are consequences per Section B?"
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from engramdb import EngramDB
from engramdb.ingestion.parser import SectionParser
from engramdb.ingestion.definitions import DefinitionExtractor
from engramdb.ingestion.references import ReferenceLinker

from cuad_loader import CUADLoader, Contract


@dataclass
class MultiHopQuestion:
    """A multi-hop question requiring reasoning across sections."""
    id: str
    contract_id: str
    question: str
    answer: str
    reasoning_chain: list[str]  # Section IDs in order
    hop_count: int
    question_type: str  # "definition_usage", "cross_reference", "conditional"
    evidence_sections: list[dict]  # Sections needed to answer

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiHopDataset:
    """Dataset of multi-hop questions."""
    contracts: list[Contract]
    questions: list[MultiHopQuestion]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "contracts": [c.to_dict() for c in self.contracts],
            "questions": [q.to_dict() for q in self.questions],
            "metadata": self.metadata,
        }

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved {len(self.questions)} multi-hop questions to {filepath}")


class MultiHopGenerator:
    """
    Generates multi-hop questions from contracts.

    Uses EngramDB to extract structure and identify cross-references,
    then generates questions that require following these connections.
    """

    # Question templates for different multi-hop patterns
    TEMPLATES = {
        "definition_usage": [
            'According to this contract, if {term} is {action}, what are the consequences as described in {target_section}?',
            'What obligations does the Receiving Party have regarding {term} as defined and used throughout the contract?',
            'How is {term} protected under this agreement, considering both its definition and the related provisions?',
        ],
        "cross_reference": [
            'Section {source} references Section {target}. What is the combined effect of these provisions?',
            'If a party violates the provisions in {source}, what remedies are available per the referenced {target}?',
            'How do the provisions in {source} and {target} work together in this agreement?',
        ],
        "conditional": [
            'If {condition} occurs under {source_section}, what actions are permitted under {target_section}?',
            'What happens to {subject} if the conditions in {source_section} are triggered, according to {target_section}?',
            'Under what circumstances described in {source_section} can a party exercise rights in {target_section}?',
        ],
        "termination_chain": [
            'What events could lead to termination of this agreement, and what are the post-termination obligations?',
            'If Party A breaches {breach_section}, can Party B terminate? What survives termination?',
            'Trace the path from a material breach to termination: what sections are involved?',
        ],
    }

    def __init__(self):
        self.parser = SectionParser()
        self.def_extractor = DefinitionExtractor()
        self.ref_linker = ReferenceLinker()

    def analyze_contract(self, contract: Contract) -> dict:
        """
        Analyze a contract's structure for multi-hop potential.

        Returns:
            Dict with sections, definitions, references, and potential multi-hop paths
        """
        text = contract.text

        # Parse structure
        sections = self.parser.parse(text)
        flat_sections = self.parser.parse_flat(text)

        # Extract definitions
        definitions, term_usages = self.def_extractor.extract_with_usages(text)

        # Extract references
        references, edges = self.ref_linker.extract_and_link(text, sections)

        # Build adjacency for path finding
        adjacency = {}
        for edge in edges:
            if edge.source_section not in adjacency:
                adjacency[edge.source_section] = []
            adjacency[edge.source_section].append({
                "target": edge.target_section,
                "type": edge.reference_type,
                "text": edge.reference_text,
            })

        return {
            "sections": flat_sections,
            "definitions": definitions,
            "term_usages": term_usages,
            "references": references,
            "edges": edges,
            "adjacency": adjacency,
        }

    def find_multihop_paths(self, adjacency: dict, max_hops: int = 3) -> list[list[str]]:
        """Find all paths of 2+ hops in the reference graph."""
        paths = []

        for start in adjacency:
            self._dfs_paths(start, [start], adjacency, paths, max_hops)

        # Filter to paths of 2+ hops
        return [p for p in paths if len(p) >= 2]

    def _dfs_paths(self, node: str, current_path: list, adjacency: dict,
                   all_paths: list, max_depth: int):
        """DFS to find all paths."""
        if len(current_path) > max_depth:
            return

        if len(current_path) >= 2:
            all_paths.append(current_path.copy())

        if node not in adjacency:
            return

        for edge in adjacency[node]:
            target = edge["target"]
            if target not in current_path:  # Avoid cycles
                current_path.append(target)
                self._dfs_paths(target, current_path, adjacency, all_paths, max_depth)
                current_path.pop()

    def generate_questions(
        self,
        contract: Contract,
        analysis: dict,
        max_questions: int = 5
    ) -> list[MultiHopQuestion]:
        """Generate multi-hop questions for a contract."""
        questions = []
        question_id = 0

        sections = analysis["sections"]
        definitions = analysis["definitions"]
        edges = analysis["edges"]
        adjacency = analysis["adjacency"]

        # Build section content map
        section_content = {}
        for s in sections:
            key = s.number or s.title
            if key:
                section_content[key] = {
                    "number": s.number,
                    "title": s.title,
                    "content": s.content[:500],  # Truncate for evidence
                }

        # 1. Generate definition-usage questions
        for defn in definitions[:3]:  # Limit per type
            term = defn.term

            # Find sections that use this term
            # Look for the term in section content
            usage_sections = []
            for s in sections:
                if term.lower() in s.content.lower() and s.number:
                    usage_sections.append(s.number)

            if len(usage_sections) >= 2:
                q = MultiHopQuestion(
                    id=f"{contract.id}_mh_{question_id}",
                    contract_id=contract.id,
                    question=f'How is "{term}" defined in this agreement, and what obligations does it create throughout the contract?',
                    answer=f'"{term}" is defined as: {defn.definition[:200]}... It is referenced in sections: {", ".join(usage_sections[:3])}',
                    reasoning_chain=usage_sections[:3],
                    hop_count=len(usage_sections[:3]),
                    question_type="definition_usage",
                    evidence_sections=[section_content.get(s, {}) for s in usage_sections[:3]]
                )
                questions.append(q)
                question_id += 1

                if len(questions) >= max_questions:
                    return questions

        # 2. Generate cross-reference questions
        paths = self.find_multihop_paths(adjacency, max_hops=3)

        for path in paths[:3]:
            if len(path) >= 2:
                source = path[0]
                target = path[-1]

                source_info = section_content.get(source, {})
                target_info = section_content.get(target, {})

                if source_info and target_info:
                    q = MultiHopQuestion(
                        id=f"{contract.id}_mh_{question_id}",
                        contract_id=contract.id,
                        question=f'Section {source} references Section {target}. How do these provisions relate to each other?',
                        answer=f'Section {source} ({source_info.get("title", "")}) references Section {target} ({target_info.get("title", "")}). The connection involves: {source_info.get("content", "")[:100]}... -> {target_info.get("content", "")[:100]}...',
                        reasoning_chain=path,
                        hop_count=len(path),
                        question_type="cross_reference",
                        evidence_sections=[section_content.get(s, {}) for s in path]
                    )
                    questions.append(q)
                    question_id += 1

                    if len(questions) >= max_questions:
                        return questions

        # 3. Generate termination-related questions
        # Find termination sections
        termination_sections = [s for s in sections if s.title and
                               'terminat' in s.title.lower()]

        for term_sec in termination_sections[:1]:
            # Find what references termination
            refs_to_termination = []
            for edge in edges:
                if term_sec.number and edge.target_section == term_sec.number:
                    refs_to_termination.append(edge.source_section)

            if refs_to_termination:
                chain = refs_to_termination[:2] + [term_sec.number]
                q = MultiHopQuestion(
                    id=f"{contract.id}_mh_{question_id}",
                    contract_id=contract.id,
                    question=f'What events or breaches in this agreement can trigger termination rights under Section {term_sec.number}?',
                    answer=f'Termination can be triggered by provisions in sections {", ".join(refs_to_termination[:2])}, which reference the termination provisions in Section {term_sec.number}.',
                    reasoning_chain=chain,
                    hop_count=len(chain),
                    question_type="termination_chain",
                    evidence_sections=[section_content.get(s, {}) for s in chain]
                )
                questions.append(q)
                question_id += 1

        return questions

    def generate_dataset(
        self,
        contracts: list[Contract],
        max_questions_per_contract: int = 5
    ) -> MultiHopDataset:
        """Generate a multi-hop QA dataset from contracts."""
        all_questions = []
        processed_contracts = []

        print(f"Generating multi-hop questions for {len(contracts)} contracts...")

        for i, contract in enumerate(contracts):
            print(f"  [{i+1}/{len(contracts)}] Processing {contract.title[:50]}...")

            try:
                # Analyze contract
                analysis = self.analyze_contract(contract)

                # Generate questions
                questions = self.generate_questions(
                    contract,
                    analysis,
                    max_questions=max_questions_per_contract
                )

                if questions:
                    all_questions.extend(questions)
                    processed_contracts.append(contract)
                    print(f"    Generated {len(questions)} questions")
                else:
                    print(f"    No multi-hop patterns found")

            except Exception as e:
                print(f"    Error: {e}")
                continue

        # Create dataset
        dataset = MultiHopDataset(
            contracts=processed_contracts,
            questions=all_questions,
            metadata={
                "source": "CUAD + EngramDB Multi-Hop Generator",
                "total_contracts": len(processed_contracts),
                "total_questions": len(all_questions),
                "avg_hops": sum(q.hop_count for q in all_questions) / len(all_questions) if all_questions else 0,
            }
        )

        print(f"\nGenerated {len(all_questions)} multi-hop questions from {len(processed_contracts)} contracts")

        return dataset


def main():
    """Generate multi-hop QA dataset from CUAD contracts."""
    # Load CUAD contracts
    print("=" * 60)
    print("Multi-Hop QA Generator")
    print("=" * 60)

    cuad_path = Path("data/cuad/cuad_contracts_50.json")
    if not cuad_path.exists():
        print("CUAD contracts not found. Run cuad_loader.py first.")
        return

    loader = CUADLoader()
    contracts = loader.load_contracts(cuad_path)
    print(f"Loaded {len(contracts)} contracts from CUAD")

    # Use all 50 contracts for robust evaluation
    # contracts = contracts[:15]  # Removed limit

    # Generate multi-hop questions
    generator = MultiHopGenerator()
    dataset = generator.generate_dataset(contracts, max_questions_per_contract=8)

    # Save dataset
    output_path = Path("data/cuad/multihop_qa_dataset.json")
    dataset.save(output_path)

    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Contracts: {dataset.metadata['total_contracts']}")
    print(f"Total questions: {dataset.metadata['total_questions']}")
    print(f"Average hops: {dataset.metadata['avg_hops']:.1f}")

    # Question types
    type_counts = {}
    for q in dataset.questions:
        type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1

    print("\n--- Question Types ---")
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {qtype}: {count}")

    # Sample questions
    if dataset.questions:
        print("\n" + "=" * 60)
        print("Sample Questions")
        print("=" * 60)
        for q in dataset.questions[:3]:
            print(f"\n[{q.question_type}] (hops: {q.hop_count})")
            print(f"Q: {q.question}")
            print(f"A: {q.answer[:200]}...")
            print(f"Chain: {' -> '.join(q.reasoning_chain)}")


if __name__ == "__main__":
    main()
