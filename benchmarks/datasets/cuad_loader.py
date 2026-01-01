"""
CUAD Dataset Loader

Downloads and processes the Contract Understanding Atticus Dataset (CUAD)
for use in EngramDB benchmarks.

CUAD contains:
- 510 commercial legal contracts from SEC EDGAR
- 13,000+ expert annotations across 41 clause categories
- Extractive QA format (SQuAD-style)

Source: https://zenodo.org/records/4595826
License: CC BY 4.0
"""

import json
import zipfile
import urllib.request
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict


@dataclass
class ContractQA:
    """A single QA pair from a contract."""
    id: str
    question: str
    answer_text: str
    answer_start: int
    clause_category: str


@dataclass
class Contract:
    """A processed contract with its text and QA annotations."""
    id: str
    title: str
    text: str
    source: str  # e.g., "SEC EDGAR"
    contract_type: Optional[str] = None
    qa_pairs: list[ContractQA] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Contract":
        qa_pairs = [ContractQA(**qa) for qa in data.pop("qa_pairs", [])]
        return cls(**data, qa_pairs=qa_pairs)


# The 41 CUAD clause categories
CUAD_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]


def extract_clause_category(question: str) -> str:
    """Extract the clause category from a CUAD question."""
    import re
    match = re.search(r"related to ['\"]([^'\"]+)['\"]", question)
    if match:
        return match.group(1)
    return "Unknown"


class CUADLoader:
    """
    Loader for the CUAD dataset from Zenodo.

    Usage:
        loader = CUADLoader()
        contracts = loader.load()

        # Or load a subset
        contracts = loader.load(max_contracts=50)
    """

    ZENODO_URL = "https://zenodo.org/api/records/4595826/files/CUAD_v1.zip/content"
    ZENODO_RECORD = "https://zenodo.org/records/4595826"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cuad")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.cache_dir / "CUAD_v1.zip"
        self.extracted_dir = self.cache_dir / "CUAD_v1"

    def download(self, force: bool = False) -> Path:
        """
        Download CUAD dataset from Zenodo.

        Args:
            force: Re-download even if already exists

        Returns:
            Path to extracted directory
        """
        if self.extracted_dir.exists() and not force:
            print(f"CUAD already downloaded at {self.extracted_dir}")
            return self.extracted_dir

        if not self.zip_path.exists() or force:
            print(f"Downloading CUAD from Zenodo (~106MB)...")
            print(f"Source: {self.ZENODO_RECORD}")

            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rProgress: {percent:.1f}% ({downloaded // 1024 // 1024}MB)", end="")

            urllib.request.urlretrieve(
                self.ZENODO_URL,
                self.zip_path,
                reporthook=report_progress
            )
            print("\nDownload complete!")

        # Extract
        print(f"Extracting to {self.extracted_dir}...")
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            zf.extractall(self.cache_dir)

        print("Extraction complete!")
        return self.extracted_dir

    def load(
        self,
        split: str = "test",
        max_contracts: Optional[int] = None
    ) -> list[Contract]:
        """
        Load contracts from CUAD.

        Args:
            split: "train" or "test"
            max_contracts: Maximum number of contracts to load

        Returns:
            List of Contract objects
        """
        # Ensure downloaded
        self.download()

        # Find the JSON file
        json_file = self.extracted_dir / "CUAD_v1.json"
        if not json_file.exists():
            # Try alternative paths
            for alt_path in [
                self.extracted_dir / "CUADv1.json",
                self.cache_dir / "CUAD_v1" / "CUAD_v1.json",
            ]:
                if alt_path.exists():
                    json_file = alt_path
                    break

        if not json_file.exists():
            # List what we have
            print(f"Contents of {self.extracted_dir}:")
            for p in self.extracted_dir.rglob("*"):
                print(f"  {p}")
            raise FileNotFoundError(f"Could not find CUADv1.json in {self.extracted_dir}")

        print(f"Loading CUAD from {json_file}...")

        with open(json_file) as f:
            data = json.load(f)

        # CUAD uses SQuAD format
        # data = {"data": [{"title": "...", "paragraphs": [{"context": "...", "qas": [...]}]}]}
        contracts_dict: dict[str, Contract] = {}

        for doc in data.get("data", []):
            title = doc["title"]

            # Combine all paragraphs into one text
            full_text = ""
            all_qas = []

            for para in doc.get("paragraphs", []):
                para_context = para.get("context", "")
                offset = len(full_text)

                # Add paragraph to full text
                if full_text:
                    full_text += "\n\n"
                    offset += 2
                full_text += para_context

                # Process QAs with adjusted offsets
                for qa in para.get("qas", []):
                    if qa.get("answers"):
                        answer = qa["answers"][0]
                        all_qas.append({
                            "id": qa.get("id", ""),
                            "question": qa.get("question", ""),
                            "answer_text": answer.get("text", ""),
                            "answer_start": answer.get("answer_start", 0) + offset,
                        })

            if not full_text:
                continue

            contract_type = self._extract_contract_type(title)

            contract = Contract(
                id=title,
                title=title,
                text=full_text,
                source="SEC EDGAR (CUAD)",
                contract_type=contract_type,
                qa_pairs=[],
                metadata={"cuad_split": split}
            )

            # Add QA pairs
            for qa_data in all_qas:
                qa = ContractQA(
                    id=qa_data["id"],
                    question=qa_data["question"],
                    answer_text=qa_data["answer_text"],
                    answer_start=qa_data["answer_start"],
                    clause_category=extract_clause_category(qa_data["question"])
                )
                contract.qa_pairs.append(qa)

            contracts_dict[title] = contract

        contracts = list(contracts_dict.values())

        # Sort by number of QA pairs (contracts with more annotations are richer)
        contracts.sort(key=lambda c: len(c.qa_pairs), reverse=True)

        if max_contracts:
            contracts = contracts[:max_contracts]

        print(f"Loaded {len(contracts)} contracts with {sum(len(c.qa_pairs) for c in contracts)} QA pairs")

        return contracts

    def _extract_contract_type(self, title: str) -> Optional[str]:
        """Extract contract type from title."""
        title_lower = title.lower()

        contract_types = [
            ("license agreement", "License Agreement"),
            ("service agreement", "Service Agreement"),
            ("distributor agreement", "Distributor Agreement"),
            ("supply agreement", "Supply Agreement"),
            ("purchase agreement", "Purchase Agreement"),
            ("employment agreement", "Employment Agreement"),
            ("consulting agreement", "Consulting Agreement"),
            ("development agreement", "Development Agreement"),
            ("collaboration agreement", "Collaboration Agreement"),
            ("marketing agreement", "Marketing Agreement"),
            ("manufacturing agreement", "Manufacturing Agreement"),
            ("hosting agreement", "Hosting Agreement"),
            ("maintenance agreement", "Maintenance Agreement"),
            ("nda", "Non-Disclosure Agreement"),
            ("non-disclosure", "Non-Disclosure Agreement"),
            ("confidentiality", "Confidentiality Agreement"),
            ("lease", "Lease Agreement"),
            ("loan", "Loan Agreement"),
            ("credit", "Credit Agreement"),
            ("partnership", "Partnership Agreement"),
            ("joint venture", "Joint Venture Agreement"),
            ("merger", "Merger Agreement"),
            ("acquisition", "Acquisition Agreement"),
            ("settlement", "Settlement Agreement"),
            ("amendment", "Amendment"),
        ]

        for pattern, ctype in contract_types:
            if pattern in title_lower:
                return ctype

        return "Other"

    def save_contracts(self, contracts: list[Contract], filepath: Path):
        """Save contracts to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in contracts]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(contracts)} contracts to {filepath}")

    def load_contracts(self, filepath: Path) -> list[Contract]:
        """Load contracts from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return [Contract.from_dict(d) for d in data]

    def get_statistics(self, contracts: list[Contract]) -> dict:
        """Get statistics about loaded contracts."""
        total_qa = sum(len(c.qa_pairs) for c in contracts)
        total_chars = sum(len(c.text) for c in contracts)

        # Count by contract type
        type_counts = defaultdict(int)
        for c in contracts:
            type_counts[c.contract_type or "Unknown"] += 1

        # Count by clause category
        category_counts = defaultdict(int)
        for c in contracts:
            for qa in c.qa_pairs:
                category_counts[qa.clause_category] += 1

        return {
            "num_contracts": len(contracts),
            "total_qa_pairs": total_qa,
            "avg_qa_per_contract": total_qa / len(contracts) if contracts else 0,
            "total_characters": total_chars,
            "avg_chars_per_contract": total_chars / len(contracts) if contracts else 0,
            "contract_types": dict(type_counts),
            "clause_categories": dict(category_counts),
        }


def main():
    """Download and process CUAD dataset."""
    loader = CUADLoader()

    # Load all contracts
    print("=" * 60)
    print("CUAD Dataset Loader")
    print("=" * 60)
    print(f"\nSource: {loader.ZENODO_RECORD}")
    print("License: CC BY 4.0")
    print()

    contracts = loader.load(max_contracts=50)

    # Print statistics
    stats = loader.get_statistics(contracts)
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Contracts: {stats['num_contracts']}")
    print(f"Total QA pairs: {stats['total_qa_pairs']}")
    print(f"Avg QA per contract: {stats['avg_qa_per_contract']:.1f}")
    print(f"Avg chars per contract: {stats['avg_chars_per_contract']:,.0f}")

    print("\n--- Contract Types ---")
    for ctype, count in sorted(stats['contract_types'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {ctype}: {count}")

    print("\n--- Top Clause Categories ---")
    for cat, count in sorted(stats['clause_categories'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")

    # Save to file
    output_path = Path("data/cuad/cuad_contracts_50.json")
    loader.save_contracts(contracts, output_path)

    # Show a sample contract
    if contracts:
        sample = contracts[0]
        print("\n" + "=" * 60)
        print(f"Sample Contract: {sample.title[:60]}...")
        print("=" * 60)
        print(f"Type: {sample.contract_type}")
        print(f"Length: {len(sample.text):,} chars")
        print(f"QA pairs: {len(sample.qa_pairs)}")
        print(f"\nFirst 500 chars:\n{sample.text[:500]}...")

        if sample.qa_pairs:
            print(f"\n--- Sample QA ---")
            qa = sample.qa_pairs[0]
            print(f"Category: {qa.clause_category}")
            print(f"Q: {qa.question[:100]}...")
            print(f"A: {qa.answer_text[:200]}...")


if __name__ == "__main__":
    main()
