"""
Standardized models and data classes for retrieval and reranking results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class RetrievalResult:
    """Unified retrieval result format across BM25, dense, and reranked pipelines."""
    chunk_id: str
    document_id: str
    document_version: str
    source_file: str
    page_number: int
    section_title: str
    chunk_text: str
    score: float
    retrieval_method: str
    rank: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d
