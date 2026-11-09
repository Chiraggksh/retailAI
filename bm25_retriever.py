"""
BM25 keyword-based retrieval module using rank-bm25.

Provides lexical search over policy chunks preserving source metadata and
returning standardized RetrievalResult structures.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from rank_bm25 import BM25Okapi

from chunking import PolicyChunk
from retriever_models import RetrievalResult

logger = logging.getLogger(__name__)


def default_bm25_tokenizer(text: str) -> List[str]:
    """
    Tokenize text into lowercase alphanumeric words and hyphens.
    Preserves policy codes, numbers, and identifiers (e.g. '30', '5-10', 'pol-ret-001').
    """
    if not text:
        return []
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if t]


class BM25Retriever:
    """Keyword retriever implementing BM25Okapi over policy chunks."""

    def __init__(
        self,
        chunks: Optional[List[PolicyChunk]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        """
        Initialize the BM25 index over policy chunks.

        Args:
            chunks: List of PolicyChunk objects.
            tokenizer: Optional custom tokenizer function.
        """
        self.tokenizer = tokenizer or default_bm25_tokenizer
        self.chunks: List[PolicyChunk] = []
        self.corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: List[PolicyChunk]) -> None:
        """Index or re-index a list of policy chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.corpus = []
            self.bm25 = None
            return

        self.corpus = [self.tokenizer(c.chunk_text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"BM25 indexed {len(self.chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve the top K most relevant chunks using BM25 scoring.

        Args:
            query: User search query.
            top_k: Number of top results to return.

        Returns:
            List of standardized RetrievalResult objects.
        """
        if not query or not query.strip():
            return []

        if self.bm25 is None or not self.chunks:
            logger.warning("BM25 index is empty; returning no results.")
            return []

        tokenized_query = self.tokenizer(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)

        # Pair chunks with their scores and original indices
        scored_pairs = list(enumerate(zip(self.chunks, doc_scores)))
        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1][1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (orig_idx, (chunk, score)) in enumerate(scored_pairs[:top_k], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_version=chunk.document_version,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    chunk_text=chunk.chunk_text,
                    score=float(score),
                    retrieval_method="bm25",
                    rank=rank,
                    metadata=chunk.to_metadata(),
                )
            )

        return results
