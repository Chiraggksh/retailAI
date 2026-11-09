"""
Dense semantic retrieval module wrapping PolicyVectorStore.

Interprets ChromaDB cosine distances into normalized similarity scores and
returns standardized RetrievalResult objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from retriever_models import RetrievalResult
from vector_store import PolicyVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Retrieves policy chunks via dense vector embeddings from ChromaDB.

    Score interpretation:
    - ChromaDB computes HNSW cosine distance d in [0, 2] for normalized vectors.
    - We map distance d to cosine similarity: score = max(0.0, 1.0 - d).
    - The returned 'score' is cosine similarity (higher is better, 1.0 is identical).
    - The underlying raw distance is preserved in metadata['cosine_distance'].
    """

    def __init__(
        self,
        vector_store: Optional[PolicyVectorStore] = None,
        persist_directory: Optional[str | Path] = "./data/chroma_db",
    ) -> None:
        """
        Initialize the dense retriever reusing the existing PolicyVectorStore.

        Args:
            vector_store: Pre-existing PolicyVectorStore instance.
            persist_directory: Storage directory if initializing a new vector store client.
        """
        self.vector_store = vector_store or PolicyVectorStore(
            persist_directory=persist_directory,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Perform dense semantic retrieval for the query.

        Args:
            query: User search query.
            top_k: Number of nearest chunks to retrieve.
            where_filter: Optional ChromaDB metadata filter.

        Returns:
            List of standardized RetrievalResult objects sorted by descending similarity.
        """
        if not query or not query.strip():
            return []

        try:
            raw_results = self.vector_store.search(
                query=query,
                top_k=top_k,
                where_filter=where_filter,
            )

            results: List[RetrievalResult] = []
            for rank_idx, item in enumerate(raw_results, start=1):
                meta = dict(item.get("metadata", {}))
                meta["cosine_distance"] = item.get("distance", 0.0)

                results.append(
                    RetrievalResult(
                        chunk_id=item["chunk_id"],
                        document_id=meta.get("document_id", "UNKNOWN"),
                        document_version=meta.get("document_version", "1.0"),
                        source_file=meta.get("source_file", ""),
                        page_number=int(meta.get("page_number", 1)),
                        section_title=meta.get("section_title", "General"),
                        chunk_text=item.get("text", ""),
                        score=float(item.get("score", 0.0)),
                        retrieval_method="dense",
                        rank=rank_idx,
                        metadata=meta,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error during dense retrieval for query '{query}': {e}")
            return []
