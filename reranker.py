"""
CrossEncoder second-stage reranking module using Sentence Transformers.

Implements optional reranking of dense retrieval candidates using
'cross-encoder/ms-marco-MiniLM-L-6-v2'.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from retriever_models import RetrievalResult

logger = logging.getLogger(__name__)

# Ensure local workspace cache is discovered
_workspace_cache = Path(__file__).resolve().parent / ".cache" / "huggingface"
if _workspace_cache.exists():
    os.environ.setdefault("HF_HOME", str(_workspace_cache))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks candidate chunks using a trained Cross-Encoder scoring model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the CrossEncoder reranker.

        Args:
            model_name: CrossEncoder HuggingFace model identifier or local path.
            device: 'cpu', 'cuda', or None.
        """
        self.model_name = (
            model_name
            or os.environ.get("RERANKER_MODEL_NAME")
            or DEFAULT_RERANKER_MODEL
        )
        self.device = device or "cpu"
        self._model: Optional[Any] = None

    def _load_model(self) -> None:
        """Lazy load the CrossEncoder model with offline-first resolution."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model '{self.model_name}' on '{self.device}'...")
                try:
                    self._model = CrossEncoder(
                        self.model_name,
                        device=self.device,
                        local_files_only=True,
                    )
                except Exception:
                    self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder model {self.model_name}: {e}")
                raise

    @property
    def model(self) -> Any:
        if self._model is None:
            self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank a list of candidate RetrievalResult objects by computing query-passage CrossEncoder scores.

        Args:
            query: User search query.
            candidates: Candidate retrieval results (e.g. from dense retrieval).
            top_k: Optional number of top reranked items to return. If None, returns all candidates sorted.

        Returns:
            Sorted list of RetrievalResult objects with CrossEncoder scores and updated ranks.
        """
        if not candidates or not query.strip():
            return candidates[:top_k] if top_k else list(candidates)

        try:
            # Prepare pairs of (query, chunk_text)
            pairs = [(query, c.chunk_text) for c in candidates]
            scores = self.model.predict(pairs)

            reranked: List[RetrievalResult] = []
            for candidate, score in zip(candidates, scores):
                # Copy metadata to preserve original dense score and distance
                meta = dict(candidate.metadata)
                meta["dense_score"] = candidate.score
                meta["cross_encoder_score"] = float(score)

                reranked.append(
                    RetrievalResult(
                        chunk_id=candidate.chunk_id,
                        document_id=candidate.document_id,
                        document_version=candidate.document_version,
                        source_file=candidate.source_file,
                        page_number=candidate.page_number,
                        section_title=candidate.section_title,
                        chunk_text=candidate.chunk_text,
                        score=float(score),
                        retrieval_method="dense+cross_encoder",
                        metadata=meta,
                    )
                )

            # Sort descending by CrossEncoder score
            reranked.sort(key=lambda x: x.score, reverse=True)

            # Update rank index (1-based)
            for rank_idx, item in enumerate(reranked, start=1):
                item.rank = rank_idx

            return reranked[:top_k] if top_k else reranked

        except Exception as e:
            logger.error(f"Error during CrossEncoder reranking: {e}; returning original candidates.")
            return candidates[:top_k] if top_k else list(candidates)
