"""
Embedding management module using Sentence Transformers.

Provides:
1. Configurable model selection (defaults to lightweight 'all-MiniLM-L6-v2').
2. Direct access to the model's underlying tokenizer for accurate token counting in chunking.
3. Document batch embedding and single-query embedding.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Ensure local workspace cache is used if present
_workspace_cache = Path(__file__).resolve().parent / ".cache" / "huggingface"
if _workspace_cache.exists():
    os.environ.setdefault("HF_HOME", str(_workspace_cache))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """Manages text embedding generation and tokenizer access using Sentence Transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the embedding manager.

        Args:
            model_name: HuggingFace model name or path.
                        Defaults to env var EMBEDDING_MODEL_NAME or 'all-MiniLM-L6-v2'.
            device: 'cpu', 'cuda', or None (auto-detect).
        """
        self.model_name = (
            model_name
            or os.environ.get("EMBEDDING_MODEL_NAME")
            or DEFAULT_MODEL_NAME
        )
        self.device = device or ("cuda" if False else "cpu")
        self._model: Optional[Any] = None

    def _load_model(self) -> None:
        """Lazy load the SentenceTransformer model and its tokenizer."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        local_files_only=True,
                    )
                except Exception:
                    self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model {self.model_name}: {e}")
                raise

    @property
    def model(self) -> Any:
        """Access the loaded SentenceTransformer instance."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """
        Access the underlying tokenizer from the SentenceTransformer model.
        Used by the chunker for model-aligned token counts.
        """
        if self._model is None:
            self._load_model()
        return self._model.tokenizer

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        if self._model is None:
            self._load_model()
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate dense vector embeddings for a list of document strings.
        """
        if not texts:
            return []
        if self._model is None:
            self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate dense vector embedding for a single search query string.
        """
        if not query.strip():
            return [0.0] * self.dimension
        if self._model is None:
            self._load_model()

        embedding = self._model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
