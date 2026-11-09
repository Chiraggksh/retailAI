"""
Vector store integration using ChromaDB for policy documents.

Implements:
1. Idempotent upsert of policy chunks with deterministic IDs.
2. Explicit handling of document version changes (purges superseded versions).
3. Dense vector similarity search with cosine distance scoring.
4. Chunk retrieval by ID.
5. Collection statistics and metadata inspection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from chunking import PolicyChunk
from embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "northstar_policy_store"


class PolicyVectorStore:
    """Manages ChromaDB storage and similarity search for policy chunks."""

    def __init__(
        self,
        persist_directory: Optional[str | Path] = None,
        collection_name: str = COLLECTION_NAME,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> None:
        """
        Initialize ChromaDB client and collection.

        Args:
            persist_directory: Path to ChromaDB storage directory. If None, uses in-memory client.
            collection_name: Target collection name.
            embedding_manager: Optional EmbeddingManager instance for embedding queries.
        """
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()

        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        else:
            self.client = chromadb.EphemeralClient()

        # Cosine distance metric for normalized sentence embeddings
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _handle_version_updates(self, incoming_chunks: List[PolicyChunk]) -> None:
        """
        Inspect incoming chunks by document_id and version.
        If a new version of an existing document is being ingested, delete old chunks
        for that document_id to avoid mixing different policy versions.
        """
        doc_versions: Dict[str, str] = {}
        for c in incoming_chunks:
            doc_versions[c.document_id] = c.document_version

        for doc_id, incoming_version in doc_versions.items():
            try:
                # Query existing chunks for this doc_id
                existing = self.collection.get(
                    where={"document_id": doc_id},
                    include=["metadatas"],
                )
                if existing and existing.get("ids"):
                    existing_metas = existing.get("metadatas", [])
                    existing_versions = {
                        str(m.get("document_version"))
                        for m in existing_metas
                        if m and "document_version" in m
                    }

                    # If existing collection has older or different versions, purge them
                    if existing_versions and incoming_version not in existing_versions:
                        logger.info(
                            f"Superseding document '{doc_id}' version(s) {existing_versions} "
                            f"with incoming version '{incoming_version}'. Purging {len(existing['ids'])} old chunks."
                        )
                        self.collection.delete(ids=existing["ids"])
            except Exception as e:
                logger.warning(f"Error checking version conflicts for doc_id {doc_id}: {e}")

    def add_chunks(
        self,
        chunks: List[PolicyChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Ingest chunks into ChromaDB with deterministic IDs and idempotent upsert.
        Handles version conflicts before upserting.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        # Purge superseded versions if applicable
        self._handle_version_updates(chunks)

        chunk_ids = [c.chunk_id for c in chunks]
        documents = [c.chunk_text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        # Generate embeddings if not pre-computed
        if embeddings is None:
            embeddings = self.embedding_manager.embed_documents(documents)

        # ChromaDB upsert guarantees idempotency: existing IDs are overwritten, new IDs added
        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        logger.info(f"Successfully upserted {len(chunks)} chunks into collection '{self.collection_name}'.")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform dense semantic vector search for a natural language query.

        Args:
            query: Query string.
            top_k: Number of nearest neighbors to return.
            where_filter: ChromaDB metadata filter dict.

        Returns:
            List of result dicts containing chunk_id, text, metadata, and similarity score.
        """
        if not query.strip():
            return []

        query_embedding = self.embedding_manager.embed_query(query)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        formatted_results: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: distance in [0, 2], similarity = 1 - (dist / 2) or 1 - dist
            # For unit vectors with cosine space: similarity = 1.0 - dist
            similarity = round(max(0.0, 1.0 - dist), 4)
            formatted_results.append({
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "score": similarity,
                "distance": round(dist, 4),
            })

        return formatted_results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by its deterministic chunk ID."""
        try:
            res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if res and res.get("ids") and len(res["ids"]) > 0:
                return {
                    "chunk_id": res["ids"][0],
                    "text": res["documents"][0],
                    "metadata": res["metadatas"][0],
                }
        except Exception as e:
            logger.error(f"Error fetching chunk {chunk_id}: {e}")
        return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics and summary statistics for the vector collection."""
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "unique_documents": 0,
                "documents": {},
            }

        all_items = self.collection.get(include=["metadatas"])
        metas = all_items.get("metadatas", [])

        doc_summary: Dict[str, Dict[str, Any]] = {}
        for m in metas:
            if not m:
                continue
            doc_id = m.get("document_id", "UNKNOWN")
            version = m.get("document_version", "1.0")
            sec_title = m.get("section_title", "General")

            if doc_id not in doc_summary:
                doc_summary[doc_id] = {
                    "version": version,
                    "chunk_count": 0,
                    "sections": set(),
                }
            doc_summary[doc_id]["chunk_count"] += 1
            doc_summary[doc_id]["sections"].add(sec_title)

        # Convert sets to sorted lists for clean serialization
        clean_docs = {
            k: {
                "version": v["version"],
                "chunk_count": v["chunk_count"],
                "sections": sorted(list(v["sections"])),
            }
            for k, v in doc_summary.items()
        }

        return {
            "collection_name": self.collection_name,
            "total_chunks": total_chunks,
            "unique_documents": len(clean_docs),
            "documents": clean_docs,
        }

    def clear(self) -> None:
        """Delete all chunks in the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning(f"Error clearing collection: {e}")
