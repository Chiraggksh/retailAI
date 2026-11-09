"""
Retrieval evaluation module for Northstar Retail Customer Support.

Implements:
1. Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
2. Per-query and aggregate metric calculations for configurable K values (e.g. K=3, K=5).
3. Latency measurement across BM25, dense retrieval, and dense + CrossEncoder reranking.
4. Export of structured evaluation results to JSON and Markdown summary tables.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from bm25_retriever import BM25Retriever
from chunking import StructureAwareChunker
from dense_retriever import DenseRetriever
from embeddings import EmbeddingManager
from ingestion import PolicyIngestor
from reranker import CrossEncoderReranker
from retriever_models import RetrievalResult
from vector_store import PolicyVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric Calculation Functions
# ---------------------------------------------------------------------------

def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Precision@K: proportion of retrieved chunks in the top K that are relevant.

    Precision@K = |Retrieved@K ∩ Relevant| / K
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return round(hits / k, 4)


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@K: proportion of all relevant chunks that appear in the top K.

    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
    If |Relevant| == 0:
        - If query is an unanswerable/negative test case with no relevant items,
          we define Recall as 1.0 (all 0 relevant items were retrieved).
    """
    if not relevant_ids:
        return 1.0  # Perfect recall on empty target set

    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def compute_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str], k: Optional[int] = None) -> float:
    """
    Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank position of the first relevant chunk.

    RR = 1 / rank_of_first_relevant_item, or 0.0 if no relevant item is found in top K.
    """
    if not relevant_ids:
        return 0.0

    items = retrieved_ids[:k] if k is not None else retrieved_ids
    for rank_1_based, cid in enumerate(items, start=1):
        if cid in relevant_ids:
            return round(1.0 / rank_1_based, 4)
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation Result Structures
# ---------------------------------------------------------------------------

@dataclass
class QueryMetrics:
    """Performance metrics for a single query."""
    query_id: str
    query_text: str
    category: str
    relevant_chunks: List[str]
    retrieved_chunks: List[str]
    precision_at_3: float
    recall_at_3: float
    mrr_at_3: float
    precision_at_5: float
    recall_at_5: float
    mrr_at_5: float
    latency_ms: float
    top_1_hit: bool


@dataclass
class MethodSummary:
    """Aggregated evaluation results for a single retrieval approach."""
    method_name: str
    query_count: int
    mean_precision_at_3: float
    mean_recall_at_3: float
    mean_mrr_at_3: float
    mean_precision_at_5: float
    mean_recall_at_5: float
    mean_mrr_at_5: float
    mean_latency_ms: float
    top_1_accuracy: float
    per_query_results: List[QueryMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "query_count": self.query_count,
            "mean_precision_at_3": round(self.mean_precision_at_3, 4),
            "mean_recall_at_3": round(self.mean_recall_at_3, 4),
            "mean_mrr_at_3": round(self.mean_mrr_at_3, 4),
            "mean_precision_at_5": round(self.mean_precision_at_5, 4),
            "mean_recall_at_5": round(self.mean_recall_at_5, 4),
            "mean_mrr_at_5": round(self.mean_mrr_at_5, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "top_1_accuracy": round(self.top_1_accuracy, 4),
            "per_query_results": [asdict(q) for q in self.per_query_results],
        }


# ---------------------------------------------------------------------------
# Retrieval Evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """Executes standardized evaluation across BM25, dense retrieval, and CrossEncoder reranking."""

    def __init__(
        self,
        ground_truth_path: str | Path = "./data/evaluation/ground_truth_relevance.json",
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.ground_truth_path = Path(ground_truth_path)
        self.test_cases: List[Dict[str, Any]] = []
        self._load_ground_truth()

        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker

    def _load_ground_truth(self) -> None:
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found at: {self.ground_truth_path}")
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.test_cases = data.get("cases", [])

    def evaluate_retrieval_function(
        self,
        method_name: str,
        retrieval_fn: Callable[[str, int], List[RetrievalResult]],
        top_k: int = 5,
    ) -> MethodSummary:
        """
        Evaluate a single retrieval function against all test cases.

        Args:
            method_name: Label for the retrieval strategy.
            retrieval_fn: Callable taking (query_string, top_k) -> List[RetrievalResult].
            top_k: Maximum candidate pool to retrieve.

        Returns:
            MethodSummary containing per-query and macro-averaged metrics.
        """
        query_metrics_list: List[QueryMetrics] = []
        latencies: List[float] = []

        for case in self.test_cases:
            qid = case["id"]
            query = case["query"]
            category = case.get("category", "general")
            relevant_chunks = set(case.get("relevant_chunk_ids", []))

            # Timed retrieval
            start_time = time.perf_counter()
            retrieved_results = retrieval_fn(query, top_k)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            latencies.append(elapsed_ms)

            retrieved_chunk_ids = [r.chunk_id for r in retrieved_results]

            # Compute metrics for K=3 and K=5
            p3 = compute_precision_at_k(retrieved_chunk_ids, relevant_chunks, k=3)
            r3 = compute_recall_at_k(retrieved_chunk_ids, relevant_chunks, k=3)
            mrr3 = compute_reciprocal_rank(retrieved_chunk_ids, relevant_chunks, k=3)

            p5 = compute_precision_at_k(retrieved_chunk_ids, relevant_chunks, k=5)
            r5 = compute_recall_at_k(retrieved_chunk_ids, relevant_chunks, k=5)
            mrr5 = compute_reciprocal_rank(retrieved_chunk_ids, relevant_chunks, k=5)

            # Top-1 accuracy (is first retrieved chunk relevant?)
            top_1_hit = bool(retrieved_chunk_ids and retrieved_chunk_ids[0] in relevant_chunks)

            query_metrics_list.append(
                QueryMetrics(
                    query_id=qid,
                    query_text=query,
                    category=category,
                    relevant_chunks=list(relevant_chunks),
                    retrieved_chunks=retrieved_chunk_ids,
                    precision_at_3=p3,
                    recall_at_3=r3,
                    mrr_at_3=mrr3,
                    precision_at_5=p5,
                    recall_at_5=r5,
                    mrr_at_5=mrr5,
                    latency_ms=round(elapsed_ms, 2),
                    top_1_hit=top_1_hit,
                )
            )

        n = len(query_metrics_list)
        if n == 0:
            return MethodSummary(
                method_name=method_name,
                query_count=0,
                mean_precision_at_3=0.0,
                mean_recall_at_3=0.0,
                mean_mrr_at_3=0.0,
                mean_precision_at_5=0.0,
                mean_recall_at_5=0.0,
                mean_mrr_at_5=0.0,
                mean_latency_ms=0.0,
                top_1_accuracy=0.0,
            )

        return MethodSummary(
            method_name=method_name,
            query_count=n,
            mean_precision_at_3=sum(q.precision_at_3 for q in query_metrics_list) / n,
            mean_recall_at_3=sum(q.recall_at_3 for q in query_metrics_list) / n,
            mean_mrr_at_3=sum(q.mrr_at_3 for q in query_metrics_list) / n,
            mean_precision_at_5=sum(q.precision_at_5 for q in query_metrics_list) / n,
            mean_recall_at_5=sum(q.recall_at_5 for q in query_metrics_list) / n,
            mean_mrr_at_5=sum(q.mrr_at_5 for q in query_metrics_list) / n,
            mean_latency_ms=sum(latencies) / n,
            top_1_accuracy=sum(1 for q in query_metrics_list if q.top_1_hit) / n,
            per_query_results=query_metrics_list,
        )

    def run_full_comparison(
        self,
        output_json_path: Optional[str | Path] = "./data/evaluation/retrieval_benchmark_results.json",
    ) -> Dict[str, MethodSummary]:
        """
        Execute benchmark over BM25, Dense Retrieval, and Dense + CrossEncoder Reranking.
        """
        # Ensure components exist
        if self.dense_retriever is None:
            self.dense_retriever = DenseRetriever()

        if self.reranker is None:
            self.reranker = CrossEncoderReranker()

        if self.bm25_retriever is None:
            # Reconstruct BM25 index from ingested chunks
            ingestor = PolicyIngestor(manifest_path="./data/manifest.json")
            raw_docs = ingestor.ingest_directory("./data/documents")
            chunker = StructureAwareChunker(tokenizer=self.dense_retriever.vector_store.embedding_manager.tokenizer)
            chunks = []
            for doc in raw_docs:
                chunks.extend(chunker.chunk_document(doc))
            self.bm25_retriever = BM25Retriever(chunks)

        summaries: Dict[str, MethodSummary] = {}

        # 1. BM25 Evaluation
        summaries["BM25"] = self.evaluate_retrieval_function(
            method_name="BM25",
            retrieval_fn=lambda q, k: self.bm25_retriever.retrieve(q, top_k=k),
            top_k=5,
        )

        # 2. Dense Retrieval Evaluation
        summaries["Dense"] = self.evaluate_retrieval_function(
            method_name="Dense (all-MiniLM-L6-v2)",
            retrieval_fn=lambda q, k: self.dense_retriever.retrieve(q, top_k=k),
            top_k=5,
        )

        # 3. Dense + CrossEncoder Evaluation (retrieve top-10 candidates, rerank to top-5)
        def dense_plus_cross_encoder(query: str, k: int) -> List[RetrievalResult]:
            candidates = self.dense_retriever.retrieve(query, top_k=max(k, 10))
            return self.reranker.rerank(query, candidates, top_k=k)

        summaries["Dense + CrossEncoder"] = self.evaluate_retrieval_function(
            method_name="Dense + CrossEncoder (ms-marco)",
            retrieval_fn=dense_plus_cross_encoder,
            top_k=5,
        )

        # Save to JSON
        if output_json_path:
            out_path = Path(output_json_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {name: s.to_dict() for name, s in summaries.items()}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"Evaluation benchmark results saved to {out_path}")

        return summaries


def format_markdown_comparison_table(summaries: Dict[str, MethodSummary]) -> str:
    """Format evaluation summaries into a clean GitHub Flavored Markdown table."""
    lines = [
        "| Retrieval Method | Precision@3 | Recall@3 | MRR@3 | Precision@5 | Recall@5 | MRR@5 | Top-1 Accuracy | Latency (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| **{name}** | {s.mean_precision_at_3:.4f} | {s.mean_recall_at_3:.4f} | {s.mean_mrr_at_3:.4f} | "
            f"{s.mean_precision_at_5:.4f} | {s.mean_recall_at_5:.4f} | {s.mean_mrr_at_5:.4f} | "
            f"{s.top_1_accuracy:.4f} | {s.mean_latency_ms:.2f} ms |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    results = evaluator.run_full_comparison()
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION BENCHMARK RESULTS")
    print("=" * 80)
    print(format_markdown_comparison_table(results))
    print("=" * 80 + "\n")
