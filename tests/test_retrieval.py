"""
Unit tests for Phase 2: BM25, Dense Retrieval, CrossEncoder Reranking, and Evaluation Metrics.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bm25_retriever import BM25Retriever, default_bm25_tokenizer
from chunking import PolicyChunk
from dense_retriever import DenseRetriever
from evaluation import (
    RetrievalEvaluator,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from reranker import CrossEncoderReranker
from retriever_models import RetrievalResult
from vector_store import PolicyVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks():
    """Sample PolicyChunk records for isolated unit testing."""
    return [
        PolicyChunk(
            chunk_id="CHUNK-RET-01",
            document_id="POL-RET-001",
            source_file="returns_policy.pdf",
            page_number=1,
            document_version="1.2",
            section_title="Return Window",
            chunk_index=0,
            chunk_text="[Return Window]\nCustomers have 30 calendar days from delivery date to return physical items.",
            token_count=20,
        ),
        PolicyChunk(
            chunk_id="CHUNK-SHIP-01",
            document_id="POL-SHIP-001",
            source_file="shipping_policy.pdf",
            page_number=1,
            document_version="1.0",
            section_title="Delivery Estimates",
            chunk_index=0,
            chunk_text="[Delivery Estimates]\nStandard domestic shipping takes 3 to 7 business days after dispatch.",
            token_count=18,
        ),
        PolicyChunk(
            chunk_id="CHUNK-REF-01",
            document_id="POL-REF-001",
            source_file="refund_policy.pdf",
            page_number=1,
            document_version="1.1",
            section_title="Refund Processing",
            chunk_index=0,
            chunk_text="[Refund Processing]\nApproved refunds take 5 to 10 business days to post to your bank account.",
            token_count=19,
        ),
    ]


# ---------------------------------------------------------------------------
# 1. BM25 Tests
# ---------------------------------------------------------------------------

def test_bm25_retriever_basic_and_metadata(sample_chunks):
    """Verify BM25 indexes chunks, scores relevant terms, and returns standardized results."""
    retriever = BM25Retriever(sample_chunks)
    assert len(retriever.chunks) == 3

    results = retriever.retrieve("return 30 calendar days", top_k=2)
    assert len(results) == 2
    top = results[0]

    assert top.chunk_id == "CHUNK-RET-01"
    assert top.document_id == "POL-RET-001"
    assert top.document_version == "1.2"
    assert top.source_file == "returns_policy.pdf"
    assert top.page_number == 1
    assert top.section_title == "Return Window"
    assert top.retrieval_method == "bm25"
    assert top.rank == 1
    assert top.score > 0.0
    assert "30 calendar days" in top.chunk_text
    assert top.metadata["document_id"] == "POL-RET-001"


def test_bm25_empty_query_and_empty_corpus(sample_chunks):
    """Verify safe handling of empty queries and empty corpora."""
    retriever = BM25Retriever(sample_chunks)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []

    empty_retriever = BM25Retriever([])
    assert empty_retriever.retrieve("shipping") == []

    unindexed_retriever = BM25Retriever()
    assert unindexed_retriever.retrieve("shipping") == []


def test_bm25_tokenizer():
    """Verify alphanumeric and hyphen preservation in BM25 tokenization."""
    tokens = default_bm25_tokenizer("POL-RET-001: 30-day window & 5–10 days!")
    assert "pol-ret-001" in tokens
    assert "30-day" in tokens or ("30" in tokens and "day" in tokens)
    assert "5" in tokens or "5-10" in tokens


# ---------------------------------------------------------------------------
# 2. Dense Retrieval Tests
# ---------------------------------------------------------------------------

def test_dense_retriever_unified_structure(sample_chunks):
    """Verify dense retriever formats output into standardized RetrievalResult objects."""
    # Use ephemeral in-memory vector store for unit test isolation
    ephemeral_store = PolicyVectorStore(persist_directory=None, collection_name="test_dense_unit")
    ephemeral_store.add_chunks(sample_chunks)

    retriever = DenseRetriever(vector_store=ephemeral_store)
    results = retriever.retrieve("How long does shipping take?", top_k=2)

    assert len(results) == 2
    top = results[0]
    assert top.retrieval_method == "dense"
    assert top.document_id == "POL-SHIP-001"
    assert 0.0 <= top.score <= 1.0
    assert "cosine_distance" in top.metadata
    assert top.rank == 1

    # Empty query handling
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


# ---------------------------------------------------------------------------
# 3. CrossEncoder Reranker Tests
# ---------------------------------------------------------------------------

def test_cross_encoder_reranker_with_mock():
    """Verify CrossEncoder properly reranks candidate items and preserves metadata using a mock model."""
    reranker = CrossEncoderReranker()

    # Mock the underlying model to avoid downloading or running heavy inference in unit test
    mock_model = MagicMock()
    # Candidate 2 receives higher score than Candidate 1
    mock_model.predict.return_value = [0.15, 0.85]
    reranker._model = mock_model

    candidates = [
        RetrievalResult(
            chunk_id="C1",
            document_id="DOC1",
            document_version="1.0",
            source_file="f1.pdf",
            page_number=1,
            section_title="Sec1",
            chunk_text="General text",
            score=0.90,
            retrieval_method="dense",
            rank=1,
            metadata={"source": "test"},
        ),
        RetrievalResult(
            chunk_id="C2",
            document_id="DOC2",
            document_version="1.0",
            source_file="f2.pdf",
            page_number=1,
            section_title="Sec2",
            chunk_text="Highly relevant text",
            score=0.75,
            retrieval_method="dense",
            rank=2,
            metadata={"source": "test"},
        ),
    ]

    reranked = reranker.rerank("query string", candidates, top_k=2)
    assert len(reranked) == 2

    # Candidate C2 should now be ranked #1
    assert reranked[0].chunk_id == "C2"
    assert reranked[0].rank == 1
    assert reranked[0].score == 0.85
    assert reranked[0].retrieval_method == "dense+cross_encoder"
    assert reranked[0].metadata["dense_score"] == 0.75
    assert reranked[0].metadata["cross_encoder_score"] == 0.85

    # Candidate C1 should now be ranked #2
    assert reranked[1].chunk_id == "C1"
    assert reranked[1].rank == 2
    assert reranked[1].score == 0.15

    # Safe handling of empty candidates
    assert reranker.rerank("query", []) == []


# ---------------------------------------------------------------------------
# 4. Evaluation Metric Tests
# ---------------------------------------------------------------------------

def test_precision_at_k():
    """Verify Precision@K calculation logic."""
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}

    # At K=3: retrieved A (hit), B (miss), C (hit) -> 2/3 = 0.6667
    assert compute_precision_at_k(retrieved, relevant, k=3) == 0.6667

    # At K=5: retrieved A (hit), B (miss), C (hit), D (miss), E (miss) -> 2/5 = 0.4000
    assert compute_precision_at_k(retrieved, relevant, k=5) == 0.4000

    # At K=0
    assert compute_precision_at_k(retrieved, relevant, k=0) == 0.0

    # No hits
    assert compute_precision_at_k(retrieved, {"X", "Y"}, k=3) == 0.0


def test_recall_at_k():
    """Verify Recall@K calculation logic."""
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}  # Total 3 relevant items

    # At K=3: retrieved A and C (2 out of 3) -> 2/3 = 0.6667
    assert compute_recall_at_k(retrieved, relevant, k=3) == 0.6667

    # At K=5: still 2 out of 3 -> 0.6667
    assert compute_recall_at_k(retrieved, relevant, k=5) == 0.6667

    # Retrieved all relevant items
    assert compute_recall_at_k(["A", "C", "F"], relevant, k=3) == 1.0

    # Query with no relevant items (e.g. negative test case)
    assert compute_recall_at_k(retrieved, set(), k=3) == 1.0


def test_reciprocal_rank():
    """Verify Reciprocal Rank (RR) calculations."""
    relevant = {"Target"}

    # First position -> RR = 1/1 = 1.0
    assert compute_reciprocal_rank(["Target", "B", "C"], relevant) == 1.0

    # Second position -> RR = 1/2 = 0.5
    assert compute_reciprocal_rank(["A", "Target", "C"], relevant) == 0.5

    # Third position -> RR = 1/3 = 0.3333
    assert compute_reciprocal_rank(["A", "B", "Target"], relevant) == 0.3333

    # Beyond K cutoff -> RR = 0.0
    assert compute_reciprocal_rank(["A", "B", "Target"], relevant, k=2) == 0.0

    # Not found at all
    assert compute_reciprocal_rank(["A", "B", "C"], relevant) == 0.0

    # Empty relevant set
    assert compute_reciprocal_rank(["A", "B", "C"], set()) == 0.0


# ---------------------------------------------------------------------------
# 5. End-to-End Evaluation Runner Integration Test
# ---------------------------------------------------------------------------

def test_evaluator_runs_on_ground_truth():
    """Verify RetrievalEvaluator loads ground truth and calculates valid metrics."""
    gt_path = Path("./data/evaluation/ground_truth_relevance.json")
    assert gt_path.exists(), "Ground truth file must exist"

    evaluator = RetrievalEvaluator(ground_truth_path=gt_path)
    assert len(evaluator.test_cases) == 10

    # Test with a mock retriever to verify end-to-end evaluation pipeline without heavy compute
    def mock_retrieval(query: str, k: int) -> list:
        return [
            RetrievalResult(
                chunk_id="POL-RET-001_v1_2_p1_idx002_e006d94317",
                document_id="POL-RET-001",
                document_version="1.2",
                source_file="returns_policy.pdf",
                page_number=1,
                section_title="Standard return window",
                chunk_text="Sample text",
                score=0.9,
                retrieval_method="mock",
                rank=1,
            )
        ]

    summary = evaluator.evaluate_retrieval_function("Mock", mock_retrieval, top_k=3)
    assert summary.query_count == 10
    assert 0.0 <= summary.mean_precision_at_3 <= 1.0
    assert 0.0 <= summary.mean_recall_at_3 <= 1.0
    assert 0.0 <= summary.mean_mrr_at_3 <= 1.0
    assert summary.mean_latency_ms >= 0.0
