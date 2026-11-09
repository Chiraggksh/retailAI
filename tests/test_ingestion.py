"""
Unit tests for Phase 1: Ingestion, Structure-aware Chunking, and Vector Store.
"""

import os
from pathlib import Path

import pytest

from chunking import PolicyChunk, StructureAwareChunker
from embeddings import EmbeddingManager
from ingestion import (
    OperationalDataIngestor,
    PolicyIngestor,
    RawDocument,
    SectionBlock,
)
from vector_store import PolicyVectorStore


DATA_DIR = Path("./data")
DOCS_DIR = DATA_DIR / "documents"
OPERATIONAL_DIR = DATA_DIR / "operational_data"
MANIFEST_PATH = DATA_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# 1. Operational Data Ingestion Tests
# ---------------------------------------------------------------------------

def test_load_operational_data():
    """Verify loading and parsing of operational CSV and JSON records."""
    assert OPERATIONAL_DIR.exists(), "Operational data directory must exist"

    dataset = OperationalDataIngestor.load_all(OPERATIONAL_DIR)

    # Customers
    assert len(dataset.customers) >= 4
    c1 = dataset.get_customer("C-001")
    assert c1 is not None
    assert c1.access_code == "DEMO-001"

    # Orders
    assert len(dataset.orders) >= 6
    o1 = dataset.get_order("NS-1001")
    assert o1 is not None
    assert o1.customer_id == "C-001"
    assert o1.status == "delivered"
    assert o1.amount_inr == 1299
    assert isinstance(o1.final_sale, bool)

    # Customer orders relation
    c1_orders = dataset.get_orders_by_customer("C-001")
    assert len(c1_orders) >= 2

    # Returns and Refunds
    assert len(dataset.returns) >= 2
    assert len(dataset.refunds) >= 1
    rf = dataset.refunds.get("RF-7001")
    assert rf is not None
    assert rf.amount_inr == 2499


# ---------------------------------------------------------------------------
# 2. PDF Ingestion Tests
# ---------------------------------------------------------------------------

def test_pdf_ingestion():
    """Verify PyMuPDF extracts text, headings, and metadata from policy documents."""
    ingestor = PolicyIngestor(manifest_path=MANIFEST_PATH)
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    assert len(pdf_files) == 3, f"Expected 3 policy PDFs, found {len(pdf_files)}"

    docs = ingestor.ingest_directory(DOCS_DIR)
    assert len(docs) == 3

    for doc in docs:
        assert doc.document_id.startswith("POL-")
        assert doc.version in ("1.0", "1.1", "1.2")
        assert len(doc.pages) > 0
        assert len(doc.sections) > 0
        # Ensure section headings are captured
        headings = {s.heading for s in doc.sections}
        assert len(headings) >= 1


# ---------------------------------------------------------------------------
# 3. Structure-Aware Chunking Tests
# ---------------------------------------------------------------------------

def test_structure_aware_chunking_preserves_metadata():
    """Verify that chunks retain all required metadata fields and deterministic IDs."""
    raw_doc = RawDocument(
        document_id="POL-TEST-001",
        source_file="test_policy.pdf",
        version="1.0",
        topic="testing",
        sections=[
            SectionBlock(heading="Return Window", text="Customers have 30 days to return physical items.", page_number=1),
            SectionBlock(heading="Return Window", text="Items must be in original condition with packaging.", page_number=1),
            SectionBlock(heading="Exceptions", text="Final sale items and digital items cannot be returned.", page_number=2),
        ],
    )

    chunker = StructureAwareChunker(max_chunk_tokens=100, overlap_tokens=20)
    chunks = chunker.chunk_document(raw_doc)

    assert len(chunks) == 2  # First two sections share heading & fit under 100 tokens, third is separate page/heading

    first_chunk = chunks[0]
    assert first_chunk.document_id == "POL-TEST-001"
    assert first_chunk.source_file == "test_policy.pdf"
    assert first_chunk.page_number == 1
    assert first_chunk.document_version == "1.0"
    assert first_chunk.section_title == "Return Window"
    assert "[Return Window]" in first_chunk.chunk_text
    assert "30 days" in first_chunk.chunk_text
    assert "original condition" in first_chunk.chunk_text

    # Verify ID determinism
    chunks_second_run = chunker.chunk_document(raw_doc)
    assert chunks[0].chunk_id == chunks_second_run[0].chunk_id
    assert chunks[1].chunk_id == chunks_second_run[1].chunk_id


def test_chunking_oversized_paragraph_with_overlap():
    """Verify sentence-boundary splitting and overlap on paragraphs exceeding token limit."""
    long_sentences = [
        "Sentence number one provides standard general context for testing purposes.",
        "Sentence number two extends the discussion with further operational details.",
        "Sentence number three adds another constraint regarding return procedures.",
        "Sentence number four concludes the paragraph with specific instructions.",
    ]
    long_paragraph = " ".join(long_sentences)

    raw_doc = RawDocument(
        document_id="POL-LONG-001",
        source_file="long.pdf",
        version="1.0",
        topic="shipping",
        sections=[
            SectionBlock(heading="Standard Shipping", text=long_paragraph, page_number=1),
        ],
    )

    # Force low token limit to trigger sentence-boundary splitting
    chunker = StructureAwareChunker(max_chunk_tokens=30, overlap_tokens=10)
    chunks = chunker.chunk_document(raw_doc)

    assert len(chunks) > 1
    for c in chunks:
        assert c.section_title == "Standard Shipping"
        assert "[Standard Shipping]" in c.chunk_text
        assert c.token_count <= 35  # Token count adheres to bounds


# ---------------------------------------------------------------------------
# 4. Vector Store & Idempotency Tests (In-Memory Ephemeral)
# ---------------------------------------------------------------------------

def test_vector_store_idempotency_and_search():
    """Verify ChromaDB upsert avoids duplicates upon repeated ingestion and performs search."""
    # Use ephemeral vector store to avoid touching persisted db during unit tests
    store = PolicyVectorStore(persist_directory=None, collection_name="test_ephemeral_store")

    chunks = [
        PolicyChunk(
            chunk_id="POL-RET_v1.0_p1_idx000_abc123",
            document_id="POL-RET-001",
            source_file="returns_policy.pdf",
            page_number=1,
            document_version="1.0",
            section_title="Eligibility",
            chunk_index=0,
            chunk_text="[Eligibility]\nPhysical items can be returned within 30 calendar days of delivery.",
            token_count=18,
        ),
        PolicyChunk(
            chunk_id="POL-SHIP_v1.0_p1_idx000_def456",
            document_id="POL-SHIP-001",
            source_file="shipping_policy.pdf",
            page_number=1,
            document_version="1.0",
            section_title="Domestic Shipping",
            chunk_index=0,
            chunk_text="[Domestic Shipping]\nStandard domestic shipping takes 3 to 7 business days after dispatch.",
            token_count=17,
        ),
    ]

    # Ingestion run 1
    count1 = store.add_chunks(chunks)
    assert count1 == 2
    assert store.collection.count() == 2

    # Ingestion run 2 (exact same chunks) -> Idempotency check
    count2 = store.add_chunks(chunks)
    assert count2 == 2
    assert store.collection.count() == 2, "Repeated ingestion must not create duplicate chunks"

    # Retrieve by ID
    retrieved = store.get_chunk_by_id("POL-RET_v1.0_p1_idx000_abc123")
    assert retrieved is not None
    assert retrieved["metadata"]["document_id"] == "POL-RET-001"
    assert "30 calendar days" in retrieved["text"]

    # Semantic search test
    search_results = store.search("How long does domestic shipping take?", top_k=1)
    assert len(search_results) == 1
    assert search_results[0]["metadata"]["document_id"] == "POL-SHIP-001"
    assert search_results[0]["score"] > 0.0

    # Collection statistics
    stats = store.get_collection_stats()
    assert stats["total_chunks"] == 2
    assert stats["unique_documents"] == 2
    assert "POL-RET-001" in stats["documents"]
    assert "POL-SHIP-001" in stats["documents"]


def test_vector_store_version_replacement():
    """Verify that ingesting a newer document version purges older chunks of that document."""
    store = PolicyVectorStore(persist_directory=None, collection_name="test_versioning_store")

    v1_chunk = PolicyChunk(
        chunk_id="POL-RET_v1.0_p1_idx000_old111",
        document_id="POL-RET-001",
        source_file="returns_policy.pdf",
        page_number=1,
        document_version="1.0",
        section_title="Eligibility",
        chunk_index=0,
        chunk_text="[Eligibility]\nOld policy: Return window is 14 days.",
        token_count=12,
    )
    store.add_chunks([v1_chunk])
    assert store.collection.count() == 1

    # Ingest newer version 1.1 for same document_id
    v2_chunk = PolicyChunk(
        chunk_id="POL-RET_v1.1_p1_idx000_new222",
        document_id="POL-RET-001",
        source_file="returns_policy.pdf",
        page_number=1,
        document_version="1.1",
        section_title="Eligibility",
        chunk_index=0,
        chunk_text="[Eligibility]\nUpdated policy: Return window is 30 days.",
        token_count=13,
    )
    store.add_chunks([v2_chunk])

    # Old version chunk must have been removed, leaving only 1 chunk for POL-RET-001
    assert store.collection.count() == 1
    chunk_after = store.get_chunk_by_id("POL-RET_v1.0_p1_idx000_old111")
    assert chunk_after is None, "Old version chunk should be purged"
    new_chunk = store.get_chunk_by_id("POL-RET_v1.1_p1_idx000_new222")
    assert new_chunk is not None, "New version chunk should exist"
    assert new_chunk["metadata"]["document_version"] == "1.1"
