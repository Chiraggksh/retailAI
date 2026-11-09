"""
Verification script for Phase 1 of the Northstar Retail Support System.

Executes:
1. Ingestion and validation of operational records.
2. PyMuPDF parsing of the 3 policy PDFs.
3. Structure-aware chunking with SentenceTransformer tokenizer token-counting.
4. ChromaDB insertion and idempotency verification.
5. Sample semantic query retrieval and metadata inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint

from chunking import StructureAwareChunker
from embeddings import EmbeddingManager
from ingestion import OperationalDataIngestor, PolicyIngestor
from vector_store import PolicyVectorStore


def main() -> None:
    print("=" * 70)
    print("PHASE 1 VERIFICATION: Ingestion, Chunking, and Vector Store")
    print("=" * 70)

    data_dir = Path("./data")
    docs_dir = data_dir / "documents"
    ops_dir = data_dir / "operational_data"
    manifest_path = data_dir / "manifest.json"
    chroma_dir = data_dir / "chroma_db"

    # 1. Operational Data Loading
    print("\n[Step 1] Loading Operational Data...")
    dataset = OperationalDataIngestor.load_all(ops_dir)
    print(f"  ✓ Customers loaded : {len(dataset.customers)}")
    print(f"  ✓ Orders loaded    : {len(dataset.orders)}")
    print(f"  ✓ Returns loaded   : {len(dataset.returns)}")
    print(f"  ✓ Refunds loaded   : {len(dataset.refunds)}")

    # 2. Embedding Model & Tokenizer Initialization
    print("\n[Step 2] Initializing Embedding Model & Tokenizer...")
    embedding_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    tokenizer = embedding_mgr.tokenizer
    print(f"  ✓ Model loaded     : {embedding_mgr.model_name}")
    print(f"  ✓ Dimensions       : {embedding_mgr.dimension}")
    print(f"  ✓ Model Tokenizer  : {tokenizer.__class__.__name__}")

    # 3. Policy PDF Extraction
    print("\n[Step 3] Ingesting Policy PDFs via PyMuPDF...")
    pdf_ingestor = PolicyIngestor(manifest_path=manifest_path)
    raw_documents = pdf_ingestor.ingest_directory(docs_dir)
    print(f"  ✓ Ingested {len(raw_documents)} policy documents:")
    for doc in raw_documents:
        print(f"    - {doc.document_id} ({doc.source_file}, v{doc.version}): {len(doc.pages)} pages, {len(doc.sections)} sections")

    # 4. Structure-Aware Chunking
    print("\n[Step 4] Performing Structure-Aware Chunking...")
    chunker = StructureAwareChunker(
        tokenizer=tokenizer,
        max_chunk_tokens=500,
        overlap_tokens=75,
    )

    all_chunks = []
    for doc in raw_documents:
        doc_chunks = chunker.chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"    - {doc.document_id}: generated {len(doc_chunks)} chunks")

    print(f"  ✓ Total policy chunks produced: {len(all_chunks)}")

    # 5. ChromaDB Upsert & Idempotency Check
    print("\n[Step 5] Ingesting into ChromaDB (Idempotency Test)...")
    store = PolicyVectorStore(
        persist_directory=chroma_dir,
        collection_name="northstar_policy_store",
        embedding_manager=embedding_mgr,
    )

    # First upsert
    added_count_1 = store.add_chunks(all_chunks)
    stats_1 = store.get_collection_stats()
    print(f"  ✓ Run 1: Upserted {added_count_1} chunks. Total in collection: {stats_1['total_chunks']}")

    # Second upsert (exact same chunks to verify idempotency)
    added_count_2 = store.add_chunks(all_chunks)
    stats_2 = store.get_collection_stats()
    print(f"  ✓ Run 2 (Repeated Ingestion): Total in collection: {stats_2['total_chunks']}")
    assert stats_1["total_chunks"] == stats_2["total_chunks"], "Idempotency failed: duplicate chunks detected!"
    print("  ✓ Idempotency verified: repeated ingestion produced 0 duplicates.")

    # 6. Sample Chunk Inspection
    print("\n[Step 6] Sample Chunk Inspection:")
    sample = all_chunks[0]
    print(f"  Chunk ID       : {sample.chunk_id}")
    print(f"  Document ID    : {sample.document_id} (v{sample.document_version})")
    print(f"  Source File    : {sample.source_file} (Page {sample.page_number})")
    print(f"  Section Title  : {sample.section_title}")
    print(f"  Token Count    : {sample.token_count}")
    print(f"  Chunk Text     :\n---")
    print(sample.chunk_text)
    print("---")
    print("  Metadata Dict  :")
    pprint(sample.to_metadata())

    # 7. Semantic Query Retrieval Test
    test_query = "How many days does a customer have to return physical items?"
    print(f"\n[Step 7] Testing Semantic Search for Query: '{test_query}'...")
    results = store.search(test_query, top_k=2)
    for idx, res in enumerate(results, start=1):
        print(f"\n  Result #{idx} (Score: {res['score']}, Distance: {res['distance']}):")
        print(f"    Doc ID  : {res['metadata'].get('document_id')} (v{res['metadata'].get('document_version')})")
        print(f"    Section : {res['metadata'].get('section_title')}")
        print(f"    Page    : {res['metadata'].get('page_number')}")
        print(f"    Text Preview: {res['text'][:150].replace(chr(10), ' ')}...")

    print("\n" + "=" * 70)
    print("PHASE 1 VERIFICATION COMPLETE: ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
