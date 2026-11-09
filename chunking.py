"""
Structure-aware chunking module for policy documents.

Implements:
1. Section-aware paragraph grouping.
2. Embedding-model-based token counting (no 1 word = 1 token assumptions).
3. Context preservation (heading prepended to chunk text).
4. Combination of related short paragraphs under the same heading.
5. Sentence-boundary fallback splitting for oversized paragraphs with configurable overlap.
6. Deterministic, stable chunk ID generation based on document identity, version, page, and content hash.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from ingestion import RawDocument, SectionBlock

logger = logging.getLogger(__name__)


@dataclass
class PolicyChunk:
    """Represents a discrete, searchable chunk of policy documentation."""
    chunk_id: str
    document_id: str
    source_file: str
    page_number: int
    document_version: str
    section_title: str
    chunk_index: int
    chunk_text: str
    token_count: int

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk fields into a ChromaDB-compatible metadata dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "document_version": self.document_version,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
        }


class StructureAwareChunker:
    """
    Splits and groups raw document sections into structure-aware chunks.
    Preserves heading context and paragraph integrity.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_chunk_tokens: int = 500,
        overlap_tokens: int = 75,
    ) -> None:
        """
        Initialize chunker with configurable limits and tokenizer.

        Args:
            tokenizer: Tokenizer instance (e.g. HuggingFace AutoTokenizer or model.tokenizer)
                       with an `encode` or `tokenize` method.
            max_chunk_tokens: Target maximum tokens per chunk (default: 500).
            overlap_tokens: Overlap tokens when splitting oversized content (default: 75).
        """
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model tokenizer, falling back to whitespace heuristics if uninitialized."""
        if not text:
            return 0

        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "encode"):
                    return len(self.tokenizer.encode(text, add_special_tokens=False))
                if hasattr(self.tokenizer, "tokenize"):
                    return len(self.tokenizer.tokenize(text))
                if callable(self.tokenizer):
                    return len(self.tokenizer(text))
            except Exception as e:
                logger.debug(f"Tokenizer invocation error: {e}; using fallback.")

        # Conservative fallback: average 1.3 tokens per word
        words = len(text.split())
        return max(1, int(words * 1.3))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences cleanly using punctuation boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _generate_chunk_id(
        self,
        document_id: str,
        document_version: str,
        page_number: int,
        chunk_index: int,
        chunk_text: str,
    ) -> str:
        """
        Generate a deterministic, stable chunk ID.
        Incorporates document ID, version, page, sequence index, and a SHA-256 content digest.
        """
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:10]
        # Clean doc id and version for URL/ID safe string
        clean_doc = re.sub(r"[^A-Za-z0-9_-]", "_", document_id)
        clean_ver = re.sub(r"[^A-Za-z0-9_-]", "_", document_version)
        return f"{clean_doc}_v{clean_ver}_p{page_number}_idx{chunk_index:03d}_{content_hash}"

    def _split_oversized_paragraph(
        self,
        heading: str,
        paragraph_text: str,
        page_number: int,
        start_chunk_idx: int,
        doc_id: str,
        source_file: str,
        doc_version: str,
    ) -> List[PolicyChunk]:
        """Split a long paragraph that exceeds max_chunk_tokens at sentence boundaries with overlap."""
        sentences = self._split_into_sentences(paragraph_text)
        if not sentences:
            sentences = [paragraph_text]

        prefix = f"[{heading}]\n"
        prefix_tokens = self.count_tokens(prefix)
        effective_max = max(5, self.max_chunk_tokens - prefix_tokens)

        chunks: List[PolicyChunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        chunk_counter = start_chunk_idx

        idx = 0
        while idx < len(sentences):
            sentence = sentences[idx]
            s_tokens = self.count_tokens(sentence)

            # If a single sentence alone is bigger than effective_max, keep it as its own chunk
            if s_tokens > effective_max:
                if current_sentences:
                    chunk_body = f"{prefix}{' '.join(current_sentences)}"
                    cid = self._generate_chunk_id(doc_id, doc_version, page_number, chunk_counter, chunk_body)
                    chunks.append(PolicyChunk(
                        chunk_id=cid,
                        document_id=doc_id,
                        source_file=source_file,
                        page_number=page_number,
                        document_version=doc_version,
                        section_title=heading,
                        chunk_index=chunk_counter,
                        chunk_text=chunk_body,
                        token_count=self.count_tokens(chunk_body),
                    ))
                    chunk_counter += 1
                    current_sentences = []
                    current_tokens = 0

                solo_body = f"{prefix}{sentence}"
                cid = self._generate_chunk_id(doc_id, doc_version, page_number, chunk_counter, solo_body)
                chunks.append(PolicyChunk(
                    chunk_id=cid,
                    document_id=doc_id,
                    source_file=source_file,
                    page_number=page_number,
                    document_version=doc_version,
                    section_title=heading,
                    chunk_index=chunk_counter,
                    chunk_text=solo_body,
                    token_count=self.count_tokens(solo_body),
                ))
                chunk_counter += 1
                idx += 1
                continue

            if current_tokens + s_tokens <= effective_max:
                current_sentences.append(sentence)
                current_tokens += s_tokens
                idx += 1
            else:
                # Flush current sentences
                chunk_body = f"{prefix}{' '.join(current_sentences)}"
                cid = self._generate_chunk_id(doc_id, doc_version, page_number, chunk_counter, chunk_body)
                chunks.append(PolicyChunk(
                    chunk_id=cid,
                    document_id=doc_id,
                    source_file=source_file,
                    page_number=page_number,
                    document_version=doc_version,
                    section_title=heading,
                    chunk_index=chunk_counter,
                    chunk_text=chunk_body,
                    token_count=self.count_tokens(chunk_body),
                ))
                chunk_counter += 1

                # Calculate overlap: backtrack sentences until overlap_tokens is met
                overlap_sentences: List[str] = []
                overlap_accum = 0
                for prev_sent in reversed(current_sentences):
                    prev_tok = self.count_tokens(prev_sent)
                    if overlap_accum + prev_tok <= self.overlap_tokens:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_accum += prev_tok
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = overlap_accum

        if current_sentences:
            chunk_body = f"{prefix}{' '.join(current_sentences)}"
            cid = self._generate_chunk_id(doc_id, doc_version, page_number, chunk_counter, chunk_body)
            chunks.append(PolicyChunk(
                chunk_id=cid,
                document_id=doc_id,
                source_file=source_file,
                page_number=page_number,
                document_version=doc_version,
                section_title=heading,
                chunk_index=chunk_counter,
                chunk_text=chunk_body,
                token_count=self.count_tokens(chunk_body),
            ))

        return chunks

    def chunk_document(self, raw_doc: RawDocument) -> List[PolicyChunk]:
        """
        Produce structure-aware chunks from a RawDocument.
        Groups adjacent paragraphs under the same heading and preserves heading context.
        """
        all_chunks: List[PolicyChunk] = []
        global_chunk_index = 0

        # Group sections by (page_number, heading)
        # To avoid splitting paragraphs unnecessarily, we process section by section
        current_heading: Optional[str] = None
        current_paragraphs: List[str] = []
        current_page: int = 1

        def flush_group() -> List[PolicyChunk]:
            nonlocal global_chunk_index, current_paragraphs, current_heading, current_page
            if not current_paragraphs or not current_heading:
                return []

            full_text = "\n\n".join(current_paragraphs)
            heading_prefix = f"[{current_heading}]\n"
            combined_text = f"{heading_prefix}{full_text}"
            token_count = self.count_tokens(combined_text)

            created_chunks = []
            if token_count <= self.max_chunk_tokens:
                cid = self._generate_chunk_id(
                    raw_doc.document_id,
                    raw_doc.version,
                    current_page,
                    global_chunk_index,
                    combined_text,
                )
                created_chunks.append(PolicyChunk(
                    chunk_id=cid,
                    document_id=raw_doc.document_id,
                    source_file=raw_doc.source_file,
                    page_number=current_page,
                    document_version=raw_doc.version,
                    section_title=current_heading,
                    chunk_index=global_chunk_index,
                    chunk_text=combined_text,
                    token_count=token_count,
                ))
                global_chunk_index += 1
            else:
                # Need sentence boundary splitting
                split_chunks = self._split_oversized_paragraph(
                    heading=current_heading,
                    paragraph_text=full_text,
                    page_number=current_page,
                    start_chunk_idx=global_chunk_index,
                    doc_id=raw_doc.document_id,
                    source_file=raw_doc.source_file,
                    doc_version=raw_doc.version,
                )
                created_chunks.extend(split_chunks)
                global_chunk_index += len(split_chunks)

            current_paragraphs = []
            return created_chunks

        for sec in raw_doc.sections:
            sec_heading = sec.heading.strip()
            sec_text = sec.text.strip()
            sec_page = sec.page_number

            if not sec_text:
                continue

            # If heading or page changes, flush previous accumulated group
            if current_heading is not None and (sec_heading != current_heading or sec_page != current_page):
                all_chunks.extend(flush_group())

            current_heading = sec_heading
            current_page = sec_page

            # Check if adding this paragraph would exceed max_chunk_tokens
            test_paragraphs = current_paragraphs + [sec_text]
            test_text = f"[{current_heading}]\n" + "\n\n".join(test_paragraphs)
            if self.count_tokens(test_text) > self.max_chunk_tokens and current_paragraphs:
                # Flush existing group and start new group with sec_text
                all_chunks.extend(flush_group())
                current_heading = sec_heading
                current_page = sec_page

            current_paragraphs.append(sec_text)

        # Flush any remaining paragraphs
        if current_paragraphs:
            all_chunks.extend(flush_group())

        return all_chunks
