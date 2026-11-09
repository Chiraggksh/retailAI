"""
Data ingestion module for Northstar Retail Customer Support.

Responsible for:
1. Extracting text, page numbers, document metadata, and section headings from policy PDFs via PyMuPDF.
2. Ingesting, validating, and structuring operational records (orders, customers, returns, refunds)
   without embedding them into the vector store.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf as fitz

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy Document Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SectionBlock:
    """Represents an extracted text block with heading association."""
    heading: str
    text: str
    page_number: int


@dataclass
class RawDocument:
    """Represents a policy document extracted from PDF."""
    document_id: str
    source_file: str
    version: str
    topic: str
    pages: List[Dict[str, Any]] = field(default_factory=list)
    sections: List[SectionBlock] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Operational Data Models
# ---------------------------------------------------------------------------

@dataclass
class Customer:
    """Synthetic customer identity record."""
    customer_id: str
    display_name: str
    access_code: str


@dataclass
class Order:
    """Synthetic order record with type-casted fields."""
    order_id: str
    customer_id: str
    order_date: str
    delivery_date: Optional[str]
    status: str
    item_name: str
    item_type: str
    final_sale: bool
    condition_reported: str
    amount_inr: int
    payment_status: str
    tracking_number: Optional[str]
    return_status: str


@dataclass
class ReturnRecord:
    """Synthetic return record."""
    return_id: str
    order_id: str
    customer_id: str
    status: str
    inspection: str
    created_at: str


@dataclass
class RefundRecord:
    """Synthetic refund record."""
    refund_id: str
    order_id: str
    customer_id: str
    status: str
    amount_inr: int
    approved_date: str


@dataclass
class OperationalDataset:
    """Container holding all loaded and validated operational records."""
    customers: Dict[str, Customer] = field(default_factory=dict)
    orders: Dict[str, Order] = field(default_factory=dict)
    returns: Dict[str, ReturnRecord] = field(default_factory=dict)
    refunds: Dict[str, RefundRecord] = field(default_factory=dict)

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        return self.customers.get(customer_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_orders_by_customer(self, customer_id: str) -> List[Order]:
        return [o for o in self.orders.values() if o.customer_id == customer_id]

    def get_returns_by_order(self, order_id: str) -> List[ReturnRecord]:
        return [r for r in self.returns.values() if r.order_id == order_id]

    def get_refunds_by_order(self, order_id: str) -> List[RefundRecord]:
        return [rf for rf in self.refunds.values() if rf.order_id == order_id]


# ---------------------------------------------------------------------------
# Policy Ingestion Pipeline
# ---------------------------------------------------------------------------

class PolicyIngestor:
    """Extracts text, hierarchy, and metadata from policy PDF documents using PyMuPDF."""

    def __init__(self, manifest_path: Optional[str | Path] = None) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.manifest_data: Dict[str, Any] = {}
        if self.manifest_path and self.manifest_path.exists():
            self._load_manifest()

    def _load_manifest(self) -> None:
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load manifest at {self.manifest_path}: {e}")

    def _get_manifest_entry(self, pdf_path: Path) -> Dict[str, str]:
        """Match PDF path to manifest metadata if available."""
        pdf_name = pdf_path.name
        for doc in self.manifest_data.get("documents", []):
            if Path(doc.get("file", "")).name == pdf_name:
                return {
                    "document_id": doc.get("document_id", "UNKNOWN"),
                    "version": str(doc.get("version", "1.0")),
                    "topic": doc.get("topic", "policy"),
                }
        return {"document_id": pdf_path.stem.upper(), "version": "1.0", "topic": pdf_path.stem}

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        document_id: Optional[str] = None,
        version: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> RawDocument:
        """
        Ingest a single PDF file and extract pages, structured text blocks, and headings.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        meta = self._get_manifest_entry(path)
        doc_id = document_id or meta["document_id"]
        doc_version = version or meta["version"]
        doc_topic = topic or meta["topic"]

        doc = fitz.open(path)
        pages_data: List[Dict[str, Any]] = []
        sections: List[SectionBlock] = []

        current_heading = f"{doc_topic.capitalize()} Policy"

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            page_text = page.get_text("text").strip()

            # Extract detailed block hierarchy
            text_page_dict = page.get_text("dict")
            blocks = text_page_dict.get("blocks", [])

            page_blocks: List[Dict[str, Any]] = []

            for b in blocks:
                # Type 0 indicates text block
                if b.get("type") == 0:
                    lines_text = []
                    max_font_size = 0.0
                    is_bold = False

                    for line in b.get("lines", []):
                        line_parts = []
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            line_parts.append(span_text)
                            font_size = span.get("size", 0.0)
                            if font_size > max_font_size:
                                max_font_size = font_size
                            flags = span.get("flags", 0)
                            # PyMuPDF font flag: 2**4 (16) is bold
                            if flags & 2 or flags & 16 or "bold" in span.get("font", "").lower():
                                is_bold = True
                        lines_text.append("".join(line_parts))

                    block_text = "\n".join(lines_text).strip()
                    if not block_text:
                        continue

                    # Heuristic heading detection:
                    # Font size > 11.5 or bold with short line or section/policy keywords
                    is_heading_candidate = (
                        (max_font_size >= 12.0 or is_bold)
                        and len(block_text.splitlines()) <= 2
                        and len(block_text) < 120
                        and not block_text.endswith(".")
                    ) or (
                        block_text.startswith(("#", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))
                        and len(block_text.splitlines()) == 1
                    )

                    if is_heading_candidate:
                        current_heading = block_text.strip("# ").strip()
                        page_blocks.append({
                            "type": "heading",
                            "heading": current_heading,
                            "text": block_text,
                            "font_size": max_font_size,
                            "page": page_num,
                        })
                    else:
                        page_blocks.append({
                            "type": "paragraph",
                            "heading": current_heading,
                            "text": block_text,
                            "page": page_num,
                        })
                        sections.append(SectionBlock(
                            heading=current_heading,
                            text=block_text,
                            page_number=page_num,
                        ))

            pages_data.append({
                "page_number": page_num,
                "text": page_text,
                "blocks": page_blocks,
            })

        doc.close()

        return RawDocument(
            document_id=doc_id,
            source_file=path.name,
            version=doc_version,
            topic=doc_topic,
            pages=pages_data,
            sections=sections,
        )

    def ingest_directory(self, documents_dir: str | Path) -> List[RawDocument]:
        """Ingest all PDF documents in a directory."""
        dir_path = Path(documents_dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {dir_path}")

        return [self.ingest_pdf(pdf) for pdf in pdf_files]


# ---------------------------------------------------------------------------
# Operational Data Ingestion Pipeline
# ---------------------------------------------------------------------------

class OperationalDataIngestor:
    """Loads, validates, and stores operational CSV and JSON data."""

    @staticmethod
    def load_customers(json_path: str | Path) -> Dict[str, Customer]:
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        customers_dict = {}
        for item in data.get("customers", []):
            cid = str(item.get("customer_id", "")).strip()
            if not cid:
                continue
            customers_dict[cid] = Customer(
                customer_id=cid,
                display_name=str(item.get("display_name", "")).strip(),
                access_code=str(item.get("access_code", "")).strip(),
            )
        return customers_dict

    @staticmethod
    def load_orders(csv_path: str | Path) -> Dict[str, Order]:
        path = Path(csv_path)
        orders_dict = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                order_id = row.get("order_id", "").strip()
                if not order_id:
                    continue

                # Parse booleans and nullable dates
                final_sale_raw = str(row.get("final_sale", "")).strip().lower()
                final_sale = final_sale_raw in ("true", "1", "yes")

                delivery_date_raw = row.get("delivery_date", "").strip()
                delivery_date = delivery_date_raw if delivery_date_raw else None

                tracking_raw = row.get("tracking_number", "").strip()
                tracking_number = tracking_raw if tracking_raw else None

                amount_raw = row.get("amount_inr", "0").strip()
                try:
                    amount_inr = int(float(amount_raw))
                except ValueError:
                    amount_inr = 0

                orders_dict[order_id] = Order(
                    order_id=order_id,
                    customer_id=row.get("customer_id", "").strip(),
                    order_date=row.get("order_date", "").strip(),
                    delivery_date=delivery_date,
                    status=row.get("status", "").strip(),
                    item_name=row.get("item_name", "").strip(),
                    item_type=row.get("item_type", "").strip(),
                    final_sale=final_sale,
                    condition_reported=row.get("condition_reported", "").strip(),
                    amount_inr=amount_inr,
                    payment_status=row.get("payment_status", "").strip(),
                    tracking_number=tracking_number,
                    return_status=row.get("return_status", "none").strip(),
                )
        return orders_dict

    @staticmethod
    def load_returns(json_path: str | Path) -> Dict[str, ReturnRecord]:
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        returns_dict = {}
        for item in data.get("returns", []):
            rid = str(item.get("return_id", "")).strip()
            if not rid:
                continue
            returns_dict[rid] = ReturnRecord(
                return_id=rid,
                order_id=str(item.get("order_id", "")).strip(),
                customer_id=str(item.get("customer_id", "")).strip(),
                status=str(item.get("status", "")).strip(),
                inspection=str(item.get("inspection", "")).strip(),
                created_at=str(item.get("created_at", "")).strip(),
            )
        return returns_dict

    @staticmethod
    def load_refunds(json_path: str | Path) -> Dict[str, RefundRecord]:
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        refunds_dict = {}
        for item in data.get("refunds", []):
            rfid = str(item.get("refund_id", "")).strip()
            if not rfid:
                continue
            amount_raw = item.get("amount_inr", 0)
            try:
                amount_inr = int(amount_raw)
            except (ValueError, TypeError):
                amount_inr = 0

            refunds_dict[rfid] = RefundRecord(
                refund_id=rfid,
                order_id=str(item.get("order_id", "")).strip(),
                customer_id=str(item.get("customer_id", "")).strip(),
                status=str(item.get("status", "")).strip(),
                amount_inr=amount_inr,
                approved_date=str(item.get("approved_date", "")).strip(),
            )
        return refunds_dict

    @classmethod
    def load_all(cls, operational_dir: str | Path) -> OperationalDataset:
        """Load and cross-validate all operational records."""
        base_dir = Path(operational_dir)
        customers = cls.load_customers(base_dir / "customers.json")
        orders = cls.load_orders(base_dir / "orders.csv")
        returns = cls.load_returns(base_dir / "returns.json")
        refunds = cls.load_refunds(base_dir / "refunds.json")

        # Validation: Verify referential integrity where possible
        for o_id, order in orders.items():
            if order.customer_id not in customers:
                logger.warning(f"Order {o_id} references unknown customer {order.customer_id}")

        for r_id, ret in returns.items():
            if ret.order_id not in orders:
                logger.warning(f"Return {r_id} references unknown order {ret.order_id}")

        for rf_id, ref in refunds.items():
            if ref.order_id not in orders:
                logger.warning(f"Refund {rf_id} references unknown order {ref.order_id}")

        return OperationalDataset(
            customers=customers,
            orders=orders,
            returns=returns,
            refunds=refunds,
        )
