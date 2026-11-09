"""
Operational business logic service for Northstar Retail Customer Support.

Provides core business functions for:
1. Order details lookup with customer authorization.
2. Return eligibility validation (return window, product type, final-sale, condition).
3. Refund status tracking across returns and refunds records.
4. Injectable date provider for deterministic testing across historical synthetic records.

Keeps business logic decoupled from MCP and LLM protocols.
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ingestion import (
    Customer,
    OperationalDataset,
    OperationalDataIngestor,
    Order,
    RefundRecord,
    ReturnRecord,
)

logger = logging.getLogger(__name__)

# Default reference date for synthetic 2026 dataset if not specified
DEFAULT_SYNTHETIC_DATE = datetime.date(2026, 10, 25)


class OperationalService:
    """Manages order inquiries, return eligibility, and refund tracking with authorization."""

    def __init__(
        self,
        dataset: Optional[OperationalDataset] = None,
        operational_dir: Optional[str | Path] = "./data/operational_data",
        date_provider: Optional[Callable[[], datetime.date]] = None,
    ) -> None:
        """
        Initialize the operational service.

        Args:
            dataset: Pre-loaded OperationalDataset (optional).
            operational_dir: Path to operational data files if dataset is None.
            date_provider: Callable returning current date (for deterministic testing).
        """
        if dataset is not None:
            self.dataset = dataset
        else:
            op_dir = Path(operational_dir or "./data/operational_data")
            self.dataset = OperationalDataIngestor.load_all(op_dir)

        self._date_provider = date_provider or self._default_date_provider

    def _default_date_provider(self) -> datetime.date:
        """Read REFERENCE_DATE from environment if set, else fall back to DEFAULT_SYNTHETIC_DATE."""
        env_ref = os.environ.get("REFERENCE_DATE")
        if env_ref:
            try:
                return datetime.date.fromisoformat(env_ref.strip())
            except ValueError:
                pass
        return DEFAULT_SYNTHETIC_DATE

    def get_current_date(self) -> datetime.date:
        """Get the current reference date."""
        return self._date_provider()

    def set_date_provider(self, provider: Callable[[], datetime.date]) -> None:
        """Set an injectable date provider for testing."""
        self._date_provider = provider

    # -----------------------------------------------------------------------
    # Tool 1: Order Details Lookup
    # -----------------------------------------------------------------------

    def get_order_details(
        self,
        order_id: str,
        authenticated_customer_id: str,
    ) -> Dict[str, Any]:
        """
        Look up order details with authenticated caller ownership validation.

        Args:
            order_id: Target order ID (e.g. 'NS-1001').
            authenticated_customer_id: Trusted caller customer ID (e.g. 'C-001').

        Returns:
            Structured order details dictionary or safe error message.
        """
        if not order_id or not isinstance(order_id, str):
            return {"success": False, "error": "Invalid order_id: must be a non-empty string."}

        clean_oid = order_id.strip()
        order = self.dataset.get_order(clean_oid)
        if not order:
            return {"success": False, "error": f"Order '{clean_oid}' not found."}

        # Ownership validation
        if order.customer_id != authenticated_customer_id:
            logger.warning(
                f"Unauthorized access attempt: caller '{authenticated_customer_id}' "
                f"attempted to access order '{clean_oid}' owned by '{order.customer_id}'."
            )
            return {
                "success": False,
                "error": f"Unauthorized: Order '{clean_oid}' does not belong to authenticated customer.",
            }

        return {
            "success": True,
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "order_date": order.order_date,
            "delivery_date": order.delivery_date,
            "status": order.status,
            "item_name": order.item_name,
            "item_type": order.item_type,
            "final_sale": order.final_sale,
            "condition_reported": order.condition_reported,
            "amount_inr": order.amount_inr,
            "payment_status": order.payment_status,
            "tracking_number": order.tracking_number,
            "return_status": order.return_status,
        }

    # -----------------------------------------------------------------------
    # Tool 2: Check Return Eligibility
    # -----------------------------------------------------------------------

    def check_return_eligibility(
        self,
        order_id: str,
        authenticated_customer_id: str,
    ) -> Dict[str, Any]:
        """
        Validate return eligibility against policy POL-RET-001 rules:
        1. Existence and ownership.
        2. Delivery status (must be 'delivered').
        3. Delivery date presence.
        4. Physical vs digital item type (digital products are strictly non-returnable).
        5. Final-sale status (final-sale items are non-returnable).
        6. Item condition (must be unused/resalable).
        7. 30 calendar days window from delivery date.

        Returns:
            Structured eligibility assessment.
        """
        if not order_id or not isinstance(order_id, str):
            return {
                "eligible": False,
                "reason": "Invalid order_id supplied.",
                "manual_review_required": False,
                "conditions_passed": [],
                "conditions_failed": ["valid_order_id"],
            }

        clean_oid = order_id.strip()
        order = self.dataset.get_order(clean_oid)
        if not order:
            return {
                "eligible": False,
                "reason": f"Order '{clean_oid}' was not found in operational records.",
                "manual_review_required": False,
                "conditions_passed": [],
                "conditions_failed": ["order_exists"],
            }

        # Ownership check
        if order.customer_id != authenticated_customer_id:
            return {
                "eligible": False,
                "reason": f"Unauthorized: Order '{clean_oid}' does not belong to authenticated customer.",
                "manual_review_required": False,
                "conditions_passed": [],
                "conditions_failed": ["customer_ownership"],
            }

        passed_conditions: List[str] = ["customer_ownership"]
        failed_conditions: List[str] = []

        # 1. Delivery status check
        if order.status.lower() != "delivered":
            failed_conditions.append("delivered_status")
            return {
                "eligible": False,
                "order_id": clean_oid,
                "reason": f"Order status is '{order.status}'. Items can only be returned after recorded delivery.",
                "delivery_date": order.delivery_date,
                "days_elapsed": None,
                "conditions_passed": passed_conditions,
                "conditions_failed": failed_conditions,
                "manual_review_required": False,
            }
        passed_conditions.append("delivered_status")

        # 2. Delivery date presence
        if not order.delivery_date:
            failed_conditions.append("delivery_date_recorded")
            return {
                "eligible": False,
                "order_id": clean_oid,
                "reason": "Delivery date is missing from order records; customer service review required.",
                "delivery_date": None,
                "days_elapsed": None,
                "conditions_passed": passed_conditions,
                "conditions_failed": failed_conditions,
                "manual_review_required": True,
            }
        passed_conditions.append("delivery_date_recorded")

        # 3. Physical vs Digital check
        if order.item_type.lower() == "digital":
            failed_conditions.append("physical_item_type")
            return {
                "eligible": False,
                "order_id": clean_oid,
                "reason": f"Item '{order.item_name}' is a digital product. Downloadable digital products are strictly non-returnable under POL-RET-001.",
                "delivery_date": order.delivery_date,
                "days_elapsed": None,
                "conditions_passed": passed_conditions,
                "conditions_failed": failed_conditions,
                "manual_review_required": False,
            }
        passed_conditions.append("physical_item_type")

        # 4. Final-sale check
        if order.final_sale:
            failed_conditions.append("not_final_sale")
            return {
                "eligible": False,
                "order_id": clean_oid,
                "reason": f"Item '{order.item_name}' was purchased as a final-sale item and cannot be returned or refunded.",
                "delivery_date": order.delivery_date,
                "days_elapsed": None,
                "conditions_passed": passed_conditions,
                "conditions_failed": failed_conditions,
                "manual_review_required": False,
            }
        passed_conditions.append("not_final_sale")

        # 5. Condition check
        if order.condition_reported and order.condition_reported.lower() not in ("unused", "not_applicable"):
            failed_conditions.append("resalable_condition")
            return {
                "eligible": False,
                "order_id": clean_oid,
                "reason": f"Item condition is reported as '{order.condition_reported}'. Only unused items in original packaging qualify for standard returns.",
                "delivery_date": order.delivery_date,
                "days_elapsed": None,
                "conditions_passed": passed_conditions,
                "conditions_failed": failed_conditions,
                "manual_review_required": True,
            }
        passed_conditions.append("resalable_condition")

        # 6. Return window calculation (30 calendar days from delivery date)
        try:
            deliv_date = datetime.date.fromisoformat(order.delivery_date)
        except ValueError:
            failed_conditions.append("valid_delivery_date_format")
            return {
                "eligible": False,
                "order_id": clean_oid,
                "reason": f"Delivery date '{order.delivery_date}' has an invalid format; manual review required.",
                "delivery_date": order.delivery_date,
                "days_elapsed": None,
                "conditions_passed": passed_conditions,
                "conditions_failed": failed_conditions,
                "manual_review_required": True,
            }

        ref_date = self.get_current_date()
        days_elapsed = (ref_date - deliv_date).days

        if days_elapsed < 0:
            failed_conditions.append("delivery_date_not_in_future")
            return {
                "eligible": False,
                "order_id": clean_oid,
                "reason": f"Recorded delivery date '{order.delivery_date}' is in the future relative to reference date '{ref_date}'.",
                "delivery_date": order.delivery_date,
                "days_elapsed": days_elapsed,
                "conditions_passed": passed_conditions,
                "conditions_failed": failed_conditions,
                "manual_review_required": True,
            }

        if days_elapsed > 30:
            failed_conditions.append("within_30_days_window")
            return {
                "eligible": False,
                "order_id": clean_oid,
                "reason": f"Return window expired: {days_elapsed} calendar days have elapsed since delivery ({order.delivery_date}); policy limit is 30 calendar days.",
                "delivery_date": order.delivery_date,
                "days_elapsed": days_elapsed,
                "conditions_passed": passed_conditions,
                "conditions_failed": failed_conditions,
                "manual_review_required": False,
            }
        passed_conditions.append("within_30_days_window")

        return {
            "eligible": True,
            "order_id": clean_oid,
            "reason": f"Eligible for return: Item was delivered on {order.delivery_date} ({days_elapsed} days ago, within the 30-day window) and meets condition requirements.",
            "delivery_date": order.delivery_date,
            "days_elapsed": days_elapsed,
            "return_window_days": 30,
            "conditions_passed": passed_conditions,
            "conditions_failed": failed_conditions,
            "manual_review_required": False,
        }

    # -----------------------------------------------------------------------
    # Tool 3: Get Refund Status
    # -----------------------------------------------------------------------

    def get_refund_status(
        self,
        order_id: str,
        authenticated_customer_id: str,
    ) -> Dict[str, Any]:
        """
        Look up refund and return records for the specified order with ownership validation.

        Returns:
            Structured refund tracking status.
        """
        if not order_id or not isinstance(order_id, str):
            return {"success": False, "error": "Invalid order_id supplied."}

        clean_oid = order_id.strip()
        order = self.dataset.get_order(clean_oid)
        if not order:
            return {"success": False, "error": f"Order '{clean_oid}' not found."}

        if order.customer_id != authenticated_customer_id:
            return {
                "success": False,
                "error": f"Unauthorized: Order '{clean_oid}' does not belong to authenticated customer.",
            }

        # Check existing refunds
        refunds = self.dataset.get_refunds_by_order(clean_oid)
        # Check existing returns
        returns = self.dataset.get_returns_by_order(clean_oid)

        if refunds:
            rf = refunds[0]
            return {
                "success": True,
                "order_id": clean_oid,
                "refund_status": rf.status,  # e.g. "approved"
                "refund_id": rf.refund_id,
                "amount_inr": rf.amount_inr,
                "approved_date": rf.approved_date,
                "return_recorded": True,
                "details": (
                    f"Refund of ₹{rf.amount_inr} was approved on {rf.approved_date} under reference {rf.refund_id}. "
                    f"In accordance with policy POL-REF-001, please allow 5–10 business days for the payment provider "
                    f"or issuing bank to credit your original payment method."
                ),
            }

        if returns:
            ret = returns[0]
            return {
                "success": True,
                "order_id": clean_oid,
                "refund_status": "pending",
                "return_id": ret.return_id,
                "return_status": ret.status,  # e.g. "received", "requested"
                "inspection_status": ret.inspection,  # e.g. "pending", "not_started"
                "amount_inr": order.amount_inr,
                "approved_date": None,
                "details": (
                    f"Return request {ret.return_id} is currently '{ret.status}' with inspection '{ret.inspection}'. "
                    f"Under policy POL-REF-001, a refund may be initiated only after the returned item has been inspected and approved."
                ),
            }

        return {
            "success": True,
            "order_id": clean_oid,
            "refund_status": "unavailable",
            "return_recorded": False,
            "details": f"No return or refund request has been initiated for order '{clean_oid}'.",
        }
