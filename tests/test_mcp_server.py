"""
Unit tests for MCP Server and OperationalService business logic.
"""

import datetime
from pathlib import Path

import pytest

from operational_service import OperationalService


@pytest.fixture
def service():
    """Create an OperationalService with synthetic operational data."""
    # Fixed reference date: 2026-10-25
    srv = OperationalService()
    srv.set_date_provider(lambda: datetime.date(2026, 10, 25))
    return srv


# ---------------------------------------------------------------------------
# 1. Order Details Tests
# ---------------------------------------------------------------------------

def test_get_order_details_authorized(service):
    """Verify authorized customer receives valid structured order details."""
    res = service.get_order_details("NS-1001", authenticated_customer_id="C-001")
    assert res["success"] is True
    assert res["order_id"] == "NS-1001"
    assert res["customer_id"] == "C-001"
    assert res["status"] == "delivered"
    assert res["amount_inr"] == 1299
    assert res["item_name"] == "Wireless Mouse"


def test_get_order_details_unauthorized(service):
    """Verify unauthorized caller cannot access another customer's order."""
    res = service.get_order_details("NS-1002", authenticated_customer_id="C-001")
    assert res["success"] is False
    assert "Unauthorized" in res["error"]
    assert "item_name" not in res  # Never leak order contents


def test_get_order_details_missing(service):
    """Verify non-existent order returns a safe not-found response."""
    res = service.get_order_details("NS-9999", authenticated_customer_id="C-001")
    assert res["success"] is False
    assert "not found" in res["error"].lower()


def test_get_order_details_invalid_input(service):
    """Verify invalid input handling."""
    assert service.get_order_details("", "C-001")["success"] is False
    assert service.get_order_details(None, "C-001")["success"] is False


# ---------------------------------------------------------------------------
# 2. Return Eligibility Tests
# ---------------------------------------------------------------------------

def test_check_return_eligibility_eligible(service):
    """Verify an eligible physical order delivered within 30 days passes."""
    # NS-1001 delivered 2026-10-05, ref date 2026-10-25 (20 days)
    res = service.check_return_eligibility("NS-1001", authenticated_customer_id="C-001")
    assert res["eligible"] is True
    assert res["days_elapsed"] == 20
    assert "within_30_days_window" in res["conditions_passed"]
    assert res["manual_review_required"] is False


def test_check_return_eligibility_digital_product(service):
    """Verify digital products are non-returnable."""
    # NS-1004 is an E-book owned by C-003
    res = service.check_return_eligibility("NS-1004", authenticated_customer_id="C-003")
    assert res["eligible"] is False
    assert "digital" in res["reason"].lower()
    assert "physical_item_type" in res["conditions_failed"]


def test_check_return_eligibility_final_sale(service):
    """Verify final-sale items are non-returnable."""
    # NS-1005 is final-sale ceramic mugs owned by C-004
    res = service.check_return_eligibility("NS-1005", authenticated_customer_id="C-004")
    assert res["eligible"] is False
    assert "final-sale" in res["reason"].lower()
    assert "not_final_sale" in res["conditions_failed"]


def test_check_return_eligibility_undelivered_order(service):
    """Verify undelivered orders cannot be returned."""
    # NS-1003 is shipped, no delivery date
    res = service.check_return_eligibility("NS-1003", authenticated_customer_id="C-001")
    assert res["eligible"] is False
    assert "delivery" in res["reason"].lower() or "delivered" in res["reason"].lower()
    assert "delivered_status" in res["conditions_failed"]


def test_exact_30_day_window_boundary(service):
    """Verify exact 30-day boundary logic using injected date providers."""
    # NS-1001 delivered on 2026-10-05
    # Day 30: 2026-10-05 + 30 days = 2026-11-04
    service.set_date_provider(lambda: datetime.date(2026, 11, 4))
    res_30 = service.check_return_eligibility("NS-1001", authenticated_customer_id="C-001")
    assert res_30["eligible"] is True
    assert res_30["days_elapsed"] == 30

    # Day 31: 2026-11-05 (expired)
    service.set_date_provider(lambda: datetime.date(2026, 11, 5))
    res_31 = service.check_return_eligibility("NS-1001", authenticated_customer_id="C-001")
    assert res_31["eligible"] is False
    assert res_31["days_elapsed"] == 31
    assert "expired" in res_31["reason"].lower()


# ---------------------------------------------------------------------------
# 3. Refund Status Tests
# ---------------------------------------------------------------------------

def test_get_refund_status_approved(service):
    """Verify approved refund record lookup."""
    # NS-1002 owned by C-002 has refund RF-7001 approved
    res = service.get_refund_status("NS-1002", authenticated_customer_id="C-002")
    assert res["success"] is True
    assert res["refund_status"] == "approved"
    assert res["refund_id"] == "RF-7001"
    assert res["amount_inr"] == 2499
    assert res["approved_date"] == "2026-09-25"
    assert "5–10 business days" in res["details"]


def test_get_refund_status_unauthorized(service):
    """Verify unauthorized caller cannot inspect another customer's refund."""
    res = service.get_refund_status("NS-1002", authenticated_customer_id="C-001")
    assert res["success"] is False
    assert "Unauthorized" in res["error"]


def test_get_refund_status_unavailable(service):
    """Verify order with no return or refund returns unavailable status."""
    # NS-1003 has no return or refund
    res = service.get_refund_status("NS-1003", authenticated_customer_id="C-001")
    assert res["success"] is True
    assert res["refund_status"] == "unavailable"
    assert res["return_recorded"] is False
