"""
Model Context Protocol (MCP) Server for Northstar Retail Customer Support.

Exposes authenticated operational tools over the official Python MCP protocol:
1. get_order_details: Look up customer order details.
2. check_return_eligibility: Evaluate return eligibility against policy rules.
3. get_refund_status: Track return and refund processing status.

Uses trusted caller identity independently of user-supplied prompt text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from mcp.server.mcpserver import MCPServer

from operational_service import OperationalService

logger = logging.getLogger(__name__)


def create_mcp_server(
    service: Optional[OperationalService] = None,
    customer_id: Optional[str] = None,
) -> MCPServer:
    """
    Factory function creating an MCPServer instance with operational tools registered.

    Args:
        service: OperationalService instance for business logic.
        customer_id: Trusted caller identity (defaults to DEMO_CUSTOMER_ID or 'C-001').
    """
    op_service = service or OperationalService()
    trusted_customer_id = customer_id or os.environ.get("DEMO_CUSTOMER_ID", "C-001").strip()

    server = MCPServer("northstar_retail_support")

    @server.tool(
        description=(
            "Retrieve verified order details by order ID (e.g. 'NS-1001'). "
            "Returns status, items, amounts, delivery dates, and tracking information. "
            "Validates customer ownership before disclosing details."
        )
    )
    def get_order_details(order_id: str) -> str:
        """Look up order details with ownership verification."""
        result = op_service.get_order_details(
            order_id=order_id,
            authenticated_customer_id=trusted_customer_id,
        )
        return json.dumps(result, indent=2)

    @server.tool(
        description=(
            "Check if an order is eligible for return under policy POL-RET-001. "
            "Evaluates ownership, delivery status, delivery date, item type, final-sale flag, "
            "condition, and the 30-day return window. Returns eligibility decision and explanation."
        )
    )
    def check_return_eligibility(order_id: str) -> str:
        """Evaluate return eligibility for an order under policy constraints."""
        result = op_service.check_return_eligibility(
            order_id=order_id,
            authenticated_customer_id=trusted_customer_id,
        )
        return json.dumps(result, indent=2)

    @server.tool(
        description=(
            "Look up refund status, amounts, and return processing progress for an order. "
            "Returns approved refund details, pending inspection status, or indicates if no return exists."
        )
    )
    def get_refund_status(order_id: str) -> str:
        """Check refund and return record status for an order."""
        result = op_service.get_refund_status(
            order_id=order_id,
            authenticated_customer_id=trusted_customer_id,
        )
        return json.dumps(result, indent=2)

    return server


# Default server instance for module execution
app = create_mcp_server()


async def main() -> None:
    """Run the MCP server over stdio for MCP clients."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger.info("Starting Northstar MCP Server over stdio transport...")
    await app.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
