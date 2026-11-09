"""
Integration and unit tests for MCP Client communicating with MCP Server over stdio.
"""

import asyncio
import pytest

from mcp_client import MCPClient, convert_mcp_schema_to_gemini


@pytest.mark.asyncio
async def test_mcp_client_connection_and_tool_discovery():
    """Verify MCPClient connects to actual mcp_server.py and discovers tools."""
    async with MCPClient() as client:
        assert client.is_connected is True
        tools = client.tools
        assert len(tools) >= 3
        assert "get_order_details" in tools
        assert "check_return_eligibility" in tools
        assert "get_refund_status" in tools

        # Test tool invocation over actual MCP protocol
        order_res = await client.call_tool("get_order_details", {"order_id": "NS-1001"})
        assert order_res.get("success") is True
        assert order_res.get("order_id") == "NS-1001"
        assert order_res.get("customer_id") == "C-001"

        # Test eligibility tool
        elig_res = await client.call_tool("check_return_eligibility", {"order_id": "NS-1001"})
        assert elig_res.get("eligible") is True
        assert elig_res.get("order_id") == "NS-1001"

        # Test refund status tool
        ref_res = await client.call_tool("get_refund_status", {"order_id": "NS-1001"})
        assert ref_res.get("success") is True


@pytest.mark.asyncio
async def test_mcp_client_gemini_schema_conversion():
    """Verify MCP tool input schemas convert accurately to Gemini FunctionDeclarations."""
    async with MCPClient() as client:
        declarations = client.get_gemini_function_declarations()
        assert len(declarations) >= 3

        decl_map = {d.name: d for d in declarations}
        assert "get_order_details" in decl_map
        order_decl = decl_map["get_order_details"]
        assert "order_id" in order_decl.parameters.properties


@pytest.mark.asyncio
async def test_mcp_client_unknown_tool_error():
    """Verify calling an unknown tool safely returns a structured error."""
    async with MCPClient() as client:
        res = await client.call_tool("non_existent_tool", {})
        assert res.get("success") is False
        assert "not an authorized" in res.get("error", "")
