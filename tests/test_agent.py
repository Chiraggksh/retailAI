"""
Unit and integration tests for the RetailSupportAgent and GeminiClient.
"""

from unittest.mock import MagicMock

import pytest
from google.genai import types

from agent import AgentState, RetailSupportAgent
from dense_retriever import DenseRetriever
from llm import GeminiClient, GeminiConfigurationError, GeminiResponse, ToolCallRequest
from mcp_client import MCPClient


# ---------------------------------------------------------------------------
# 1. Gemini Client Unit Tests
# ---------------------------------------------------------------------------

def test_gemini_client_missing_key_error(monkeypatch):
    """Verify GeminiClient raises clear configuration error when API key is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient(api_key=None)
    with pytest.raises(GeminiConfigurationError) as exc_info:
        client._get_client()
    assert "GEMINI_API_KEY is missing" in str(exc_info.value)


def test_gemini_client_mock_text_response():
    """Verify GeminiClient handles text responses properly."""
    mock_sdk_client = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_part = types.Part.from_text(text="Here is policy information.")
    mock_candidate.content = types.Content(role="model", parts=[mock_part])
    mock_response = MagicMock(candidates=[mock_candidate], usage_metadata=None)
    mock_sdk_client.models.generate_content.return_value = mock_response

    client = GeminiClient(client=mock_sdk_client)
    res = client.generate(contents=["hello"])
    assert res.text == "Here is policy information."
    assert res.has_tool_calls is False
    assert res.finish_reason == "STOP"


def test_gemini_client_mock_function_call():
    """Verify GeminiClient parses function calls requested by the model."""
    mock_sdk_client = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_part = types.Part.from_function_call(name="get_order_details", args={"order_id": "NS-1001"})
    mock_candidate.content = types.Content(role="model", parts=[mock_part])
    mock_response = MagicMock(candidates=[mock_candidate], usage_metadata=None)
    mock_sdk_client.models.generate_content.return_value = mock_response

    client = GeminiClient(client=mock_sdk_client)
    res = client.generate(contents=["check NS-1001"])
    assert res.has_tool_calls is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "get_order_details"
    assert res.tool_calls[0].args == {"order_id": "NS-1001"}


# ---------------------------------------------------------------------------
# 2. Stateful Agent Scenarios & Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_policy_question_scenario():
    """Scenario A: General policy question retrieves evidence without calling operational order tools."""
    async with MCPClient() as mcp:
        agent = RetailSupportAgent(
            mcp_client=mcp,
            demo_customer_id="C-001",
        )
        state = await agent.run("How many days do I have to return a physical item?")

        assert state.completion_status == "completed"
        assert len(state.retrieved_policy_chunks) > 0
        assert "POL-RET-001" in [c.document_id for c in state.retrieved_policy_chunks]
        # Order operational tools must not be called for pure policy questions
        operational_calls = [tc for tc in state.tool_calls if tc["name"] in ("get_order_details", "get_refund_status")]
        assert len(operational_calls) == 0
        assert len(state.citations) > 0


@pytest.mark.asyncio
async def test_agent_return_eligibility_scenario():
    """Scenario B: Return question retrieves order details, checks eligibility, and provides evidence."""
    async with MCPClient() as mcp:
        agent = RetailSupportAgent(
            mcp_client=mcp,
            demo_customer_id="C-001",
        )
        state = await agent.run("Can I return order NS-1001?")

        assert state.completion_status == "completed"
        tool_names = [tc["name"] for tc in state.tool_calls]
        assert "get_order_details" in tool_names
        assert "check_return_eligibility" in tool_names
        assert "eligible for return" in state.final_answer.lower()
        # Verify read-only notice
        assert "cannot automatically create" in state.final_answer.lower()


@pytest.mark.asyncio
async def test_agent_refund_status_scenario():
    """Scenario C: Refund question checks refund status and cites processing window."""
    async with MCPClient() as mcp:
        agent = RetailSupportAgent(
            mcp_client=mcp,
            demo_customer_id="C-002",
        )
        state = await agent.run("What is happening with my refund for order NS-1002?")

        assert state.completion_status == "completed"
        tool_names = [tc["name"] for tc in state.tool_calls]
        assert "get_refund_status" in tool_names
        assert "2499" in state.final_answer
        assert "RF-7001" in state.final_answer
        assert "5–10 business days" in state.final_answer or "5-10 business days" in state.final_answer


@pytest.mark.asyncio
async def test_agent_shipping_question_scenario():
    """Scenario D: Shipping inquiry retrieves order and cites transit estimates (not guarantees)."""
    async with MCPClient() as mcp:
        agent = RetailSupportAgent(
            mcp_client=mcp,
            demo_customer_id="C-001",
        )
        state = await agent.run("What is the expected delivery time for order NS-1003?")

        assert state.completion_status == "completed"
        assert "shipped" in state.final_answer.lower()
        assert "estimate" in state.final_answer.lower()
        assert "POL-SHIP-001" in state.final_answer or any(c["document_id"] == "POL-SHIP-001" for c in state.citations)


@pytest.mark.asyncio
async def test_agent_unrelated_query_refusal():
    """Scenario E: Out-of-scope query is politely declined with zero tool calls."""
    async with MCPClient() as mcp:
        agent = RetailSupportAgent(
            mcp_client=mcp,
            demo_customer_id="C-001",
        )
        state = await agent.run("Who won yesterday's cricket match?")

        assert state.completion_status == "completed"
        assert len(state.tool_calls) == 0
        assert "retail customer support" in state.final_answer.lower()


@pytest.mark.asyncio
async def test_agent_unauthorized_access_barrier():
    """Verify caller C-001 is denied access to order NS-1002 belonging to C-002."""
    async with MCPClient() as mcp:
        agent = RetailSupportAgent(
            mcp_client=mcp,
            demo_customer_id="C-001",
        )
        state = await agent.run("Please return order NS-1002 for me.")

        assert state.completion_status == "completed"
        assert "unauthorized" in state.final_answer.lower()
        # Ensure private items belonging to C-002 are not disclosed
        assert "Desk Lamp" not in state.final_answer


@pytest.mark.asyncio
async def test_agent_max_steps_safeguard():
    """Verify agent terminates gracefully when max steps limit is reached."""
    # Force max_steps = 1
    async with MCPClient() as mcp:
        agent = RetailSupportAgent(
            mcp_client=mcp,
            max_steps=0,
            demo_customer_id="C-001",
        )
        state = await agent.run("Can I return order NS-1001?")
        assert state.current_step >= state.max_steps
