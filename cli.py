"""
CLI Demonstration Entry Point for Northstar Retail Customer Support.

Demonstrates:
1. Authenticated customer session identity (DEMO_CUSTOMER_ID).
2. Pure policy knowledge queries with source citations.
3. Multi-step return resolution via MCP tools and policy retrieval.
4. Operational refund status tracking.
5. Exception handling: unauthorized order access and non-returnable digital products.
6. Refusal of out-of-domain inquiries.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from agent import RetailSupportAgent
from dense_retriever import DenseRetriever
from llm import GeminiClient
from mcp_client import MCPClient

load_dotenv()


def print_banner() -> None:
    print("=" * 80)
    print("NORTHSTAR RETAIL CUSTOMER SUPPORT — AGENTIC RESOLUTION SYSTEM")
    print("=" * 80)


def print_state_trace(state) -> None:
    """Print clean observable action trace and final answer with citations."""
    print(f"\n[Customer Request] : {state.user_query}")
    print(f"[Caller Identity]  : {state.authenticated_customer_id}")
    print(f"[Execution Status] : {state.completion_status} (Steps: {state.current_step}/{state.max_steps})")
    print(f"[Total Latency]    : {state.total_latency_ms:.2f} ms")
    if state.total_tokens > 0:
        print(f"[Tokens Consumed]  : {state.total_tokens}")

    # Tool calls & results
    if state.tool_calls:
        print("\n--- Observable Action Trace ---")
        for idx, (tc, tr) in enumerate(zip(state.tool_calls, state.tool_results), start=1):
            print(f"  Step {idx} [Action] : Invoking '{tc['name']}'")
            print(f"         [Arguments]: {json.dumps(tc['args'])}")
            result_preview = json.dumps(tr.get("result", {}), indent=2)
            if len(result_preview) > 300:
                result_preview = result_preview[:300] + "... (truncated)"
            print(f"         [Result]   :\n{result_preview}")

    # Retrieved Policy Evidence
    if state.retrieved_policy_chunks:
        print("\n--- Retrieved Policy Evidence ---")
        for chunk in state.retrieved_policy_chunks[:2]:
            print(f"  • [{chunk.document_id} v{chunk.document_version}] Section: {chunk.section_title} (Page {chunk.page_number})")
            print(f"    Excerpt: {chunk.chunk_text[:140].replace(chr(10), ' ')}...")

    # Final Answer
    print("\n--- Final Grounded Answer ---")
    print(state.final_answer)

    # Citations
    if state.citations:
        print("\n--- Policy Source Citations ---")
        for cit in state.citations:
            print(f"  ✓ {cit['document_id']} (v{cit['document_version']}, Page {cit['page_number']}) — {cit['section_title']}")

    # Diagnostic Errors or Warnings
    if state.errors:
        print("\n--- Diagnostic Warnings / Errors ---")
        for err in state.errors:
            print(f"  ⚠ {err}")

    print("-" * 80)


async def run_scenario(agent: RetailSupportAgent, query: str, customer_id: Optional[str] = None) -> None:
    """Run a single test scenario."""
    if customer_id:
        agent.authenticated_customer_id = customer_id
    state = await agent.run(query)
    print_state_trace(state)


async def run_predefined_demos(agent: RetailSupportAgent) -> None:
    """Run canonical demonstration scenarios covering all assignment requirements."""
    print("\nRunning automated demonstration suite across canonical retail scenarios...\n")

    # Scenario 1: Policy Knowledge Query
    print("\n>>> DEMO 1: Policy Knowledge Inquiry (Returns Window)")
    await run_scenario(
        agent,
        "How many days does a customer have to return physical items, and what starts the window?",
        customer_id="C-001",
    )

    # Scenario 2: Multi-step Return Eligibility (Eligible order)
    print("\n>>> DEMO 2: Multi-Step Return Resolution (Physical item, eligible)")
    await run_scenario(
        agent,
        "I received order NS-1001 and want to return it. What can you do?",
        customer_id="C-001",
    )

    # Scenario 3: Operational Refund Status (Customer Two owns NS-1002)
    print("\n>>> DEMO 3: Refund Status Inquiry (Approved refund)")
    await run_scenario(
        agent,
        "Where is my refund for order NS-1002?",
        customer_id="C-002",
    )

    # Scenario 4: Exception Scenario 1 (Unauthorized access attempt by Customer One on NS-1002)
    print("\n>>> DEMO 4: Exception Scenario — Unauthorized Order Ownership")
    await run_scenario(
        agent,
        "Please look up order NS-1002 and process a return for me.",
        customer_id="C-001",
    )

    # Scenario 5: Exception Scenario 2 (Non-returnable Digital item NS-1004 owned by Customer Three)
    print("\n>>> DEMO 5: Exception Scenario — Non-Returnable Digital Product")
    await run_scenario(
        agent,
        "Can I return my e-book order NS-1004?",
        customer_id="C-003",
    )

    # Scenario 6: Out-of-Scope Inquiry
    print("\n>>> DEMO 6: Out-of-Scope Query Handling")
    await run_scenario(
        agent,
        "Who won yesterday's cricket match?",
        customer_id="C-001",
    )


async def interactive_mode(agent: RetailSupportAgent) -> None:
    """Run interactive REPL loop."""
    print("\nEntering interactive mode. Type 'exit' or 'quit' to terminate.\n")
    while True:
        try:
            query = input(f"[{agent.authenticated_customer_id}] Enter question > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            state = await agent.run(query)
            print_state_trace(state)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


async def main() -> None:
    parser = argparse.ArgumentParser(description="Northstar Retail Support Agent CLI")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat mode")
    parser.add_argument("--customer", type=str, default=None, help="Set caller customer ID (e.g. C-001, C-002)")
    parser.add_argument("--query", type=str, default=None, help="Execute a single query")
    args = parser.parse_args()

    print_banner()

    demo_customer = args.customer or os.environ.get("DEMO_CUSTOMER_ID", "C-001")
    os.environ["DEMO_CUSTOMER_ID"] = demo_customer
    print(f"Authenticated Session Customer ID: {demo_customer}")

    retriever = DenseRetriever()
    llm = GeminiClient()
    if llm.is_configured:
        print(f"LLM Provider: Google Gemini ({llm.model}) [LIVE MODE]")
    else:
        print("LLM Provider: Offline Deterministic Fallback Mode (GEMINI_API_KEY not configured)")

    # Initialize MCP client with caller identity
    mcp_env = dict(os.environ)
    mcp_env["DEMO_CUSTOMER_ID"] = demo_customer
    async with MCPClient(env=mcp_env) as mcp:
        agent = RetailSupportAgent(
            llm_client=llm,
            mcp_client=mcp,
            retriever=retriever,
            demo_customer_id=demo_customer,
        )

        if args.query:
            await run_scenario(agent, args.query, demo_customer)
        elif args.interactive:
            await interactive_mode(agent)
        else:
            await run_predefined_demos(agent)


if __name__ == "__main__":
    asyncio.run(main())
