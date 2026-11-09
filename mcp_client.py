"""
MCP Client module connecting to the Northstar Retail MCP Server.

Provides:
1. Tool discovery over the official MCP protocol (reading schemas, names, descriptions).
2. Protocol-level tool invocation over stdio with timeout and error handling.
3. Conversion of MCP tool definitions into Gemini-compatible FunctionDeclaration objects.
4. Robust lifecycle management (async context manager) to avoid zombie subprocesses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


def convert_mcp_schema_to_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively sanitize and convert an MCP JSON schema to Gemini-compatible Schema dict.
    Converts type strings (e.g. 'object' -> 'OBJECT', 'string' -> 'STRING').
    """
    if not isinstance(schema, dict):
        return schema

    converted: Dict[str, Any] = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            converted[k] = v.upper()
        elif k == "properties" and isinstance(v, dict):
            converted[k] = {prop: convert_mcp_schema_to_gemini(p_schema) for prop, p_schema in v.items()}
        elif k == "items" and isinstance(v, dict):
            converted[k] = convert_mcp_schema_to_gemini(v)
        elif k in ("title", "$schema", "additionalProperties"):
            # Omit non-standard Gemini schema annotations
            continue
        else:
            converted[k] = v

    if "type" not in converted:
        converted["type"] = "OBJECT"

    return converted


class MCPClient:
    """Manages the connection, tool discovery, and tool execution with the MCP Server."""

    def __init__(
        self,
        server_command: Optional[str] = None,
        server_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        """
        Initialize the MCP Client configuration.

        Args:
            server_command: Executable binary (defaults to sys.executable).
            server_args: Arguments passed to launch MCP server (defaults to ['mcp_server.py']).
            env: Custom environment variables for server subprocess.
            timeout_seconds: Timeout for tool calls and connection initialization.
        """
        self.server_command = server_command or sys.executable
        self.server_args = server_args or ["mcp_server.py"]
        self.env = env or dict(os.environ)
        self.timeout_seconds = timeout_seconds

        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._discovered_tools: Dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        """Check if an active session is established."""
        return self._session is not None

    async def connect(self) -> None:
        """Start the MCP server subprocess, initialize session, and discover tools."""
        if self.is_connected:
            return

        logger.info(f"Connecting to MCP server via: {self.server_command} {' '.join(self.server_args)}")
        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
            env=self.env,
        )

        self._exit_stack = AsyncExitStack()
        try:
            # Enter stdio transport context
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            # Enter ClientSession context
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize session with timeout
            await asyncio.wait_for(self._session.initialize(), timeout=self.timeout_seconds)

            # Discover available tools
            await self.refresh_tools()
            logger.info(f"MCP client connected. Discovered {len(self._discovered_tools)} tools.")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.close()
            raise ConnectionError(f"Could not connect to MCP server: {e}") from e

    async def refresh_tools(self) -> List[Any]:
        """Fetch and cache available tools from the server."""
        if not self._session:
            raise RuntimeError("MCP client is not connected.")

        tools_result = await asyncio.wait_for(
            self._session.list_tools(),
            timeout=self.timeout_seconds,
        )
        self._discovered_tools = {t.name: t for t in tools_result.tools}
        return list(self._discovered_tools.values())

    @property
    def tools(self) -> Dict[str, Any]:
        """Return the dictionary of discovered tools keyed by tool name."""
        return self._discovered_tools

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an operational tool through the MCP protocol.

        Args:
            name: Tool name (e.g. 'get_order_details').
            arguments: Dictionary of keyword arguments matching tool input schema.

        Returns:
            Parsed JSON result dictionary or structured error response.
        """
        if not self._session:
            raise RuntimeError("MCP client is not connected.")

        if name not in self._discovered_tools:
            return {
                "success": False,
                "error": f"Tool '{name}' is not an authorized or discovered MCP tool.",
            }

        args = arguments or {}
        try:
            call_res = await asyncio.wait_for(
                self._session.call_tool(name=name, arguments=args),
                timeout=self.timeout_seconds,
            )

            # Parse content from MCP TextContent
            if call_res.content and len(call_res.content) > 0:
                raw_text = call_res.content[0].text
                try:
                    return json.loads(raw_text)
                except json.JSONDecodeError:
                    return {"success": True, "raw_content": raw_text}

            return {"success": True, "result": getattr(call_res, "structured_content", {})}

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling MCP tool '{name}' ({self.timeout_seconds}s)")
            return {"success": False, "error": f"Tool execution for '{name}' timed out."}
        except Exception as e:
            logger.error(f"Error executing MCP tool '{name}': {e}")
            return {"success": False, "error": f"MCP tool error: {str(e)}"}

    def get_gemini_function_declarations(self) -> List[types.FunctionDeclaration]:
        """
        Convert all discovered MCP tools into Gemini FunctionDeclaration objects.
        """
        declarations: List[types.FunctionDeclaration] = []
        for name, tool in self._discovered_tools.items():
            desc = tool.description or f"Operational tool {name}"
            raw_schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", {}) or {"type": "object"}
            gemini_schema = convert_mcp_schema_to_gemini(raw_schema)

            decl = types.FunctionDeclaration(
                name=name,
                description=desc,
                parameters=gemini_schema,
            )
            declarations.append(decl)
        return declarations

    async def close(self) -> None:
        """Clean up the MCP session and terminate server subprocesses."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.debug(f"Exception during MCP cleanup: {e}")
            finally:
                self._exit_stack = None
                self._session = None
                self._discovered_tools.clear()
                logger.info("MCP client session closed.")

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
