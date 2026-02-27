
import asyncio
import json
from contextlib import AsyncExitStack
from typing import AsyncGenerator, List, Dict, Any, Awaitable, Callable, cast
from openai.types import ResponseFormatJSONSchema
from openai.types.shared.response_format_json_schema import JSONSchema

import httpx
from mcp import ClientSession
from mcp.client.stdio import (  # For JSON-RPC stdio transport
    StdioServerParameters,
    stdio_client,
)
from mcp.client.streamable_http import streamablehttp_client  # For HTTP transport
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

from app.config.settings import BaseConfig
from app.constants import ChatCompletionTypeEnum, AGENT_DESCRIPTION
from app.lib.llm.groq import GroqLLMProvider
from app.lib.llm.ollama import OllamaLLMProvider
from app.lib.llm.openai import OpenAILLMProvider
from app.types import ChatCompletionStreamResponseType
from app.utils.logger import logger

# Type alias for concurrent tasks that run alongside the LLM call.
# Each task is a coroutine that returns an arbitrary result (or None).
ConcurrentTask = Awaitable[Any]


class LoggingHTTPClient(httpx.AsyncClient):
    """Custom HTTP client that logs all requests"""

    async def send(self, request, **kwargs):
        logger.info(f"🌐 HTTP REQUEST to {request.url}")
        logger.info(f"Method: {request.method}")
        logger.info(f"Headers: {dict(request.headers)}")

        if request.content:
            try:
                # Try to parse and pretty-print JSON content
                content = json.loads(request.content.decode())
                logger.info("📤 Request Body:")
                logger.info(json.dumps(content, indent=2))

                # Specifically highlight messages and tools
                if "messages" in content:
                    logger.info("💬 MESSAGES TO LLM:")
                    for i, msg in enumerate(content["messages"]):
                        logger.info(f"Message {i + 1}: {json.dumps(msg, indent=2)}")

                if "tools" in content:
                    logger.info("🛠️  TOOLS SCHEMA:")
                    logger.info(json.dumps(content["tools"], indent=2))

            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning(f"Request Body (raw): {request.content}")

        logger.info("-" * 80)

        response = await super().send(request, **kwargs)

        logger.success(f"✅ HTTP RESPONSE from {request.url}")
        logger.info(f"Status: {response.status_code}")
        if response.content:
            try:
                response_content = json.loads(response.content.decode())
                logger.info("📥 Response Body:")
                logger.info(json.dumps(response_content, indent=2))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning(f"Response Body (raw): {response.content}")
        logger.info("=" * 80)

        return response


class AgentServer:
    def __init__(self):
        # Initialize session and client objects for multiple servers
        self.servers: dict[str, ClientSession] = {}  # Map server names to sessions
        self.exit_stack = AsyncExitStack()

        # Create custom HTTP client for logging
        self.http_client = LoggingHTTPClient()

        self.llm = GroqLLMProvider(api_key=BaseConfig.GROQ_API_KEY, model_name="openai/gpt-oss-20b")
        # self.llm = OpenAILLMProvider(api_key=BaseConfig.OPENAI_API_KEY, model_name="gpt-4.1-mini")
        # self.llm = OllamaLLMProvider(api_key="", model_name="functiongemma:latest")

    async def connect_to_server(self, server_name: str, url: str):
        """Connect to an MCP server over HTTP

        Args:
            server_name: A unique name for this server connection
            url: The HTTP endpoint URL of the running MCP server (e.g., "http://127.0.0.1:5000/mcp")
        """
        logger.info(f"🔌 Connecting to MCP server '{server_name}' at: {url}")

        # Connect using Streamable HTTP transport
        http_transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(url)
        )
        read, write, _ = http_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))

        await session.initialize()

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        logger.success(
            f"✅ Connected to server '{server_name}' with tools: {[tool.name for tool in tools]}"
        )
        print(
            f"\nConnected to server '{server_name}' with tools:",
            [tool.name for tool in tools],
        )

        # Store the session
        self.servers[server_name] = session

    async def connect_to_stdio_server(self, server_name: str, command: list[str]):
        """Connect to an MCP server over JSON-RPC stdio transport

        Args:
            server_name: A unique name for this server connection
            command: Command to start the server (e.g., ["python", "music-agent.py"])
        """
        logger.info(
            f"🔌 Connecting to JSON-RPC MCP server '{server_name}' with command: {' '.join(command)}"
        )

        # Create server parameters with proper structure
        server_params = StdioServerParameters(
            command=command[0],  # First element is the executable
            args=command[1:] if len(command) > 1 else [],  # Rest are arguments
        )

        # Connect using stdio transport
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))

        await session.initialize()

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        logger.success(
            f"✅ Connected to JSON-RPC server '{server_name}' with tools: {[tool.name for tool in tools]}"
        )
        print(
            f"\nConnected to JSON-RPC server '{server_name}' with tools:",
            [tool.name for tool in tools],
        )

        # Store the session
        self.servers[server_name] = session

    async def process_query(
        self,
        messages: List[ChatCompletionMessageParam],
        concurrent_tasks: Dict[str, ConcurrentTask] | None = None,
        tool_args_enhancer: Callable[[str, Any, Dict[str, Any]], Any] | None = None,
    ) -> AsyncGenerator[ChatCompletionStreamResponseType, None]:
        """Process a query using the LLM and available MCP tools.

        This is a **standalone, reusable** method.  It knows nothing about any
        specific remote agent (datetime-parser, etc.).  External callers can
        optionally inject:

        Args:
            messages: The conversation history (user / assistant / system).
            concurrent_tasks: A dict of ``{name: awaitable}`` that will be
                executed **simultaneously** with the LLM call via
                ``asyncio.gather``.  Their results are collected and forwarded
                to *tool_args_enhancer*.
            tool_args_enhancer: An optional callback
                ``(tool_name, tool_args, concurrent_results) -> tool_args``
                that is called **before** each MCP tool execution, allowing
                callers to merge concurrent-task results into the tool
                arguments.
        """
        logger.info("🚀 Processing new query")

        instruction = AGENT_DESCRIPTION + "\n"
        instruction += (
            "You are not permitted to answer any user questions beyond your "
            "primary task, if a user asks you, simply notify them that you do "
            "not have sufficient information to answer that question."
        )
        system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": instruction,
        }
        messages = [system_message] + messages

        logger.info(f"📝 Messages: {messages}")

        # ── Collect tools from all connected MCP servers ──
        available_tools: list = []
        tool_to_server_map: Dict[str, tuple] = {}

        for server_name, session in self.servers.items():
            response = await session.list_tools()
            for tool in response.tools:
                available_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
                tool_to_server_map[tool.name] = (server_name, session)

        logger.info(
            f"🛠️  Available tools from all servers: "
            f"{[t['function']['name'] for t in available_tools]}"
        )

        # ── Run LLM + concurrent tasks SIMULTANEOUSLY ──
        async def _collect_llm() -> tuple[list[ChatCompletionStreamResponseType], list]:
            """Consume the LLM stream and return (content_chunks, function_calls)."""
            chunks: list[ChatCompletionStreamResponseType] = []
            fn_calls: list = []
            async for rc in self.llm.chat_completion(
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=1,
                reasoning_effort="low"
            ):
                logger.debug(f"Response chunk: {rc}")
                if rc["type"] == ChatCompletionTypeEnum.CONTENT:
                    chunks.append(rc)
                elif rc["type"] == ChatCompletionTypeEnum.FUNCTION_CALLING:
                    if (
                        rc.get("data")
                        and isinstance(rc["data"], dict)
                        and rc["data"].get("function")
                    ):
                        fn_calls = rc["data"]["function"]
                        logger.info(f"🔧 LLM requested {len(fn_calls)} tool call(s)")
                elif rc["type"] == ChatCompletionTypeEnum.DONE:
                    break
            return chunks, fn_calls

        # Build the list of awaitables: LLM first, then any concurrent tasks.
        task_names: list[str] = []
        awaitables: list[asyncio.Future] = [_collect_llm()]
        if concurrent_tasks:
            for name, coro in concurrent_tasks.items():
                task_names.append(name)
                awaitables.append(coro)

        logger.info(
            f"📞 Running LLM + {len(task_names)} concurrent task(s) "
            f"{task_names} simultaneously..."
        )

        gather_results = await asyncio.gather(*awaitables, return_exceptions=True)

        # Unpack LLM result (always index 0).
        llm_result = gather_results[0]
        if isinstance(llm_result, BaseException):
            logger.error(f"LLM call failed: {llm_result}")
            yield ChatCompletionStreamResponseType(
                type=ChatCompletionTypeEnum.ERROR,
                data=str(llm_result),
            )
            return

        # Check if llm_result is a tuple or list as expected
        if isinstance(llm_result, (tuple, list)) and len(llm_result) == 2:
            content_chunks, function_calls = llm_result
        else:
            logger.error(f"Unexpected LLM result format: {llm_result}")
            content_chunks, function_calls = [], []

        # Unpack concurrent-task results into a name→result dict.
        concurrent_results: Dict[str, Any] = {}
        for idx, name in enumerate(task_names):
            # concurrency results start at index 1
            res = gather_results[idx + 1]
            if isinstance(res, BaseException):
                logger.error(f"Concurrent task '{name}' failed: {res}")
                concurrent_results[name] = None
            else:
                concurrent_results[name] = res

        logger.info(
            f"✅ All tasks completed — LLM chunks: {len(content_chunks)}, "
            f"function_calls: {len(function_calls)}, "
            f"concurrent results: {list(concurrent_results.keys())}"
        )

        # ── Yield collected LLM content chunks ──
        for chunk in content_chunks:
            yield chunk

        # ── Process tool calls if any ──
        if function_calls:
            # Add assistant message with tool calls to conversation
            tool_calls = []

            for func_call in function_calls:
                tool_calls.append(
                    {
                        "id": func_call.get("id", f"call_{func_call['name']}"),
                        "type": "function",
                        "function": {
                            "name": func_call["name"],
                            "arguments": (
                                str(func_call["arguments"])
                                if isinstance(func_call["arguments"], dict)
                                else func_call["arguments"]
                            ),
                        },
                    }
                )

            assistant_message: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_message)

            tool_results = {}
            for func_call in function_calls:
                tool_name = func_call["name"]
                tool_args = func_call["arguments"]

                # ── Apply argument enhancer if provided ──
                if tool_args_enhancer:
                    try:
                        tool_args = tool_args_enhancer(
                            tool_name, tool_args, concurrent_results
                        )
                    except Exception as e:
                        logger.error(
                            f"Argument enhancer failed for {tool_name}: {e}"
                        )

                # Find which server has this tool
                if tool_name in tool_to_server_map:
                    server_name, session = tool_to_server_map[tool_name]
                    logger.info(
                        f"⚙️  Executing tool: {tool_name} on server '{server_name}' "
                        f"with args: {tool_args}"
                    )

                    # Execute tool call on the appropriate server
                    result = await session.call_tool(tool_name, tool_args)
                    logger.info(
                        f"✅ Tool result from '{server_name}': {result.content}"
                    )
                    tool_results[tool_name] = result
                    yield ChatCompletionStreamResponseType(
                        type=ChatCompletionTypeEnum.DATA,
                        data=tool_results)
                else:
                    logger.error(
                        f"❌ Tool {tool_name} not found in any connected server"
                    )
                    tool_results[tool_name] = f"Error: Tool {tool_name} not available"
                    yield ChatCompletionStreamResponseType(
                        type=ChatCompletionTypeEnum.DATA,
                        data=tool_results)
