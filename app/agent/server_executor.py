import json
import os
import sys
import uuid
import asyncio
from typing import List, cast, Dict, Any, Optional
from urllib.parse import urlencode, urlparse, parse_qs

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
    TaskState,
    TextPart,
    Part,
    DataPart,
    UnsupportedOperationError,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message, new_task

from openai.types.chat import ChatCompletionMessageParam

from app.auth import get_google_creds, store_google_creds, create_session_token, verify_session_token
from .server_agent import AgentServer
from app.config.settings import BaseConfig
from app.remote_agents import RoutingAgent
from app.utils.logger import logger

from app.constants import ChatCompletionTypeEnum


class CalendarAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Agent for calendar event and reminder retrieval."""

    _awaiting_auth: Dict[str, asyncio.Future]
    _credentials: Dict[str, Dict[str, Any]]

    def __init__(self, runner: AgentServer, card: AgentCard, routing_agent: RoutingAgent | None = None):
        logger.debug("Initializing CalendarAgentExecutor...")
        self.runner = runner
        self._card = card
        self._routing_agent = routing_agent
        self._active_sessions: set[str] = set()
        self._awaiting_auth = {}
        self._credentials = {}

    async def on_auth_callback(self, state: str, url: str):
        if state not in self._awaiting_auth:
            logger.warning(
                'Received auth callback for unknown or already processed state: %s. '
                'Available states: %s',
                state,
                list(self._awaiting_auth.keys())
            )
            return
        self._awaiting_auth[state].set_result(url)

    def _convert_task_history_to_messages(self, task_history) -> List[ChatCompletionMessageParam]:
        """Convert task history to ChatCompletionMessageParam format"""
        messages: List[ChatCompletionMessageParam] = []

        for message in task_history:
            # Extract text content from message parts
            content_parts = []
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        content_parts.append(part.root.text)

            content = " ".join(content_parts) if content_parts else ""

            # Convert role: agent -> assistant, keep user as user
            if hasattr(message, 'role'):
                if message.role == Role.agent:
                    role = "assistant"
                elif message.role == Role.user:
                    role = "user"
                else:
                    role = "user"  # fallback
            else:
                role = "user"  # fallback

            if content.strip():  # Only add messages with content
                if role == "assistant":
                    messages.append(cast(ChatCompletionMessageParam, {
                        "role": "assistant",
                        "content": content
                    }))
                else:  # user role
                    messages.append(cast(ChatCompletionMessageParam, {
                        "role": "user",
                        "content": content
                    }))

        return messages

    def _get_user_id(self, context: RequestContext) -> str:
        if context.call_context and context.call_context.user:
            # We expect the AuthMiddleware to populate 'user_name' with the internal user_id
            return context.call_context.user.user_name or "anonymous"
        return "anonymous"

    async def _handle_auth_flow(self, context: RequestContext, updater: TaskUpdater) -> Optional[Dict[str, Any]]:
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            logger.error("GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not set")
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message("Server configuration error: OAuth credentials missing.", context.context_id)
            )
            return None

        # Ensure correct redirect URI
        base_url = self._card.url.rstrip('/')
        redirect_uri = f"{base_url}/authenticate"

        flow = InstalledAppFlow.from_client_config(
            {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uris": [redirect_uri],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=['https://www.googleapis.com/auth/calendar']
        )
        flow.redirect_uri = redirect_uri
        auth_url, state = flow.authorization_url(prompt='consent', access_type='offline')

        future = asyncio.get_running_loop().create_future()
        self._awaiting_auth[state] = future

        logger.info(f"Initiating auth flow, state: {state}")
        # Send auth required status with URL
        await updater.update_status(
            TaskState.auth_required,
            message=new_agent_text_message(f"Please authorize the application by visiting this URL: {auth_url}", context.context_id)
        )

        try:
            logger.debug(f"Waiting for auth callback...")
            # Wait for callback
            callback_url = await asyncio.wait_for(future, timeout=300)
        except asyncio.TimeoutError:
            logger.warning("Auth timeout")
            self._awaiting_auth.pop(state, None)
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message("Timed out waiting for authorization.", context.context_id)
            )
            return None

        self._awaiting_auth.pop(state, None)
        logger.info("Auth callback received")

        # Parse code
        try:
            parsed = urlparse(callback_url)
            params = parse_qs(parsed.query)
            code = params.get('code', [None])[0]

            if not code:
                logger.error("No code in callback URL")
                return None

            flow.fetch_token(code=code)
            creds = flow.credentials
            creds_json = json.loads(creds.to_json())

            user_id = self._get_user_id(context)
            if user_id == "anonymous":
                user_id = str(uuid.uuid4())

            # Store in-memory and in Redis
            self._credentials[user_id] = creds_json
            store_google_creds(user_id, creds_json)

            # Generate session token to return
            session_token = create_session_token(user_id)
            return {"token": session_token}
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return None

    # ── Remote-agent helpers (calendar-specific orchestration) ──

    @staticmethod
    def _extract_user_query(messages: List[ChatCompletionMessageParam]) -> str | None:
        """Extract the last user message text from a message list."""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                return str(msg["content"])
        return None

    async def _call_datetime_parser(self, user_query: str) -> dict | None:
        """Call the remote datetime-parser agent and return the parsed result.

        Returns:
            On success: ``{"time_range": {...}}`` (the ``datetime_parser`` payload).
            On error  : ``{"error": "<message>"}`` so the caller can propagate it.
            If the agent is unavailable: ``None``.
        """
        if not self._routing_agent:
            return None

        # Use the configured agent name from settings
        datetime_agent_name = BaseConfig.DATETIME_PARSER_AGENT
        if not datetime_agent_name:
            logger.warning("DATETIME_PARSER_AGENT not configured in settings")
            return None

        if datetime_agent_name not in self._routing_agent.agents_info:
            logger.warning(f"Datetime parser agent '{datetime_agent_name}' not found in routing agent")
            return None

        try:
            datetime_result = None
            error_message = None

            async for event in self._routing_agent.request(datetime_agent_name, user_query, metadata={"single_time_mode": False}):
                if isinstance(event, TaskArtifactUpdateEvent):
                    if event.artifact and event.artifact.parts:
                        for part in event.artifact.parts:
                            if hasattr(part, "root") and hasattr(part.root, "data"):
                                data = part.root.data
                                if isinstance(data, dict) and "datetime_parser" in data:
                                    datetime_result = data["datetime_parser"]
                elif isinstance(event, TaskStatusUpdateEvent):
                    # Capture error/status messages when no artifact is produced
                    if (
                        event.status
                        and event.status.message
                        and event.status.message.parts
                    ):
                        for part in event.status.message.parts:
                            if hasattr(part, "root") and hasattr(part.root, "text") and part.root.text:
                                error_message = part.root.text

            if datetime_result:
                logger.info(f"📅 Datetime parser result: {datetime_result}")
                return datetime_result

            # No artifact received – treat the status message as an error
            if error_message:
                logger.warning(f"📅 Datetime parser returned error: {error_message}")
                return {"error": error_message}

            return None
        except Exception as e:
            logger.error(f"Failed to call datetime parser agent: {e}")
            return {"error": str(e)}

    def _build_concurrent_tasks(
        self, messages: List[ChatCompletionMessageParam]
    ) -> Dict[str, Any]:
        """Build the dict of concurrent awaitables to run alongside the LLM.

        Add new remote-agent tasks here as they become available.
        """
        tasks: Dict[str, Any] = {}
        user_query = self._extract_user_query(messages)

        if user_query and self._routing_agent:
            tasks["datetime_parser"] = self._call_datetime_parser(user_query)

        return tasks

    @staticmethod
    def _calendar_tool_args_enhancer(
        tool_name: str, tool_args: Any, concurrent_results: Dict[str, Any],
        auth_info: Optional[Dict[str, Any]] = None,
        timezone: int | float | None = None,
    ) -> Any:
        # Normalize tool_args to dict
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                pass

        if not isinstance(tool_args, dict):
            return tool_args

        # 1. Pass datetime_parser (or its error message) to every tool.
        #    Each tool's run() decides whether it needs the parsed data.
        datetime_result = concurrent_results.get("datetime_parser")
        if datetime_result:
            if isinstance(datetime_result, dict) and "error" in datetime_result:
                # Parser returned an error — pass the message so the
                # tool can log it or decide to ignore it.
                tool_args["datetime_parser_message"] = datetime_result["error"]
                logger.info(
                    f"📅 Passing datetime_parser_message to {tool_name}: "
                    f"{datetime_result['error']}"
                )
            else:
                tool_args["datetime_parser"] = datetime_result
                logger.info(f"📅 Merged datetime_parser into {tool_name} args")

        # 2. Merge auth_info
        if auth_info:
            tool_args["__auth_info"] = auth_info

        # 3. Merge timezone
        if timezone is not None:
            tool_args["__timezone"] = timezone

        return tool_args

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        logger.debug("[calendar-agent] execute entered")
        # dump context for debugging
        if context._params:
            logger.debug(context._params.metadata if context._params.metadata else "No metadata")
        logger.debug(context.context_id)
        logger.debug(context.task_id)

        # Extract timezone from metadata
        timezone_offset = None
        if context._params and context._params.metadata:
            if 'timezone' in context._params.metadata:
                timezone_offset = context._params.metadata['timezone']
                logger.info(f"Timezone offset from metadata: {timezone_offset}")

        query = context.get_user_input()
        task = context.current_task

        if not task:
            if context.message:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
            else:
                logger.error("No task available and no message to create task from")
                return

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        user_id = self._get_user_id(context)
        logger.debug(f"User ID: {user_id}")

        # Convert task history to messages
        messages = self._convert_task_history_to_messages(task.history)
        if not messages and query:
            messages.append(cast(ChatCompletionMessageParam, {
                "role": "user",
                "content": query
            }))

        max_retries = 1
        for attempt in range(max_retries + 1):
            # Retrieve Google Credentials from Vault (Redis) or memory
            auth_info = get_google_creds(user_id)
            if not auth_info:
                auth_info = self._credentials.get(user_id) or {}

            logger.debug(f"Auth info found for user {user_id}")
            retry_needed = False

            # Prepare concurrent tasks and enhancer
            concurrent_tasks = self._build_concurrent_tasks(messages)

            def tool_args_enhancer_wrapper(tool_name, tool_args, concurrent_results):
                return self._calendar_tool_args_enhancer(
                    tool_name,
                    tool_args,
                    concurrent_results,
                    auth_info=auth_info,
                    timezone=timezone_offset)

            async for response in self.runner.process_query(
                messages,
                concurrent_tasks=concurrent_tasks,
                tool_args_enhancer=tool_args_enhancer_wrapper
            ):
                logger.debug(f"[calendar-agent] response type: {response['type']}")

                if response["type"] == ChatCompletionTypeEnum.CONTENT:
                    if response["data"]:
                        await updater.update_status(
                            TaskState.completed,
                            new_agent_text_message(response["data"], task.context_id, task.id)
                        )

                elif response["type"] == ChatCompletionTypeEnum.DATA:
                    data = response.get("data", {})
                    if not data:
                        continue

                    auth_error = False
                    combined_response_text = []

                    for tool_name, tool_result in data.items():
                        # Check content text for auth error
                        content_text = ""
                        if tool_result and hasattr(tool_result, 'content'):
                            content_text = " ".join([part.text for part in tool_result.content if part.type == "text"])

                        if "Missing authorization information" in content_text or "Authorization required" in content_text:
                            auth_error = True
                            break  # Stop processing if auth error found

                        # structuredContent → artifact
                        if tool_result and tool_result.structuredContent:
                            await updater.add_artifact(
                                [Part(root=DataPart(data={tool_name: tool_result.structuredContent}, kind="data", metadata=None))],
                                name=f"{tool_name} Data"
                            )
                        # text content → text message (no artifact)
                        elif tool_result:
                            if content_text.strip():
                                combined_response_text.append(content_text)
                        else:
                            combined_response_text.append(f"No result from {tool_name}")

                    if not auth_error:
                        logger.debug(f"[status] {TaskState.completed}")
                        final_message = " ".join(combined_response_text)
                        await updater.update_status(
                            TaskState.completed,
                            new_agent_text_message(final_message, task.context_id, task.id) if final_message.strip() else None
                        )

                    if auth_error:
                        logger.info("Tool returned auth error. Initiating auth flow.")
                        auth_result = await self._handle_auth_flow(context, updater)
                        if auth_result:
                            # Send token artifact
                            if 'token' in auth_result:
                                await updater.add_artifact([Part(root=TextPart(text=auth_result['token']))], name="token")

                                # Update user_id for the retry attempt
                                decoded = verify_session_token(auth_result['token'])
                                if decoded and "sub" in decoded:
                                    user_id = decoded["sub"]
                                    logger.info(f"Updated session user_id to {user_id}")

                            await updater.update_status(
                                TaskState.working,
                                new_agent_text_message("Auth received, continuing...", task.context_id, task.id)
                            )
                            retry_needed = True
                            break  # Break 'async for process_query' to retry outer loop
                        else:
                            # Auth failed or timed out
                            return

                elif response["type"] == ChatCompletionTypeEnum.DONE:
                    # If we reach here successfully, we are done
                    pass

            if retry_needed:
                continue
            else:
                break

        logger.debug("[calendar-agent] execute exiting")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.debug("[calendar-agent] cancel entered")
        """Cancel the execution for the given context.

        Currently logs the cancellation attempt as the underlying ADK runner
        doesn't support direct cancellation of ongoing tasks.
        """
        session_id = context.context_id
        if session_id in self._active_sessions:
            logger.info(
                f"Cancellation requested for active calendar-agent session: {session_id}"
            )
            # TODO: Implement proper cancellation when ADK supports it
            self._active_sessions.discard(session_id)
        else:
            logger.debug(
                f"Cancellation requested for inactive calendar-agent session: {session_id}"
            )

        raise ServerError(error=UnsupportedOperationError())
