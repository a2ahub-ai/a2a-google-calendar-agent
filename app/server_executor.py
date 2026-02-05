import json
import os
import sys
import asyncio
from typing import List, cast, Dict, Any, Optional
from urllib.parse import urlencode, urlparse, parse_qs

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
)
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message, new_task

from openai.types.chat import ChatCompletionMessageParam

from app.auth import get_google_creds
from .server_agent import MCPClient
from app.utils.logger import logger

from app.constants import ChatCompletionTypeEnum


class CalendarAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Agent for calendar event and reminder retrieval."""

    def __init__(self, runner: MCPClient, card: AgentCard):
        logger.debug("Initializing CalendarAgentExecutor...")
        self.runner = runner
        self._card = card
        self._active_sessions: set[str] = set()

    async def on_auth_callback(self, state: str, url: str):
        # Deprecated
        pass

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

        logger.debug(f"[status] {TaskState.working}")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "I'm retrieving your calendar events and reminders...",
                task.context_id,
                task.id,
            ),
        )

        # Retrieve Google Credentials from Vault (Redis)
        auth_info = get_google_creds(user_id)
        if not auth_info:
            logger.warning(f"No credentials found for user {user_id}")
            # The client should have handled auth, but if we are here and have no creds
            # it means either the session maps to no user (anonymous) or the user has no Google Creds yet.
            # We should return a failure or instructions to authenticate.
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    "Authentication missing or expired. Please run the client with --profile to login.",
                    task.context_id,
                    task.id
                )
            )
            return

        # Convert task history to messages
        messages = self._convert_task_history_to_messages(task.history)
        if not messages and query:
            messages.append(cast(ChatCompletionMessageParam, {
                "role": "user",
                "content": query
            }))

        logger.debug(f"Auth info found for user {user_id}")

        async for response in self.runner.process_query(messages, auth_info=auth_info):
            logger.debug(f"[calendar-agent] response type: {response['type']}")

            if response["type"] == ChatCompletionTypeEnum.CONTENT:
                if response["data"]:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(response["data"], task.context_id, task.id)
                    )
                    await updater.add_artifact([Part(root=TextPart(text=response["data"]))], name="Text Response")

            elif response["type"] == ChatCompletionTypeEnum.DATA:
                # Check for tool results
                data = response.get("data", {})
                if not data:
                    continue

                for tool_name, tool_result in data.items():
                    # Check content text for auth error
                    content_text = ""
                    if tool_result and hasattr(tool_result, 'content'):
                        content_text = " ".join([part.text for part in tool_result.content if part.type == "text"])

                    # Normal processing
                    if tool_result and tool_result.structuredContent:
                        await updater.add_artifact([Part(root=DataPart(data={tool_name: tool_result.structuredContent}, kind="data", metadata=None))], name="Calendar Events Data")
                        response_text = f"Retrieved calendar events: {tool_result.structuredContent}"
                    elif tool_result:
                        await updater.add_artifact([Part(root=TextPart(text=f"{tool_name}: {content_text}"))], name="Text Response")
                        response_text = content_text.strip()
                    else:
                        response_text = "No result from tool"

                    logger.debug(f"[status] {TaskState.completed}")
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(response_text, task.context_id, task.id)
                    )

            elif response["type"] == ChatCompletionTypeEnum.DONE:
                # If we reach here successfully, we are done
                pass

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
