import json
import uuid

from typing import Any, AsyncGenerator, Dict, TypedDict

import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    Message,
    TaskState
)

from app.constants import ChatCompletionTypeEnum
from app.types import AgentInfo, ChatCompletionStreamResponseType
from .remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback,
)

from app.utils import logger


class RoutingAgent:
    """The Routing agent.

    This is the agent responsible for choosing which remote seller agents
    to send tasks to and coordinate their work.
    """

    def __init__(
        self,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.agents_info: dict[str, AgentInfo] = {}

    async def _async_init_components(
        self, remote_agent_addresses: list[str]
    ) -> None:
        """Asynchronous part of initialization."""
        # Use a single httpx.AsyncClient for all card resolutions for efficiency
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(
                    client, address
                )  # Constructor is sync
                try:
                    card = (
                        await card_resolver.get_agent_card()
                    )  # get_agent_card is async

                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    await remote_connection.authenticate()

                    self.agents_info[card.name] = {
                        'remote_agent_connections': remote_connection,
                        'context_storage': {},
                        'card': card,
                    }
                except httpx.ConnectError as e:
                    logger.error(
                        f'ERROR: Failed to get agent card from {address}: {e}'
                    )
                except Exception as e:  # Catch other potential errors
                    logger.error(
                        f'ERROR: Failed to initialize connection for {address}: {e}'
                    )

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ) -> 'RoutingAgent':
        """Create and asynchronously initialize an instance of the RoutingAgent."""
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    async def send_message_streaming(self, agent_name: str, message: str, context_id: str |
                                     None = None, task_id: str | None = None, metadata: dict | None = None) -> AsyncGenerator[Any, None]:
        """Send a message with streaming support"""
        agents_info = self.agents_info[agent_name]
        if not agents_info:
            raise ValueError(f"Agent connection info not available for {agent_name}")

        client = agents_info['remote_agent_connections']
        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        context_storage = agents_info['context_storage']

        request_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
                "messageId": message_id,
                "contextId": context_id,
                "taskId": task_id,
                # "referenceTaskIds": [task_id] if task_id else None
            },
            'metadata': metadata or {}
        }

        message_request = SendStreamingMessageRequest(
            id=request_id, params=MessageSendParams.model_validate(payload)
        )

        # logger.info(f"Starting streaming message to agent: {agent_name}")

        async for chunk in client.send_message_streaming(message_request):
            if hasattr(chunk.root, 'error') and chunk.root.error:
                 logger.error(f"JSONRPC Error from agent {agent_name}: {chunk.root.error}")
                 # You might want to yield an error event here
                 continue

            if hasattr(chunk.root, 'result'):
                event = chunk.root.result
            else:
                logger.warning(f"Chunk from {agent_name} has no result: {chunk}")
                continue

            # logger.debug(f"Received streaming chunk: {chunk}")
            # logger.debug(chunk.model_dump_json(exclude_none=True, indent=2))

            # Yield the actual event objects for the agent mode
            if isinstance(event, TaskStatusUpdateEvent):
                # Check if task is completed and remove from context_storage
                if hasattr(
                        event,
                        'status') and event.status and hasattr(
                        event.status,
                        'state') and event.status.state == "completed":
                    # Extract context_id from the event to remove from storage
                    event_context_id = getattr(event, 'context_id', None)
                    if event_context_id and event_context_id in context_storage:
                        removed_task_id = context_storage.pop(event_context_id)
                        logger.debug(
                            f"Removed completed task from context_storage: context_id: {event_context_id}, task_id: {removed_task_id}")

                # logger.info(f"Received TaskStatusUpdateEvent: {event.model_dump_json(exclude_none=True, indent=2)}")
                yield event
            elif isinstance(event, TaskArtifactUpdateEvent):
                # logger.info(f"Received TaskArtifactUpdateEvent: {event.model_dump_json(exclude_none=True, indent=2)}")
                yield event
            elif isinstance(event, Task):
                # Extract context_id and task_id from Task if not provided
                task_context_id = getattr(event, 'context_id', None)
                task_task_id = getattr(event, 'id', None)

                if context_id is None and task_context_id:
                    context_id = task_context_id
                if task_id is None and task_task_id:
                    task_id = task_task_id

                # Store in context_storage if both are available
                if context_id and task_id:
                    context_storage[context_id] = task_id
                    logger.debug(f"Stored context_id: {context_id}, task_id: {task_id} in context_storage from Task")

                # logger.info(f"Received Task: {event.model_dump_json(exclude_none=True, indent=2)}")
                yield event
            elif isinstance(event, Message):
                # logger.info(f"Received Message: {event}")
                yield event

    async def request(
            self,
            agent_name: str,
            query: str,
            context_id: str | None = None,
            metadata: dict | None = None) -> AsyncGenerator[Any, None]:
        """Process a single agent's response and stream events."""
        
        # Get task_id from context_storage if context_id is provided
        task_id = None
        if context_id and agent_name in self.agents_info:
            agents_info = self.agents_info[agent_name]
            context_storage = agents_info['context_storage']
            task_id = context_storage.get(context_id)

        async for event in self.send_message_streaming(
            agent_name, query, context_id, task_id, metadata
        ):
            logger.debug(f"[{agent_name}] Received event: {event}")
            yield event

            # Check for terminal states to stop consuming the stream
            if isinstance(event, TaskStatusUpdateEvent):
                if event.status.state in [
                        TaskState.completed,
                        TaskState.failed,
                        # TaskState.auth_required, # auth_required is NOT a terminal state if we want to handle auth
                        TaskState.canceled,
                        TaskState.unknown]:
                    logger.debug(f"[{agent_name}] Task reached terminal state: {event.status.state}")
                    break
