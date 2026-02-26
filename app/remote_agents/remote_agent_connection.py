from collections.abc import Callable
from typing import AsyncGenerator

import httpx
from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendStreamingMessageRequest,
    SendMessageResponse,
    SendStreamingMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv

from app.config.settings import BaseConfig
from app.remote_agents.auth import OAuthClient

load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        self._httpx_client = httpx.AsyncClient(timeout=30)

        # Initialize OAuth Client
        profile = BaseConfig.PROFILE
        self.oauth_client = OAuthClient(agent_card, agent_card.name, profile=profile)

        # Try to get existing token
        token = self.oauth_client.get_token()
        if token:
            self._httpx_client.headers["Authorization"] = f"Bearer {token}"

        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card

    async def authenticate(self):
        """Perform authentication if no token exists."""
        if not self.oauth_client.token and BaseConfig.AUTO_AUTH_MODE:
            await self.oauth_client.authenticate()
            token = self.oauth_client.get_token()
            if token:
                self._httpx_client.headers["Authorization"] = f"Bearer {token}"
            else:
                pass  # logger.warning("Warning: No authentication token available even after attempt.")
        elif not self.oauth_client.token:
            # If token is missing and auto-auth is disabled, we proceed without auth header
            pass

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        return await self.agent_client.send_message(message_request)

    async def send_message_streaming(
            self, message_request: SendStreamingMessageRequest) -> AsyncGenerator[SendStreamingMessageResponse, None]:
        """Send a message with streaming support"""
        async for chunk in self.agent_client.send_message_streaming(
            message_request
        ):
            yield chunk
