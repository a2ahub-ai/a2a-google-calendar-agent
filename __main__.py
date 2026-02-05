
import asyncio
import sys
import os
import uvicorn
import contextlib
import base64
import json

# Force UTF-8 encoding for Windows to handle Vietnamese characters
if sys.platform == 'win32':
    # Set environment variable for subprocesses
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    BaseUser,
    SimpleUser,
)
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection, Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AuthorizationCodeOAuthFlow,
    OAuth2SecurityScheme,
    OAuthFlows,
    SecurityScheme,
)
from app.constants import AGENT_DESCRIPTION
from app.server_agent import (
    MCPClient,
)
from app.server_executor import (
    CalendarAgentExecutor,
)

from app.utils.logger import logger
from app.config.settings import BaseConfig
from app.auth import (
    verify_session_token,
    handle_authorize,
    handle_auth_callback,
    handle_token,
    REDIRECT_URI
)

DEFAULT_HOST = BaseConfig.HOST
DEFAULT_PORT = BaseConfig.PORT


class SessionJWTAuthBackend(AuthenticationBackend):
    async def authenticate(
        self, conn: HTTPConnection
    ) -> tuple[AuthCredentials, BaseUser] | None:
        if "Authorization" not in conn.headers:
            return None

        auth_header = conn.headers['Authorization']
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != 'bearer':
                return None

            payload = verify_session_token(token)
            if payload:
                # sub is the user_id
                user_id = payload.get("sub", "unknown")
                return AuthCredentials(["authenticated"]), SimpleUser(user_id)
        except Exception as e:
            logger.error(f"Authentication error: {e}")

        return None


async def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    skill = AgentSkill(
        id=BaseConfig.AGENT_ID,
        name="Calendar Skill",
        description="A skill for retrieving calendar events and reminders for today.",
        tags=[
            "calendar",
            "events",
            "reminders",
            "schedule"],
        examples=[
            "tell me my events today",
            "what's on my calendar today",
            "summarize my reminders today",
            "show me today's schedule"]
    )

    # Define OAuth2 security scheme.
    OAUTH_SCHEME_NAME = 'CalendarGoogleOAuth'
    oauth_scheme = OAuth2SecurityScheme(
        type='oauth2',
        description='OAuth2 for Google Calendar API',
        flows=OAuthFlows(
            authorization_code=AuthorizationCodeOAuthFlow(
                authorization_url=f'{BaseConfig.APP_URL}/authorize',
                token_url=f'{BaseConfig.APP_URL}/token',
                scopes={
                    'https://www.googleapis.com/auth/calendar': 'Access Google Calendar'
                },
            )
        ),
    )

    agent_card = AgentCard(
        name=BaseConfig.AGENT_NAME,
        description=AGENT_DESCRIPTION,
        url=BaseConfig.APP_URL,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        security_schemes={OAUTH_SCHEME_NAME: SecurityScheme(root=oauth_scheme)},
        # Declare that this scheme is required to use the agent's skills
        security=[
            {OAUTH_SCHEME_NAME: ['https://www.googleapis.com/auth/calendar']}
        ],
    )

    runner = MCPClient()
    # Use -X utf8 flag to ensure UTF-8 encoding for the subprocess on Windows
    python_cmd = ["python",
                  "-X",
                  "utf8",
                  "app/server_mcp.py"] if sys.platform == 'win32' else ["python",
                                                                        "app/server_mcp.py"]
    await runner.connect_to_stdio_server("calendar-agent", python_cmd)

    agent_executor = CalendarAgentExecutor(runner, agent_card)

    async def handle_auth(request: Request) -> PlainTextResponse:
        logger.info(f"Auth callback received: {request.url}")
        state = request.query_params.get('state')
        if state:
            await agent_executor.on_auth_callback(
                str(state), str(request.url)
            )
            return PlainTextResponse('Authentication successful. You can close this window.')
        return PlainTextResponse('Authentication failed: Missing state parameter.', status_code=400)

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    routes = a2a_app.routes()
    # Add OAuth routes
    routes.extend([
        Route('/authorize', endpoint=handle_authorize, methods=['GET']),
        Route('/auth/callback', endpoint=handle_auth_callback, methods=['GET']),
        Route('/token', endpoint=handle_token, methods=['POST']),
    ])

    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(
                AuthenticationMiddleware, backend=SessionJWTAuthBackend()
            )
        ],
    )

    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main(DEFAULT_HOST, DEFAULT_PORT))
