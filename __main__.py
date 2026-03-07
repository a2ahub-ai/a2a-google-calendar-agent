
import asyncio
import sys
import os
import uvicorn
import contextlib
import base64
import json

from app.agent.server_mcp import (
    ListCalendarEvents,
    AddCalendarEvent,
    UpdateCalendarEvent,
    DeleteCalendarEvent,
    GetEventDetails,
)

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
from app.agent.server_agent import (
    AgentServer,
)
from app.agent.server_executor import (
    CalendarAgentExecutor,
)
from app.remote_agents import RoutingAgent

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
    list_calendar_events = ListCalendarEvents()
    list_calendar_events_skill = AgentSkill(
        id=BaseConfig.AGENT_ID,
        name=f"{list_calendar_events.name} Skill",
        description=f"{list_calendar_events.description}",
        tags=[
            "calendar",
            "events",
            "reminders",
            "schedule"],
        examples=[
            "tell me my events today",
            "show me today's schedule",
            "list my calendar events for tomorrow",
        ],
    )

    add_calendar_event = AddCalendarEvent()
    add_calendar_event_skill = AgentSkill(
        id=f"{BaseConfig.AGENT_ID}-add",
        name=f"{add_calendar_event.name} Skill",
        description=f"{add_calendar_event.description}",
        tags=[
            "calendar",
            "events",
            "create",
            "add",
            "schedule"
        ],
        examples=[
            "schedule a meeting tomorrow at 2pm",
            "add a dentist appointment for next Friday",
            "create an event called Team Standup"
        ],
    )

    update_calendar_event = UpdateCalendarEvent()
    update_calendar_event_skill = AgentSkill(
        id=f"{BaseConfig.AGENT_ID}-update",
        name=f"{update_calendar_event.name} Skill",
        description=f"{update_calendar_event.description}",
        tags=[
            "calendar",
            "events",
            "update",
            "modify",
            "change"
        ],
        examples=[
            "move my 2pm meeting to 3pm",
            "rename the Team Standup event",
            "change location of meeting"
        ],
    )

    delete_calendar_event = DeleteCalendarEvent()
    delete_calendar_event_skill = AgentSkill(
        id=f"{BaseConfig.AGENT_ID}-delete",
        name=f"{delete_calendar_event.name} Skill",
        description=f"{delete_calendar_event.description}",
        tags=[
            "calendar",
            "events",
            "delete",
            "remove",
            "cancel"
        ],
        examples=[
            "cancel my meeting at 2pm",
            "delete the Team Standup event",
            "remove the dentist appointment"
        ],
    )

    get_event_details = GetEventDetails()
    get_event_details_skill = AgentSkill(
        id=f"{BaseConfig.AGENT_ID}-get-details",
        name=f"{get_event_details.name} Skill",
        description=f"{get_event_details.description}",
        tags=[
            "calendar",
            "events",
            "details",
            "information",
            "query"
        ],
        examples=[
            "get details for my 2pm meeting",
            "show me the location of the Team Standup event",
            "when is my next dentist appointment"
        ],
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
        skills=[
            list_calendar_events_skill,
            add_calendar_event_skill,
            update_calendar_event_skill,
            delete_calendar_event_skill,
            get_event_details_skill,
        ],
        security_schemes={OAUTH_SCHEME_NAME: SecurityScheme(root=oauth_scheme)},
        # Declare that this scheme is required to use the agent's skills
        security=[
            {OAUTH_SCHEME_NAME: ['https://www.googleapis.com/auth/calendar']}
        ],
    )

    # Connect to remote agents (e.g., datetime parser) for concurrent processing
    routing_agent = None
    if BaseConfig.REMOTE_AGENT_ADDRESSES:
        addresses = [addr.strip() for addr in BaseConfig.REMOTE_AGENT_ADDRESSES.split(",") if addr.strip()]
        if addresses:
            try:
                routing_agent = await RoutingAgent.create(addresses)
                logger.info(f"📡 Connected to remote agents: {list(routing_agent.agents_info.keys())}")
            except Exception as e:
                logger.error(f"Failed to connect to remote agents: {e}")

    runner = AgentServer()
    # Use -X utf8 flag to ensure UTF-8 encoding for the subprocess on Windows
    python_cmd = ["python",
                  "-X",
                  "utf8",
                  "app/agent/server_mcp.py"] if sys.platform == 'win32' else ["python",
                                                                              "app/agent/server_mcp.py"]
    await runner.connect_to_stdio_server("calendar-agent", python_cmd)

    agent_executor = CalendarAgentExecutor(runner, agent_card, routing_agent=routing_agent)

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
        Route('/authenticate', endpoint=handle_auth, methods=['GET']),
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
