import asyncio
import base64
import os
import urllib.parse
import webbrowser

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from pathlib import Path
from uuid import uuid4

import asyncclick as click
import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.types import (
    FilePart,
    FileWithBytes,
    GetTaskRequest,
    JSONRPCErrorResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskQueryParams,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

from app.utils.logger import logger

# --- OAuth Client Logic ---


class OAuthClient:
    def __init__(self, agent_card, profile: str = "default"):
        self.agent_card = agent_card
        self.profile = profile
        self.storage_path = Path(".client_storage") / profile / "session_token"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.token = None
        self._auth_code = None
        self._auth_event = asyncio.Event()

    def get_token(self) -> str:
        if self.token:
            return self.token

        # Try loading from disk
        if self.storage_path.exists():
            try:
                self.token = self.storage_path.read_text().strip()
                return self.token
            except Exception:
                pass
        return ""

    def _find_oauth_flow(self):
        # Look for OAuth2 security scheme in agent card
        if not self.agent_card.security_schemes:
            return None

        for scheme_name, scheme in self.agent_card.security_schemes.items():
            if scheme.root.type == 'oauth2':
                flows = scheme.root.flows
                if flows.authorization_code:
                    return flows.authorization_code
        return None

    async def authenticate(self):
        flow_config = self._find_oauth_flow()
        if not flow_config:
            logger.error("No OAuth2 authorization code flow found in Agent Card.")
            return

        logger.info("Initiating Authentication...")

        # Start local callback server
        import socket
        sock = socket.socket()
        sock.bind(('localhost', 0))
        port = sock.getsockname()[1]
        sock.close()

        callback_uri = f"http://localhost:{port}/callback"

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args): pass

            def do_GET(self):
                try:
                    query = urllib.parse.urlparse(self.path).query
                    params = urllib.parse.parse_qs(query)
                    code = params.get('code', [None])[0]
                    if code:
                        loop.call_soon_threadsafe(future.set_result, code)
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b"<h1>Authentication successful!</h1><p>You can close this window.</p>")
                    else:
                        loop.call_soon_threadsafe(future.set_result, None)
                        self.send_response(400)
                except Exception as e:
                    loop.call_soon_threadsafe(future.set_exception, e)

        server = HTTPServer(('localhost', port), Handler)
        server_thread = Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            state = base64.urlsafe_b64encode(os.urandom(16)).decode()
            params = {
                "client_id": "a2a-cli",
                "redirect_uri": callback_uri,
                "response_type": "code",
                "state": state
            }

            # Use authorization endpoint from Agent Card
            auth_endpoint = flow_config.authorization_url
            auth_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"

            logger.info(f"Opening browser: {auth_url}")
            webbrowser.open(auth_url)

            logger.info("Waiting for callback...")
            code = await future

            if not code:
                raise Exception("Authentication failed: No code received")

            # Exchange code for token using token endpoint from Agent Card
            token_endpoint = flow_config.token_url
            async with httpx.AsyncClient() as client:
                resp = await client.post(token_endpoint, data={"code": code})
                resp.raise_for_status()
                data = resp.json()
                self.token = data["access_token"]
                self.storage_path.write_text(self.token)
                logger.info("Authentication successful & token saved.")

        finally:
            server.shutdown()
            server_thread.join()


@click.command()
@click.option('--agent', default='http://localhost:8083')
@click.option(
    '--bearer-token',
    help='Bearer token for authentication.',
    envvar='A2A_CLI_BEARER_TOKEN',
)
@click.option('--session', default=0)
@click.option('--profile', default='default', help='Profile name for session storage')
@click.option('--history', default=False)
@click.option('--use_push_notifications', default=False)
@click.option('--push_notification_receiver', default='http://localhost:5000')
@click.option('--header', multiple=True)
@click.option(
    '--automatic-authentication',
    default=True,
    type=bool,
    help='Whether to automatically open browser for authentication.',
)
@click.option(
    '--enabled_extensions',
    default='',
    help='Comma-separated list of extension URIs to enable (sets X-A2A-Extensions header).',
)
async def cli(
    agent,
    bearer_token,
    session,
    profile,
    history,
    use_push_notifications: bool,
    push_notification_receiver: str,
    header,
    automatic_authentication: bool,
    enabled_extensions,
):
    headers = {h.split('=')[0]: h.split('=')[1] for h in header}

    # Auth Logic
    if bearer_token:
        headers['Authorization'] = f'Bearer {bearer_token}'
    elif 'Authorization' in headers:
        pass  # Token already provided via header
    else:
        # Try to get token from storage or flow
        try:
            # We initialize OAuthClient differently now.
            # First, fetch the card to get auth endpoints.
            async with httpx.AsyncClient(timeout=30, headers=headers) as httpx_client:
                card_resolver = A2ACardResolver(httpx_client, agent)
                card = await card_resolver.get_agent_card()

            auth_client = OAuthClient(agent_card=card, profile=profile)
            token = auth_client.get_token()

            if not token and automatic_authentication:
                await auth_client.authenticate()
                token = auth_client.get_token()

            if token:
                headers['Authorization'] = f'Bearer {token}'
            else:
                logger.warning("Warning: No authentication token available.")

        except Exception as e:
            logger.error(f"Authentication warning/error: {e}")
            logger.info("Continuing without auth header (or with whatever was provided)...")

    # --- Add enabled_extensions support ---
    # If the user provided a comma-separated list of extensions,
    # we set the X-A2A-Extensions header.
    # This allows the server to know which extensions are activated.
    # Note: We assume the extensions are supported by the server.
    # This headers will be used by the server to activate the extensions.
    # If the server does not support the extensions, it will ignore them.
    if enabled_extensions:
        ext_list = [
            ext.strip() for ext in enabled_extensions.split(',') if ext.strip()
        ]
        if ext_list:
            headers[HTTP_EXTENSION_HEADER] = ', '.join(ext_list)
    logger.info(f'Will use headers: {headers}')
    async with httpx.AsyncClient(timeout=30, headers=headers) as httpx_client:
        card_resolver = A2ACardResolver(httpx_client, agent)
        card = await card_resolver.get_agent_card()

        logger.info('======= Agent Card ========')
        logger.info(card.model_dump_json(exclude_none=True))

        notif_receiver_parsed = urllib.parse.urlparse(
            push_notification_receiver
        )
        notification_receiver_host = notif_receiver_parsed.hostname
        notification_receiver_port = notif_receiver_parsed.port

        if use_push_notifications:
            from .push_notification_listener import (
                PushNotificationListener,
            )

            push_notification_listener = PushNotificationListener(
                host=notification_receiver_host,
                port=notification_receiver_port,
            )
            push_notification_listener.start()

        client = A2AClient(httpx_client, agent_card=card)

        continue_loop = True
        streaming = card.capabilities.streaming
        # If user explicitly provided 0 (default) as session, generate one.
        # But if they provided a string via command line that looks like an integer > 0, we use it...
        # Wait, session arg is default=0.
        context_id = str(session) if session != 0 else uuid4().hex

        while continue_loop:
            logger.info('=========  starting a new task ======== ')
            continue_loop, _, task_id = await completeTask(
                client,
                streaming,
                use_push_notifications,
                notification_receiver_host,
                notification_receiver_port,
                None,
                context_id,
                profile,
            )

            if history and continue_loop:
                logger.info('========= history ======== ')
                task_response = await client.get_task(
                    {'id': task_id, 'historyLength': 10}
                )
                logger.info(
                    task_response.model_dump_json(
                        include={'result': {'history': True}}
                    )
                )


async def completeTask(
    client: A2AClient,
    streaming,
    use_push_notifications: bool,
    notification_receiver_host: str,
    notification_receiver_port: int,
    task_id,
    context_id,
    profile="default",
):
    prompt_text = await click.prompt(
        '\nWhat do you want to send to the agent? (:q or quit to exit)'
    )
    if prompt_text == ':q' or prompt_text == 'quit':
        return False, None, None

    message = Message(
        role='user',
        parts=[TextPart(text=prompt_text)],
        message_id=str(uuid4()),
        task_id=task_id,
        context_id=context_id,
    )

    file_path = await click.prompt(
        'Select a file path to attach? (press enter to skip)',
        default='',
        show_default=False,
    )
    if file_path and file_path.strip() != '':
        with open(file_path, 'rb') as f:
            file_content = base64.b64encode(f.read()).decode('utf-8')
            file_name = os.path.basename(file_path)

        message.parts.append(
            Part(
                root=FilePart(
                    file=FileWithBytes(name=file_name, bytes=file_content)
                )
            )
        )

    payload = MessageSendParams(
        id=str(uuid4()),
        message=message,
        configuration=MessageSendConfiguration(
            accepted_output_modes=['text'],
        ),
    )

    if use_push_notifications:
        payload['pushNotification'] = {
            'url': f'http://{notification_receiver_host}:{notification_receiver_port}/notify',
            'authentication': {
                'schemes': ['bearer'],
            },
        }

    taskResult = None
    message = None
    task_completed = False
    if streaming:
        response_stream = client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=payload,
            )
        )
        async for result in response_stream:
            if isinstance(result.root, JSONRPCErrorResponse):
                logger.error(
                    f'Error: {result.root.error}, context_id: {context_id}, task_id: {task_id}'
                )
                return False, context_id, task_id
            event = result.root.result
            context_id = event.context_id
            if isinstance(event, Task):
                task_id = event.id
            elif isinstance(event, TaskStatusUpdateEvent) or isinstance(
                event, TaskArtifactUpdateEvent
            ):
                task_id = event.task_id
                if isinstance(event, TaskArtifactUpdateEvent):
                    if event.artifact and event.artifact.name == "token" and event.artifact.parts:
                        for part in event.artifact.parts:
                            if hasattr(part.root, 'text'):
                                # Save the token
                                token = part.root.text
                                storage_path = Path(".client_storage") / profile / "session_token"
                                storage_path.parent.mkdir(parents=True, exist_ok=True)
                                storage_path.write_text(token)
                                logger.info(f"\n[Artifact] Token received and saved to {storage_path}")
                                break

                if (
                    isinstance(event, TaskStatusUpdateEvent)
                    and event.status.state == 'completed'
                ):
                    task_completed = True
            elif isinstance(event, Message):
                message = event
            logger.info(f'stream event => {event.model_dump_json(exclude_none=True)}')
        # Upon completion of the stream. Retrieve the full task if one was made.
        if task_id and not task_completed:
            taskResultResponse = await client.get_task(
                GetTaskRequest(
                    id=str(uuid4()),
                    params=TaskQueryParams(id=task_id),
                )
            )
            if isinstance(taskResultResponse.root, JSONRPCErrorResponse):
                logger.error(
                    f'Error: {taskResultResponse.root.error}, context_id: {context_id}, task_id: {task_id}'
                )
                return False, context_id, task_id
            taskResult = taskResultResponse.root.result
    else:
        try:
            # For non-streaming, assume the response is a task or message.
            event = await client.send_message(
                SendMessageRequest(
                    id=str(uuid4()),
                    params=payload,
                )
            )
            event = event.root.result
        except Exception as e:
            logger.error(f'Failed to complete the call: {e}')
        if not context_id:
            context_id = event.context_id
        if isinstance(event, Task):
            if not task_id:
                task_id = event.id
            taskResult = event
        elif isinstance(event, Message):
            message = event

    if message:
        logger.info(f'\n{message.model_dump_json(exclude_none=True)}')
        return True, context_id, task_id
    if taskResult:
        # Don't print the contents of a file.
        task_content = taskResult.model_dump_json(
            exclude={
                'history': {
                    '__all__': {
                        'parts': {
                            '__all__': {'file'},
                        },
                    },
                },
            },
            exclude_none=True,
        )
        logger.info(f'\n{task_content}')
        # if the result is that more input is required, loop again.
        state = TaskState(taskResult.status.state)
        if state.name == TaskState.input_required.name:
            return (
                await completeTask(
                    client,
                    streaming,
                    use_push_notifications,
                    notification_receiver_host,
                    notification_receiver_port,
                    task_id,
                    context_id,
                ),
                context_id,
                task_id,
            )
        # task is complete
        return True, context_id, task_id
    # Failure case, shouldn't reach
    return True, context_id, task_id


if __name__ == '__main__':
    asyncio.run(cli())
