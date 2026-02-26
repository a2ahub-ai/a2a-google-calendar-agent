import asyncio
import base64
import os
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from pathlib import Path

import httpx
from a2a.types import AgentCard

from app.utils.logger import logger


class OAuthClient:
    def __init__(self, agent_card: AgentCard, agent_name: str, profile: str = "default"):
        self.agent_card = agent_card
        self.profile = profile
        self.agent_name = agent_name
        # Sanitize agent name for folder path
        safe_agent_name = "".join([c for c in agent_name if c.isalnum() or c in (' ', '-', '_')]).strip()
        # Modified structure: .client_storage/<agent_name>/<profile>/session_token
        self.storage_path = Path(".client_storage") / safe_agent_name / self.profile / "session_token"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.token = None

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
            logger.info("No OAuth2 authorization code flow found in Agent Card.")
            return

        logger.info(f"Initiating Authentication for {self.agent_name}...")

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
                "client_id": "olli-agent-client",  # Changed client_id just in case, but usually depends on provider
                "redirect_uri": callback_uri,
                "response_type": "code",
                "state": state
            }

            # Use authorization endpoint from Agent Card
            auth_endpoint = flow_config.authorization_url
            auth_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"

            logger.info(f"Opening browser: {auth_url}")
            webbrowser.open(auth_url)

            logger.info(f"Waiting for callback from {self.agent_name}...")
            # Wait for the future with a timeout (e.g. 5 minutes)
            code = await asyncio.wait_for(future, timeout=300)

            if not code:
                raise Exception("Authentication failed: No code received")

            # Exchange code for token using token endpoint from Agent Card
            token_endpoint = flow_config.token_url
            async with httpx.AsyncClient() as client:
                resp = await client.post(token_endpoint, data={
                    "code": code,
                    "client_id": "olli-agent-client",  # Should match the one sent in auth url
                    "grant_type": "authorization_code",
                    "redirect_uri": callback_uri
                })
                resp.raise_for_status()
                data = resp.json()
                self.token = data.get("access_token")
                if self.token:
                    self.storage_path.write_text(self.token)
                    logger.info(f"Authentication successful for {self.agent_name} & token saved.")
                else:
                    logger.error("No access_token in response")

        except asyncio.TimeoutError:
            logger.error("Authentication timed out.")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
        finally:
            server.shutdown()
            server_thread.join()
