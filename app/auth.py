import os
import json
import uuid
import time
import base64
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlencode, parse_qs, quote, unquote

import redis
import jwt
from starlette.responses import RedirectResponse, JSONResponse
from starlette.requests import Request
from google_auth_oauthlib.flow import Flow
from dotenv import load_dotenv

from app.config.settings import BaseConfig

load_dotenv()

# Logger
logger = logging.getLogger("a2a.auth")

# Config
GOOGLE_CLIENT_ID = BaseConfig.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = BaseConfig.GOOGLE_CLIENT_SECRET
# The URL where this server is running
SERVER_URL = BaseConfig.APP_URL
# The internal callback URL that Google redirects to
REDIRECT_URI = f"{SERVER_URL}/auth/callback"

REDIS_HOST = BaseConfig.REDIS_HOST
REDIS_PORT = BaseConfig.REDIS_PORT
REDIS_DB = BaseConfig.REDIS_DB

JWT_SECRET = BaseConfig.JWT_SECRET
JWT_ALGORITHM = "HS256"

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
SESSION_EXPIRY_SECONDS = BaseConfig.SESSION_EXPIRY_SECONDS

# Redis Client
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None


def create_session_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iss": SERVER_URL,
        "iat": int(time.time()),
        "exp": int(time.time()) + SESSION_EXPIRY_SECONDS
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except Exception as e:
        logger.warning(f"Invalid token: {e}")
        return None


def store_google_creds(user_id: str, creds: Dict[str, Any]):
    if not redis_client:
        return
    # Store the full credentials dict JSON serialization
    key = f"google_creds:{user_id}"
    redis_client.set(key, json.dumps(creds), ex=SESSION_EXPIRY_SECONDS)


def get_google_creds(user_id: str) -> Optional[Dict[str, Any]]:
    if not redis_client:
        return None
    key = f"google_creds:{user_id}"
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return None

# OAuth Routes Logic


async def handle_authorize(request: Request):
    """
    Initiates the OAuth flow.
    Query params:
    - client_id: The ID of the A2A client (cli)
    - redirect_uri: Where to redirect the CLI after success
    - state: Client state
    - code_challenge / method: PKCE (optional, can implement later)
    """
    params = request.query_params
    cli_redirect_uri = params.get("redirect_uri")
    cli_state = params.get("state")

    if not cli_redirect_uri:
        return JSONResponse({"error": "Missing redirect_uri"}, status_code=400)

    # We encode the CLI's return info into the 'state' param passed to Google
    # So when Google calls back, we know where to send the user eventually.
    internal_state = json.dumps({
        "cli_redirect_uri": cli_redirect_uri,
        "cli_state": cli_state,
        "original_client_id": params.get("client_id")
    })
    encoded_state = base64.urlsafe_b64encode(internal_state.encode()).decode()

    # Create flow instance to generate the authorization URL
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        state=encoded_state,
        prompt='consent'  # Force consent to ensure we get refresh token
    )

    return RedirectResponse(auth_url)


async def handle_auth_callback(request: Request):
    """
    Callback from Google.
    """
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code:
        return JSONResponse({"error": "Missing code"}, status_code=400)

    try:
        # 1. Exchange code for Google Tokens
        flow = Flow.from_client_config(
            client_config={
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        # Allow scope mismatch since Google sometimes adds extra scopes (like openid, email)
        os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
        flow.fetch_token(code=code)
        creds = flow.credentials

        # 2. Decode state to find where to go next
        state_json = base64.urlsafe_b64decode(state).decode()
        state_data = json.loads(state_json)
        cli_redirect_uri = state_data.get("cli_redirect_uri")
        cli_state = state_data.get("cli_state")

        # 3. Create a User ID (or use email from id_token if available)
        # Generate a unique session ID (user_id) for this authentication event
        user_id = str(uuid.uuid4())

        # Store Google Creds in Redis (Vault)
        creds_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes
        }
        store_google_creds(user_id, creds_data)

        # 4. Generate Session Token (for the CLI)
        session_token = create_session_token(user_id)

        # 5. Generate a temporary "Authorization Code" for the CLI (which exchanges it for the session token)
        # This emulates standard OAuth implementation
        auth_code = f"auth_code_{uuid.uuid4()}"
        if redis_client:
            redis_client.setex(f"auth_code:{auth_code}", 300, session_token)

        # 6. Redirect to CLI
        redirect_url = f"{cli_redirect_uri}?code={auth_code}&state={cli_state}"
        return RedirectResponse(redirect_url)

    except Exception as e:
        logger.error(f"Auth callback error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def handle_token(request: Request):
    """
    Exchange auth_code for session_token
    """
    form = await request.form()
    code = form.get("code")

    if not code:
        return JSONResponse({"error": "Missing code"}, status_code=400)

    # Retrieve session token from Redis using auth_code
    if not redis_client:
        return JSONResponse({"error": "Server error"}, status_code=500)

    session_token = redis_client.get(f"auth_code:{code}")
    if not session_token:
        return JSONResponse({"error": "Invalid code"}, status_code=400)

    redis_client.delete(f"auth_code:{code}")

    return JSONResponse({
        "access_token": session_token,
        "token_type": "Bearer",
        "expires_in": 3600 * 24 * 365
    })
