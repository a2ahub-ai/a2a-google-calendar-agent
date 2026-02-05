import os
import sys
from dotenv import load_dotenv

dotenv_path = os.path.join("app/../.env")
load_dotenv(dotenv_path)


def boolean_parser(input_string):
    if input_string is not None:
        input_string = input_string.lower()
        if input_string == "true":
            return True
        elif input_string == "false":
            return False
    return False


class BaseConfig:
    SERVICE_NAME = os.environ.get("SERVICE_NAME", "a2a-google-calendar-agent")
    AGENT_ID = os.environ.get("AGENT_ID", "calendar_agent")
    AGENT_NAME = os.environ.get("AGENT_NAME", "Calendar Agent")
    ENV = os.environ.get("ENV", "production")
    HOST = os.environ.get("HOST", "localhost")
    PORT = int(os.environ.get("PORT", 10001))
    APP_URL = os.environ.get("APP_URL", f"http://{HOST}:{PORT}")
    DEBUG = boolean_parser(os.environ.get("DEBUG", "false"))
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    DISABLE_LOG = boolean_parser(os.environ.get("DISABLE_LOG", "false"))
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
    REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
    JWT_SECRET = os.environ.get("JWT_SECRET", "super-secret-jwt-key")
    SESSION_EXPIRY_SECONDS = int(os.environ.get("SESSION_EXPIRY_SECONDS", 365 * 24 * 60 * 60))
