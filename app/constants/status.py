from enum import Enum


class Status(Enum):
    ERROR = 0
    SUCCESS = 1
    UNKNOWN = 1000

    LLM_CHAT_COMPLETION_ERROR = 2001
