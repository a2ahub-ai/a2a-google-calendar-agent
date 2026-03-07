from enum import Enum


class ChatCompletionTypeEnum(Enum):
    CONTENT = 0
    DATA = 1
    FUNCTION_CALLING = 2
    THINK = 3
    DONE = 4
    ERROR = 5
    TIMEOUT = 6


AGENT_DESCRIPTION = (
    "A Google Calendar assistant that can list, create, update, delete, "
    "and search calendar events."
)
