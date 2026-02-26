from a2a.types import (
    AgentCard
)

from typing import Any, Optional, TypedDict, Required, NotRequired, TypedDict, Dict, TYPE_CHECKING

from app.constants import ChatCompletionTypeEnum

if TYPE_CHECKING:
    from app.remote_agents import RemoteAgentConnections

class AgentInfo(TypedDict):
    remote_agent_connections: "RemoteAgentConnections"
    context_storage: Dict[str, str]
    card: AgentCard


class ChatCompletionStreamResponseType(TypedDict):
    type: ChatCompletionTypeEnum
    data: Required[Optional[Any]]
    input_tokens: NotRequired[Optional[int]]
    output_tokens: NotRequired[Optional[int]]


class FunctionCallingResponseType(TypedDict):
    name: str
    index: int
    id: str
    arguments: str