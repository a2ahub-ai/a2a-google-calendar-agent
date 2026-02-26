from typing import AsyncGenerator, List, Optional, Dict, Any, Union, AsyncIterator

from openai.types.chat import ChatCompletionMessageParam

from app.constants import ChatCompletionTypeEnum
from app.types import ChatCompletionStreamResponseType
from app.utils.logger import logger
from .base import LLMProvider


async def llm_quick_prompt(
    llm: LLMProvider,
    instruction: str,
    prompt: str,
) -> Optional[Dict[str, Any]]:
    """
    Quick prompt function for single completion

    Args:
        llm: LLM provider instance
        instruction: System instruction/context
        prompt: User prompt

    Returns:
        Dictionary with data, input_tokens, and output_tokens or None
    """
    llm_messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": instruction,
        },
        {"role": "user", "content": prompt},
    ]

    # Direct iteration works at runtime despite type checker confusion
    async for response in llm.chat_completion(  # type: ignore
        messages=llm_messages,
        response_format=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        temperature=0.7,
        top_p=1.0,
        reasoning_effort=None,
        max_tokens=None,
        stop=None,
        retry=None,
        args=None,
    ):
        if response["type"] == ChatCompletionTypeEnum.DONE:
            return {
                "data": response.get("data"),
                "input_tokens": response.get("input_tokens"),
                "output_tokens": response.get("output_tokens"),
            }

    return None


async def llm_stream_quick_prompt(
    llm: LLMProvider,
    instruction: Optional[str] = None,
    prompt: Optional[str] = None,
    messages: Optional[List[ChatCompletionMessageParam]] = None,
) -> AsyncGenerator[ChatCompletionStreamResponseType, None]:
    """
    Streaming quick prompt function

    Args:
        llm: LLM provider instance
        instruction: System instruction/context (optional)
        prompt: User prompt (optional)
        messages: Pre-built messages list (optional)

    Yields:
        ChatCompletionStreamResponseType dictionaries
    """
    llm_messages: List[ChatCompletionMessageParam] = []

    if instruction:
        llm_messages.append({
            "role": "system",
            "content": instruction,
        })

    if prompt:
        llm_messages.append({
            "role": "user",
            "content": prompt,
        })

    if messages:
        llm_messages = messages

    # Direct iteration works at runtime despite type checker confusion
    async for response in llm.chat_completion_stream(  # type: ignore
        messages=llm_messages,
        response_format=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        temperature=0.7,
        top_p=1.0,
        reasoning_effort=None,
        max_tokens=None,
        stop=None,
        retry=None,
        args=None,
    ):
        if response["type"] == ChatCompletionTypeEnum.CONTENT:
            yield {"type": ChatCompletionTypeEnum.CONTENT, "data": response.get("data")}
        elif response["type"] == ChatCompletionTypeEnum.DONE:
            yield {
                "type": ChatCompletionTypeEnum.DONE,
                "data": response.get("data"),
                "input_tokens": response.get("input_tokens"),
                "output_tokens": response.get("output_tokens"),
            }
            return


# Export functions for easier imports
__all__ = ["llm_quick_prompt", "llm_stream_quick_prompt", "LLMProvider"]
