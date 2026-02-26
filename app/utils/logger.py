import asyncio
import inspect
import json
import os
import sys
from typing import Any, Dict

from loguru import logger

# Project root for relative paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))
)  # Adjust based on your structure (app/utils/logger.py -> project root)


def get_caller_info(stack_index: int = 2) -> str:
    """Get caller file:line (relative path). stack_index=2 skips cloud and this func."""
    stack = inspect.stack()
    if len(stack) <= stack_index:
        return "unknown:0"
    frame = stack[stack_index]
    filename = frame.filename
    relative_path = os.path.relpath(filename, PROJECT_ROOT).replace("\\", "/")
    line = frame.lineno
    return f"{relative_path}:{line}"


# Remove default sink and add configured ones
logger.remove()
# Stdout with pretty format (includes file:function:line by default)
logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# File logger
log_file_path = os.path.join(PROJECT_ROOT, "logs", "app.log")
logger.add(
    log_file_path,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
)

# Export
__all__ = ["logger"]
