from datetime import datetime, timezone, timedelta
from typing import Any, Dict
import json
import sys

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.config.settings import BaseConfig
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult

# Initialize FastMCP server
mcp = FastMCP(f"{BaseConfig.SERVICE_NAME}-mcp-server")


class ListCalendarEvents(Tool):
    name: str = "list_calendar_events"
    description: str = "Retrieve calendar events and reminders. Supports natural language queries for time ranges. If no time provided, defaults to today."
    parameters: Dict[str, Any] = {
        "type": "object",
        "description": "Get calendar events",
        "properties": {
            "time_min": {"type": "string", "description": "Start time in ISO 8601 format (e.g. 2026-01-08T00:00:00Z). Optional."},
            "time_max": {"type": "string", "description": "End time in ISO 8601 format. Optional."},
            "max_results": {"type": "integer", "description": "Max events to return. Default 10."}
        },
        "additionalProperties": True  # Allow hidden auth args
    }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        sys.stderr.write(f"[calendar-agent-mcp-server] Retrieving calendar events: {arguments}\n")

        auth_info = arguments.get("__auth_info")
        if not auth_info:
            return ToolResult(
                content=[{"type": "text", "text": "Error: Missing authorization information. Please authenticate first."}])

        try:
            creds = Credentials.from_authorized_user_info(auth_info)
            service = build('calendar', 'v3', credentials=creds)

            time_min = arguments.get('time_min')
            if not time_min:
                # Default to start of today in UTC if not specified
                now = datetime.now(timezone.utc)
                start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
                time_min = start_of_day.isoformat()

            # If time_max not specified, maybe end of day? Or just next 24h?
            # Let's leave time_max open if not provided, or default to end of day.

            kwargs = {
                'calendarId': 'primary',
                'timeMin': time_min,
                'maxResults': arguments.get('max_results', 10),
                'singleEvents': True,
                'orderBy': 'startTime'
            }

            if arguments.get('time_max'):
                kwargs['timeMax'] = arguments.get('time_max')

            events_result = service.events().list(**kwargs).execute()
            events = events_result.get('items', [])

            if not events:
                return ToolResult(content=[{"type": "text", "text": "No events found."}])

            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                formatted_events.append(f"- {event.get('summary', 'No Title')} ({start})")

            return ToolResult(structured_content=events, content=[
                              {"type": "text", "text": "\n".join(formatted_events)}])

        except Exception as e:
            sys.stderr.write(f"Error calling Google Calendar API: {e}\n")
            return ToolResult(content=[{"type": "text", "text": f"Error retrieving calendar events: {str(e)}"}])


# Add the ListCalendarEvents to the server
mcp.add_tool(ListCalendarEvents())

if __name__ == "__main__":
    sys.stderr.write("Starting Calendar Agent MCP Server with stdio transport\n")
    mcp.run(transport="stdio")
