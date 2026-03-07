from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
import json
import sys

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.config.settings import BaseConfig
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from app.utils.logger import logger

# Initialize FastMCP server
mcp = FastMCP(f"{BaseConfig.SERVICE_NAME}-mcp-server")


def _get_calendar_service(auth_info: Dict[str, Any]):
    """Build and return a Google Calendar API service from auth info."""
    creds = Credentials.from_authorized_user_info(auth_info)
    return build('calendar', 'v3', credentials=creds)


def _format_timezone_offset(tz_offset: int | float) -> str:
    """Format a numeric timezone offset (e.g. 7, -5, 5.5) into an RFC 3339 suffix like '+07:00', '-05:00', '+05:30'."""
    sign = '+' if tz_offset >= 0 else '-'
    abs_offset = abs(tz_offset)
    hours = int(abs_offset)
    minutes = int((abs_offset - hours) * 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _ensure_rfc3339(dt_str: Optional[str], tz_offset: int | float | None = None) -> Optional[str]:
    """Ensure a datetime string is RFC 3339 compliant for the Google Calendar API.

    The datetime parser returns naive timestamps like '2026-02-26T00:00:00'.
    Google Calendar API requires a timezone offset.  When *tz_offset* is
    provided (e.g. 7 for UTC+7) we append the corresponding offset instead
    of defaulting to 'Z' (UTC).
    """
    if not dt_str:
        return dt_str
    # Already has timezone info (Z, +HH:MM, -HH:MM)
    if dt_str.endswith('Z') or '+' in dt_str[10:] or dt_str[10:].count('-') > 0:
        tail = dt_str.split('T')[-1] if 'T' in dt_str else dt_str
        if 'Z' in tail or '+' in tail or tail.count('-') > 0:
            return dt_str
    if tz_offset is not None:
        return dt_str + _format_timezone_offset(tz_offset)
    return dt_str + 'Z'


def _extract_datetime_range(arguments: Dict[str, Any]) -> tuple[Optional[str], Optional[str], int | float | None]:
    """Extract start/end times from the datetime_parser object.

    The datetime_parser is injected by the executor from the remote
    a2a-datetime-parser-agent (single_time_mode=False).  Its format is:
        {
            "time_range": {
                "start_date": {"datetime": "2026-02-27T12:00:00"},
                "end_date":   {"datetime": "2026-02-27T12:59:59"}
            }
        }

    The datetime_parser data is **optional** for every tool.  If it is
    absent or malformed the caller is expected to fall back to its own
    default time range.

    Returns (time_min, time_max, tz_offset) – RFC 3339-compliant timestamps
    (with timezone) or None, plus the numeric timezone offset.
    """
    time_min: Optional[str] = None
    time_max: Optional[str] = None

    # Extract timezone offset injected by the executor
    tz_offset: int | float | None = arguments.get('timezone_offset')

    datetime_parser = arguments.get('datetime_parser')
    if datetime_parser and isinstance(datetime_parser, dict):
        time_range = datetime_parser.get('time_range')
        if time_range and isinstance(time_range, dict):
            start_date = time_range.get('start_date')
            end_date = time_range.get('end_date')
            if start_date and isinstance(start_date, dict):
                time_min = start_date.get('datetime')
            if end_date and isinstance(end_date, dict):
                time_max = end_date.get('datetime')

    return _ensure_rfc3339(time_min, tz_offset), _ensure_rfc3339(time_max, tz_offset), tz_offset


def _filter_event_fields(event: Dict[str, Any]) -> Dict[str, Any]:
    """Filter event dictionary to keep only essential fields."""
    keep_keys = {'kind', 'id', 'status', 'summary', 'creator', 'start', 'end'}
    return {k: v for k, v in event.items() if k in keep_keys}


def _format_event_summary(event: Dict[str, Any]) -> str:
    """Return a human-readable one-line summary of a calendar event."""
    summary = event.get('summary', 'No Title')
    start = event.get('start', {}).get('dateTime', event.get('start', {}).get('date', ''))
    end = event.get('end', {}).get('dateTime', event.get('end', {}).get('date', ''))
    location = event.get('location', '')
    event_id = event.get('id', '')
    parts = [f"- {summary}"]
    if start:
        parts.append(f"  Start: {start}")
    if end:
        parts.append(f"  End: {end}")
    if location:
        parts.append(f"  Location: {location}")
    parts.append(f"  ID: {event_id}")
    return "\n".join(parts)


# ─────────────────────────────────────────────
#  1. List Calendar Events
# ─────────────────────────────────────────────
class ListCalendarEvents(Tool):
    name: str = "list_calendar_events"
    description: str = (
        "Retrieve calendar events within a time range. "
        "Use this for queries like 'show my schedule today', "
        "'what meetings do I have tomorrow', 'events this week', etc. "
        "If no time is provided, defaults to today."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "description": "List calendar events in a time range",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "Maximum number of events to return. Default 10."
            },
            "timezone_offset": {
                "type": "number",
            },
        },
        "required": ["timezone_offset"],
        "additionalProperties": True,
    }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        logger.debug(f"[calendar-mcp] list_calendar_events args: {arguments}")

        auth_info = arguments.get("__auth_info")
        if not auth_info:
            return ToolResult(
                content=TextContent(
                    type="text",
                    text="Error: Missing authorization information. Please authenticate first."))

        try:
            service = _get_calendar_service(auth_info)
            time_min, time_max, tz_offset = _extract_datetime_range(arguments)

            # Default to today if no time range resolved
            if not time_min:
                if tz_offset is not None:
                    tz = timezone(timedelta(hours=tz_offset))
                else:
                    tz = timezone.utc
                now = datetime.now(tz)
                time_min = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

            kwargs = {
                'calendarId': 'primary',
                'timeMin': time_min,
                'maxResults': arguments.get('max_results', 10),
                'singleEvents': True,
                'orderBy': 'startTime',
            }
            if time_max:
                kwargs['timeMax'] = time_max

            events_result = service.events().list(**kwargs).execute()
            logger.debug(f"Google Calendar API response: {events_result}")
            events = events_result.get('items', [])

            if not events:
                return ToolResult(content=TextContent(type="text", text="No events found in the specified time range."))

            formatted = [_format_event_summary(e) for e in events]
            filtered_events = [_filter_event_fields(e) for e in events]
            return ToolResult(
                structured_content={
                    "status": "success",
                    "msg": f"Retrieved successfully {len(events)} event(s).",
                    "events": filtered_events},
                content=TextContent(type="text", text="\n".join(formatted)),
            )

        except Exception as e:
            sys.stderr.write(f"[calendar-mcp] list error: {e}\n")
            return ToolResult(content=TextContent(type="text", text=f"Error retrieving calendar events: {str(e)}"))


mcp.add_tool(ListCalendarEvents())


# ─────────────────────────────────────────────
#  2. Create / Add Calendar Event
# ─────────────────────────────────────────────
class AddCalendarEvent(Tool):
    name: str = "add_calendar_event"
    description: str = (
        "Create a new event on the calendar. "
        "Use this for requests like 'schedule a meeting tomorrow at 2pm', "
        "'create an event called Team Standup on Monday from 9am to 9:30am', etc."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Title/name of the event."
            },
            "description": {
                "type": "string",
                "description": "Optional description / notes for the event."
            },
            "start_time": {
                "type": "string",
                "description": "Event start time in ISO 8601 format."
            },
            "end_time": {
                "type": "string",
                "description": "Event end time in ISO 8601 format."
            },
            "location": {
                "type": "string",
                "description": "Optional location of the event."
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of attendee email addresses."
            },
            "timezone_offset": {
                "type": "number",
            },
        },
        "required": ["summary", "timezone_offset"],
        "additionalProperties": True,
    }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        sys.stderr.write(f"[calendar-mcp] add_calendar_event args: {arguments}\n")

        auth_info = arguments.get("__auth_info")
        if not auth_info:
            return ToolResult(content=TextContent(type="text", text="Error: Missing authorization information."))

        try:
            service = _get_calendar_service(auth_info)

            # Get times exclusively from datetime_parser
            start_time, end_time, tz_offset = _extract_datetime_range(arguments)
            tz_str = _format_timezone_offset(tz_offset) if tz_offset is not None else '+00:00'

            # If still missing times, default to now + 1 hour
            if not start_time:
                if tz_offset is not None:
                    tz = timezone(timedelta(hours=tz_offset))
                else:
                    tz = timezone.utc
                now = datetime.now(tz)
                start_time = now.isoformat()
                if not end_time:
                    end_time = (now + timedelta(hours=1)).isoformat()
            if not end_time:
                try:
                    st = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_time = (st + timedelta(hours=1)).isoformat()
                except Exception:
                    end_time = start_time

            event_body: Dict[str, Any] = {
                'summary': arguments.get('summary', 'Untitled Event'),
                'start': {'dateTime': start_time, 'timeZone': tz_str},
                'end': {'dateTime': end_time, 'timeZone': tz_str},
            }

            if arguments.get('description'):
                event_body['description'] = arguments['description']
            if arguments.get('location'):
                event_body['location'] = arguments['location']
            if arguments.get('attendees'):
                event_body['attendees'] = [{'email': e} for e in arguments['attendees']]

            logger.debug(f"Creating event with body: {event_body}")

            created = service.events().insert(calendarId='primary', body=event_body).execute()
            return ToolResult(
                structured_content={
                    "status": "success",
                    "msg": f"Event created successfully: {created.get('summary')}",
                    "event": _filter_event_fields(created)
                }, content=TextContent(
                    type="text", text=f"Event created: {
                        created.get('summary')} | Link: {
                        created.get('htmlLink')}"))

        except Exception as e:
            sys.stderr.write(f"[calendar-mcp] add error: {e}\n")
            return ToolResult(content=TextContent(type="text", text=f"Error creating event: {str(e)}"))


mcp.add_tool(AddCalendarEvent())


# ─────────────────────────────────────────────
#  3. Update / Modify Calendar Event
# ─────────────────────────────────────────────
class UpdateCalendarEvent(Tool):
    name: str = "update_calendar_event"
    description: str = (
        "Update an existing calendar event. Requires the event ID. "
        "Use this for requests like 'move my 2pm meeting to 3pm', "
        "'rename the Team Standup event', 'change location of meeting', etc. "
        "You can update the title, start/end time, description, location, or attendees. "
        "Please use get_event_details to find the event ID before updating if you don't have it."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "The ID of the event to update (obtained from list_calendar_events)."
            },
            "summary": {
                "type": "string",
                "description": "New title for the event."
            },
            "description": {
                "type": "string",
                "description": "New description for the event."
            },
            "start_time": {
                "type": "string",
                "description": "Event start time in ISO 8601 format."
            },
            "end_time": {
                "type": "string",
                "description": "Event end time in ISO 8601 format."
            },
            "location": {
                "type": "string",
                "description": "New location for the event."
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New list of attendee email addresses (replaces existing)."
            },
            "timezone_offset": {
                "type": "number",
            },
        },
        "required": ["event_id", "timezone_offset"],
        "additionalProperties": True,
    }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        sys.stderr.write(f"[calendar-mcp] update_calendar_event args: {arguments}\n")

        auth_info = arguments.get("__auth_info")
        if not auth_info:
            return ToolResult(content=TextContent(type="text", text="Error: Missing authorization information."))

        event_id = arguments.get('event_id')
        if not event_id:
            return ToolResult(content=TextContent(type="text", text="Error: event_id is required to update an event."))

        try:
            service = _get_calendar_service(auth_info)

            # Fetch the existing event first
            existing = service.events().get(calendarId='primary', eventId=event_id).execute()

            # Apply updates
            if arguments.get('summary'):
                existing['summary'] = arguments['summary']
            if arguments.get('description'):
                existing['description'] = arguments['description']
            if arguments.get('location'):
                existing['location'] = arguments['location']

            # Time updates from datetime_parser
            start_time, end_time, tz_offset = _extract_datetime_range(arguments)
            tz_str = _format_timezone_offset(tz_offset) if tz_offset is not None else None

            if start_time:
                existing['start'] = {
                    'dateTime': start_time,
                    'timeZone': tz_str or existing.get(
                        'start',
                        {}).get(
                        'timeZone',
                        '+00:00')}
            if end_time:
                existing['end'] = {
                    'dateTime': end_time,
                    'timeZone': tz_str or existing.get(
                        'end',
                        {}).get(
                        'timeZone',
                        '+00:00')}

            if arguments.get('attendees'):
                existing['attendees'] = [{'email': e} for e in arguments['attendees']]

            logger.debug(f"Updating event {event_id} with data: {existing}")

            updated = service.events().update(
                calendarId='primary', eventId=event_id, body=existing
            ).execute()

            return ToolResult(
                structured_content={
                    "status": "success",
                    "msg": f"Event updated successfully: {updated.get('summary')}",
                    "event": _filter_event_fields(updated)
                },
                content=[{"type": "text", "text": f"Event updated: {updated.get('summary')} | Link: {updated.get('htmlLink')}"}],
            )

        except Exception as e:
            sys.stderr.write(f"[calendar-mcp] update error: {e}\n")
            return ToolResult(content=TextContent(type="text", text=f"Error updating event: {str(e)}"))


mcp.add_tool(UpdateCalendarEvent())


# ─────────────────────────────────────────────
#  4. Delete Calendar Event
# ─────────────────────────────────────────────
class DeleteCalendarEvent(Tool):
    name: str = "delete_calendar_event"
    description: str = (
        "Delete an event from the calendar. Requires the event ID. "
        "Please use get_event_details to find the event ID before updating if you don't have it."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "The ID of the event to delete (obtained from list_calendar_events)."
            },
        },
        "required": ["event_id"],
        "additionalProperties": True,
    }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        sys.stderr.write(f"[calendar-mcp] delete_calendar_event args: {arguments}\n")

        auth_info = arguments.get("__auth_info")
        if not auth_info:
            return ToolResult(content=TextContent(type="text", text="Error: Missing authorization information."))

        event_id = arguments.get('event_id')
        if not event_id:
            return ToolResult(content=TextContent(type="text", text="Error: event_id is required to delete an event."))

        try:
            service = _get_calendar_service(auth_info)

            # Fetch event summary before deletion for confirmation message
            try:
                event = service.events().get(calendarId='primary', eventId=event_id).execute()
                event_name = event.get('summary', 'Untitled Event')
            except Exception:
                event_name = event_id

            service.events().delete(calendarId='primary', eventId=event_id).execute()

            return ToolResult(
                # structured_content={"deleted": True, "event_id": event_id, "summary": event_name},
                structured_content={
                    "status": "success",
                    "msg": f"Event '{event_name}' deleted successfully.",
                    "event_id": event_id,
                    "summary": event_name
                },
                content=TextContent(type="text", text=f"Event '{event_name}' has been deleted successfully."),
            )

        except Exception as e:
            sys.stderr.write(f"[calendar-mcp] delete error: {e}\n")
            return ToolResult(content=TextContent(type="text", text=f"Error deleting event: {str(e)}"))


mcp.add_tool(DeleteCalendarEvent())


# ─────────────────────────────────────────────
#  5. Get Detailed Event Information
# ─────────────────────────────────────────────
class GetEventDetails(Tool):
    name: str = "get_event_details"
    description: str = (
        "Get detailed information about a specific calendar event. "
        "Use this for requests like 'get details for my 2pm meeting', "
        "'show me the location of the Team Standup event', etc. "
        "Optionally narrow results to a specific time range."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text search query to match against event titles, descriptions, locations, etc."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of events to return. Default 10."
            },
            "timezone_offset": {
                "type": "number",
            },
        },
        "required": ["query", "timezone_offset"],
        "additionalProperties": True,
    }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        sys.stderr.write(f"[calendar-mcp] get_event_details args: {arguments}\n")

        auth_info = arguments.get("__auth_info")
        if not auth_info:
            return ToolResult(content=TextContent(type="text", text="Error: Missing authorization information."))

        query = arguments.get('query', '')
        if not query:
            return ToolResult(content=TextContent(type="text", text="Error: A search query is required."))

        try:
            service = _get_calendar_service(auth_info)
            time_min, time_max, tz_offset = _extract_datetime_range(arguments)

            kwargs: Dict[str, Any] = {
                'calendarId': 'primary',
                'q': query,
                'maxResults': arguments.get('max_results', 10),
                'singleEvents': True,
                'orderBy': 'startTime',
            }

            if time_min:
                kwargs['timeMin'] = time_min
            else:
                # Default: search from 1 year ago
                if tz_offset is not None:
                    tz = timezone(timedelta(hours=tz_offset))
                else:
                    tz = timezone.utc
                one_year_ago = datetime.now(tz) - timedelta(days=365)
                kwargs['timeMin'] = one_year_ago.isoformat()

            if time_max:
                kwargs['timeMax'] = time_max

            events_result = service.events().list(**kwargs).execute()
            events = events_result.get('items', [])

            if not events:
                return ToolResult(content=TextContent(type="text", text=f"No events found matching '{query}'."))

            formatted = [_format_event_summary(e) for e in events]
            filtered_events = [_filter_event_fields(e) for e in events]
            return ToolResult(
                # structured_content={"events": filtered_events},
                structured_content={
                    "status": "success",
                    "msg": f"Found {len(events)} event(s) matching '{query}'.",
                    "events": filtered_events
                },
                content=TextContent(
                    type="text",
                    text=f"Found {len(events)} event(s) matching '{query}':\n" +
                    "\n".join(formatted)),
            )

        except Exception as e:
            sys.stderr.write(f"[calendar-mcp] search error: {e}\n")
            return ToolResult(content=TextContent(type="text", text=f"Error searching events: {str(e)}"))


mcp.add_tool(GetEventDetails())


if __name__ == "__main__":
    sys.stderr.write("Starting Calendar Agent MCP Server with stdio transport\n")
    mcp.run(transport="stdio")
