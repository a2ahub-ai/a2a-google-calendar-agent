# A2A Calendar Agent

A calendar agent that retrieves events and reminders using the Google Calendar API. It is built with the A2A SDK and supports natural language queries.

## Features

- **Google Calendar Integration**: Retrieve events and reminders from your Google Calendar.
- **Natural Language Query**: Ask about your schedule in plain English (e.g., "What are my events today?").
- **A2A Protocol**: Fully compliant with the Agent-to-Agent protocol.
- **CLI Client**: Includes a command-line interface for testing and interaction.

## Prerequisites

- Python 3.12+
- `uv` package manager (recommended)
- Google Cloud Project with Calendar API enabled
- OpenAI or Groq API Key

## Setup

1.  **Install dependencies**:

    ```bash
    uv sync
    ```

2.  **Environment Configuration**:

    Create a `.env` file from the example:

    ```bash
    cp .env.example .env
    ```

    Edit `.env` and provide your API keys and configuration:
    - `OPENAI_API_KEY` or `GROQ_API_KEY`: Required for the LLM.
    - `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`: Required for Google Calendar OAuth.
    - `REMOTE_AGENT_ADDRESSES`: Comma-separated list of remote agent URLs (e.g., `http://localhost:10001`).
    - `DATETIME_PARSER_AGENT`: Name of the datetime parser agent.

    *Note: Ensure your Google Cloud OAuth consent screen is configured and the redirect URI `http://localhost:10010/auth/callback` is added to your OAuth client credentials.*

## Running the Agent Server

Start the calendar agent server:

```bash
uv run __main__.py
```

The agent will be available at `http://localhost:10010` (or the port specified in your `.env` file).

## Running the CLI Client

You can use the provided CLI tool to interact with the agent for testing.

```bash
uv run cli --agent "http://localhost:10010"
```

### Authentication

The CLI supports both automatic and manual authentication.

**Automatic Authentication (Default)**:
By default, the CLI will attempt to authenticate automatically. If a valid session token is not found for the specified profile, it will open your default web browser to the Google OAuth login page.

```bash
uv run cli --agent "http://localhost:10010" --profile user1
```

**Manual Authentication**:
You can provide an existing token manually using the `--header` option or disable automatic authentication if needed.

```bash
# Provide a token manually
uv run cli --agent "http://localhost:10010" --header "Authorization=Bearer <token>"

# Disable automatic authentication
uv run cli --agent "http://localhost:10010" --profile user1 --automatic-authentication false
```

### CLI Usage

Once the CLI is running, you can wrap your queries directly.

Examples:
- "Tell me my events today"
- "What's on my calendar today?"
- "Summarize my reminders for the week"
- "Show me today's schedule"

To exit the CLI, type `:q` or `quit`.

## Development

- **Project Structure**:
    - `app/`: Contains the server and agent logic.
        - `server_mcp.py`: The FastMCP server handling tool execution.
        - `server_agent.py`: The MCP client and agent orchestration.
    - `cli/`: The command-line interface client.

- **Dependencies**: Managed via `pyproject.toml`.

