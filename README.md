# Ollama MCP Agent

Ollama MCP Agent allows you to use LLM models locally on your PC for free. Using Ollama's locally installed LLM models along with MCP (Model Context Protocol) additional features, you can easily extend LLM functionality.

- contains Ollama(main.py), Gemini(gemini.py) example
- Inspired by: [Teddynote-lab's mcp agents](https://github.com/teddynote-lab/langgraph-mcp-agents), [langchain mcp adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- Contributor: [odeothx](https://github.com/odeothx?tab=repositories)

## Key Features

- Run LLM models locally on your PC (no additional costs)
- Extend LLM capabilities through MCP
- Streaming response output
- Tool call information monitoring

## System Requirements

- Python 3.12 or higher
- [Ollama](https://ollama.ai) installation
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- MCP server (optional)

## Installation

1. Clone repository
```bash
git clone https://github.com/godstale/ollama-mcp-agent
cd ollama-mcp-agent
```

2. Install uv (if not installed)
```bash
# Using pip
pip install uv

# Or using curl (Unix-like systems)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using PowerShell (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Create virtual environment and install dependencies
```bash
# Install dependencies
uv sync
```

4. Install Ollama and download model
```bash
# Install Ollama (refer to https://ollama.ai for platform-specific installation)
# Download LLM model which supports Tool calling feature
ollama pull qwen3:14b
```

## Configuration

### MCP Configuration (mcp_config.json)

You can extend LLM functionality through the MCP configuration file. You can implement MCP servers directly in Python or add MCP servers found on [smithery.ai](https://smithery.ai/). Add settings to the `mcp_config.json` file:

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["./mcp_server/mcp_server_weather.py"],
      "transport": "stdio"
    },
    "fetch": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@smithery-ai/fetch",
        "--key",
        "your_unique_uuid"
      ]
    }
  }
}
```

## Running the Application (with Ollama)

Basic execution:
```bash
uv run main.py
```

With options:
```bash
uv run main.py --temp 0.7 --timeout 300 --show-tools
```

## Using Google Gemini Model

Ollama MCP Agent now supports Google's Gemini model as an alternative to Ollama. (written by: [odeothx](https://github.com/odeothx?tab=repositories)) To use Gemini:

1. Set up Google API Key
```bash
# Create .env file and add your Google API key
echo GOOGLE_API_KEY=your_google_api_key_here > .env

# Or set environment variable directly
export GOOGLE_API_KEY=your_google_api_key_here  # For Unix-like systems
# Or
set GOOGLE_API_KEY=your_google_api_key_here     # For Windows
```

2. Run with Gemini
```bash
uv run gemini.py
```

### Gemini Run Options

- `--temp`: Set temperature value (0.0 ~ 1.0, default: 0.5)
- `--system-prompt`: Set custom system prompt
- `--timeout`: Response generation timeout (seconds, default: 300)
- `--show-tools`: Display tool call information

### Important Notes for Gemini

- Requires valid Google API key
- Uses Gemini 1.5 Flash model by default
- Supports all MCP tools like the Ollama version
- Streaming responses are enabled by default

### Run Options

- `--temp`: Set temperature value (0.0 ~ 1.0, default: 0.1)
- `--system-prompt`: Set system prompt
- `--timeout`: Response generation timeout (seconds, default: 300)
- `--show-tools`: Display tool call information

## Key Files

- `main.py`: Main application file
- `mcp_manager.py`: MCP client management
- `mcp_config.json`: MCP server configuration file

## Extending MCP Tools

1. Add new MCP server and tools to `mcp_config.json`
2. Implement and run MCP server
3. Restart application

Refer to [smithery.ai](https://smithery.ai/) to add and use various MCP servers.

## Exit Commands

Enter one of the following commands to exit the program:
- quit
- exit
- bye

## Important Notes

- Basic LLM functionality works even without MCP server configuration
- Response speed may vary depending on local PC performance
- Be mindful of memory usage (especially when using large LLM models)

## License

MIT License
