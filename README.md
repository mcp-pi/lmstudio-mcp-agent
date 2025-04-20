# Ollama MCP Agent

Ollama MCP Agent allows you to use LLM models locally on your PC for free. Using Ollama's locally installed LLM models along with MCP (Model Context Protocol) additional features, you can easily extend LLM functionality.

- Inspired by: [Teddynote-lab's mcp agents](https://github.com/teddynote-lab/langgraph-mcp-agents), [langchain mcp adapters](https://github.com/langchain-ai/langchain-mcp-adapters)

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
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # For Unix-like systems
# Or
.venv\Scripts\activate     # For Windows

# Install dependencies
uv sync
```

4. Install Ollama and download model
```bash
# Install Ollama (refer to https://ollama.ai for platform-specific installation)
# Download LLM model which supports Tool calling feature
ollama pull MFDoom/deepseek-r1-tool-calling:14b
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

## Running the Application

Basic execution:
```bash
python main.py
```

With options:
```bash
python main.py --temp 0.7 --timeout 300 --show-tools
```

### Run Options

- `--temp`: Set temperature value (0.0 ~ 1.0, default: 1.0)
- `--no-stream`: Disable streaming
- `--system-prompt`: Set system prompt
- `--timeout`: Response generation timeout (seconds, default: 300)
- `--show-tools`: Display tool call information

## Key Files

- `main.py`: Main application file
- `mcp_manager.py`: MCP client management
- `query_handler.py`: Query processing and streaming implementation
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
