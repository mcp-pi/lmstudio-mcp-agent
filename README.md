# LM Studio MCP Agent

LM Studio MCP Agent allows you to use LLM models locally on your PC for free through LM Studio. Using LM Studio's locally hosted LLM models along with MCP (Model Context Protocol) additional features, you can easily extend LLM functionality.

- contains LM Studio(main.py), Gemini(gemini.py) example
- Inspired by: [Teddynote-lab's mcp agents](https://github.com/teddynote-lab/langgraph-mcp-agents), [langchain mcp adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- Contributor: [odeothx](https://github.com/odeothx?tab=repositories)

## Key Features

- Run LLM models locally on your PC through LM Studio (no additional costs)
- Extend LLM capabilities through MCP
- Streaming response output
- Tool call information monitoring

## System Requirements

- Python 3.12 or higher
- [LM Studio](https://lmstudio.ai) installation and running local server
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- MCP server (optional)

## Installation

1. Clone repository
```bash
git clone https://github.com/godstale/lmstudio-mcp-agent
cd lmstudio-mcp-agent
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

4. Install and setup LM Studio
```bash
# Download and install LM Studio from https://lmstudio.ai
# Start LM Studio and load a model
# Start the local server (usually on http://localhost:1234)
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

## Running the Application (with LM Studio)

Basic execution:
```bash
uv run main.py
```

With options:
```bash
uv run main.py --temp 0.7 --timeout 300 --show-tools --model "your-model-name" --base-url "http://localhost:1234/v1"
```

### Command Line Options:
- `--temp`: Temperature setting (0.0-1.0, default: 0.1)
- `--model`: Model name to use with LM Studio (default: "local-model")
- `--base-url`: LM Studio API base URL (default: "http://localhost:1234/v1")
- `--timeout`: Response timeout in seconds (default: 300)
- `--show-tools`: Show tool execution information
- `--system-prompt`: Custom system prompt

## Using Google Gemini Model

LM Studio MCP Agent also supports Google's Gemini model as an alternative to LM Studio. (written by: [odeothx](https://github.com/odeothx?tab=repositories)) To use Gemini:

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
- Supports all MCP tools like the LM Studio version
- Streaming responses are enabled by default

## LM Studio Setup Instructions

1. **Download and Install LM Studio**
   - Download from [https://lmstudio.ai](https://lmstudio.ai)
   - Install and launch LM Studio

2. **Download a Model**
   - In LM Studio, go to the "Discover" tab
   - Search for and download a model that supports function calling (e.g., models like Llama 3.2, Qwen, or similar)
   - Recommended models for function calling:
     - `microsoft/Phi-3.5-mini-instruct`
     - `Qwen/Qwen2.5-7B-Instruct`
     - `meta-llama/Llama-3.2-3B-Instruct`

3. **Start the Local Server**
   - Go to the "Local Server" tab in LM Studio
   - Load your downloaded model
   - Click "Start Server"
   - The server will typically run on `http://localhost:1234`

4. **Verify Connection**
   - Make sure the server is running before using this application
   - You can test the connection by visiting `http://localhost:1234/v1/models` in your browser

### Run Options

- `--temp`: Set temperature value (0.0 ~ 1.0, default: 0.1)
- `--model`: Model name to use with LM Studio (default: "local-model")
- `--base-url`: LM Studio API base URL (default: "http://localhost:1234/v1")
- `--system-prompt`: Set system prompt
- `--timeout`: Response generation timeout (seconds, default: 300)
- `--show-tools`: Display tool call information

## Key Files

- `main.py`: Main application file (LM Studio integration)
- `gemini.py`: Gemini model integration
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
- Make sure LM Studio server is running before using this application
- Response speed may vary depending on local PC performance and model size
- Be mindful of memory usage (especially when using large LLM models)
- For best results with tool calling, use models that explicitly support function calling

## Troubleshooting

### LM Studio Connection Issues
- Ensure LM Studio is running and the server is started
- Check that the correct port (default: 1234) is being used
- Verify the model is loaded in LM Studio
- Test the connection: `curl http://localhost:1234/v1/models`

### Model Compatibility
- Some models may not support function calling properly
- If tool calling doesn't work, try a different model
- Recommended models are listed in the setup instructions above

## License

MIT License
