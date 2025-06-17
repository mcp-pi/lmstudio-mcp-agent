import asyncio
import json
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient

# Load MCP configuration
DEFAULT_MCP_CONFIG = {"mcpServers": {}}


# MCP configuration is stored in mcp_config.json file
def load_mcp_config() -> Dict[str, Any]:
    try:
        with open("mcp_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config["mcpServers"]
    except FileNotFoundError:
        return DEFAULT_MCP_CONFIG


# Clean up MCP client
async def cleanup_mcp_client(client=None):
    if client is not None:
        try:
            # New API doesn't use context manager, so no cleanup needed
            pass
        except Exception as e:
            print(f"Error occurred while cleaning up MCP client: {str(e)}")


# Initialize MCP client
async def initialize_mcp_client():
    mcp_config = load_mcp_config()

    try:
        client = MultiServerMCPClient(mcp_config)
        await client.__aenter__()
        tools = client.get_tools()
        return client, tools
    except Exception as e:
        print(f"Error occurred while initializing MCP client: {str(e)}")
        if "client" in locals():
            await cleanup_mcp_client(client)
        raise


# Test MCP tool calls
async def test_mcp_tool(mcp_tools):
    try:
        # Test calls
        for tool in mcp_tools:
            print(f"[Tool] {tool.name}")
            # print(f"  - Description: {tool.description}")
            # print(f"  - Schema: {tool.args_schema}")

            # Check MCP server status
            # if tool.name == "get_weather":
            #     try:
            #         # Test call
            #         test_response = await tool.ainvoke({"location": "Seoul"})
            #         print("\n=== MCP Server Status ===")
            #         print("✅ MCP server is running normally.")
            #         print(f"Test response: {test_response}")
            #     except Exception as e:
            #         print("\n❌ MCP server status check failed:")
            #         print(f"- Error message: {str(e)}")
            #         return
    except Exception as e:
        print(f"Error occurred during test call: {str(e)}")
