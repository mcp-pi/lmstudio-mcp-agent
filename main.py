import argparse
import asyncio
import signal
import sys
import uuid
import warnings
from typing import List, Optional

import nest_asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from mcp_manager import cleanup_mcp_client, initialize_mcp_client
from query_handler import astream_graph

# Constants
QUERY_TIMEOUT_SECONDS = 60 * 5
RECURSION_LIMIT = 100
MCP_CHAT_PROMPT = """
    You are a helpful AI assistant that can use tools to answer questions.
    When using tools, think step by step:
    1. Understand what tools are available
    2. Decide which tool is most appropriate
    3. Check tool's parameters and make appropriate input values for parameters
    4. Use the tool and analyze its output
    5. Do not use the tool if it is not necessary
    """
DEFAULT_SYSTEM_PROMPT = MCP_CHAT_PROMPT
QUERY_THREAD_ID = str(uuid.uuid4())
DEFAULT_TEMPERATURE = 1.0

# LLM model settings that support Tool calling
DEEPSEEK_R1_14B_TOOL_CALLING = "MFDoom/deepseek-r1-tool-calling:14b"
DEEPSEEK = DEEPSEEK_R1_14B_TOOL_CALLING


# Signal handler for Ctrl+C on Windows
def handle_sigint(signum, frame):
    print("\n\nProgram terminated. Goodbye!")
    sys.exit(0)


# Create chat model
def create_chat_model(
    temperature: float = 0.7,
    streaming: bool = True,
    system_prompt: Optional[str] = None,
    mcp_tools: Optional[List] = None,
) -> ChatOllama | CompiledGraph:
    # Create Chat model: Requires LLM with Tool support
    chat_model = ChatOllama(
        model=DEEPSEEK,
        temperature=temperature,
    )

    # Create ReAct agent (when MCP tools are available)
    if mcp_tools:
        chat_model = create_react_agent(
            model=chat_model,
            tools=mcp_tools,
            checkpointer=MemorySaver(),
            prompt=MCP_CHAT_PROMPT,
        )

    return chat_model


# Process user input
def process_input(user_input: str) -> Optional[str]:
    # Clean input string
    cleaned_input = user_input.strip()
    # Handle exit commands
    if cleaned_input.lower() in ["quit", "exit", "bye"]:
        return None

    return cleaned_input


# Return streaming callback function and accumulated data
def get_streaming_callback():
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        if isinstance(message_content, dict) and "messages" in message_content:
            for msg in message_content["messages"]:
                if isinstance(msg, AIMessage):
                    accumulated_text.append(msg.content)
                    print(
                        msg.content.encode("utf-8", "replace").decode("utf-8"),
                        end="",
                        flush=True,
                    )
                elif isinstance(msg, ToolMessage):
                    print(f"Processing tool: {msg.name}\n---------------------")
                    tool_response = msg.content.encode("utf-8", "replace").decode(
                        "utf-8"
                    )
                    accumulated_tool.append(tool_response)

        return None

    return callback_func, accumulated_text, accumulated_tool


# Process user query and generate response
async def process_query(agent, query: str, timeout: int = QUERY_TIMEOUT_SECONDS):
    try:
        # Set up streaming callback
        streaming_callback, accumulated_text, accumulated_tool = (
            get_streaming_callback()
        )

        # Convert input to dictionary format
        inputs = {
            "messages": [HumanMessage(content=query)],
            "config": RunnableConfig(
                recursion_limit=RECURSION_LIMIT,
                configurable={"thread_id": QUERY_THREAD_ID},
            ),
        }

        # Generate response
        await asyncio.wait_for(
            astream_graph(
                graph=agent,
                inputs=inputs,
                callback=streaming_callback,
                config=RunnableConfig(
                    recursion_limit=RECURSION_LIMIT,
                    configurable={"thread_id": QUERY_THREAD_ID},
                ),
                stream_mode="updates",
            ),
            timeout=timeout,
        )

        # Return accumulated text
        full_response = (
            "".join(accumulated_text)
            if accumulated_text
            else "Unable to generate response."
        )
        tool_info = "".join(accumulated_tool) if accumulated_tool else ""
        return {"output": full_response, "tool_calls": tool_info}

    except asyncio.TimeoutError:
        return {
            "error": f"⏱️ Request exceeded timeout of {timeout} seconds. Please try again."
        }
    except Exception as e:
        import traceback

        print(f"\nDebug info: {traceback.format_exc()}")
        return {"error": f"❌ An error occurred: {str(e)}"}


async def amain(args):
    """Async main function"""

    mcp_client = None
    try:
        # Initialize MCP client
        print("\n=== Initializing MCP client... ===")
        mcp_client, mcp_tools = await initialize_mcp_client()
        print(f"Loaded {len(mcp_tools)} MCP tools.")

        # Print MCP tool information
        for tool in mcp_tools:
            print(f"[Tool] {tool.name}")

        # Initialize model
        chat_model = create_chat_model(
            temperature=args.temp,
            streaming=not args.no_stream,
            system_prompt=args.system_prompt,
            mcp_tools=mcp_tools,
        )

        # Start chat
        print("\n=== Starting Ollama Chat ===")
        print("Enter 'quit', 'exit', or 'bye' to exit.")
        print("=" * 40 + "\n")

        message_history = []
        message_history.append(
            SystemMessage(content=args.system_prompt or DEFAULT_SYSTEM_PROMPT)
        )

        while True:
            try:
                # Get user input
                user_input = input("\nUser: ")

                # Process input
                processed_input = process_input(user_input)
                if processed_input is None:
                    print("\nChat ended. Goodbye!")
                    break

                # Generate response
                print("AI:\n", end="", flush=True)

                response = await process_query(
                    chat_model, processed_input, timeout=int(args.timeout)
                )

                if "error" in response:
                    print(response["error"])
                    continue

                # Display tool call information
                if (
                    args.show_tools
                    and "tool_calls" in response
                    and response["tool_calls"].strip()
                ):
                    print("\n🔧 Tool Call Information:")
                    print(response["tool_calls"])

                # Update message history
                message_history.append(HumanMessage(content=processed_input))
                message_history.append(AIMessage(content=response["output"]))
                # CLI output is handled in streaming_callback

            except KeyboardInterrupt:
                print("\n\nProgram terminated. Goodbye!")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {str(e)}")
                continue

    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")
        raise
    finally:
        # Clean up MCP client
        if mcp_client is not None:
            await cleanup_mcp_client(mcp_client)


def main():
    """Main function"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Ollama Chat CLI")
        parser.add_argument(
            "--temp",
            type=float,
            default=DEFAULT_TEMPERATURE,
            help="Temperature value (0.0 ~ 1.0)",
        )
        parser.add_argument(
            "--no-stream", action="store_true", help="Disable streaming"
        )
        parser.add_argument("--system-prompt", type=str, help="Set system prompt")
        parser.add_argument(
            "--timeout",
            type=int,
            default=QUERY_TIMEOUT_SECONDS,
            help="Response generation timeout (seconds)",
        )
        parser.add_argument(
            "--show-tools", action="store_true", help="Show tool call information"
        )
        args = parser.parse_args()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply()

        # Run main function
        loop.run_until_complete(amain(args))
    except Exception as e:
        print(f"\n\nAn error occurred during program execution: {str(e)}")
        sys.exit(1)
    finally:
        # Handle remaining tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # Process cancelled tasks
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        # Close event loop
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


# Warning filter settings
warnings.filterwarnings(
    "ignore", category=ResourceWarning, message="unclosed.*<socket.socket.*>"
)

signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    main()
