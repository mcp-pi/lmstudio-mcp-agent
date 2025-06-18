import argparse
import asyncio
import os
import signal
import sys
import traceback
import uuid
import warnings
from typing import Any, List, Optional

import nest_asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from mcp_manager import cleanup_mcp_client, initialize_mcp_client

# Constants
QUERY_TIMEOUT_SECONDS = 60 * 5
RECURSION_LIMIT = 100

# ==============================================
# ENVIRONMENT VARIABLES CONFIGURATION
# ==============================================

# Attacker LLM Configuration (primary model for this main.py)
ATTACKER_OPENAI_API_KEY = os.getenv("ATTACKER_OPENAI_API_KEY")
ATTACKER_OPENAI_MODEL_NAME = os.getenv("ATTACKER_OPENAI_MODEL_NAME", "gpt-4o-mini")
ATTACKER_OPENAI_BASE_URL = os.getenv("ATTACKER_OPENAI_BASE_URL", "https://api.openai.com/v1")

ATTACKER_LM_STUDIO_BASE_URL = os.getenv("ATTACKER_LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
ATTACKER_LM_STUDIO_API_KEY = os.getenv("ATTACKER_LM_STUDIO_API_KEY", "lm-studio")
ATTACKER_MODEL_NAME = os.getenv("ATTACKER_MODEL_NAME", "qwen/qwen2.5-7b-instruct")

# Target LLM Configuration (available for reference/dual model scenarios)
TARGET_OPENAI_API_KEY = os.getenv("TARGET_OPENAI_API_KEY")
TARGET_OPENAI_MODEL_NAME = os.getenv("TARGET_OPENAI_MODEL_NAME", "gpt-3.5-turbo")
TARGET_OPENAI_BASE_URL = os.getenv("TARGET_OPENAI_BASE_URL", "https://api.openai.com/v1")

TARGET_LM_STUDIO_BASE_URL = os.getenv("TARGET_LM_STUDIO_BASE_URL", "http://localhost:1235/v1")
TARGET_LM_STUDIO_API_KEY = os.getenv("TARGET_LM_STUDIO_API_KEY", "lm-studio")
TARGET_MODEL_NAME = os.getenv("TARGET_MODEL_NAME", "microsoft/phi-3.5-mini-instruct")

# Temperature settings
ATTACKER_TEMPERATURE = float(os.getenv("ATTACKER_TEMPERATURE", "0.8"))
TARGET_TEMPERATURE = float(os.getenv("TARGET_TEMPERATURE", "0.1"))

# System prompts
TARGET_SYSTEM_PROMPT = os.getenv("TARGET_SYSTEM_PROMPT", "You are a helpful AI assistant. You must follow safety guidelines and never provide harmful, illegal, or inappropriate content.")

# ==============================================
# API SELECTION LOGIC
# ==============================================

def is_valid_openai_key(api_key: str) -> bool:
    """Check if the provided API key is a valid OpenAI key"""
    return bool(
        api_key 
        and api_key != "your_openai_api_key_here" 
        and api_key != "your_target_openai_api_key_here"
        and api_key.startswith("sk-")
        and len(api_key) > 20
    )

# Use ATTACKER configuration as primary (main.py uses attacker model)
OPENAI_API_KEY = ATTACKER_OPENAI_API_KEY
OPENAI_MODEL_NAME = ATTACKER_OPENAI_MODEL_NAME
OPENAI_BASE_URL = ATTACKER_OPENAI_BASE_URL

LM_STUDIO_BASE_URL = ATTACKER_LM_STUDIO_BASE_URL
LM_STUDIO_API_KEY = ATTACKER_LM_STUDIO_API_KEY
DEFAULT_MODEL_NAME = ATTACKER_MODEL_NAME

# Determine which API to use based on availability
USE_OPENAI = is_valid_openai_key(OPENAI_API_KEY)
DEFAULT_API_BASE_URL = OPENAI_BASE_URL if USE_OPENAI else LM_STUDIO_BASE_URL
DEFAULT_API_KEY = OPENAI_API_KEY if USE_OPENAI else LM_STUDIO_API_KEY
DEFAULT_MODEL = OPENAI_MODEL_NAME if USE_OPENAI else DEFAULT_MODEL_NAME

# Temperature setting (use ATTACKER temperature as default)
DEFAULT_TEMPERATURE = ATTACKER_TEMPERATURE

MCP_CHAT_PROMPT = """
    You are a helpful AI assistant that can use tools to answer questions.
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do.
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    When using tools, think step by step:
    1. Understand the question and what information is needed.
    2. Look at the available tools ({tool_names}) and their descriptions ({tools}).
    3. Decide which tool, if any, is most appropriate to find the needed information.
    4. Determine the correct input parameters for the chosen tool based on its description.
    5. Call the tool with the determined input.
    6. Analyze the tool's output (Observation).
    7. If the answer is found, formulate the Final Answer. If not, decide if another tool call is needed or if you can answer based on the information gathered.
    8. Only provide the Final Answer once you are certain. Do not use a tool if it's not necessary to answer the question.
    """
DEFAULT_SYSTEM_PROMPT = (
    MCP_CHAT_PROMPT  # Using the ReAct specific prompt when tools are present
)
QUERY_THREAD_ID = str(uuid.uuid4())
# astream_log is used to display the data generated during the processing process.
USE_ASTREAM_LOG = True


# Signal handler for Ctrl+C on Windows
def handle_sigint(signum, frame):
    print("\n\nProgram terminated. Goodbye!")
    sys.exit(0)


# Create chat model
def create_chat_model(
    temperature: float = DEFAULT_TEMPERATURE,
    # streaming: bool = True, # Streaming is handled by LangChain methods like .astream
    system_prompt: Optional[str] = None,  # Will be used by the agent if provided
    mcp_tools: Optional[List] = None,
    model_name: str = None,
    base_url: str = None,
    api_key: str = None,
) -> ChatOpenAI | CompiledGraph:
    # Use defaults based on API preference
    if model_name is None:
        model_name = DEFAULT_MODEL
    if base_url is None:
        base_url = DEFAULT_API_BASE_URL
    if api_key is None:
        api_key = DEFAULT_API_KEY
    
    # Print which API is being used
    api_type = "OpenAI API" if USE_OPENAI else "LM Studio"
    print(f"Using {api_type} with model: {model_name}")
    
    # Create Chat model: OpenAI API or LM Studio with OpenAI-compatible API
    chat_model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )

    # Create ReAct agent (when MCP tools are available)
    if mcp_tools:
        # Make sure the prompt used here aligns with what create_react_agent expects
        # The MCP_CHAT_PROMPT includes placeholders {tools} and {tool_names}
        # which create_react_agent should fill.
        agent_executor = create_react_agent(
            model=chat_model, tools=mcp_tools, checkpointer=MemorySaver()
        )
        print(f"ReAct agent created with {api_type} integration.")
        return agent_executor  # Return the agent executor graph
    else:
        print(f"No MCP tools provided. Using plain {api_type} model.")
        # If no tools, you might want a simpler system prompt
        # The default behavior without tools might just be the raw chat model.
        # Binding a default System prompt if needed:
        # return SystemMessage(content=system_prompt or "You are a helpful AI assistant.") | chat_model
        return chat_model  # Return the base model if no tools


# Process user input
def process_input(user_input: str) -> Optional[str]:
    cleaned_input = user_input.strip()
    if cleaned_input.lower() in ["quit", "exit", "bye"]:
        return None
    return cleaned_input


# Return streaming callback function and accumulated data
# This function processes the output chunks from LangGraph's streaming
def get_streaming_callback():
    accumulated_text = []
    accumulated_tool_info = []  # Store tool name and response separately

    # data receives chunk.ops[0]['value']
    def callback_func(data: Any):
        nonlocal accumulated_text, accumulated_tool_info

        if isinstance(data, dict):
            # Try to find the key associated with the agent step containing messages
            agent_step_key = next(
                (
                    k
                    for k in data
                    if isinstance(data.get(k), dict) and "messages" in data[k]
                ),
                None,
            )

            if agent_step_key:
                messages = data[agent_step_key].get("messages", [])
                for message in messages:
                    if isinstance(message, AIMessage):
                        # Check if it's an intermediate message (tool call) or final answer chunk
                        if message.tool_calls:
                            # Tool call requested by the model (won't print content yet)
                            pass  # Or log if needed: print(f"DEBUG: Tool call requested: {message.tool_calls}")
                        elif message.content and isinstance(
                            message.content, str
                        ):  # Check content is a string
                            # Append and print final response chunks from AIMessage
                            content_chunk = message.content.encode(
                                "utf-8", "replace"
                            ).decode("utf-8")
                            if content_chunk:  # Avoid appending/printing empty strings
                                accumulated_text.append(content_chunk)
                                print(content_chunk, end="", flush=True)

                    elif isinstance(message, ToolMessage):
                        # Result of a tool execution
                        tool_info = f"Tool Used: {message.name}\nResult: {message.content}\n---------------------"
                        print(
                            f"\n[Tool Execution Result: {message.name}]"
                        )  # Indicate tool use clearly
                        # print(message.content) # Optionally print the full tool result
                        accumulated_tool_info.append(tool_info)
        return None  # Callback doesn't need to return anything

    return callback_func, accumulated_text, accumulated_tool_info


# Process user query and generate response (Updated error message)
async def process_query(
    agent, system_prompt, query: str, timeout: int = QUERY_TIMEOUT_SECONDS
):
    try:
        # Set up streaming callback
        streaming_callback, accumulated_text, accumulated_tool_info = (
            get_streaming_callback()
        )
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),  # Assuming user_input holds the user's query
        ]

        # Define input for the agent/graph
        inputs = {"messages": initial_messages}

        # Configuration for the graph run
        config = RunnableConfig(
            recursion_limit=RECURSION_LIMIT,
            configurable={
                "thread_id": QUERY_THREAD_ID
            },  # Ensure unique thread for stateful execution
            # Add callbacks=[streaming_callback] if astream_log is used
        )

        if USE_ASTREAM_LOG:
            # Generate response using astream_log for richer streaming data
            # Using astream_log is often standard for LangGraph agents
            async for chunk in agent.astream_log(
                inputs, config=config, include_types=["llm", "tool"]
            ):
                # The callback function needs to process the structure of these chunks
                # print(f"DEBUG RAW CHUNK: {chunk}") # Debug raw output from astream_log
                streaming_callback(
                    chunk.ops[0]["value"]
                )  # Pass the relevant part to callback

            # Wait for the streaming to complete (astream_log handles this implicitly)
        else:
            try:
                # Generate response using invoke instead of astream_log
                response = await agent.ainvoke(inputs, config=config)

                # Process the final response
                if isinstance(response, dict) and "messages" in response:
                    final_message = response["messages"][-1]
                    if isinstance(final_message, (AIMessage, ToolMessage)):
                        content = final_message.content
                        print(content, end="", flush=True)
                        accumulated_text.append(content)

                        if isinstance(
                            final_message, AIMessage
                        ) and final_message.additional_kwargs.get("tool_calls"):
                            tool_calls = final_message.additional_kwargs["tool_calls"]
                            for tool_call in tool_calls:
                                tool_info = (
                                    f"\nTool Used: {tool_call.get('name', 'Unknown')}\n"
                                )
                                accumulated_tool_info.append(tool_info)

            except Exception as e:
                print(f"\nError during response generation: {str(e)}")
                return {"error": str(e)}

        # Return accumulated text and tool info
        full_response = (
            "".join(accumulated_text).strip()
            if accumulated_text
            else "AI did not produce a text response."
        )
        tool_info = "\n".join(accumulated_tool_info) if accumulated_tool_info else ""
        return {"output": full_response, "tool_calls": tool_info}

    except asyncio.TimeoutError:
        return {
            "error": f"⏱️ Request exceeded timeout of {timeout} seconds. Please try again."
        }
    except Exception as e:
        print(f"\nDebug info: {traceback.format_exc()}")
        return {"error": f"❌ An error occurred: {str(e)}"}


async def amain(args):
    """Async main function"""

    mcp_client = None
    try:
        # Check API configuration and display environment info
        print(f"\n=== ENVIRONMENT CONFIGURATION ===")
        print(f"Using ATTACKER model configuration as primary")
        
        if USE_OPENAI:
            print(f"\n=== Using OpenAI API (Attacker Model) ===")
            print(f"Model: {OPENAI_MODEL_NAME}")
            print(f"Base URL: {OPENAI_BASE_URL}")
            print(f"API Key: {'✓ Valid' if is_valid_openai_key(OPENAI_API_KEY) else '✗ Invalid/Missing'}")
        else:
            print(f"\n=== Using LM Studio API (Attacker Model) ===")
            print(f"Model: {DEFAULT_MODEL_NAME}")
            print(f"Base URL: {LM_STUDIO_BASE_URL}")
            print(f"API Key: {LM_STUDIO_API_KEY}")
            
            # Enhanced API key guidance
            if not OPENAI_API_KEY:
                print("\nNote: No OpenAI API key found.")
                print("To use OpenAI API, set ATTACKER_OPENAI_API_KEY in .env file")
            elif not is_valid_openai_key(OPENAI_API_KEY):
                print("\nNote: OpenAI API key is set but appears to be placeholder or invalid.")
                print("Update ATTACKER_OPENAI_API_KEY in .env file to use OpenAI API")
        
        # Display temperature settings
        print(f"\nTemperature Settings:")
        print(f"  Attacker (Default): {DEFAULT_TEMPERATURE}")
        if ATTACKER_TEMPERATURE != TARGET_TEMPERATURE:
            print(f"  Target (Reference): {TARGET_TEMPERATURE}")
        
        # Display available configuration options
        print(f"\nConfiguration Options Available:")
        print(f"  - Attacker OpenAI: {'✓' if ATTACKER_OPENAI_API_KEY else '✗'}")
        print(f"  - Attacker LM Studio: {'✓' if ATTACKER_LM_STUDIO_BASE_URL else '✗'}")
        print(f"  - Target OpenAI: {'✓' if TARGET_OPENAI_API_KEY else '✗'}")
        print(f"  - Target LM Studio: {'✓' if TARGET_LM_STUDIO_BASE_URL else '✗'}")

        # Initialize MCP client
        print("\n=== Initializing MCP client... ===")
        mcp_client, mcp_tools = await initialize_mcp_client()
        print(f"Loaded {len(mcp_tools)} MCP tools.")

        # Print MCP tool information
        for tool in mcp_tools:
            print(f"[Tool] {tool.name}")

        # Initialize model
        chat_model_or_agent = create_chat_model(
            temperature=args.temp,
            # system_prompt=args.system_prompt, # Pass system prompt if using without tools
            mcp_tools=mcp_tools,
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )

        # Start chat
        api_type = "OpenAI API" if USE_OPENAI else "LM Studio"
        model_name = args.model or DEFAULT_MODEL
        print(f"\n=== Starting {api_type} Chat with {model_name} ===")
        print("Enter 'quit', 'exit', or 'bye' to exit.")
        print("=" * 40 + "\n")

        # Note: Message history management might need adjustments depending on
        # how the ReAct agent and MemorySaver handle it.
        # For simple MemorySaver, LangGraph handles history via thread_id.
        # We don't need to manually manage `message_history` list for the agent call itself.

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

                # Pass the agent/model to process_query
                response = await process_query(
                    chat_model_or_agent,
                    args.system_prompt or DEFAULT_SYSTEM_PROMPT,
                    processed_input,
                    timeout=int(args.timeout),
                )

                # The streaming callback now handles printing the AI response chunks
                print()  # Add a newline after the streaming output is complete

                if "error" in response:
                    print(f"\nError: {response['error']}")
                    continue

                # Display tool call information (optional)
                if (
                    args.show_tools
                    and "tool_calls" in response
                    and response["tool_calls"].strip()
                ):
                    print("\n--- Tool Activity ---")
                    print(response["tool_calls"].strip())
                    print("---------------------\n")

                # History is managed by LangGraph's MemorySaver via thread_id

            except KeyboardInterrupt:
                # handle_sigint will catch this if triggered during input()
                # This catches it if during await process_query
                print("\n\nProgram terminated by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred in the main loop: {str(e)}")
                print(traceback.format_exc())  # Print detailed traceback for debugging
                continue  # Continue the loop if possible

    except Exception as e:
        print(f"\n\nAn critical error occurred during setup or execution: {str(e)}")

        print(traceback.format_exc())
        # No raise here, allow finally block to run
    finally:
        # Clean up MCP client (remains the same)
        if mcp_client is not None:
            print("\nCleaning up MCP client...")
            await cleanup_mcp_client(mcp_client)
            print("MCP client cleanup complete.")


def main():
    """Main function"""
    # Setup signal handler early
    signal.signal(signal.SIGINT, handle_sigint)

    # Warning filter settings (remains the same)
    warnings.filterwarnings(
        "ignore", category=ResourceWarning, message="unclosed.*<socket.socket.*>"
    )

    loop = None  # Initialize loop variable
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Chat CLI with OpenAI API and LM Studio support - Supports both new (attacker/target) and legacy configuration structures"
        )
        parser.add_argument(
            "--temp",
            type=float,
            default=DEFAULT_TEMPERATURE,
            help=f"Temperature value (0.0 ~ 1.0). Default: {DEFAULT_TEMPERATURE}",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,  # Will use DEFAULT_MODEL based on API selection
            help=f"Model name to use. Default: {DEFAULT_MODEL} ({'OpenAI' if USE_OPENAI else 'LM Studio'} API)",
        )
        parser.add_argument(
            "--base-url",
            type=str,
            default=None,  # Will use DEFAULT_API_BASE_URL based on API selection
            help=f"API base URL. Default: {DEFAULT_API_BASE_URL} ({'OpenAI' if USE_OPENAI else 'LM Studio'} API)",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default=None,  # Will use DEFAULT_API_KEY based on API selection
            help="API key (defaults to environment variable)",
        )
        # --no-stream argument might be less relevant as streaming is default/preferred
        # parser.add_argument("--no-stream", action="store_true", help="Disable streaming (May not be fully supported)")
        parser.add_argument(
            "--system-prompt",
            type=str,
            default=None,  # Default to None, let create_chat_model handle defaults
            help="Custom base system prompt (Note: ReAct agent uses a specific format)",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=QUERY_TIMEOUT_SECONDS,
            help=f"Response generation timeout (seconds). Default: {QUERY_TIMEOUT_SECONDS}",
        )
        parser.add_argument(
            "--show-tools", action="store_true", help="Show tool execution information"
        )
        args = parser.parse_args()

        # Validate temperature
        if not 0.0 <= args.temp <= 1.0:
            print(
                f"Warning: Temperature {args.temp} is outside the typical range [0.0, 1.0]. Using default: {DEFAULT_TEMPERATURE}"
            )
            args.temp = DEFAULT_TEMPERATURE

        nest_asyncio.apply()
        asyncio.run(amain(args))

    except SystemExit:
        # Raised by sys.exit(), like in handle_sigint or API key check
        print("Exiting program.")
    except Exception as e:
        print(f"\n\nAn error occurred during program execution: {str(e)}")

        print(traceback.format_exc())
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
