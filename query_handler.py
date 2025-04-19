from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


# Derived from:
# https://github.com/teddynote-lab/langgraph-mcp-agents
async def astream_graph(
    graph: CompiledStateGraph,
    inputs: List[BaseMessage] | Dict[str, Any],
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    stream_mode: str = "messages",
    include_subgraphs: bool = False,
) -> Dict[str, Any]:
    """
    Execute LangGraph and stream results

    Args:
        graph (CompiledStateGraph): Compiled LangGraph object to execute
        inputs (dict): Dictionary of inputs to pass to the graph
        config (Optional[RunnableConfig]): Execution configuration (optional)
        node_names (List[str], optional): List of node names to output. Defaults to empty list
        callback (Optional[Callable], optional): Callback function for processing each chunk. Defaults to None
            Callback function takes a dictionary of {"node": str, "content": Any} as argument
        stream_mode (str, optional): Streaming mode ("messages" or "updates"). Defaults to "messages"
        include_subgraphs (bool, optional): Whether to include subgraphs. Defaults to False

    Returns:
        Dict[str, Any]: Final result (optional)
    """
    config = config or {}
    final_result = {}

    if stream_mode == "updates":
        # Error fix: Change unpacking method
        # Some graphs like REACT agent only return a single dictionary
        async for chunk in graph.astream(
            inputs, config, stream_mode=stream_mode, subgraphs=include_subgraphs
        ):
            # Branch processing based on return format
            if isinstance(chunk, tuple) and len(chunk) == 2:
                # Expected format: (namespace, chunk_dict)
                namespace, node_chunks = chunk
            else:
                # Single dictionary return case (REACT agent etc.)
                namespace = []  # Empty namespace (root graph)
                node_chunks = chunk  # chunk itself is the node chunks dictionary

            # Check if dictionary and process items
            if isinstance(node_chunks, dict):
                for node_name, node_chunk in node_chunks.items():
                    final_result = {
                        "node": node_name,
                        "content": node_chunk,
                        "namespace": namespace,
                    }

                    # Only filter if node_names is not empty
                    if len(node_names) > 0 and node_name not in node_names:
                        continue

                    # Execute callback function if exists
                    if callback is not None:
                        result = callback({"node": node_name, "content": node_chunk})
                        if hasattr(result, "__await__"):
                            await result

            else:
                final_result = {"content": node_chunks}
                if callback is not None:
                    callback({"node": "", "content": node_chunks})

    elif stream_mode == "messages":
        async for chunk_msg, metadata in graph.astream(
            inputs, config, stream_mode=stream_mode
        ):
            curr_node = metadata["langgraph_node"]
            final_result = {
                "node": curr_node,
                "content": chunk_msg,
                "metadata": metadata,
            }

            # Only process if node_names is empty or current node is in node_names
            if not node_names or curr_node in node_names:
                # Execute callback function if exists
                if callback:
                    result = callback({"node": curr_node, "content": chunk_msg})
                    if hasattr(result, "__await__"):
                        await result

    else:
        raise ValueError(
            f"Invalid stream_mode: {stream_mode}. Must be 'messages' or 'updates'."
        )

    # Return final result if needed
    return final_result
