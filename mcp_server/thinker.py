#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP
import json
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Initialize FastMCP server
mcp = FastMCP(
    "Sequential_Thinking_MCP",  # Name of the MCP server
    instructions="A tool for dynamic and reflective problem-solving through a structured thinking process.",
    host="0.0.0.0",  # Host address (0.0.0.0 allows connections from any IP)
    port=1108,  # Port number for the server
)

@dataclass
class ThoughtData:
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    is_revision: Optional[bool] = None
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: Optional[bool] = None

class SequentialThinkingServer:
    def __init__(self):
        self.thought_history: List[ThoughtData] = []
        self.branches: Dict[str, List[ThoughtData]] = {}

    def log_error(self, message: str):
        """Log error messages to stderr for debugging"""
        print(f"ERROR: {message}", file=sys.stderr)

    def log_info(self, message: str):
        """Log informational messages to stderr for debugging"""
        print(f"INFO: {message}", file=sys.stderr)

    def validate_thought_data(self, **kwargs) -> ThoughtData:
        """Validate and create ThoughtData from input parameters"""
        try:
            # Required fields
            thought = kwargs.get('thought')
            thought_number = kwargs.get('thoughtNumber') or kwargs.get('thought_number')
            total_thoughts = kwargs.get('totalThoughts') or kwargs.get('total_thoughts')
            next_thought_needed = kwargs.get('nextThoughtNeeded') 
            if next_thought_needed is None:
                next_thought_needed = kwargs.get('next_thought_needed')

            if not thought or not isinstance(thought, str):
                raise ValueError('Invalid thought: must be a string')
            if not thought_number or not isinstance(thought_number, int):
                raise ValueError('Invalid thoughtNumber: must be a number')
            if not total_thoughts or not isinstance(total_thoughts, int):
                raise ValueError('Invalid totalThoughts: must be a number')
            if next_thought_needed is None or not isinstance(next_thought_needed, bool):
                raise ValueError('Invalid nextThoughtNeeded: must be a boolean')

            return ThoughtData(
                thought=thought,
                thought_number=thought_number,
                total_thoughts=total_thoughts,
                next_thought_needed=next_thought_needed,
                is_revision=kwargs.get('isRevision') or kwargs.get('is_revision'),
                revises_thought=kwargs.get('revisesThought') or kwargs.get('revises_thought'),
                branch_from_thought=kwargs.get('branchFromThought') or kwargs.get('branch_from_thought'),
                branch_id=kwargs.get('branchId') or kwargs.get('branch_id'),
                needs_more_thoughts=kwargs.get('needsMoreThoughts') or kwargs.get('needs_more_thoughts'),
            )
        except Exception as e:
            self.log_error(f"Error validating thought data: {str(e)}")
            raise

    def format_thought(self, thought_data: ThoughtData) -> str:
        """Format thought data for display"""
        thought_number = thought_data.thought_number
        total_thoughts = thought_data.total_thoughts
        thought = thought_data.thought
        is_revision = thought_data.is_revision
        revises_thought = thought_data.revises_thought
        branch_from_thought = thought_data.branch_from_thought
        branch_id = thought_data.branch_id

        prefix = ''
        context = ''

        if is_revision:
            prefix = '🔄 Revision'
            context = f' (revising thought {revises_thought})'
        elif branch_from_thought:
            prefix = '🌿 Branch'
            context = f' (from thought {branch_from_thought}, ID: {branch_id})'
        else:
            prefix = '💭 Thought'
            context = ''

        header = f"{prefix} {thought_number}/{total_thoughts}{context}"
        border_length = max(len(header), len(thought)) + 4
        border = '─' * border_length

        return f"""
┌{border}┐
│ {header} │
├{border}┤
│ {thought.ljust(border_length - 2)} │
└{border}┘"""

    def process_thought(self, **kwargs) -> Dict[str, Any]:
        """Process a thought and return formatted response"""
        try:
            validated_input = self.validate_thought_data(**kwargs)

            # Adjust total thoughts if current thought number exceeds it
            if validated_input.thought_number > validated_input.total_thoughts:
                validated_input.total_thoughts = validated_input.thought_number

            # Add to history
            self.thought_history.append(validated_input)

            # Handle branching
            if validated_input.branch_from_thought and validated_input.branch_id:
                if validated_input.branch_id not in self.branches:
                    self.branches[validated_input.branch_id] = []
                self.branches[validated_input.branch_id].append(validated_input)

            # Format and log thought
            formatted_thought = self.format_thought(validated_input)
            self.log_info(formatted_thought)

            return {
                "thoughtNumber": validated_input.thought_number,
                "totalThoughts": validated_input.total_thoughts,
                "nextThoughtNeeded": validated_input.next_thought_needed,
                "branches": list(self.branches.keys()),
                "thoughtHistoryLength": len(self.thought_history),
                "status": "success"
            }

        except Exception as error:
            self.log_error(f"Error processing thought: {str(error)}")
            return {
                "error": str(error),
                "status": "failed"
            }

# Create global instance
thinking_server = SequentialThinkingServer()

@mcp.tool()
async def sequentialthinking(
    thought: str,
    nextThoughtNeeded: bool,
    thoughtNumber: int,
    totalThoughts: int,
    isRevision: Optional[bool] = None,
    revisesThought: Optional[int] = None,
    branchFromThought: Optional[int] = None,
    branchId: Optional[str] = None,
    needsMoreThoughts: Optional[bool] = None
) -> str:
    """A detailed tool for dynamic and reflective problem-solving through thoughts.
    This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
    Each thought can build on, question, or revise previous insights as understanding deepens.

    When to use this tool:
    - Breaking down complex problems into steps
    - Planning and design with room for revision
    - Analysis that might need course correction
    - Problems where the full scope might not be clear initially
    - Problems that require a multi-step solution
    - Tasks that need to maintain context over multiple steps
    - Situations where irrelevant information needs to be filtered out

    Key features:
    - You can adjust total_thoughts up or down as you progress
    - You can question or revise previous thoughts
    - You can add more thoughts even after reaching what seemed like the end
    - You can express uncertainty and explore alternative approaches
    - Not every thought needs to build linearly - you can branch or backtrack
    - Generates a solution hypothesis
    - Verifies the hypothesis based on the Chain of Thought steps
    - Repeats the process until satisfied
    - Provides a correct answer

    Args:
        thought: Your current thinking step, which can include:
            * Regular analytical steps
            * Revisions of previous thoughts
            * Questions about previous decisions
            * Realizations about needing more analysis
            * Changes in approach
            * Hypothesis generation
            * Hypothesis verification
        nextThoughtNeeded: True if you need more thinking, even if at what seemed like the end
        thoughtNumber: Current number in sequence (can go beyond initial total if needed)
        totalThoughts: Current estimate of thoughts needed (can be adjusted up/down)
        isRevision: A boolean indicating if this thought revises previous thinking
        revisesThought: If isRevision is true, which thought number is being reconsidered
        branchFromThought: If branching, which thought number is the branching point
        branchId: Identifier for the current branch (if any)
        needsMoreThoughts: If reaching end but realizing more thoughts needed

    Returns:
        JSON string containing the thought processing result and metadata
    """
    try:
        result = thinking_server.process_thought(
            thought=thought,
            nextThoughtNeeded=nextThoughtNeeded,
            thoughtNumber=thoughtNumber,
            totalThoughts=totalThoughts,
            isRevision=isRevision,
            revisesThought=revisesThought,
            branchFromThought=branchFromThought,
            branchId=branchId,
            needsMoreThoughts=needsMoreThoughts
        )
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        thinking_server.log_error(f"Error in sequentialthinking tool: {str(e)}")
        return json.dumps({
            "error": str(e),
            "status": "failed"
        }, indent=2)

@mcp.tool()
async def get_thought_history() -> str:
    """Get the complete history of thoughts processed so far.
    
    Returns:
        JSON string containing the complete thought history and branch information
    """
    try:
        history_data = []
        for thought in thinking_server.thought_history:
            history_data.append(asdict(thought))
        
        result = {
            "thoughtHistory": history_data,
            "branches": {k: [asdict(t) for t in v] for k, v in thinking_server.branches.items()},
            "totalThoughts": len(thinking_server.thought_history),
            "status": "success"
        }
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        thinking_server.log_error(f"Error getting thought history: {str(e)}")
        return json.dumps({
            "error": str(e),
            "status": "failed"
        }, indent=2)

@mcp.tool()
async def reset_thinking_session() -> str:
    """Reset the current thinking session, clearing all thought history and branches.
    
    Returns:
        Confirmation message that the session has been reset
    """
    try:
        thinking_server.thought_history.clear()
        thinking_server.branches.clear()
        thinking_server.log_info("Thinking session reset")
        
        return json.dumps({
            "message": "Thinking session has been reset successfully",
            "status": "success"
        }, indent=2)
    
    except Exception as e:
        thinking_server.log_error(f"Error resetting thinking session: {str(e)}")
        return json.dumps({
            "error": str(e),
            "status": "failed"
        }, indent=2)

def main():
    """pip을 통해 패키지가 설치될 때의 진입점"""
    thinking_server.log_info("Starting Sequential Thinking MCP Server")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    # Initialize and run the server
    main()
