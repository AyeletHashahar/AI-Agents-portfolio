"""Agent layer: connects the LLM to the business-level tools in agent_tools/.

Current scope:
    agent/tools.py         tool registry + dispatcher (execute_tool)
    agent/instructions.py  system instruction / grounding policy (SYSTEM_INSTRUCTION)
    agent/loop.py          bounded Gemini tool-calling loop for one user turn
    agent/session.py       AgentSession: in-memory multi-turn conversation
    agent/memory.py        on-demand lookup of earlier turns (retrieve_conversation_context)
    agent/trace.py         execution trace of one turn (AgentSession.last_trace)
No summarization, persistence or cross-session memory yet.

Session shape:
    AgentSession
        |-- stored conversation    (all completed turns, original Gemini Content)
        |     model context = last RECENT_TURNS turns + new message; older turns on demand
        `-- AgentToolContext       (agent_tools: clients + raw-data cache)
"""

from agent.instructions import SYSTEM_INSTRUCTION
from agent.loop import (
    MAX_LLM_CALLS,
    MAX_TOOL_ROUNDS,
    RECENT_TURNS,
    AgentError,
    EmptyModelResponse,
    FinalizationFailed,
    create_gemini_client,
    run_agent_turn,
    run_conversation_turn,
)
from agent.memory import MAX_MEMORY_RESULTS, MEMORY_TOOL_NAME, ConversationMemory
from agent.session import AgentSession
from agent.trace import LLMCallTrace, RequestedCall, ToolCallTrace, TurnTrace
from agent.tools import TOOLS, RegisteredTool, execute_tool, gemini_tools, tool_declarations

__all__ = ["TOOLS", "RegisteredTool", "execute_tool", "gemini_tools", "tool_declarations",
           "run_agent_turn", "run_conversation_turn", "AgentSession", "create_gemini_client",
           "MAX_TOOL_ROUNDS", "MAX_LLM_CALLS", "RECENT_TURNS",
           "ConversationMemory", "MEMORY_TOOL_NAME", "MAX_MEMORY_RESULTS",
           "TurnTrace", "LLMCallTrace", "ToolCallTrace", "RequestedCall",
           "AgentError", "EmptyModelResponse", "FinalizationFailed", "SYSTEM_INSTRUCTION"]
