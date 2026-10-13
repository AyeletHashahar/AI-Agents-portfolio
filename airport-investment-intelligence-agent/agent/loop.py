"""Minimal single-turn agent loop: Gemini chooses tools, our dispatcher executes them.

    user message -> Gemini -> final text            -> return it
                          -> function call(s)        -> execute_tool(...) for each
                                                     -> send structured results back
                                                     -> Gemini again (bounded)

Gemini decides every next step; there is no hard-coded airport workflow here. Tools run
ONLY through agent.tools.execute_tool with the caller's AgentToolContext. The SDK's
automatic function calling stays disabled.

Entry points (one loop, _run_turn, underneath both):
- run_agent_turn(...)               standalone turn, no prior conversation.
- run_conversation_turn(turns, ...) one turn on top of earlier COMPLETED turns; returns
                                    (text, this_turn) and never modifies `turns`.
  agent.session.AgentSession uses the latter to hold a multi-turn conversation.

Model context: only the last RECENT_TURNS completed turns (whole turns) + the new message.
Older turns stay stored; Gemini can look them up with retrieve_conversation_context
(agent/memory.py). _execute_call routes that name to the memory layer and every other
name to the business-tool dispatcher.

Budgets (per user turn)
- llm_calls:   NORMAL model invocations (decision calls). At most max_llm_calls.
- tool_rounds: model responses whose tool call(s) were actually EXECUTED (one round may
               execute several calls). At most max_tool_rounds.
A response's tool calls are executed only if, afterwards, both budgets still allow a
normal call that can use their results. Otherwise nothing more is executed and ONE extra
terminal finalization call (tools disabled) asks the model to answer from the evidence
already collected. Worst case: max_llm_calls + 1 model requests.

Tracing (agent/trace.py): the loop reports model calls, tool calls and budget events to a
TurnRecorder. This is instrumentation only; it never changes what the loop does.
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.instructions import SYSTEM_INSTRUCTION
from agent.memory import MEMORY_DECLARATION, MEMORY_TOOL_NAME, ConversationMemory
from agent.tools import TOOLS, execute_tool, tool_declarations
from agent.trace import ERROR, FINAL_TEXT, FINALIZATION, FUNCTION_CALLS, TurnRecorder
from config import MAX_LLM_CALLS, MAX_TOOL_ROUNDS, MODEL, RECENT_TURNS, TIMEOUT_MS  # noqa: F401 (re-exported)

# MODEL, TIMEOUT_MS, MAX_TOOL_ROUNDS, MAX_LLM_CALLS and RECENT_TURNS: see config.py.

if MEMORY_TOOL_NAME in TOOLS:
    raise RuntimeError(f"'{MEMORY_TOOL_NAME}' must not also be a business tool name.")

FINALIZATION_INSTRUCTION = (
    "EXECUTION BUDGET REACHED: no more tools can be run for this request. These requested tool "
    "calls were NOT executed: {calls}. Answer the user's original question now, using only the "
    "information already available above. Do not claim that any unexecuted tool was run and do not "
    "invent its results. If missing information prevents a complete answer, state that limitation."
)


class AgentError(Exception):
    """Agent-level failure (not a tool or business error)."""


class EmptyModelResponse(AgentError):
    """A normal model response had neither tool calls nor usable text."""


class FinalizationFailed(AgentError):
    """The terminal finalization call (budget exhausted) produced no usable final answer."""


def create_gemini_client(api_key=None):
    """Real Gemini client (GEMINI_API_KEY from .env). Only created when a turn runs."""
    load_dotenv()
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise AgentError("GEMINI_API_KEY was not found in .env")
    return genai.Client(api_key=api_key, http_options={"timeout": TIMEOUT_MS})


def agent_tools():
    """Everything Gemini may call: the registered business tools plus the conversation-memory
    lookup, as one Gemini Tool. Routing happens in _execute_call."""
    return [types.Tool(function_declarations=tool_declarations() + [MEMORY_DECLARATION])]


def build_config():
    """Normal calls: grounding policy, business tools + memory lookup, model decides (AUTO),
    NO automatic Python execution."""
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=agent_tools(),
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def build_finalization_config():
    """Terminal call: function calling disabled (mode NONE = 'as if no function declarations
    were provided'). The declarations stay only because earlier turns in `contents` contain
    function-call/response parts; nothing is executed in this step either way. The same
    grounding policy applies to the best-effort answer."""
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=agent_tools(),
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def _parts(response):
    candidate = response.candidates[0] if response.candidates else None
    content = candidate.content if candidate else None
    return candidate, content, (content.parts or []) if content else []


def _visible_text(parts):
    return "".join(part.text for part in parts if part.text and not part.thought).strip()


def _finalize(client, model, contents, unexecuted_calls, trace):
    """ONE terminal request: answer from the evidence already in `contents`, no tools.
    The model turn that requested `unexecuted_calls` is NOT appended and no function
    response is created for them; they are only named in the instruction. On success the
    instruction and the model's answer are appended to `contents` (the working turn)."""
    requested = ", ".join(f"{call.name}({json.dumps(dict(call.args or {}), default=str)})"
                          for call in unexecuted_calls)
    contents.append(types.Content(
        role="user", parts=[types.Part(text=FINALIZATION_INSTRUCTION.format(calls=requested))]))
    trace.llm_started("finalization")
    response = client.models.generate_content(model=model, contents=contents,
                                              config=build_finalization_config())
    candidate, model_content, parts = _parts(response)
    requested = [part.function_call.name for part in parts if part.function_call]
    if requested:
        trace.llm_finished(ERROR, requested)
        raise FinalizationFailed("Model requested tools during finalization although tools were disabled.")
    text = _visible_text(parts)
    if not text:
        trace.llm_finished(ERROR)
        reason = candidate.finish_reason if candidate else "no candidates"
        raise FinalizationFailed(f"Finalization returned no usable text (finish_reason: {reason}).")
    trace.llm_finished(FINALIZATION)
    contents.append(model_content)
    return text


def _execute_call(call, context, memory):
    """Route one model-requested call: the conversation-memory lookup goes to the session's
    read-only ConversationMemory; every other name goes to the business-tool dispatcher with
    the AgentToolContext (which rejects unknown names)."""
    if call.name == MEMORY_TOOL_NAME:
        return memory.retrieve(call.args)
    return execute_tool(call.name, call.args, context)


def _traced_call(call, context, memory, trace):
    """_execute_call, timed and reported to the trace (exceptions still propagate)."""
    start = time.perf_counter()
    try:
        result = _execute_call(call, context, memory)
    except Exception as exc:
        trace.tool_called(call, time.perf_counter() - start, exception=exc)
        raise
    trace.tool_called(call, time.perf_counter() - start, result)
    return result


def _run_turn(contents, context, client, model, max_tool_rounds, max_llm_calls, memory=None, trace=None):
    """The agent loop for ONE user turn, run on `contents` - a WORKING list of Gemini Content:
    the recent prior turns plus the new user message. Every content of this turn is appended
    to it (never trimmed while the turn runs), ending with the model's final answer (the
    original Content object). Returns the final text. Memory lookups are ordinary tool
    calls here: they count toward tool rounds like business tools."""
    memory = memory if memory is not None else ConversationMemory()
    trace = trace if trace is not None else TurnRecorder()
    config = build_config()
    llm_calls = 0      # normal model invocations in THIS turn (finalization not counted)
    tool_rounds = 0    # model responses in THIS turn whose tool calls were executed

    while True:
        trace.llm_started("normal")
        response = client.models.generate_content(model=model, contents=contents, config=config)
        llm_calls += 1
        candidate, model_content, parts = _parts(response)
        calls = [part.function_call for part in parts if part.function_call]

        if not calls:                                   # final answer
            text = _visible_text(parts)
            if not text:
                trace.llm_finished(ERROR)
                reason = candidate.finish_reason if candidate else "no candidates"
                raise EmptyModelResponse(f"Model returned no tool calls and no text (finish_reason: {reason}).")
            trace.llm_finished(FINAL_TEXT)
            contents.append(model_content)
            return text

        trace.llm_finished(FUNCTION_CALLS, [call.name for call in calls])
        if tool_rounds >= max_tool_rounds or llm_calls >= max_llm_calls:
            # No budget to execute these calls AND let a normal call use the results:
            # execute nothing more, answer from what has been collected.
            trace.not_executed(calls)
            return _finalize(client, model, contents, calls, trace)

        # Keep the model's own turn exactly as returned (function calls + thought signatures),
        # then answer every call in it, in order.
        contents.append(model_content)
        results = [
            types.Part(function_response=types.FunctionResponse(
                id=call.id, name=call.name, response=_traced_call(call, context, memory, trace)))
            for call in calls
        ]
        # Function responses go in a "user" turn: Content.role must be "user" or "model", and the
        # SDK's own automatic function calling appends Content(role="user", parts=<responses>).
        # ("tool" from the README example is rejected by the API: 400 INVALID_ARGUMENT.)
        contents.append(types.Content(role="user", parts=results))
        tool_rounds += 1
        trace.round_completed()


def run_conversation_turn(turns, user_message, context, *, client=None, model=MODEL,
                          max_tool_rounds=MAX_TOOL_ROUNDS, max_llm_calls=MAX_LLM_CALLS,
                          recent_turns=RECENT_TURNS, trace=None):
    """Run one user turn on top of earlier COMPLETED turns of a conversation.

    turns: committed completed turns, oldest first; each a sequence of Gemini Content from
           the user message to the model's final answer. Not modified.
    Model context = the last `recent_turns` completed turns (whole turns, so no function
    call is ever separated from its response) + the new message. Older turns are NOT sent;
    the model can look them up on demand through retrieve_conversation_context.
    Returns (final_text, turn): turn = tuple of this turn's Content (user message ... final
    answer). If the turn raises, nothing is returned and `turns` is unchanged.
    Budgets apply to this turn only. Other arguments: see run_agent_turn.
    trace: optional agent.trace.TurnRecorder that receives this turn's execution events
           (also on failure); it does not affect execution.
    """
    if max_llm_calls < 1 or max_tool_rounds < 0 or recent_turns < 0:
        raise ValueError("max_llm_calls must be >= 1, max_tool_rounds and recent_turns >= 0.")
    if client is None:
        client = create_gemini_client()
    turns = list(turns)
    recent = turns[max(len(turns) - recent_turns, 0):] if recent_turns else []
    working = [content for turn in recent for content in turn]
    start = len(working)
    working.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    memory = ConversationMemory(turns, visible_turns=len(recent))
    text = _run_turn(working, context, client, model, max_tool_rounds, max_llm_calls, memory, trace)
    return text, tuple(working[start:])


def run_agent_turn(user_message, context, *, client=None, model=MODEL,
                   max_tool_rounds=MAX_TOOL_ROUNDS, max_llm_calls=MAX_LLM_CALLS):
    """Answer one standalone user message (no prior conversation), letting Gemini call
    registered tools as needed. A memory lookup in a standalone turn safely reports that no
    earlier conversation exists.

    context: the session's AgentToolContext, passed to every tool execution unchanged.
    client:  a google-genai Client (or any object with .models.generate_content); a real
             client is created only if none is given.
    Returns the model's final text - from a normal call, or from the single terminal
    finalization call when a budget is exhausted (see module docstring). Raises
    EmptyModelResponse / FinalizationFailed; Gemini/network errors and exceptions inside
    tools propagate unchanged. For multi-turn conversations use agent.session.AgentSession.
    """
    text, _ = run_conversation_turn([], user_message, context, client=client, model=model,
                                    max_tool_rounds=max_tool_rounds, max_llm_calls=max_llm_calls)
    return text
