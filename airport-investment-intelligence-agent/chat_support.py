"""UI-side helpers for the Chainlit chat (chat_app.py): friendly error messages and a safe
execution summary. Pure Python, no Chainlit import, so it is testable offline.

Nothing here touches the Agent: it only interprets exceptions raised by AgentSession.send()
and the tool-level part of AgentSession.last_trace. Never returns exception text, tool
arguments, tool results, prompts or memory queries.
"""

import httpx
import requests
from google.genai import errors as genai_errors

from agent import AgentError, EmptyModelResponse, FinalizationFailed

APP_TITLE = "Airport Investment Intelligence Agent"

BUSY = "The AI service is busy or its quota is exhausted. Please try again later."
UNAVAILABLE = "The AI service is temporarily unavailable. Please try again."
NO_ANSWER = "I couldn't produce an answer to that. Please try rephrasing."
NOT_CONFIGURED = "The service is not configured correctly."
UNEXPECTED = "Something went wrong. Please try again."

_TEMPORARY_CODES = frozenset({408, 500, 502, 503, 504})
_TEMPORARY_STATUSES = frozenset({"UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"})
_CONFIGURATION_CODES = frozenset({400, 401, 403, 404})
_NETWORK_ERRORS = (httpx.TransportError, requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError)

# Executed tool -> what the user may see. Tools not listed here are never shown.
TOOL_LABELS = {
    "resolve_airports": "Resolved airport",
    "compare_airports": "Compared airports",
    "analyze_airport": "Analyzed airport",
    "discover_airports": "Found airports in the region",
    "retrieve_conversation_context": "Checked earlier conversation",
}

STARTERS = tuple((question, question) for question in (       # (visible label, message sent)
    "Compare Anchorage, Fairbanks and Juneau for airport investment.",
    "Which New England airports show the strongest investment signals?",
    "What percentage of flights from Anchorage are long-haul?",
))


def friendly_error(exc):
    """Fixed, user-safe message for an exception raised by AgentSession.send()."""
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429 or exc.status == "RESOURCE_EXHAUSTED":
            return BUSY
        if exc.code in _TEMPORARY_CODES or exc.status in _TEMPORARY_STATUSES:
            return UNAVAILABLE
        if exc.code in _CONFIGURATION_CODES:
            return NOT_CONFIGURED
        return UNEXPECTED
    if isinstance(exc, (EmptyModelResponse, FinalizationFailed)):
        return NO_ANSWER
    if isinstance(exc, AgentError):              # e.g. GEMINI_API_KEY missing on the server
        return NOT_CONFIGURED
    if isinstance(exc, _NETWORK_ERRORS):
        return UNAVAILABLE
    return UNEXPECTED


def execution_summary(trace):
    """Lines describing the tools that were executed in the last turn, in order; repeated
    activity is counted ("Analyzed airport (3x)"). Only names from TOOL_LABELS and a
    'not completed' marker for calls that ended in an error; nothing else from the trace."""
    if trace is None:
        return []
    counts, failed = {}, set()
    for call in getattr(trace, "tool_calls", ()) or ():
        label = TOOL_LABELS.get(getattr(call, "name", None))
        if label is None:
            continue
        counts[label] = counts.get(label, 0) + 1
        if getattr(call, "status", None) in ("error", "exception"):
            failed.add(label)
    return [label + (f" ({n}x)" if n > 1 else "") + (" - not completed" if label in failed else "")
            for label, n in counts.items()]
