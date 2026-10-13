"""Agent tool layer (V1): the ONLY four functions the LLM Agent may call.

    resolve_airports(references)
    discover_airports(country, states=None)
    compare_airports(airport_codes, year=None, month=None, allow_large_group=False)
    analyze_airport(airport_code, year=None, month=None)

Plain Python functions returning compact, JSON-serializable, structured data (no prose,
no questions to the user). resolve_airports turns user airport names/cities into codes
from local OurAirports data; the other tools take codes. Deterministic work - data
retrieval, identifier resolution,
metrics, KPIs, normalization, scoring - happens in the layers below; these tools only
validate inputs, call that pipeline and shape the output. Pass one AgentToolContext per
session (context=...) so follow-up calls reuse data already retrieved.
See docs/AGENT_TOOLS.md.
"""

from agent_tools.analyze import analyze_airport
from agent_tools.common import AgentToolContext, ToolContext
from agent_tools.compare import AUTOMATIC_ANALYSIS_LIMIT, TOP_K, compare_airports
from agent_tools.discover import discover_airports
from agent_tools.resolve import resolve_airports

__all__ = ["resolve_airports", "discover_airports", "compare_airports", "analyze_airport", "AgentToolContext",
           "ToolContext",
           "AUTOMATIC_ANALYSIS_LIMIT", "TOP_K"]
