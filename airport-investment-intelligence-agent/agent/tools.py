"""Tool registry and dispatcher: connects LLM-visible tool schemas to the business tools.

One RegisteredTool per business-level tool is the single source of truth for
    LLM tool name (= the Python function's name) -> Gemini schema -> Python callable.
Only these registered tools can ever be executed; low-level metric/KPI/data functions are
never exposed. The internal `context` (AgentToolContext) is NOT part of any LLM-visible
schema: execute_tool() injects the session's context itself.

To add a tool: write the business function in agent_tools/ (with a keyword-only
`context`), then add one register(...) entry below. The dispatcher does not change.
"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

from google.genai import types

from agent_tools import (AUTOMATIC_ANALYSIS_LIMIT, TOP_K, AgentToolContext, analyze_airport, compare_airports,
                         discover_airports, resolve_airports)
from agent_tools.common import error_response, issue

UNKNOWN_TOOL = "unknown_tool"
INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
CONTEXT_PARAMETER = "context"     # injected by execute_tool, never visible to the LLM


@dataclass(frozen=True)
class RegisteredTool:
    function: Callable
    declaration: types.FunctionDeclaration

    @property
    def name(self):
        return self.declaration.name

    @property
    def parameters(self):
        """LLM-visible argument names (from the schema)."""
        return set(self.declaration.parameters_json_schema["properties"])

    @property
    def required(self):
        return set(self.declaration.parameters_json_schema.get("required", []))


def _check_contract(function, schema):
    """Fail at import if a schema drifts from its Python function."""
    signature = inspect.signature(function)
    context = signature.parameters.get(CONTEXT_PARAMETER)
    if context is None or context.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise TypeError(f"{function.__name__} must take a keyword-only '{CONTEXT_PARAMETER}' parameter.")
    if CONTEXT_PARAMETER in schema["properties"]:
        raise TypeError(f"{function.__name__}: '{CONTEXT_PARAMETER}' must not be LLM-visible.")
    unknown = set(schema["properties"]) - set(signature.parameters)
    if unknown:
        raise TypeError(f"{function.__name__}: schema arguments {sorted(unknown)} are not function parameters.")


def register(function, description, properties, required=()):
    """Build the RegisteredTool for a business function; the tool name is the function name."""
    schema = {"type": "object", "properties": properties, "required": list(required)}
    _check_contract(function, schema)
    declaration = types.FunctionDeclaration(name=function.__name__, description=description,
                                            parameters_json_schema=schema)
    return RegisteredTool(function=function, declaration=declaration)


_PERIOD = {
    "year": {"type": "integer", "description": "Optional analysis year; give with month. Omit for the latest "
                                               "available month."},
    "month": {"type": "integer", "description": "Optional analysis month 1-12; give with year."},
}

TOOLS = {tool.name: tool for tool in (
    register(
        resolve_airports,
        "Resolve airport NAMES or CITIES the user mentioned (e.g. 'Ted Stevens Anchorage International Airport', "
        "'Fairbanks', 'Portland, OR') to airport codes, deterministically from airport reference data. Call it "
        "BEFORE analyze_airport / compare_airports whenever the user did not give an airport code; never use codes "
        "from memory. Each reference returns status 'found' (use airport.code), 'ambiguous' (ask the user to choose "
        "among the candidates; do not pick one yourself) or 'not_found'. Not needed when the user gave a code such "
        "as 'ANC'. For a region or state ('airports in Alaska') use discover_airports instead.",
        {
            "references": {"type": "array", "items": {"type": "string"},
                           "description": "Airport references as the user gave them, one per airport or city, "
                                          "e.g. ['Anchorage', 'Fairbanks International Airport']. Add ', <state>' "
                                          "to narrow a city, e.g. 'Portland, OR'."},
        },
        required=["references"],
    ),
    register(
        discover_airports,
        "Discover candidate U.S. airports in a structured geography when the airport set is not yet known. "
        "Returns lightweight commercial-activity and passenger context per airport plus candidate counts; it does "
        "NOT rank or score airports. It does not interpret region names: translate places such as 'New England' "
        "into a country code and state codes yourself (e.g. country 'US', states ['ME','NH','VT','MA','RI','CT']).",
        {
            "country": {"type": "string", "description": "ISO 3166-1 alpha-2 country code, e.g. 'US'."},
            "states": {"type": "array", "items": {"type": "string"},
                       "description": "Optional state/region codes, e.g. ['MA','CT']. Omit for the whole country."},
        },
        required=["country"],
    ),
    register(
        compare_airports,
        "Compare a known group of U.S. airports as investment candidates and rank them. Use it for questions that "
        "compare airports. Its result already contains the comparison-level metrics needed to answer them: family "
        f"scores, signal values with percentiles and key context for the top {TOP_K} airports, and score, rank and "
        "coverage for every airport. Investment Opportunity "
        "Scores are percentile ranks RELATIVE to the supplied comparison group, not absolute values. Up to "
        f"{AUTOMATIC_ANALYSIS_LIMIT} airports are analysed directly; set allow_large_group=true ONLY after the user "
        f"has explicitly authorised analysing a group larger than {AUTOMATIC_ANALYSIS_LIMIT} airports.",
        {
            "airport_codes": {"type": "array", "items": {"type": "string"},
                              "description": "Airport codes to compare as one group, e.g. ['BOS','BDL','PVD'] "
                                             "(from the user, resolve_airports or discover_airports)."},
            **_PERIOD,
            "allow_large_group": {"type": "boolean",
                                  "description": f"Default false. True only after explicit user authorisation for "
                                                 f"more than {AUTOMATIC_ANALYSIS_LIMIT} airports."},
        },
        required=["airport_codes"],
    ),
    register(
        analyze_airport,
        "Detailed deep dive on ONE airport: passenger demand and history, operational pressure, scale and "
        "connectivity, data quality. Use when the user asks what is happening at one airport. After "
        "compare_airports, do NOT call it for an airport included in that comparison merely to retrieve information "
        "already present in the comparison result. It is still appropriate for an airport that was not in the "
        "previous comparison, or when the question requires airport-level information the comparison result did "
        "not provide. It does NOT return an Investment Opportunity Score or rank: relative scoring requires a "
        "comparison group (use compare_airports).",
        {
            "airport_code": {"type": "string", "description": "One airport code, e.g. 'ANC' (from the user, "
                                                              "resolve_airports or discover_airports)."},
            **_PERIOD,
        },
        required=["airport_code"],
    ),
)}


def tool_declarations():
    """LLM-visible function declarations of all registered tools."""
    return [tool.declaration for tool in TOOLS.values()]


def gemini_tools():
    """Registered tools in the form Gemini's GenerateContentConfig(tools=...) expects."""
    return [types.Tool(function_declarations=tool_declarations())]


def execute_tool(tool_name, arguments, context):
    """Execute one LLM-requested tool call and return the tool's structured result.

    tool_name: name chosen by the LLM; only registered tools are executed.
    arguments: the LLM-generated arguments (a mapping; None = no arguments).
    context:   the session's AgentToolContext, injected as the tool's `context` argument.
               Required: the LLM never provides or controls it, and no new context is
               created here.
    Invalid calls (unknown tool, non-mapping or unexpected/missing arguments) return a
    structured error result in the tools' own format and execute nothing. Exceptions raised
    inside a tool are NOT caught: they are programming errors, not model errors.
    """
    if not isinstance(context, AgentToolContext):
        raise TypeError("execute_tool requires the session's AgentToolContext.")

    tool = TOOLS.get(tool_name) if isinstance(tool_name, str) else None
    if tool is None:
        return error_response(str(tool_name), issue(UNKNOWN_TOOL, "No registered tool with this name.",
                                                    details={"available_tools": sorted(TOOLS)}))

    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        return error_response(tool.name, issue(INVALID_TOOL_ARGUMENTS, "Tool arguments must be an object.",
                                               details={"received_type": type(arguments).__name__}))
    arguments = dict(arguments)
    unexpected = sorted(str(name) for name in arguments if name not in tool.parameters)
    missing = sorted(tool.required - set(arguments))
    if unexpected or missing:
        return error_response(tool.name, issue(INVALID_TOOL_ARGUMENTS, "Arguments do not match the tool schema.",
                                               details={"unexpected": unexpected, "missing": missing,
                                                        "allowed": sorted(tool.parameters)}))
    try:
        inspect.signature(tool.function).bind(**arguments, **{CONTEXT_PARAMETER: context})
    except TypeError as exc:
        return error_response(tool.name, issue(INVALID_TOOL_ARGUMENTS, str(exc)))

    return tool.function(**arguments, **{CONTEXT_PARAMETER: context})
