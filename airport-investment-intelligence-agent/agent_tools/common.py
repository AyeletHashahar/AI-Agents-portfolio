"""Shared helpers for the Agent tools: client context, structured issues, periods, lookup.

Every tool returns a JSON-serializable dict:
    {"tool": name, "status": "ok" | "partial" | "error", "errors": [...], "warnings": [...], ...}
An issue is {"code": machine-readable code, "message": short factual text, optional
"airport", optional "details"}. Tools never write prose recommendations and never ask the
user questions; the Agent decides what to do next.
"""

from dataclasses import dataclass, field

import requests

from data_sources.bts_ontime import BtsOnTimeClient, TranStatsFormError
from data_sources.bts_t100 import BtsT100Client
from data_sources.ourairports import OurAirportsClient
from data_sources.session_cache import SessionCache

# Issue codes used by the tools.
AIRPORT_NOT_FOUND = "airport_not_found"
AIRPORT_AMBIGUOUS = "airport_code_ambiguous"
AIRPORT_NOT_SUPPORTED = "airport_not_supported"
INSUFFICIENT_DATA = "insufficient_data"
IDENTIFIER_AMBIGUOUS = "identifier_ambiguous"
IDENTIFIER_NOT_FOUND_IN_BTS = "identifier_not_found_in_bts"
REQUESTED_PERIOD_UNAVAILABLE = "requested_period_unavailable"
INVALID_PERIOD = "invalid_period"
INVALID_ARGUMENT = "invalid_argument"
DATA_SOURCE_UNAVAILABLE = "data_source_unavailable"

# BTS data covers U.S. airports (states and territories with a TranStats geography).
SUPPORTED_COUNTRIES = frozenset({"US", "PR", "VI", "GU", "MP", "AS"})
FIRST_ONTIME_YEAR = 1987


@dataclass
class AgentToolContext:
    """One session's state, shared by the tools across calls (e.g. one conversation).

    Holds the data clients and an in-memory SessionCache of raw retrieved data:
      - On-Time form rows per (state, year, month),
      - T-100 passenger rows per airport code, and the latest T-100 month,
      - parsed OurAirports CSVs (inside the OurAirports client).
    Follow-up calls reuse it instead of downloading again. Nothing is written to disk and
    nothing outlives the process. Create a new context per session; clear() empties it.
    Clients passed in explicitly are used as given (they share this cache only if they were
    built with it).
    """
    ourairports: OurAirportsClient = field(default_factory=OurAirportsClient)
    t100: BtsT100Client = None
    ontime: BtsOnTimeClient = None
    cache: SessionCache = field(default_factory=SessionCache)

    def __post_init__(self):
        if self.t100 is None:
            self.t100 = BtsT100Client(cache=self.cache)
        if self.ontime is None:
            self.ontime = BtsOnTimeClient(airports_client=self.ourairports, cache=self.cache)

    def clear(self):
        self.cache.clear()


ToolContext = AgentToolContext   # earlier name, kept for compatibility


_DEFAULT_CONTEXT = None


def get_context(context=None):
    """The given context, or one process-wide default (in memory only). A future Agent
    should create one AgentToolContext per conversation and pass it to every tool call."""
    global _DEFAULT_CONTEXT
    if context is not None:
        return context
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = AgentToolContext()
    return _DEFAULT_CONTEXT


def issue(code, message, airport=None, details=None, **extra):
    """Structured issue. Details can be passed as details={...} and/or keyword arguments."""
    item = {"code": code, "message": message}
    if airport is not None:
        item["airport"] = airport
    merged = {**(details or {}), **extra}
    if merged:
        item["details"] = merged
    return item


def response(tool, errors=None, warnings=None, **payload):
    """Successful result. "partial" when some requested items produced errors."""
    errors, warnings = errors or [], warnings or []
    return {"tool": tool, "status": "partial" if errors else "ok",
            "errors": errors, "warnings": warnings, **payload}


def error_response(tool, *errors):
    """Nothing could be produced; only structured errors."""
    return {"tool": tool, "status": "error", "errors": list(errors), "warnings": []}


SOURCE_ERRORS = (requests.RequestException, TranStatsFormError)


def rnd(value, digits=3):
    """Round floats for compact LLM output (presentation only; None stays None)."""
    return round(value, digits) if isinstance(value, float) else value


def resolve_period(context, year, month):
    """Choose the On-Time analysis month. Returns (period dict, error issue or None).

    - year and month given: validated against the latest month BTS has published.
    - neither given: the latest published month, used for EVERY airport in the call.
    """
    if (year is None) != (month is None):
        return None, issue(INVALID_PERIOD, "Provide both year and month, or neither.")
    try:
        latest = context.ontime.latest_available_month()
    except SOURCE_ERRORS as exc:
        return None, issue(DATA_SOURCE_UNAVAILABLE, "Could not read the latest available BTS On-Time month.",
                           details={"source": "BTS TranStats", "error": type(exc).__name__})
    if year is None:
        return {"year": latest[0], "month": latest[1], "selection": "latest_available",
                "latest_available": f"{latest[0]}-{latest[1]:02d}"}, None
    if not (isinstance(year, int) and isinstance(month, int) and 1 <= month <= 12):
        return None, issue(INVALID_PERIOD, "year and month must be integers, month 1-12.")
    if (year, month) > latest or year < FIRST_ONTIME_YEAR:
        return None, issue(REQUESTED_PERIOD_UNAVAILABLE,
                           f"BTS On-Time data is available from {FIRST_ONTIME_YEAR}-01 to "
                           f"{latest[0]}-{latest[1]:02d}.",
                           details={"requested": f"{year}-{month:02d}",
                                    "latest_available": f"{latest[0]}-{latest[1]:02d}"})
    return {"year": year, "month": month, "selection": "requested",
            "latest_available": f"{latest[0]}-{latest[1]:02d}"}, None


def lookup_airports(context, codes):
    """Map input codes to OurAirports rows. Returns (rows by input code, issues).
    Unknown, ambiguous and non-U.S. codes become issues; nothing is guessed."""
    rows, issues = {}, []
    for code in codes:
        found = context.ourairports.lookup_code(code)
        if found["status"] == "not_found":
            issues.append(issue(AIRPORT_NOT_FOUND, "No airport with this code in OurAirports.", airport=code))
        elif found["status"] == "ambiguous":
            issues.append(issue(AIRPORT_AMBIGUOUS, "Code matches several airports.", airport=code,
                                matches=found["matches"], matched_on=found["matched_on"]))
        elif found["airport"]["iso_country"] not in SUPPORTED_COUNTRIES:
            issues.append(issue(AIRPORT_NOT_SUPPORTED, "BTS data covers U.S. airports only.", airport=code,
                                country=found["airport"]["iso_country"]))
        else:
            rows[code] = found["airport"]
    return rows, issues


def identifier_issue(resolution, airport):
    """Structured issue for an unresolved OurAirports -> BTS identifier, or None."""
    if resolution["status"] == "ambiguous":
        return issue(IDENTIFIER_AMBIGUOUS, "Several BTS codes could match; none was used.",
                     airport=airport, details={"candidates": resolution["candidates"], "detail": resolution["detail"]})
    if resolution["status"] == "not_found_in_bts":
        return issue(IDENTIFIER_NOT_FOUND_IN_BTS,
                     "No candidate code exists in BTS. This is missing data, not zero activity.",
                     airport=airport, details={"candidates": resolution["candidates"]})
    return None


def state_of(airport):
    region = airport.get("iso_region") or ""
    return region.split("-", 1)[1] if "-" in region else None
