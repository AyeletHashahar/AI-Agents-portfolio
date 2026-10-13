"""Agent tool: resolve_airports - which airport does a user's name/city reference mean?

Entity resolution only, kept apart from the analytics: user references ("Anchorage",
"Fairbanks International Airport", "ANC") are resolved deterministically from the local
OurAirports data (OurAirportsClient.search_airports: code -> exact name -> place, with
data-only narrowing; see its docstring). No BTS request, no LLM, no fuzzy matching.
Ambiguous or unknown references are returned as such, with candidates; nothing is guessed.

The returned `code` is the code to pass to analyze_airport / compare_airports: an
identifier their own lookup (OurAirportsClient.lookup_code) maps back to exactly this
airport. The OurAirports -> BTS code mapping then happens inside those tools as before.
"""

from collections import Counter

from agent_tools.common import (
    AIRPORT_NOT_FOUND,
    INVALID_ARGUMENT,
    error_response,
    get_context,
    issue,
    state_of,
)
from agent_tools.compare import AUTOMATIC_ANALYSIS_LIMIT

TOOL = "resolve_airports"
COUNTRY = "US"
MAX_REFERENCES = AUTOMATIC_ANALYSIS_LIMIT
MAX_REFERENCE_CHARS = 200
MAX_CANDIDATES = 5                       # candidates listed per reference
AIRPORT_REFERENCE_AMBIGUOUS = "airport_reference_ambiguous"

FOUND, AMBIGUOUS, NOT_FOUND = "found", "ambiguous", "not_found"


def _analysis_code(ctx, row):
    """IATA code if it identifies exactly this airport for lookup_code, else the OurAirports
    ident (unique). None if neither round-trips (never expected; reported, not guessed)."""
    for code in (row["iata_code"], row["ident"]):
        found = ctx.ourairports.lookup_code(code) if code else None
        if found and found["status"] == "found" and found["airport"]["ident"] == row["ident"]:
            return code
    return None


def _airport(ctx, row):
    return {
        "code": _analysis_code(ctx, row),
        "name": row["name"],
        "municipality": row["municipality"] or None,
        "state": state_of(row),
        "airport_type": row["type"],
        "scheduled_service": row["scheduled_service"] == "yes",
        "ourairports_ident": row["ident"],
    }


def _resolve(ctx, reference):
    found = ctx.ourairports.search_airports(reference, country=COUNTRY)
    result = {"reference": reference, "status": found["status"], "matched_on": found["matched_on"]}
    if found["region"]:
        result["state_filter"] = found["region"].split("-", 1)[1]
    candidates = [row for row in found["candidates"] if row is not found["airport"]]
    if found["status"] == FOUND:
        result["selection"] = found["selection"]
        result["airport"] = _airport(ctx, found["airport"])
        others = [row for row in candidates if row["scheduled_service"] == "yes"]
        if others:                       # e.g. a smaller scheduled airport in the same city
            result["other_scheduled_candidates"] = [_airport(ctx, row) for row in others[:MAX_CANDIDATES]]
    elif found["status"] == AMBIGUOUS:
        result["reason"] = found["reason"]
        result["candidate_count"] = len(candidates)
        result["candidates"] = [_airport(ctx, row) for row in candidates[:MAX_CANDIDATES]]
    return result


def resolve_airports(references, *, context=None):
    """Resolve natural-language airport references (names, cities or codes) to airports.

    references: list of strings as the user gave them, e.g. ["Anchorage", "Fairbanks"],
    ["Ted Stevens Anchorage International Airport"], ["Portland, OR"], ["ANC"].
    Returns one result per reference, in input order, with status "found" (airport.code is
    the code for analyze_airport / compare_airports), "ambiguous" (candidates; the user
    should choose) or "not_found". Top-level status: "ok" if every reference was found,
    "partial" if some were, "error" if none was.
    """
    if (not isinstance(references, (list, tuple)) or not references
            or not all(isinstance(r, str) and r.strip() for r in references)):
        return error_response(TOOL, issue(INVALID_ARGUMENT, "references must be a non-empty list of strings."))
    if len(references) > MAX_REFERENCES:
        return error_response(TOOL, issue(INVALID_ARGUMENT, f"At most {MAX_REFERENCES} references per call."))
    if any(len(r) > MAX_REFERENCE_CHARS for r in references):
        return error_response(TOOL, issue(INVALID_ARGUMENT,
                                          f"Each reference must be at most {MAX_REFERENCE_CHARS} characters."))

    ctx = get_context(context)
    results = [_resolve(ctx, reference.strip()) for reference in references]
    errors = []
    for result in results:
        if result["status"] == AMBIGUOUS:
            errors.append(issue(AIRPORT_REFERENCE_AMBIGUOUS, "No airport could be chosen safely; see the candidates.",
                                airport=result["reference"], reason=result["reason"]))
        elif result["status"] == NOT_FOUND:
            errors.append(issue(AIRPORT_NOT_FOUND, "No U.S. airport matches this reference.",
                                airport=result["reference"]))
        elif result["airport"]["code"] is None:
            errors.append(issue(AIRPORT_NOT_FOUND, "No code identifies this airport uniquely.",
                                airport=result["reference"]))
    counts = Counter(result["status"] for result in results)
    found = counts[FOUND]
    return {
        "tool": TOOL,
        "status": "ok" if not errors else ("partial" if found else "error"),
        "errors": errors,
        "warnings": [],
        "counts": {status: counts.get(status, 0) for status in (FOUND, AMBIGUOUS, NOT_FOUND)},
        "results": results,
        "notes": ["Pass airport.code of found results to analyze_airport / compare_airports.",
                  "Ambiguous: ask the user which candidate they mean; do not choose one yourself."],
    }
