"""Agent tool 1: discover_airports - which airports in a geography are analysis candidates.

The Agent turns place names ("New England") into structured country/state codes; this tool
never interprets named regions. Airports come from OurAirports; evidence of commercial
passenger activity comes from ONE batched BTS T-100 query. No passenger-volume threshold is
applied: small airports are kept. Missing BTS evidence is reported as "unknown", never as
zero activity.
"""

from collections import Counter

from data_sources.identifiers import bts_code_candidates, resolve_bts_code
from metrics.passengers import annual_passenger_totals, passenger_cagr

from agent_tools.common import (
    DATA_SOURCE_UNAVAILABLE,
    INVALID_ARGUMENT,
    SOURCE_ERRORS,
    SUPPORTED_COUNTRIES,
    error_response,
    get_context,
    identifier_issue,
    issue,
    response,
    rnd,
    state_of,
)
from agent_tools.compare import AUTOMATIC_ANALYSIS_LIMIT
from config import RECENT_ACTIVITY_MONTHS

TOOL = "discover_airports"
RECENT_MONTHS = RECENT_ACTIVITY_MONTHS          # see config.py

# One compact row per listed airport (column order of `airports`). Only identity, geography,
# type, activity and lightweight narrowing fields; detailed KPIs belong to compare/analyze.
DISCOVERY_COLUMNS = [
    "code",                      # pass to compare_airports / analyze_airport
    "name",
    "municipality",
    "state",
    "airport_type",
    "commercial_activity",
    "recent_12m_passengers",     # T-100 passengers in the recent window (null = unknown)
    "latest_annual_year",
    "latest_annual_passengers",  # latest complete calendar year (null = none)
    "passenger_cagr",            # existing Demand Growth CAGR (null = not computable)
    "cagr_years",
    "latest_bts_month",
    "identifier_status",         # exact | alternate | ambiguous | not_found_in_bts
]

# commercial_activity values, in output order.
CONFIRMED = "confirmed"            # BTS reports passengers in the recent window
REPORTED_ZERO = "reported_zero"    # BTS reports records in the window, all with 0 passengers
HISTORICAL_ONLY = "historical_only"  # BTS has records, but none in the recent window
UNKNOWN = "unknown"                # no usable BTS evidence: NOT the same as zero activity
_ORDER = {CONFIRMED: 0, REPORTED_ZERO: 1, HISTORICAL_ONLY: 2, UNKNOWN: 3}


def _whole(value):
    """Passenger counts as whole numbers (None stays None)."""
    return int(round(value)) if isinstance(value, float) else value


def _recent_window(latest_month):
    """The RECENT_MONTHS 'YYYY-MM' strings ending at latest_month (inclusive)."""
    year, month = int(latest_month[:4]), int(latest_month[5:7])
    months = []
    for _ in range(RECENT_MONTHS):
        months.append(f"{year}-{month:02d}")
        year, month = (year, month - 1) if month > 1 else (year - 1, 12)
    return set(months)


def _activity(resolution, rows, window, scheduled_service):
    """(commercial_activity, recent passengers or None, listed?) for one airport."""
    if resolution["bts_code"]:
        recent = [r for r in rows if r["reporting_month"][:7] in window]
        known = [r["total_passengers"] for r in recent if "total_passengers" in r]
        if not recent:
            return HISTORICAL_ONLY, None, True
        if not known:
            return UNKNOWN, None, True
        total = sum(known)
        return (CONFIRMED if total > 0 else REPORTED_ZERO), total, True
    # No BTS code: list only if there is other evidence or an identifier problem to report.
    listed = resolution["status"] == "ambiguous" or scheduled_service
    return UNKNOWN, None, listed


def discover_airports(country, states=None, *, context=None):
    """Discover candidate airports in a country and optional list of states/regions.

    country: ISO 3166-1 alpha-2 code, e.g. "US". states: optional list such as
    ["MA", "CT"]; None = whole country. Returns structured candidates with commercial-
    activity evidence (see module docstring); the Agent decides what to do with them.
    """
    ctx = get_context(context)
    try:
        found = ctx.ourairports.find_airports(country=country, states=states)
    except (ValueError, TypeError, AttributeError) as exc:
        return error_response(TOOL, issue(INVALID_ARGUMENT, str(exc)))

    geography = {"country": country.strip().upper(),
                 "states": sorted({s.strip().upper() for s in ([states] if isinstance(states, str) else states)})
                 if states else None}
    warnings = []
    if geography["country"] not in SUPPORTED_COUNTRIES:
        warnings.append(issue(DATA_SOURCE_UNAVAILABLE, "BTS activity evidence covers U.S. airports only; "
                                                       "activity is unknown for other countries."))

    codes = {code for airport in found for code, _ in bts_code_candidates(airport)}
    try:
        t100_by_code = ctx.t100.get_monthly_passengers_for_airports(codes) if codes else {}
        latest_month = ctx.t100.latest_reporting_month()
    except SOURCE_ERRORS as exc:
        return error_response(TOOL, issue(DATA_SOURCE_UNAVAILABLE, "BTS T-100 activity data could not be retrieved.",
                                          details={"source": "BTS T-100", "error": type(exc).__name__}))
    window = _recent_window(latest_month) if latest_month else set()
    bts_codes = {code for code, rows in t100_by_code.items() if rows}
    owners = ctx.ourairports.code_owners()

    listed, excluded = [], Counter()
    for airport in found:
        resolution = resolve_bts_code(airport, bts_codes, owners)
        rows = t100_by_code.get(resolution["bts_code"], []) if resolution["bts_code"] else []
        scheduled = airport["scheduled_service"] == "yes"
        activity, recent_passengers, keep = _activity(resolution, rows, window, scheduled)
        if not keep:
            excluded[airport["type"]] += 1
            continue
        totals = annual_passenger_totals(rows) if rows else {}
        complete_years = [y for y, e in totals.items() if e["complete"]]
        latest_year = max(complete_years) if complete_years else None
        cagr = passenger_cagr(rows) if rows else None        # existing Demand Growth definition
        code = resolution["bts_code"] or airport["iata_code"] or airport["ident"]
        listed.append([
            code,
            airport["name"],
            airport["municipality"] or None,
            state_of(airport),
            airport["type"],
            activity,
            _whole(recent_passengers),
            latest_year,
            _whole(totals[latest_year]["passengers"]) if latest_year else None,
            rnd(cagr["cagr"], 4) if cagr and cagr["cagr"] is not None else None,
            f"{cagr['first_year']}-{cagr['last_year']}" if cagr and cagr["cagr"] is not None else None,
            rows[-1]["reporting_month"][:7] if rows else None,
            resolution["status"],
        ])
        problem = identifier_issue(resolution, code)
        if problem:
            warnings.append(problem)

    columns = DISCOVERY_COLUMNS
    activity_col, recent_col, name_col = columns.index("commercial_activity"), \
        columns.index("recent_12m_passengers"), columns.index("name")
    listed.sort(key=lambda r: (_ORDER[r[activity_col]], -(r[recent_col] or 0), r[name_col]))
    activity_counts = Counter(r[activity_col] for r in listed)
    return response(
        TOOL,
        warnings=warnings,
        geography=geography,
        candidate_count=len(listed),
        activity_counts={value: activity_counts.get(value, 0) for value in _ORDER},
        excluded_counts_by_type=dict(excluded),
        automatic_analysis_limit=AUTOMATIC_ANALYSIS_LIMIT,
        exceeds_automatic_analysis_limit=len(listed) > AUTOMATIC_ANALYSIS_LIMIT,
        t100_latest_month=latest_month,
        recent_window=f"{min(window)}..{max(window)}" if window else None,
        definitions={
            CONFIRMED: f"BTS T-100 reports passengers in the latest {RECENT_MONTHS} months of T-100 data.",
            REPORTED_ZERO: f"BTS T-100 has records in the latest {RECENT_MONTHS} months, all with 0 passengers.",
            HISTORICAL_ONLY: f"BTS T-100 has records, but none in the latest {RECENT_MONTHS} months.",
            UNKNOWN: "No usable BTS evidence (identifier not found/ambiguous or values missing). Not zero activity.",
            "passenger_cagr": "Existing Demand Growth CAGR (complete years from 2022, 2020-2021 excluded).",
            "excluded": "Not listed: no BTS T-100 record and no OurAirports scheduled service. Heliports, "
                        "balloonports and closed facilities are never discovered.",
        },
        columns=columns,
        airports=listed,
    )
