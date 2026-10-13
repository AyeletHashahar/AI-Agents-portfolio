"""Agent tool 3: analyze_airport - detailed raw analysis of ONE airport.

Reuses the same deterministic pipeline as compare_airports with a single airport. A single
airport has NO Investment Opportunity Score or rank: those are percentile ranks within a
comparison group (use compare_airports). All values here are raw metrics with coverage
and limitations; missing data stays null, never 0.
"""

from analysis import analyze_comparison_group
from kpis.build import ONTIME_SCOPE
from kpis.config import DEMAND_GROWTH, INVESTMENT_SCORE_WEIGHTS, OPERATIONAL_PRESSURE, SCALE_CONNECTIVITY

from agent_tools.common import (
    DATA_SOURCE_UNAVAILABLE,
    INSUFFICIENT_DATA,
    INVALID_ARGUMENT,
    SOURCE_ERRORS,
    error_response,
    get_context,
    identifier_issue,
    issue,
    lookup_airports,
    resolve_period,
    response,
    rnd,
    state_of,
)

TOOL = "analyze_airport"


def _rounded(values, digits=4):
    return {k: (_rounded(v, digits) if isinstance(v, dict) else rnd(v, digits)) for k, v in values.items()}


def analyze_airport(airport_code, year=None, month=None, *, context=None):
    """Detailed raw analysis of one airport (code as returned by discover_airports, an IATA
    code or an ICAO code). year/month: optional On-Time month; default = latest available."""
    if not isinstance(airport_code, str) or not airport_code.strip():
        return error_response(TOOL, issue(INVALID_ARGUMENT, "airport_code must be a non-empty string."))
    code = airport_code.strip().upper()
    ctx = get_context(context)
    period, period_error = resolve_period(ctx, year, month)
    if period_error:
        return error_response(TOOL, period_error)
    rows, lookup_errors = lookup_airports(ctx, [code])
    if lookup_errors:
        return error_response(TOOL, *lookup_errors)
    airport = rows[code]

    try:
        out = analyze_comparison_group([airport], year=period["year"], month=period["month"],
                                       ourairports=ctx.ourairports, t100=ctx.t100, ontime=ctx.ontime,
                                       allow_bulk_fallback=True)
    except SOURCE_ERRORS as exc:
        return error_response(TOOL, issue(DATA_SOURCE_UNAVAILABLE, "BTS data could not be retrieved.",
                                          details={"error": type(exc).__name__}))

    result = out["comparison"]["airports"][0]
    record = result["raw_kpis"]
    ident = record["identifiers"]
    kpis = record["kpis"]
    demand, ops, scale = (kpis[f] for f in (DEMAND_GROWTH, OPERATIONAL_PRESSURE, SCALE_CONNECTIVITY))
    capacity = dict(record["capacity_pressure_inputs"])
    seasonality = capacity.pop("seasonality")
    capacity.pop("operational_pressure")    # identical to the operational_pressure section

    warnings = []
    if (problem := identifier_issue(ident, record["airport"])):
        warnings.append(problem)
    if not record["coverage"]["bts_ontime"]["available"]:
        warnings.append(issue(INSUFFICIENT_DATA, "No BTS On-Time rows for this airport in the period: "
                                                 "operational and connectivity metrics are unavailable "
                                                 "(missing data, not zero activity).", airport=record["airport"]))
    if demand["signals"]["passenger_cagr"] is None:
        warnings.append(issue(INSUFFICIENT_DATA, f"No passenger CAGR: {demand['context']['cagr_reason_if_missing']}",
                              airport=record["airport"]))
    if any(method == "bulk" for method, *_ in getattr(ctx.ontime, "last_requests", [])):
        warnings.append(issue("ontime_bulk_fallback_used", "TranStats form failed; the BTS monthly file was used."))

    return response(
        TOOL,
        warnings=warnings,
        airport={
            "code": record["airport"],
            "name": airport["name"],
            "municipality": airport["municipality"] or None,
            "state": state_of(airport),
            "country": airport["iso_country"],
            "airport_type": airport["type"],
            "ourairports_ident": airport["ident"],
            "ourairports_iata": ident["ourairports_iata"],
            "bts_code": ident["bts_code"],
            "identifier_status": ident["status"],
            "identifier_source": ident["source"],
        },
        analysis_period={
            "operations_month": f"{period['year']}-{period['month']:02d}",
            "selection": period["selection"],
            "latest_available": period["latest_available"],
            "passenger_trend_years": [demand["context"]["cagr_first_year"], demand["context"]["cagr_last_year"]],
            "latest_complete_passenger_year": demand["context"]["latest_complete_year"],
        },
        relative_scoring={
            "investment_score": None,
            "rank": None,
            "reason": "The Investment Opportunity Score is a percentile rank within a comparison group; "
                      "a single airport has none. Use compare_airports.",
        },
        demand={**_rounded(demand["signals"]), **_rounded(demand["context"]),
                "seasonality": _rounded(seasonality, 3) if seasonality else None},
        operational_pressure={"signals": _rounded(ops["signals"]), "context": _rounded(ops["context"])},
        scale_connectivity={"signals": _rounded(scale["signals"]), "context": _rounded(scale["context"])},
        capacity_pressure={**_rounded(capacity),
                           "seasonality_peak_month": seasonality and seasonality["peak_month"],
                           "seasonality_peak_index": seasonality and rnd(seasonality["peak_index"], 3)},
        data_quality={
            "data_coverage": rnd(result["coverage"]),
            "data_coverage_definition": "Share of the configured KPI signal weight with data for this airport.",
            "family_coverage": {f: rnd(result["families"][f]["coverage"]) for f in INVESTMENT_SCORE_WEIGHTS},
            "missing_signals": {f: s for f, s in record["missing_signals"].items() if s},
            "sources": record["coverage"],
            "limitations": [ONTIME_SCOPE,
                            "T-100 passengers are enplaned (departing) passengers.",
                            "Operational runway count is a proxy (no length, configuration or capacity)."],
        },
    )
