"""Agent tool 2: compare_airports - how do these airports compare as investment candidates?

Runs the existing deterministic pipeline (analysis.analyze_comparison_group: data -> KPIs
-> comparison-group percentile scoring) and returns a COMPACT LLM-facing result:
metadata, the top-K candidates with the evidence behind their score, a compact ranking of
every scored airport and the unscored airports with reasons. The full internal comparison
object (~200 KB for 30 airports) is never returned.
"""

from collections import Counter

from analysis import analyze_comparison_group
from config import AUTOMATIC_ANALYSIS_LIMIT, TOP_K
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

TOOL = "compare_airports"
# AUTOMATIC_ANALYSIS_LIMIT (automatic-analysis guardrail) and TOP_K (detailed top candidates):
# see config.py.
COMPARISON_GROUP_REQUIRED = "comparison_group_required"
COMPARISON_GROUP_REQUIRES_CONFIRMATION = "comparison_group_requires_confirmation"

# Supporting context returned for each top candidate (raw values, no extra calculation).
TOP_CONTEXT = {
    DEMAND_GROWTH: ["cagr_first_year", "cagr_last_year", "latest_complete_year", "latest_annual_passengers"],
    OPERATIONAL_PRESSURE: ["scheduled_p95_movements_per_hour", "operational_runways",
                           "median_departure_delay_minutes", "median_arrival_delay_minutes",
                           "cancellation_rate"],
    SCALE_CONNECTIVITY: [],
}
NOTES = [
    "Scores are percentile ranks within THIS comparison group (0-100); they are not comparable "
    "across different groups.",
    "BTS On-Time covers domestic scheduled passenger flights of reporting carriers only; "
    "destinations and long-haul values are not the complete network.",
    "Capacity Pressure is not scored (proxy only; no terminal capacity data). Use analyze_airport "
    "for detailed single-airport metrics.",
]


def _family(airport, family):
    fam = airport["families"][family]
    raw = airport["raw_kpis"]["kpis"][family]
    excluded = {label: names for label, names in (("missing", fam["missing_signals"]),
                                                  ("non_discriminative", fam["non_discriminative_signals"]),
                                                  ("not_rankable", fam["not_rankable_signals"])) if names}
    summary = {
        "score": rnd(fam["score"], 1),
        "coverage": rnd(fam["coverage"]),
        "signals": {s: {"value": rnd(v, 4), "percentile": rnd(airport["normalized_signals"][family][s], 1)}
                    for s, v in raw["signals"].items()},
    }
    if TOP_CONTEXT[family]:
        summary["context"] = {k: rnd(raw["context"][k], 4) for k in TOP_CONTEXT[family]}
    if excluded:
        summary["excluded_signals"] = excluded
    if fam["reason_if_no_score"]:
        summary["reason_if_no_score"] = fam["reason_if_no_score"]
    return summary


def _top_candidate(airport, info):
    ident = airport["raw_kpis"]["identifiers"]
    warnings = [issue(INSUFFICIENT_DATA, reason, airport=airport["airport"], family=family)
                for family, reason in airport["missing_families"].items()]
    if ident["status"] == "alternate":
        warnings.append(issue("identifier_alternate", f"BTS code {ident['bts_code']} used for OurAirports "
                                                      f"code {ident['ourairports_iata']} ({ident['source']}).",
                              airport=airport["airport"]))
    return {
        "code": airport["airport"],
        "name": info["name"],
        "state": state_of(info),
        "rank": airport["rank"],
        "investment_score": rnd(airport["investment_score"], 1),
        "score_status": airport["score_status"],
        "coverage": rnd(airport["coverage"]),
        "families": {f: _family(airport, f) for f in INVESTMENT_SCORE_WEIGHTS},
        "warnings": warnings,
    }


def _name(airport, info):
    """Canonical OurAirports name of an analysed airport (None if unavailable)."""
    row = info.get(airport["raw_kpis"]["identifiers"]["ident"])
    return row["name"] if row else None


def compare_airports(airport_codes, year=None, month=None, allow_large_group=False, *, context=None):
    """Compare airports as ONE comparison group (codes as returned by discover_airports,
    e.g. ["BOS", "BDL", "PVD"]). year/month: optional On-Time month; default = latest month
    BTS has published, used for every airport.

    Up to AUTOMATIC_ANALYSIS_LIMIT airports are analysed directly. A larger group returns
    "comparison_group_requires_confirmation" unless allow_large_group=True (set only after
    the user explicitly chose to analyse the larger group). With the override, EVERY
    airport is analysed and scored in the same single comparison group - never truncated,
    pre-selected or split.
    """
    if (not isinstance(airport_codes, (list, tuple)) or not airport_codes
            or not all(isinstance(c, str) and c.strip() for c in airport_codes)):
        return error_response(TOOL, issue(INVALID_ARGUMENT, "airport_codes must be a non-empty list of codes."))
    codes = list(dict.fromkeys(c.strip().upper() for c in airport_codes))
    warnings = []
    if len(codes) < len(airport_codes):
        warnings.append(issue("duplicate_airport_codes", "Duplicate codes were analysed once."))
    if len(codes) > AUTOMATIC_ANALYSIS_LIMIT:
        if not allow_large_group:
            return error_response(TOOL, issue(
                COMPARISON_GROUP_REQUIRES_CONFIRMATION,
                f"{len(codes)} airports exceed the automatic analysis limit of {AUTOMATIC_ANALYSIS_LIMIT}. "
                "The group can be analysed in full with allow_large_group=True (higher latency).",
                details={"candidate_count": len(codes), "automatic_analysis_limit": AUTOMATIC_ANALYSIS_LIMIT}))
        warnings.append(issue("large_group_confirmed",
                              f"{len(codes)} airports analysed in one comparison group (above the automatic "
                              f"analysis limit of {AUTOMATIC_ANALYSIS_LIMIT}; expect higher latency).",
                              details={"candidate_count": len(codes),
                                       "automatic_analysis_limit": AUTOMATIC_ANALYSIS_LIMIT}))

    ctx = get_context(context)
    period, period_error = resolve_period(ctx, year, month)
    if period_error:
        return error_response(TOOL, period_error)

    rows, errors = lookup_airports(ctx, codes)
    candidates, requested_as = [], {}
    for code, row in rows.items():
        if row["ident"] in requested_as:
            warnings.append(issue("duplicate_airport_codes", f"{code} is the same airport as "
                                                             f"{requested_as[row['ident']]}; analysed once.", airport=code))
            continue
        requested_as[row["ident"]] = code
        candidates.append(row)
    if not candidates:
        return error_response(TOOL, *errors)

    try:
        out = analyze_comparison_group(candidates, year=period["year"], month=period["month"],
                                       ourairports=ctx.ourairports, t100=ctx.t100, ontime=ctx.ontime,
                                       allow_bulk_fallback=True)
    except SOURCE_ERRORS as exc:
        return error_response(TOOL, *errors, issue(DATA_SOURCE_UNAVAILABLE, "BTS data could not be retrieved.",
                                                   details={"error": type(exc).__name__}))
    if any(method == "bulk" for method, *_ in getattr(ctx.ontime, "last_requests", [])):
        warnings.append(issue("ontime_bulk_fallback_used", "TranStats form failed; the BTS monthly file was used."))

    comparison = out["comparison"]
    info = {c["ident"]: c for c in candidates}       # OurAirports row of each analysed airport
    identifier_issues = [problem for a in comparison["airports"]
                         if (problem := identifier_issue(a["raw_kpis"]["identifiers"], a["airport"]))]
    if comparison["status"] == COMPARISON_GROUP_REQUIRED:
        warnings.append(issue(COMPARISON_GROUP_REQUIRED, comparison["message"]))

    scored = [a for a in comparison["airports"] if a["investment_score"] is not None]
    trend_windows = Counter((a["raw_kpis"]["kpis"][DEMAND_GROWTH]["context"]["cagr_first_year"],
                             a["raw_kpis"]["kpis"][DEMAND_GROWTH]["context"]["cagr_last_year"])
                            for a in comparison["airports"])
    if len([w for w in trend_windows if w != (None, None)]) > 1:
        warnings.append(issue("passenger_trend_period_differs",
                              "Passenger CAGR years differ between airports (complete-year rule).",
                              details={"windows": {f"{f}-{l}": n for (f, l), n in trend_windows.items()}}))

    return response(
        TOOL,
        errors=errors,
        warnings=warnings,
        metadata={
            "requested_airports": codes,
            "analyzed_count": len(comparison["airports"]),
            "scored_count": len(scored),
            "analysis_period": {
                "operations_month": f"{period['year']}-{period['month']:02d}",
                "selection": period["selection"],
                "latest_available": period["latest_available"],
                "same_month_for_all_airports": True,
                "passenger_trend_windows": sorted(f"{f}-{l}" for f, l in trend_windows if f is not None),
            },
            "coverage_summary": dict(Counter(a["score_status"] for a in comparison["airports"])),
            "non_discriminative_signals": comparison["non_discriminative_signals"],
            "investment_score_weights": INVESTMENT_SCORE_WEIGHTS,
            "identifier_issues": identifier_issues,
            "notes": NOTES,
        },
        top_candidates=[_top_candidate(a, info[a["raw_kpis"]["identifiers"]["ident"]]) for a in scored[:TOP_K]],
        ranking=[{"code": a["airport"], "name": _name(a, info), "rank": a["rank"],
                  "investment_score": rnd(a["investment_score"], 1), "coverage": rnd(a["coverage"])} for a in scored],
        not_scored=[{"code": a["airport"], "name": _name(a, info), "score_status": a["score_status"],
                     "coverage": rnd(a["coverage"]),
                     "identifier_status": a["raw_kpis"]["identifiers"]["status"],
                     "missing_families": sorted(a["missing_families"]),
                     "data_available": {
                         "bts_ontime_rows": a["raw_kpis"]["coverage"]["bts_ontime"]["rows_for_airport"],
                         "t100_complete_years": len(a["raw_kpis"]["coverage"]["bts_t100"]["complete_years"]),
                     }}
                    for a in comparison["airports"] if a["investment_score"] is None],
    )
