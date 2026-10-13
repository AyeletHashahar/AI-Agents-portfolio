"""Deterministic comparison-group pipeline: data acquisition -> KPIs -> scoring.

analyze_comparison_group() is the backend a future compare_airports() Agent tool will
call. It makes no LLM calls and changes no KPI or scoring rule; it only organises data
retrieval efficiently:
  - ONE batched T-100 query for the whole group (only the passenger fields), which also
    tells us which candidate codes exist in BTS (identifier resolution);
  - On-Time data once per state/month, using the state already known from discovery;
  - OurAirports CSVs parsed once per client and runways indexed by airport;
  - flight rows indexed by airport once, so each airport's KPIs only see its own rows.
"""

import calendar
from datetime import date

from data_sources.bts_ontime import BtsOnTimeClient
from data_sources.bts_t100 import BtsT100Client
from data_sources.identifiers import bts_code_candidates, resolve_bts_code
from data_sources.ourairports import OurAirportsClient
from kpis import build_airport_kpis, score_comparison_group


def index_flights_by_airport(flights):
    """{airport code: [rows where it is Origin or Dest]} built in one pass. A row appears
    under both its Origin and its Dest; for every airport this is exactly the subset its
    metrics would select from the full list."""
    index = {}
    for flight in flights:
        for code in {flight.get("Origin"), flight.get("Dest")} - {None, ""}:
            index.setdefault(code, []).append(flight)
    return index


def analyze_comparison_group(candidates, *, year, month, ourairports=None, t100=None, ontime=None,
                             allow_bulk_fallback=False):
    """Retrieve data for, build and score a comparison group of airports.

    candidates: OurAirports airport rows, e.g. OurAirportsClient.find_airports(...) output
                (must include ident, iata_code, local_code, keywords, iso_country,
                iso_region).
    year, month: the On-Time month analysed (the KPI analysis period).
    ourairports / t100 / ontime: optional client instances (reused if given).
    allow_bulk_fallback: passed to BtsOnTimeClient.get_flights (monthly PREZIP file in a
                         self-deleting temp file if the TranStats form fails).

    Returns {"period", "identifiers", "identifier_issues", "comparison"}:
      identifiers        one resolution per candidate (data_sources.identifiers): the
                         OurAirports identifiers and the BTS code actually used.
      identifier_issues  candidates whose BTS code could not be resolved (ambiguous or not
                         found). They are still listed in the comparison, with no BTS data,
                         so they surface as insufficient coverage - never as zero activity.
      comparison         score_comparison_group(...) result. Each raw KPI record carries
                         its "identifiers" entry.
    """
    ourairports = ourairports or OurAirportsClient()
    t100 = t100 or BtsT100Client()
    ontime = ontime or BtsOnTimeClient(airports_client=ourairports)
    first_date = date(year, month, 1)
    last_date = date(year, month, calendar.monthrange(year, month)[1])

    # 1. One batched T-100 request for every candidate code (passenger data + existence).
    all_codes = {code for airport in candidates for code, _ in bts_code_candidates(airport)}
    t100_by_code = t100.get_monthly_passengers_for_airports(all_codes) if all_codes else {}
    bts_codes = {code for code, rows in t100_by_code.items() if rows}

    # 2. Identifier resolution (conservative; see data_sources.identifiers).
    owners = ourairports.code_owners()
    identifiers, used = [], set()
    for airport in candidates:
        resolution = resolve_bts_code(airport, bts_codes, owners)
        if resolution["bts_code"] in used:   # never merge two candidates into one BTS code
            resolution.update(bts_code=None, source=None, status="ambiguous",
                              detail=f"BTS code {resolution['bts_code']!r} already used by another candidate.")
        if resolution["bts_code"]:
            used.add(resolution["bts_code"])
        identifiers.append(resolution)

    resolved = {r["bts_code"]: airport for r, airport in zip(identifiers, candidates) if r["bts_code"]}

    # 3. On-Time once per state, with the state taken from the discovery rows.
    flights = (ontime.get_flights(list(resolved), year, month, airport_rows=resolved,
                                  allow_bulk_fallback=allow_bulk_fallback)
               if resolved else [])
    flights_by_airport = index_flights_by_airport(flights)

    # 4. KPIs per airport, each with only its own rows.
    records, labels = [], set()
    for resolution, airport in zip(identifiers, candidates):
        code = resolution["bts_code"]
        label = code or resolution["ourairports_iata"] or airport["ident"]
        if label in labels:
            label = airport["ident"]
        labels.add(label)
        record = build_airport_kpis(
            label,
            flights=flights_by_airport.get(code, []) if code else [],
            t100_rows=t100_by_code.get(code, []) if code else [],
            runways=ourairports.get_runways(airport["ident"]),
            first_date=first_date,
            last_date=last_date,
        )
        record["identifiers"] = resolution
        records.append(record)

    return {
        "period": {"first_date": first_date.isoformat(), "last_date": last_date.isoformat()},
        "identifiers": identifiers,
        "identifier_issues": [r for r in identifiers if r["bts_code"] is None],
        "comparison": score_comparison_group(records),
    }
