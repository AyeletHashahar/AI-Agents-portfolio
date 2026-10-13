"""Raw KPI construction for ONE airport from already-retrieved data (no API calls).

build_airport_kpis() combines existing deterministic metrics (metrics/) into the three
KPI families plus supporting context, coverage metadata and Capacity Pressure PROXY
inputs. It returns raw values only; comparison-group normalization and the Investment
Opportunity Score are done in kpis.scoring.
"""

from metrics.airports import operational_runway_count, runway_count
from metrics.flights import (
    cancellation_rate,
    delay_metrics,
    flight_counts,
    long_haul_summary,
    peak_hour_traffic,
    peak_period_spacing,
    unique_destinations,
)
from metrics.passengers import annual_passenger_totals, passenger_cagr, seasonality_index

from kpis.config import DEMAND_GROWTH, OPERATIONAL_PRESSURE, SCALE_CONNECTIVITY

ONTIME_SCOPE = (
    "BTS Reporting Carrier On-Time Performance: domestic scheduled passenger flights of "
    "reporting carriers only. Cargo, international and non-reporting carriers are NOT "
    "included, so destinations and long-haul values are not the airport's complete network."
)
CAPACITY_PRESSURE_NOTE = (
    "PROXY / synthesis inputs only. No terminal capacity, gate capacity or terminal "
    "utilization data exists in our sources; this is not a measurement of unmet demand. "
    "Not scored separately: these signals already feed the base KPI families."
)


def _operational_runways(operational):
    """(count, reason): the operational runway count used as divisor, or (None, why not)."""
    if not operational["count_is_certain"]:
        return None, "Runway count uncertain (unknown open/closed status or unknown surface)."
    if operational["operational"] == 0:
        return None, "No open hard-surface runway in OurAirports."
    return operational["operational"], None


def _seasonality_summary(t100_rows, cagr):
    """Seasonal index over the CAGR trend years (complete, contiguous, post-COVID), or None."""
    if cagr["first_year"] is None:
        return None
    years = list(range(cagr["first_year"], cagr["last_year"] + 1))
    index = seasonality_index(t100_rows, years=years)
    peak_month = max(index, key=index.get)
    trough_month = min(index, key=index.get)
    return {
        "years": years,
        "index": index,
        "peak_month": peak_month,
        "peak_index": index[peak_month],
        "trough_month": trough_month,
        "trough_index": index[trough_month],
    }


def build_airport_kpis(airport, *, flights, t100_rows, runways, first_date, last_date,
                       airports_by_code=None):
    """Raw KPI record for one airport.

    Inputs (all already retrieved by data_sources):
      airport      BTS 3-letter code, e.g. "ANC".
      flights      BTS On-Time rows (may include other airports; filtered by code).
      t100_rows    BTS T-100 monthly rows for THIS airport.
      runways      OurAirports runways.csv rows for THIS airport.
      first_date / last_date   local dates covered by `flights` (analysis period).
      airports_by_code  optional OurAirports rows by code (long-haul distance fallback).

    Returns a dict:
      "kpis": {family: {"signals": {...}, "context": {...}}} - "signals" are the values
         that are normalized and scored; "context" is supporting information only.
      "capacity_pressure_inputs": Capacity Pressure PROXY inputs (not scored).
      "coverage": data availability and source scope.
      "missing_signals": {family: [signal names whose raw value is None]}.
    Missing data is None, never 0. If the airport has no On-Time rows at all, every
    On-Time-based signal is None (source unavailable), not 0.
    """
    airport = airport.strip().upper()
    ontime_rows = [f for f in flights if airport in (f.get("Origin"), f.get("Dest"))]
    has_ontime = bool(ontime_rows)
    period = dict(first_date=first_date, last_date=last_date)

    # --- passenger data (T-100) ---
    cagr = passenger_cagr(t100_rows)
    totals = annual_passenger_totals(t100_rows)
    complete_years = [year for year, entry in totals.items() if entry["complete"]]
    latest_year = max(complete_years) if complete_years else None
    latest_passengers = totals[latest_year]["passengers"] if latest_year else None
    seasonality = _seasonality_summary(t100_rows, cagr)

    # --- runways (OurAirports) ---
    runway_status = runway_count(runways)                     # raw: every row with closed == 0
    operational = operational_runway_count(runways)           # hard surface, not helipad
    operational_runways, runway_reason = _operational_runways(operational)

    # --- flights (BTS On-Time) ---
    if has_ontime:
        sched_peak = peak_hour_traffic(ontime_rows, airport, time_basis="scheduled", **period)
        actual_peak = peak_hour_traffic(ontime_rows, airport, time_basis="actual", **period)
        sched_spacing = peak_period_spacing(ontime_rows, airport, time_basis="scheduled", **period)
        actual_spacing = peak_period_spacing(ontime_rows, airport, time_basis="actual", **period)
        delays = delay_metrics(ontime_rows, airport)
        cancellations = cancellation_rate(ontime_rows, airport)
        counts = flight_counts(ontime_rows, airport)
        destinations = unique_destinations(ontime_rows, airport)
        long_haul = long_haul_summary(ontime_rows, airport, airports_by_code)
    else:
        sched_peak = actual_peak = sched_spacing = actual_spacing = None
        delays = cancellations = counts = destinations = long_haul = None

    def get(result, *keys):
        """Nested lookup that returns None when the result (source) is unavailable."""
        for key in keys:
            if result is None:
                return None
            result = result[key]
        return result

    p95 = get(sched_peak, "value")
    p95_per_runway = p95 / operational_runways if p95 is not None and operational_runways else None
    long_haul_known = get(long_haul, "known_distance")

    kpis = {
        DEMAND_GROWTH: {
            "signals": {"passenger_cagr": cagr["cagr"]},
            "context": {
                "cagr_first_year": cagr["first_year"],
                "cagr_last_year": cagr["last_year"],
                "cagr_years": cagr["years"],
                "cagr_first_passengers": cagr["first_passengers"],
                "cagr_last_passengers": cagr["last_passengers"],
                "cagr_reason_if_missing": cagr["reason"],
                "latest_complete_year": latest_year,
                "latest_annual_passengers": latest_passengers,
                # annual totals the passenger metrics are built from (year -> sum, months, complete)
                "annual_passengers_by_year": totals,
            },
        },
        OPERATIONAL_PRESSURE: {
            "signals": {
                "scheduled_p95_per_operational_runway": p95_per_runway,
                "scheduled_peak_spacing_minutes": get(sched_spacing, "median_spacing_minutes"),
                "departure_delayed_15_rate": get(delays, "departures", "delayed_15_plus_rate"),
                "arrival_delayed_15_rate": get(delays, "arrivals", "delayed_15_plus_rate"),
                "departure_delayed_60_rate": get(delays, "departures", "delayed_60_plus_rate"),
            },
            "context": {
                "scheduled_p95_movements_per_hour": p95,
                "operational_runways": operational_runways,
                "operational_runways_reason_if_missing": runway_reason,
                "operational_runway_detail": operational,
                "open_runways_all_surfaces": runway_status["open"],
                "median_departure_delay_minutes": get(delays, "departures", "median_delay_minutes"),
                "median_arrival_delay_minutes": get(delays, "arrivals", "median_delay_minutes"),
                "mean_departure_delay_minutes": get(delays, "departures", "mean_delay_minutes"),
                "mean_arrival_delay_minutes": get(delays, "arrivals", "mean_delay_minutes"),
                "eligible_departures": get(delays, "departures", "eligible"),
                "eligible_arrivals": get(delays, "arrivals", "eligible"),
                "cancellation_rate": get(cancellations, "rate"),
                "actual_p95_movements_per_hour": get(actual_peak, "value"),
                "actual_peak_spacing_minutes": get(actual_spacing, "median_spacing_minutes"),
                "note": "Delays and cancellations have many causes (weather, airlines, "
                        "network); they are not proof of airport infrastructure congestion. "
                        "Operational runway count (open, hard-surface, not helipad) is a crude "
                        "proxy: it ignores runway length, configuration, simultaneous-use "
                        "restrictions and runway capacity.",
            },
        },
        SCALE_CONNECTIVITY: {
            "signals": {
                "annual_passengers": latest_passengers,
                "scheduled_departures": get(counts, "departures", "scheduled"),
                "unique_destinations": get(destinations, "count"),
                "long_haul_flights": get(long_haul, "long_haul") if long_haul_known else None,
                "long_haul_share": get(long_haul, "long_haul_share"),
            },
            "context": {
                "annual_passengers_year": latest_year,
                "destinations": get(destinations, "destinations"),
                "long_haul_threshold_miles": get(long_haul, "threshold_miles"),
                "long_haul_known_distance": long_haul_known,
                "long_haul_unknown_distance": get(long_haul, "unknown_distance"),
                "long_haul_distance_source": get(long_haul, "distance_source"),
                "connectivity_is_complete_network": False,
                "connectivity_source_scope": ONTIME_SCOPE,
            },
        },
    }

    return {
        "airport": airport,
        "period": {"first_date": first_date.isoformat(), "last_date": last_date.isoformat()},
        "kpis": kpis,
        "capacity_pressure_inputs": {
            "is_proxy": True,
            "scored": False,
            "note": CAPACITY_PRESSURE_NOTE,
            "passenger_cagr": cagr["cagr"],
            "latest_annual_passengers": latest_passengers,
            "seasonality": seasonality,
            "operational_pressure": dict(kpis[OPERATIONAL_PRESSURE]["signals"]),
        },
        "coverage": {
            "bts_ontime": {
                "available": has_ontime,
                "rows_for_airport": len(ontime_rows),
                "scope": ONTIME_SCOPE,
            },
            "bts_t100": {
                "months": len(t100_rows),
                "complete_years": sorted(complete_years),
            },
            "ourairports_runways": {"all_rows": runway_status, "operational": operational},
        },
        "missing_signals": {
            family: [name for name, value in data["signals"].items() if value is None]
            for family, data in kpis.items()
        },
    }
