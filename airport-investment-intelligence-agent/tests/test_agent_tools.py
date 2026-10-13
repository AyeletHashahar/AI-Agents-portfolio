"""Offline tests for the three Agent tools (agent_tools).

A real OurAirportsClient reads synthetic CSV files; the BTS clients are small fakes with the
same interface. The real pipeline (analysis -> kpis -> scoring) runs underneath. No network.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from agent_tools import (AUTOMATIC_ANALYSIS_LIMIT, AgentToolContext, ToolContext, analyze_airport,
                         compare_airports, discover_airports)
from data_sources.bts_ontime import FORM_COLUMNS, BtsOnTimeClient
from data_sources.bts_t100 import BtsT100Client
from data_sources.bts_ontime import TranStatsFormError
from data_sources.ourairports import OurAirportsClient

AIRPORT_COLUMNS = ["id", "ident", "type", "name", "latitude_deg", "longitude_deg", "elevation_ft", "continent",
                   "iso_country", "iso_region", "municipality", "scheduled_service", "icao_code", "iata_code",
                   "gps_code", "local_code", "home_link", "wikipedia_link", "keywords"]
# ident, type, name, country, region, scheduled, iata, local, keywords
AIRPORTS = [
    ("PANC", "large_airport", "Anchorage Intl", "US", "US-AK", "yes", "ANC", "ANC", ""),
    ("PAFA", "medium_airport", "Fairbanks Intl", "US", "US-AK", "yes", "FAI", "FAI", ""),
    ("PAJN", "medium_airport", "Juneau Intl", "US", "US-AK", "yes", "JNU", "JNU", ""),
    ("PAXX", "small_airport", "Unverified Commuter Field", "US", "US-AK", "yes", "XXA", "XXA", ""),
    ("PAHO", "small_airport", "Old Service Field", "US", "US-AK", "yes", "HOA", "HOA", ""),
    ("PACG", "medium_airport", "Cargo Only Field", "US", "US-AK", "no", "CGO", "CGO", ""),
    ("1AK1", "small_airport", "Private Strip", "US", "US-AK", "no", "", "", ""),
    ("2AK2", "heliport", "City Heliport", "US", "US-AK", "yes", "HHA", "HHA", ""),
    ("KSEA", "large_airport", "Seattle Intl", "US", "US-WA", "yes", "SEA", "SEA", ""),
    ("KPBI", "large_airport", "President Donald J. Trump Intl", "US", "US-FL", "yes", "DJT", "DJT", "PBI, KPBI, MFW"),
    ("CYVR", "large_airport", "Vancouver Intl", "CA", "CA-BC", "yes", "YVR", "", ""),
]
# 40 extra small airports in Oregon (no BTS data) for group-size tests; no discovery test uses US-OR.
EXTRA_CODES = [f"Q{i:02d}" for i in range(40)]
AIRPORTS += [(f"KQ{i:02d}", "small_airport", f"Extra Field {i}", "US", "US-OR", "yes", f"Q{i:02d}", f"Q{i:02d}", "")
             for i in range(40)]
REGIONS = ["US-AK", "US-WA", "US-FL", "US-VT", "US-OR", "CA-BC"]
RUNWAYS = {"PANC": 3, "PAFA": 2, "PAJN": 1, "KSEA": 3, "KPBI": 2, "PAXX": 1, "PACG": 1, "PAHO": 1}


def t100_rows(code, first_year_total, growth, recent_zero=False, years=range(2022, 2026), partial=True):
    rows = []
    for i, year in enumerate(years):
        total = first_year_total * (1 + growth) ** i
        rows += [{"origin_airport_code": code, "reporting_month": f"{year}-{m:02d}-01T00:00:00.000",
                  "total_passengers": total / 12 * (1.3 if m in (6, 7, 8) else 0.88)} for m in range(1, 13)]
    if partial:
        rows += [{"origin_airport_code": code, "reporting_month": f"2026-{m:02d}-01T00:00:00.000",
                  "total_passengers": 0.0 if recent_zero else first_year_total / 12} for m in range(1, 5)]
    if recent_zero:
        for row in rows:
            if row["reporting_month"] >= "2025-05":
                row["total_passengers"] = 0.0
    return rows


T100 = {
    "ANC": t100_rows("ANC", 2_500_000, 0.02), "FAI": t100_rows("FAI", 500_000, 0.03),
    "JNU": t100_rows("JNU", 400_000, -0.001), "SEA": t100_rows("SEA", 25_000_000, 0.04),
    "PBI": t100_rows("PBI", 3_000_000, 0.06), "CGO": t100_rows("CGO", 10_000, 0.0, recent_zero=True),
    "HOA": t100_rows("HOA", 5_000, 0.0, years=range(2018, 2020), partial=False),
}


def flight(origin, dest, day, hhmm, delay, distance="1000.00"):
    return {"FlightDate": f"2026-07-0{day}", "Origin": origin, "Dest": dest, "CRSDepTime": hhmm,
            "CRSArrTime": "2300", "DepDelay": f"{delay:.2f}", "ArrDelay": f"{delay - 5:.2f}",
            "Cancelled": "0.00", "Diverted": "0.00", "Distance": distance}


def build_flights():
    rows = []
    plan = {"ANC": (40, 8, ["SEA", "FAI", "JNU", "ORD"]), "FAI": (14, 3, ["ANC", "SEA"]),
            "JNU": (10, 12, ["ANC", "SEA"]), "SEA": (60, 15, ["ANC", "PBI", "XLH", "LAX", "JNU"]),
            "PBI": (30, 20, ["SEA", "ATL", "JFK"])}
    for origin, (count, delay, dests) in plan.items():
        for k in range(count):
            dest = dests[k % len(dests)]
            rows.append(flight(origin, dest, 1 + k % 3, f"{6 + (k % 10):02d}{(k * 7) % 60:02d}",
                               delay * (k % 4) - 3, "4000.00" if dest == "XLH" else "1000.00"))
    return rows


FLIGHTS = build_flights()


class FakeT100:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def get_monthly_passengers_for_airports(self, codes):
        self.calls.append(sorted(codes))
        if self.fail:
            raise requests.ConnectionError("offline")
        return {code: [dict(r) for r in T100.get(code, [])] for code in codes}

    def latest_reporting_month(self):
        return "2026-04"


class FakeOnTime:
    def __init__(self, fail=False):
        self.fail, self.calls, self.last_requests = fail, [], []

    def latest_available_month(self):
        return (2026, 7)

    def get_flights(self, codes, year, month, allow_bulk_fallback=False, airport_rows=None):
        self.calls.append({"codes": sorted(codes), "year": year, "month": month,
                           "fallback": allow_bulk_fallback, "rows": sorted(airport_rows or {})})
        if self.fail:
            raise TranStatsFormError("form down")
        self.last_requests = [("form", "test", year, month)]
        return [f for f in FLIGHTS if f["Origin"] in codes or f["Dest"] in codes]


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("requests.sessions.Session.send",
                             side_effect=AssertionError("Real HTTP request attempted"))
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cache = Path(tmp.name)
        with open(cache / "airports.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AIRPORT_COLUMNS)
            writer.writeheader()
            for i, (ident, typ, name, country, region, sched, iata, local, kw) in enumerate(AIRPORTS):
                icao = "KDJT" if ident == "KPBI" else (ident if ident[0] in "PKC" else "")
                writer.writerow({"id": i, "ident": ident, "type": typ, "name": name, "latitude_deg": "60",
                                 "longitude_deg": "-150", "iso_country": country, "iso_region": region,
                                 "municipality": f"{name} City", "scheduled_service": sched, "icao_code": icao,
                                 "iata_code": iata, "gps_code": icao, "local_code": local, "keywords": kw})
        with open(cache / "regions.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "code", "name", "iso_country"])
            writer.writeheader()
            for i, code in enumerate(REGIONS):
                writer.writerow({"id": i, "code": code, "name": code, "iso_country": code[:2]})
        with open(cache / "runways.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["airport_ident", "le_ident", "closed", "surface", "length_ft"])
            writer.writeheader()
            for ident, count in RUNWAYS.items():
                for k in range(count):
                    writer.writerow({"airport_ident": ident, "le_ident": f"{k + 1:02d}", "closed": "0",
                                     "surface": "ASP", "length_ft": "9000"})
        self.t100, self.ontime = FakeT100(), FakeOnTime()
        self.ctx = ToolContext(ourairports=OurAirportsClient(cache_dir=cache), t100=self.t100, ontime=self.ontime)

    def assert_serializable(self, result):
        return json.loads(json.dumps(result))


# --- discover_airports -------------------------------------------------------------------

class DiscoverAirportsTests(ToolTestCase):
    def by_code(self, result):
        """Rows -> {code: {column: value}} using the result's own column list."""
        return {row[0]: dict(zip(result["columns"], row)) for row in result["airports"]}

    def test_single_state_with_activity_states(self):
        result = discover_airports("US", ["AK"], context=self.ctx)
        self.assert_serializable(result)
        self.assertEqual(result["status"], "ok")
        airports = self.by_code(result)
        self.assertEqual(set(airports), {"ANC", "FAI", "JNU", "XXA", "HOA", "CGO"})
        self.assertEqual(result["candidate_count"], 6)
        self.assertEqual(airports["ANC"]["commercial_activity"], "confirmed")
        self.assertEqual(airports["CGO"]["commercial_activity"], "reported_zero")
        self.assertEqual(airports["CGO"]["recent_12m_passengers"], 0)
        self.assertEqual(airports["HOA"]["commercial_activity"], "historical_only")
        self.assertEqual(airports["HOA"]["latest_bts_month"], "2019-12")
        self.assertEqual(result["activity_counts"],
                         {"confirmed": 3, "reported_zero": 1, "historical_only": 1, "unknown": 1})
        self.assertEqual(len(self.t100.calls), 1)                      # one batched T-100 call

    def test_missing_evidence_is_unknown_not_zero(self):
        xxa = self.by_code(discover_airports("US", ["AK"], context=self.ctx))["XXA"]
        self.assertEqual(xxa["commercial_activity"], "unknown")
        self.assertIsNone(xxa["latest_annual_passengers"])
        self.assertIsNone(xxa["recent_12m_passengers"])
        self.assertIsNone(xxa["passenger_cagr"])
        self.assertEqual(xxa["identifier_status"], "not_found_in_bts")

    def test_identifier_warning_and_excluded_counts(self):
        result = discover_airports("US", ["AK"], context=self.ctx)
        self.assertEqual([(w["code"], w["airport"]) for w in result["warnings"]],
                         [("identifier_not_found_in_bts", "XXA")])
        self.assertEqual(result["excluded_counts_by_type"], {"small_airport": 1})    # private strip

    def test_compact_row_with_narrowing_fields(self):
        result = discover_airports("US", ["AK"], context=self.ctx)
        self.assertEqual([row[0] for row in result["airports"]][:3], ["ANC", "FAI", "JNU"])
        anc = self.by_code(result)["ANC"]
        self.assertEqual((anc["name"], anc["state"], anc["airport_type"]), ("Anchorage Intl", "AK", "large_airport"))
        self.assertEqual(anc["latest_annual_year"], 2025)
        self.assertIsInstance(anc["latest_annual_passengers"], int)
        self.assertAlmostEqual(anc["passenger_cagr"], 0.02, places=4)   # existing Demand Growth CAGR
        self.assertEqual(anc["cagr_years"], "2022-2025")
        for row in result["airports"]:
            self.assertEqual(len(row), len(result["columns"]))
        # no detailed operational KPIs in discovery
        text = json.dumps(result)
        for kpi in ("scheduled_p95", "delayed_15", "operational_runways", "unique_destinations", "investment_score"):
            self.assertNotIn(kpi, text)

    def test_counts_and_automatic_limit_flag(self):
        result = discover_airports("US", ["AK"], context=self.ctx)
        self.assertEqual(result["automatic_analysis_limit"], AUTOMATIC_ANALYSIS_LIMIT)
        self.assertFalse(result["exceeds_automatic_analysis_limit"])
        large = discover_airports("US", ["OR"], context=self.ctx)        # 40 airports with no BTS data
        self.assertEqual(large["candidate_count"], 40)                  # all listed (scheduled service), none dropped
        self.assertTrue(large["exceeds_automatic_analysis_limit"])
        self.assertEqual(large["activity_counts"]["unknown"], 40)

    def test_multiple_states_and_alternate_identifier(self):
        result = discover_airports("US", ["AK", "FL", "WA"], context=self.ctx)
        airports = self.by_code(result)
        self.assertIn("SEA", airports)
        self.assertEqual(airports["PBI"]["identifier_status"], "alternate")
        self.assertEqual(result["geography"], {"country": "US", "states": ["AK", "FL", "WA"]})

    def test_empty_result(self):
        result = discover_airports("US", ["VT"], context=self.ctx)
        self.assertEqual((result["status"], result["candidate_count"], result["airports"]), ("ok", 0, []))
        self.assertEqual(result["activity_counts"], {"confirmed": 0, "reported_zero": 0, "historical_only": 0, "unknown": 0})

    def test_invalid_geography(self):
        result = discover_airports("US", ["XX"], context=self.ctx)
        self.assertEqual((result["status"], result["errors"][0]["code"]), ("error", "invalid_argument"))

    def test_t100_failure_is_structured(self):
        self.ctx.t100 = FakeT100(fail=True)
        result = discover_airports("US", ["AK"], context=self.ctx)
        self.assertEqual(result["errors"][0]["code"], "data_source_unavailable")


# --- compare_airports --------------------------------------------------------------------

GROUP = ["ANC", "FAI", "JNU", "SEA", "PBI", "XXA"]


class CompareAirportsTests(ToolTestCase):
    def test_normal_comparison(self):
        result = compare_airports(GROUP, context=self.ctx)
        self.assert_serializable(result)
        meta = result["metadata"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual((meta["analyzed_count"], meta["scored_count"]), (6, 5))
        self.assertEqual(meta["analysis_period"]["operations_month"], "2026-07")
        self.assertEqual(meta["analysis_period"]["selection"], "latest_available")
        self.assertTrue(meta["analysis_period"]["same_month_for_all_airports"])
        self.assertEqual(self.ontime.calls[0]["month"], 7)
        self.assertTrue(self.ontime.calls[0]["fallback"])
        self.assertEqual(len(self.t100.calls), 1)

    def test_top5_explains_the_score(self):
        top = compare_airports(GROUP, context=self.ctx)["top_candidates"]
        self.assertEqual(len(top), 5)
        self.assertEqual([t["rank"] for t in top], sorted(t["rank"] for t in top))
        first = top[0]
        for key in ("code", "name", "rank", "investment_score", "coverage", "families", "warnings"):
            self.assertIn(key, first)
        for family in ("demand_growth", "operational_pressure", "scale_connectivity"):
            self.assertIn("score", first["families"][family])
            for signal in first["families"][family]["signals"].values():
                self.assertEqual(set(signal), {"value", "percentile"})
        self.assertIn("latest_annual_passengers", first["families"]["demand_growth"]["context"])
        self.assertIn("operational_runways", first["families"]["operational_pressure"]["context"])

    def test_fewer_than_five_scored(self):
        result = compare_airports(["ANC", "FAI", "JNU"], context=self.ctx)
        self.assertEqual(len(result["top_candidates"]), 3)
        self.assertEqual(len(result["ranking"]), 3)

    def test_ranking_contains_every_scored_airport_compactly(self):
        result = compare_airports(GROUP, context=self.ctx)
        self.assertEqual({r["code"] for r in result["ranking"]}, {"ANC", "FAI", "JNU", "SEA", "PBI"})
        for row in result["ranking"]:
            self.assertEqual(set(row), {"code", "name", "rank", "investment_score", "coverage"})

    def test_missing_data_and_identifier_issues_propagate(self):
        result = compare_airports(GROUP, context=self.ctx)
        not_scored = {r["code"]: r for r in result["not_scored"]}
        self.assertEqual(not_scored["XXA"]["score_status"], "insufficient_coverage")
        self.assertEqual(not_scored["XXA"]["identifier_status"], "not_found_in_bts")
        self.assertEqual(not_scored["XXA"]["data_available"], {"bts_ontime_rows": 0, "t100_complete_years": 0})
        self.assertEqual([(i["code"], i["airport"]) for i in result["metadata"]["identifier_issues"]],
                         [("identifier_not_found_in_bts", "XXA")])
        self.assertEqual(result["metadata"]["identifier_issues"][0]["details"], {"candidates": ["XXA"]})
        pbi = next(t for t in result["top_candidates"] if t["code"] == "PBI")
        self.assertIn("identifier_alternate", [w["code"] for w in pbi["warnings"]])

    def test_output_is_compact_not_the_internal_payload(self):
        result = compare_airports(GROUP, context=self.ctx)
        for internal in ("raw_kpis", "normalized_signals", "capacity_pressure_inputs", "annual_passengers_by_year",
                         "operational_runway_detail", "effective_family_weights", "destinations", "families_raw"):
            self.assertEqual(list(find_key_values(result, internal)), [], internal)   # no such key anywhere
        self.assertLess(len(json.dumps(result)), 20_000)

    def test_thirty_accepted_normally(self):
        thirty = GROUP[:5] + EXTRA_CODES[:AUTOMATIC_ANALYSIS_LIMIT - 5]
        result = compare_airports(thirty, context=self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metadata"]["analyzed_count"], 30)
        self.assertNotIn("large_group_confirmed", [w["code"] for w in result["warnings"]])

    def test_thirty_one_without_override_requires_confirmation(self):
        thirty_one = GROUP[:5] + EXTRA_CODES[:26]
        result = compare_airports(thirty_one, context=self.ctx)
        self.assertEqual(result["status"], "error")
        error = result["errors"][0]
        self.assertEqual(error["code"], "comparison_group_requires_confirmation")
        self.assertEqual(error["details"], {"candidate_count": 31, "automatic_analysis_limit": 30})
        self.assertNotIn("unsupported", error["message"].lower())
        self.assertEqual((self.t100.calls, self.ontime.calls), ([], []))   # nothing retrieved

    def test_large_group_with_override_is_analysed_in_full_as_one_group(self):
        codes = GROUP[:5] + EXTRA_CODES                                      # 45 airports
        import agent_tools.compare as compare_module
        with mock.patch.object(compare_module, "analyze_comparison_group",
                               wraps=compare_module.analyze_comparison_group) as pipeline:
            result = compare_airports(codes, allow_large_group=True, context=self.ctx)
        self.assertEqual(result["status"], "ok")
        pipeline.assert_called_once()                                        # one shared scoring group
        self.assertEqual(len(pipeline.call_args.args[0]), 45)                # no truncation / selection
        meta = result["metadata"]
        self.assertEqual(meta["analyzed_count"], 45)
        self.assertEqual(len(result["ranking"]) + len(result["not_scored"]), 45)
        self.assertEqual(len(self.ontime.calls), 1)                          # one retrieval for the group
        self.assertEqual(len(self.t100.calls), 1)
        self.assertEqual([r["rank"] for r in result["ranking"]], [1, 2, 3, 4, 5])
        warning = next(w for w in result["warnings"] if w["code"] == "large_group_confirmed")
        self.assertEqual(warning["details"], {"candidate_count": 45, "automatic_analysis_limit": 30})

    def test_override_does_not_change_small_group_results(self):
        normal = compare_airports(GROUP, context=self.ctx)
        override = compare_airports(GROUP, allow_large_group=True, context=self.ctx)
        self.assertEqual(normal, override)

    def test_requested_period(self):
        result = compare_airports(["ANC", "FAI"], 2026, 3, context=self.ctx)
        self.assertEqual(result["metadata"]["analysis_period"]["operations_month"], "2026-03")
        self.assertEqual(result["metadata"]["analysis_period"]["selection"], "requested")
        self.assertEqual(self.ontime.calls[-1]["month"], 3)
        self.assertEqual(compare_airports(["ANC"], 2026, 9, context=self.ctx)["errors"][0]["code"],
                         "requested_period_unavailable")
        self.assertEqual(compare_airports(["ANC"], 2026, context=self.ctx)["errors"][0]["code"], "invalid_period")

    def test_duplicates_unsupported_and_single_airport(self):
        result = compare_airports(["ANC", "anc", "PANC", "YVR"], context=self.ctx)
        self.assertEqual(result["metadata"]["analyzed_count"], 1)
        self.assertEqual([e["code"] for e in result["errors"]], ["airport_not_supported"])
        self.assertIn("comparison_group_required", [w["code"] for w in result["warnings"]])
        self.assertEqual(result["ranking"], [])
        self.assertEqual(result["not_scored"][0]["score_status"], "comparison_group_required")

    def test_source_failure_is_structured(self):
        self.ctx.ontime = FakeOnTime(fail=True)
        result = compare_airports(["ANC", "FAI"], context=self.ctx)
        self.assertEqual((result["status"], result["errors"][0]["code"]), ("error", "data_source_unavailable"))

    def test_invalid_input(self):
        for bad in ([], "ANC", [""], None):
            self.assertEqual(compare_airports(bad, context=self.ctx)["errors"][0]["code"], "invalid_argument")


# --- analyze_airport ---------------------------------------------------------------------

def find_key_values(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from find_key_values(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from find_key_values(item, key)


class AnalyzeAirportTests(ToolTestCase):
    def test_detailed_raw_analysis(self):
        result = analyze_airport("ANC", context=self.ctx)
        self.assert_serializable(result)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["airport"]["code"], "ANC")
        self.assertEqual(result["airport"]["identifier_status"], "exact")
        self.assertEqual(result["analysis_period"]["operations_month"], "2026-07")
        ops = result["operational_pressure"]
        self.assertEqual(set(ops["signals"]), {"scheduled_p95_per_operational_runway", "scheduled_peak_spacing_minutes",
                                               "departure_delayed_15_rate", "arrival_delayed_15_rate",
                                               "departure_delayed_60_rate"})
        for key in ("operational_runways", "scheduled_p95_movements_per_hour", "cancellation_rate",
                    "mean_departure_delay_minutes", "actual_p95_movements_per_hour", "actual_peak_spacing_minutes"):
            self.assertIn(key, ops["context"])
        self.assertEqual(ops["context"]["operational_runways"], 3)
        demand = result["demand"]
        self.assertIsNotNone(demand["passenger_cagr"])
        self.assertIn("2025", {str(y) for y in demand["annual_passengers_by_year"]})
        self.assertEqual(demand["seasonality"]["peak_month"], 6)
        self.assertEqual(result["scale_connectivity"]["signals"]["scheduled_departures"], 40)
        self.assertTrue(result["capacity_pressure"]["is_proxy"])
        self.assertEqual(result["data_quality"]["data_coverage"], 1.0)

    def test_no_comparison_relative_score_or_rank(self):
        result = analyze_airport("ANC", context=self.ctx)
        self.assertEqual(result["relative_scoring"]["investment_score"], None)
        self.assertEqual(result["relative_scoring"]["rank"], None)
        self.assertTrue(all(v is None for v in find_key_values(result, "investment_score")))
        self.assertTrue(all(v is None for v in find_key_values(result, "rank")))
        self.assertEqual(list(find_key_values(result, "percentile")), [])        # no normalized values

    def test_missing_data_is_null_not_zero(self):
        result = analyze_airport("XXA", context=self.ctx)
        self.assertEqual(result["status"], "ok")
        codes = [w["code"] for w in result["warnings"]]
        self.assertIn("identifier_not_found_in_bts", codes)
        self.assertIn("insufficient_data", codes)
        self.assertTrue(all(v is None for v in result["operational_pressure"]["signals"].values()))
        self.assertIsNone(result["demand"]["passenger_cagr"])
        self.assertEqual(result["data_quality"]["data_coverage"], 0)

    def test_identifier_resolution(self):
        for code in ("PBI", "DJT", "KPBI"):
            airport = analyze_airport(code, context=self.ctx)["airport"]
            self.assertEqual((airport["code"], airport["bts_code"], airport["ourairports_iata"],
                              airport["identifier_status"]), ("PBI", "PBI", "DJT", "alternate"))

    def test_requested_period(self):
        result = analyze_airport("ANC", 2026, 5, context=self.ctx)
        self.assertEqual(result["analysis_period"]["operations_month"], "2026-05")
        self.assertEqual(result["analysis_period"]["selection"], "requested")
        self.assertEqual(self.ontime.calls[-1]["month"], 5)

    def test_errors(self):
        self.assertEqual(analyze_airport("ZZZ", context=self.ctx)["errors"][0]["code"], "airport_not_found")
        self.assertEqual(analyze_airport("YVR", context=self.ctx)["errors"][0]["code"], "airport_not_supported")
        self.assertEqual(analyze_airport("", context=self.ctx)["errors"][0]["code"], "invalid_argument")
        self.assertEqual(analyze_airport("ANC", 2031, 1, context=self.ctx)["errors"][0]["code"],
                         "requested_period_unavailable")


# --- in-session cache ---------------------------------------------------------------------

def form_rows(flights):
    """Synthetic TranStats form rows (form column names) for the given flight records."""
    rows = []
    for f in flights:
        row = {FORM_COLUMNS[k]: v for k, v in f.items() if k in FORM_COLUMNS}
        year, month, day = f["FlightDate"].split("-")
        row["FL_DATE"] = f"{int(month)}/{int(day)}/{year} 12:00:00 AM"
        for column in FORM_COLUMNS.values():
            row.setdefault(column, "")
        rows.append(row)
    return rows


class SessionCacheTests(ToolTestCase):
    """Real BtsOnTimeClient / BtsT100Client with a shared session cache; only their network
    functions are replaced by counting fakes."""

    def setUp(self):
        super().setUp()
        self.downloads, self.fetches = [], []
        self.fail_downloads = 0

        def download(client, state, year, month):
            self.downloads.append((state, year, month))
            if self.fail_downloads:
                self.fail_downloads -= 1
                raise TranStatsFormError("form down")
            return form_rows(FLIGHTS)

        def fetch(client, codes):
            self.fetches.append(sorted(codes))
            return {code: [dict(r) for r in T100.get(code, [])] for code in codes}

        for target, name, fake in ((BtsOnTimeClient, "_download_form_rows", download),
                                   (BtsT100Client, "_fetch_passengers", fetch)):
            patcher = mock.patch.object(target, name, autospec=True, side_effect=fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        bulk = mock.patch.object(BtsOnTimeClient, "_get_flights_from_bulk",
                                 side_effect=requests.ConnectionError("bulk offline"))
        bulk.start()
        self.addCleanup(bulk.stop)
        self.session = self.new_session()

    def new_session(self):
        session = AgentToolContext(ourairports=self.ctx.ourairports)
        session.ontime._latest_month = (2026, 7)          # skip the latest-month page request
        return session

    def test_compare_then_analyze_downloads_state_month_once(self):
        compare_airports(["ANC", "FAI", "JNU"], context=self.session)
        analyze_airport("FAI", context=self.session)
        analyze_airport("JNU", context=self.session)
        self.assertEqual(self.downloads, [("US-AK", 2026, 7)])               # one download for 3 calls
        self.assertEqual(self.session.ontime.last_requests, [("form_cached", "US-AK", 2026, 7)])
        self.assertEqual([codes for codes in self.fetches if codes], [["ANC", "FAI", "JNU"]])  # T-100 reused
        self.assertEqual(self.session.cache.stats()["entries"]["ontime_form"], 1)

    def test_different_month_and_state_are_fetched_separately(self):
        analyze_airport("FAI", context=self.session)
        analyze_airport("FAI", 2026, 5, context=self.session)                 # other month
        analyze_airport("SEA", context=self.session)                          # other state
        analyze_airport("JNU", 2026, 5, context=self.session)                 # cached (AK, May)
        self.assertEqual(self.downloads, [("US-AK", 2026, 7), ("US-AK", 2026, 5), ("US-WA", 2026, 7)])

    def test_failed_retrieval_is_not_cached(self):
        self.fail_downloads = 1
        failed = compare_airports(["ANC", "FAI"], context=self.session)
        self.assertEqual(failed["errors"][-1]["code"], "data_source_unavailable")
        self.assertFalse(self.session.cache.contains("ontime_form", ("US-AK", 2026, 7)))
        ok = compare_airports(["ANC", "FAI"], context=self.session)
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(self.downloads, [("US-AK", 2026, 7), ("US-AK", 2026, 7)])   # retried, not cached

    def test_cached_and_uncached_results_are_identical(self):
        compare_airports(["ANC", "FAI", "JNU"], context=self.session)
        cached = analyze_airport("FAI", context=self.session)
        uncached = analyze_airport("FAI", context=self.new_session())
        self.assertEqual(cached, uncached)
        self.assertEqual(compare_airports(["ANC", "FAI", "JNU"], context=self.session),
                         compare_airports(["ANC", "FAI", "JNU"], context=self.new_session()))

    def test_sessions_do_not_share_cache(self):
        analyze_airport("FAI", context=self.session)
        analyze_airport("FAI", context=self.new_session())
        self.assertEqual(len(self.downloads), 2)
        self.session.clear()
        analyze_airport("FAI", context=self.session)
        self.assertEqual(len(self.downloads), 3)


# --- composition -------------------------------------------------------------------------

class CompositionTests(ToolTestCase):
    def test_discover_codes_feed_compare(self):
        discovered = discover_airports("US", ["AK", "FL", "WA"], context=self.ctx)
        code_column = discovered["columns"].index("code")
        codes = [row[code_column] for row in discovered["airports"]]
        self.assertFalse(discovered["exceeds_automatic_analysis_limit"])
        result = compare_airports(codes, context=self.ctx)
        self.assertNotIn("airport_not_found", [e["code"] for e in result["errors"]])
        self.assertEqual(result["metadata"]["analyzed_count"], len(codes))

    def test_compare_codes_feed_analyze(self):
        result = compare_airports(GROUP, context=self.ctx)
        for row in result["ranking"] + result["not_scored"]:
            detail = analyze_airport(row["code"], context=self.ctx)
            self.assertEqual(detail["status"], "ok", row["code"])
            self.assertEqual(detail["airport"]["code"], row["code"])

    def test_same_raw_metrics_in_both_views(self):
        compared = compare_airports(["ANC", "FAI", "JNU"], context=self.ctx)
        fai_top = next(t for t in compared["top_candidates"] if t["code"] == "FAI")
        fai_detail = analyze_airport("FAI", context=self.ctx)
        self.assertEqual(fai_top["families"]["operational_pressure"]["signals"]["departure_delayed_15_rate"]["value"],
                         fai_detail["operational_pressure"]["signals"]["departure_delayed_15_rate"])


if __name__ == "__main__":
    unittest.main()
