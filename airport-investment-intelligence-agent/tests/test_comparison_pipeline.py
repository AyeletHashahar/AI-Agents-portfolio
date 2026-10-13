"""Offline tests for the optimizations: batched T-100, OurAirports in-memory cache, On-Time
state pass-through, flight indexing, and the analyze_comparison_group pipeline."""

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import requests

from analysis import analyze_comparison_group, index_flights_by_airport
from data_sources.bts_ontime import BtsOnTimeClient
from data_sources.bts_t100 import PASSENGER_FIELDS, BtsT100Client
from data_sources.ourairports import OurAirportsClient
from kpis import build_airport_kpis
from metrics.passengers import annual_passenger_totals, passenger_cagr, seasonality_index


def fake_response(body):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(body).encode()
    return response


def t100_full_rows(code, totals):
    """Rows as the per-airport API returns them (many fields, numbers as strings)."""
    return [{"origin_airport_code": code, "reporting_month": f"{y}-{m:02d}-01T00:00:00.000",
             "total_passengers": str(total / 12 + m), "total_seats": "999", "year": str(y)}
            for y, total in totals.items() for m in range(1, 13)]


class NoNetwork(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("requests.sessions.Session.send",
                             side_effect=AssertionError("Real HTTP request attempted"))
        patcher.start()
        self.addCleanup(patcher.stop)


class T100BatchTests(NoNetwork):
    def test_one_request_with_only_needed_fields(self):
        rows = [{k: v for k, v in r.items() if k in PASSENGER_FIELDS}
                for r in t100_full_rows("AAA", {2022: 1200}) + t100_full_rows("BBB", {2022: 600})]
        with mock.patch("data_sources.bts_t100.requests.get", return_value=fake_response(rows)) as get:
            result = BtsT100Client().get_monthly_passengers_for_airports(["bbb", "AAA", "CCC"])
        self.assertEqual(get.call_count, 1)
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["$select"], "origin_airport_code,reporting_month,total_passengers")
        self.assertEqual(params["$where"], "origin_airport_code in ('AAA','BBB','CCC')")
        self.assertEqual(sorted(result), ["AAA", "BBB", "CCC"])
        self.assertEqual(len(result["AAA"]), 12)
        self.assertEqual(result["CCC"], [])                       # requested, no data
        self.assertIsInstance(result["AAA"][0]["total_passengers"], float)

    def test_pagination(self):
        page1 = [{"origin_airport_code": "AAA", "reporting_month": "2024-01-01T00:00:00.000",
                  "total_passengers": "1"}] * 3
        page2 = page1[:1]
        with mock.patch("data_sources.bts_t100._PAGE_SIZE", 3), \
                mock.patch("data_sources.bts_t100.requests.get",
                           side_effect=[fake_response(page1), fake_response(page2)]) as get:
            result = BtsT100Client().get_monthly_passengers_for_airports(["AAA"])
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args.kwargs["params"]["$offset"], 3)
        self.assertEqual(len(result["AAA"]), 4)

    def test_invalid_code_rejected_before_request(self):
        with mock.patch("data_sources.bts_t100.requests.get") as get:
            with self.assertRaises(ValueError):
                BtsT100Client().get_monthly_passengers_for_airports(["AAA", "x'); drop"])
        get.assert_not_called()

    def test_batched_rows_give_identical_passenger_metrics(self):
        full = t100_full_rows("AAA", {2022: 1200, 2023: 1300, 2024: 1500, 2025: 1550})
        batched_raw = [{k: v for k, v in r.items() if k in PASSENGER_FIELDS} for r in full]
        with mock.patch("data_sources.bts_t100.requests.get", return_value=fake_response(full)):
            per_airport = BtsT100Client().get_monthly_airport_stats("AAA")
        with mock.patch("data_sources.bts_t100.requests.get", return_value=fake_response(batched_raw)):
            batched = BtsT100Client().get_monthly_passengers_for_airports(["AAA"])["AAA"]
        self.assertEqual(annual_passenger_totals(per_airport), annual_passenger_totals(batched))
        self.assertEqual(passenger_cagr(per_airport), passenger_cagr(batched))
        years = [2022, 2023, 2024, 2025]
        self.assertEqual(seasonality_index(per_airport, years), seasonality_index(batched, years))


class OurAirportsCacheTests(NoNetwork):
    def test_csv_parsed_once_and_runways_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runways.csv"
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["airport_ident", "le_ident", "closed", "surface"])
                writer.writeheader()
                writer.writerow({"airport_ident": "KAAA", "le_ident": "09", "closed": "0", "surface": "ASP"})
                writer.writerow({"airport_ident": "KBBB", "le_ident": "18", "closed": "0", "surface": "CON"})
            client = OurAirportsClient(cache_dir=tmp)
            self.assertEqual(len(client.get_runways("kaaa")), 1)
            path.unlink()                                      # served from memory afterwards
            self.assertEqual(client.get_runways("KBBB")[0]["le_ident"], "18")
            self.assertEqual(client.get_runways("KZZZ"), [])
            returned = client.get_runways("KAAA")
            returned.clear()                                   # callers cannot corrupt the cache
            self.assertEqual(len(client.get_runways("KAAA")), 1)


class OnTimeStatePassThroughTests(NoNetwork):
    def test_known_rows_avoid_airport_lookup(self):
        airports_client = mock.Mock()
        airports_client.get_airport.side_effect = AssertionError("lookup not expected")
        airports_client.get_region_name.side_effect = lambda region: {"US-FL": "Florida"}[region]
        client = BtsOnTimeClient(airports_client=airports_client)
        with mock.patch.object(client, "_download_form_rows", return_value=[]) as download:
            client.get_flights(["PBI"], 2026, 4,
                               airport_rows={"PBI": {"iso_country": "US", "iso_region": "US-FL"}})
        download.assert_called_once_with("Florida", 2026, 4)


def flight(origin, dest, crs_dep="0600", dep_delay="0.00", distance="1000.00"):
    return {"FlightDate": "2026-04-01", "Origin": origin, "Dest": dest, "CRSDepTime": crs_dep,
            "CRSArrTime": "0900", "DepDelay": dep_delay, "ArrDelay": "0.00", "Cancelled": "0.00",
            "Diverted": "0.00", "Distance": distance}


FLIGHTS = [flight("AAA", "BBB"), flight("AAA", "CCC", "0610", "20.00"), flight("BBB", "AAA", "0700"),
           flight("CCC", "DDD", "0800", distance="4000.00"), flight("BBB", "CCC", "0630", "70.00")]
RUNWAYS = [{"closed": "0", "surface": "ASP", "le_ident": "09"}]
T100 = [{"reporting_month": f"{y}-{m:02d}-01T00:00:00.000", "total_passengers": 100.0 + y - 2000}
        for y in (2022, 2023, 2024, 2025) for m in range(1, 13)]


class FlightIndexTests(NoNetwork):
    def test_index_contains_each_airports_rows(self):
        index = index_flights_by_airport(FLIGHTS)
        self.assertEqual(len(index["AAA"]), 3)
        self.assertEqual(len(index["DDD"]), 1)
        self.assertEqual(sum(len(rows) for rows in index.values()), 2 * len(FLIGHTS))

    def test_indexed_rows_give_identical_kpis(self):
        index = index_flights_by_airport(FLIGHTS)
        period = dict(first_date=date(2026, 4, 1), last_date=date(2026, 4, 1))
        for code in ("AAA", "BBB", "CCC"):
            full = build_airport_kpis(code, flights=FLIGHTS, t100_rows=T100, runways=RUNWAYS, **period)
            indexed = build_airport_kpis(code, flights=index[code], t100_rows=T100, runways=RUNWAYS, **period)
            self.assertEqual(full, indexed, code)


class PipelineTests(NoNetwork):
    def candidates(self):
        base = {"iso_country": "US", "keywords": ""}
        return [
            {**base, "ident": "KAAA", "iata_code": "AAA", "local_code": "AAA", "iso_region": "US-AK"},
            {**base, "ident": "KBBB", "iata_code": "BBB", "local_code": "BBB", "iso_region": "US-AK"},
            {**base, "ident": "KDJT", "iata_code": "DJT", "local_code": "DJT", "iso_region": "US-FL",
             "keywords": "PBI, KPBI, MFW"},                                      # resolved via keyword
            {**base, "ident": "KNNN", "iata_code": "NNN", "local_code": "NNN", "iso_region": "US-AK"},  # not in BTS
        ]

    def test_end_to_end_with_fake_clients(self):
        ourairports = mock.Mock()
        ourairports.code_owners.return_value = {"AAA": {"KAAA"}, "BBB": {"KBBB"}, "DJT": {"KDJT"},
                                                "NNN": {"KNNN"}, "MFW": {"KMFW"}}
        ourairports.get_runways.return_value = RUNWAYS
        t100 = mock.Mock()
        t100.get_monthly_passengers_for_airports.side_effect = lambda codes: {
            code: (T100 if code in ("AAA", "BBB", "PBI", "MFW") else []) for code in codes}
        ontime = mock.Mock()
        ontime.get_flights.return_value = FLIGHTS + [flight("PBI", "AAA", "0900")]

        out = analyze_comparison_group(self.candidates(), year=2026, month=4,
                                       ourairports=ourairports, t100=t100, ontime=ontime)

        t100.get_monthly_passengers_for_airports.assert_called_once()   # one batched call
        self.assertEqual(set(t100.get_monthly_passengers_for_airports.call_args.args[0]),
                         {"AAA", "BBB", "DJT", "PBI", "MFW", "NNN"})
        args, kwargs = ontime.get_flights.call_args
        self.assertEqual(sorted(args[0]), ["AAA", "BBB", "PBI"])
        self.assertEqual(kwargs["airport_rows"]["PBI"]["ident"], "KDJT")   # state passed forward
        self.assertEqual(out["period"], {"first_date": "2026-04-01", "last_date": "2026-04-30"})

        ids = {r["ident"]: r for r in out["identifiers"]}
        self.assertEqual((ids["KDJT"]["bts_code"], ids["KDJT"]["source"]), ("PBI", "keywords"))
        self.assertEqual([r["ident"] for r in out["identifier_issues"]], ["KNNN"])

        airports = {a["airport"]: a for a in out["comparison"]["airports"]}
        self.assertEqual(set(airports), {"AAA", "BBB", "PBI", "NNN"})
        self.assertEqual(airports["PBI"]["raw_kpis"]["identifiers"]["ourairports_iata"], "DJT")
        nnn = airports["NNN"]
        self.assertEqual(nnn["score_status"], "insufficient_coverage")   # exposed, not zero activity
        self.assertEqual(nnn["raw_kpis"]["identifiers"]["status"], "not_found_in_bts")
        self.assertFalse(nnn["raw_kpis"]["coverage"]["bts_ontime"]["available"])

    def test_two_candidates_never_merged_into_one_bts_code(self):
        ourairports = mock.Mock()
        ourairports.code_owners.return_value = {}
        ourairports.get_runways.return_value = RUNWAYS
        t100 = mock.Mock()
        t100.get_monthly_passengers_for_airports.side_effect = lambda codes: {c: T100 for c in codes}
        ontime = mock.Mock()
        ontime.get_flights.return_value = FLIGHTS
        base = {"iso_country": "US", "iso_region": "US-AK", "keywords": ""}
        dupes = [{**base, "ident": "K1", "iata_code": "AAA", "local_code": "AAA"},
                 {**base, "ident": "K2", "iata_code": "AAA", "local_code": "AAA"}]
        out = analyze_comparison_group(dupes, year=2026, month=4,
                                       ourairports=ourairports, t100=t100, ontime=ontime)
        self.assertEqual([r["status"] for r in out["identifiers"]], ["exact", "ambiguous"])
        self.assertEqual(len(out["comparison"]["airports"]), 2)


if __name__ == "__main__":
    unittest.main()
