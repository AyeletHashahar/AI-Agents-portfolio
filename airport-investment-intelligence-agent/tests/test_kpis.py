"""Offline tests for the KPI layer (kpis.build, kpis.scoring) with synthetic data."""

import copy
import unittest
from datetime import date
from unittest import mock

from kpis import build_airport_kpis, score_comparison_group
from kpis.config import (
    DEMAND_GROWTH,
    FAMILY_SIGNAL_WEIGHTS,
    INVESTMENT_SCORE_WEIGHTS,
    MIN_FAMILY_COVERAGE,
    MIN_INVESTMENT_SCORE_COVERAGE,
    OPERATIONAL_PRESSURE,
    SCALE_CONNECTIVITY,
    SIGNAL_DIRECTIONS,
)
from kpis.scoring import percentile_scores, signal_group_status

APRIL_1 = date(2026, 4, 1)


# --- synthetic raw data ------------------------------------------------------------------

def flight(origin="ANC", dest="SEA", crs_dep="0600", crs_arr="0900", dep_delay="0.00",
           arr_delay="0.00", cancelled="0.00", diverted="0.00", distance="1448.00"):
    return {"FlightDate": "2026-04-01", "Origin": origin, "Dest": dest,
            "CRSDepTime": crs_dep, "CRSArrTime": crs_arr, "DepDelay": dep_delay,
            "ArrDelay": arr_delay, "Cancelled": cancelled, "Diverted": diverted,
            "Distance": distance}


def t100(totals, partial_year=None):
    """Monthly T-100 rows: complete years from {year: annual total}, plus 4 months of partial_year."""
    rows = []
    for year, total in totals.items():
        rows += [{"reporting_month": f"{year}-{m:02d}-01T00:00:00.000",
                  "total_passengers": total / 12 * (1.5 if m == 7 else 0.5 if m == 1 else 1)}
                 for m in range(1, 13)]
    if partial_year:
        rows += [{"reporting_month": f"{partial_year}-{m:02d}-01T00:00:00.000", "total_passengers": 1.0}
                 for m in range(1, 5)]
    return rows


OPEN = {"closed": "0", "surface": "ASP", "le_ident": "07"}   # open, hard surface, not a helipad
WATER = {"closed": "0", "surface": "WATER", "le_ident": "07W"}
GRAVEL = {"closed": "0", "surface": "GRVL-G", "le_ident": "02"}
T100_GROWTH = t100({2022: 1000, 2023: 1100, 2024: 1210, 2025: 1331}, partial_year=2026)

# Departures from ANC 06:00-06:40 (delays 0, 20, 70, -5); arrivals 07:00 (+15) and 07:05 (0).
ANC_FLIGHTS = [
    flight(crs_dep="0600", dep_delay="0.00", dest="SEA"),
    flight(crs_dep="0610", dep_delay="20.00", dest="ORD", distance="2846.00"),
    flight(crs_dep="0620", dep_delay="70.00", dest="XLH", distance="3500.00"),   # long-haul
    flight(crs_dep="0640", dep_delay="-5.00", dest="SEA"),
    flight(origin="SEA", dest="ANC", crs_dep="0330", crs_arr="0700", arr_delay="15.00"),
    flight(origin="FAI", dest="ANC", crs_dep="0600", crs_arr="0705", arr_delay="0.00", distance="261.00"),
]


def build(flights=ANC_FLIGHTS, t100_rows=T100_GROWTH, runways=(OPEN, OPEN), airport="ANC", **kw):
    return build_airport_kpis(airport, flights=flights, t100_rows=t100_rows, runways=list(runways),
                              first_date=APRIL_1, last_date=APRIL_1, **kw)


class NoNetwork(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("requests.sessions.Session.send",
                             side_effect=AssertionError("Real HTTP request attempted"))
        patcher.start()
        self.addCleanup(patcher.stop)


# --- KPI construction --------------------------------------------------------------------

class DemandGrowthKpiTests(NoNetwork):
    def test_cagr_signal_and_latest_volume_context(self):
        record = build()
        growth = record["kpis"][DEMAND_GROWTH]
        self.assertAlmostEqual(growth["signals"]["passenger_cagr"], 0.10)   # 1000 -> 1331 over 3 years
        self.assertEqual(list(growth["signals"]), ["passenger_cagr"])        # YoY / seasonality not scored
        self.assertEqual((growth["context"]["cagr_first_year"], growth["context"]["cagr_last_year"]),
                         (2022, 2025))
        self.assertEqual(growth["context"]["latest_complete_year"], 2025)    # partial 2026 ignored
        self.assertAlmostEqual(growth["context"]["latest_annual_passengers"], 1331)

    def test_missing_passenger_data_is_none(self):
        record = build(t100_rows=[])
        self.assertIsNone(record["kpis"][DEMAND_GROWTH]["signals"]["passenger_cagr"])
        self.assertIsNone(record["kpis"][SCALE_CONNECTIVITY]["signals"]["annual_passengers"])
        self.assertEqual(record["missing_signals"][DEMAND_GROWTH], ["passenger_cagr"])


class OperationalPressureKpiTests(NoNetwork):
    def test_signals(self):
        ops = build()["kpis"][OPERATIONAL_PRESSURE]
        # 24 hourly counts: 06h = 4, 07h = 2, 22 zeros -> P95 rank 21.85 -> 0 + 0.85 * 2 = 1.7
        self.assertAlmostEqual(ops["context"]["scheduled_p95_movements_per_hour"], 1.7)
        self.assertAlmostEqual(ops["signals"]["scheduled_p95_per_operational_runway"], 0.85)  # 1.7 / 2
        # peak hours 06 + 07 -> gaps 10, 10, 20, 20, 5 -> median 10
        self.assertEqual(ops["signals"]["scheduled_peak_spacing_minutes"], 10)
        # departure delays 0, 20, 70, -5: two >= 15, one >= 60; arrivals 15, 0: one >= 15
        self.assertEqual(ops["signals"]["departure_delayed_15_rate"], 0.5)
        self.assertEqual(ops["signals"]["departure_delayed_60_rate"], 0.25)
        self.assertEqual(ops["signals"]["arrival_delayed_15_rate"], 0.5)

    def test_context_is_not_scored(self):
        ops = build()["kpis"][OPERATIONAL_PRESSURE]
        self.assertEqual(ops["context"]["median_departure_delay_minutes"], 10)   # median of 0,20,70,-5
        self.assertEqual(ops["context"]["median_arrival_delay_minutes"], 7.5)
        self.assertEqual(ops["context"]["cancellation_rate"], 0)
        self.assertIsNotNone(ops["context"]["actual_p95_movements_per_hour"])
        for name in ("median_departure_delay_minutes", "cancellation_rate", "actual_p95_movements_per_hour"):
            self.assertNotIn(name, FAMILY_SIGNAL_WEIGHTS[OPERATIONAL_PRESSURE])

    def test_runway_normalization_uses_operational_runways_only(self):
        # one hard runway + water lane + gravel strip -> divide by 1, not 3
        ops = build(runways=[OPEN, WATER, GRAVEL])["kpis"][OPERATIONAL_PRESSURE]
        self.assertAlmostEqual(ops["signals"]["scheduled_p95_per_operational_runway"], 1.7)
        self.assertEqual(ops["context"]["operational_runways"], 1)
        self.assertEqual(ops["context"]["open_runways_all_surfaces"], 3)        # raw count kept
        three = build(runways=[OPEN] * 3)["kpis"][OPERATIONAL_PRESSURE]
        self.assertAlmostEqual(three["signals"]["scheduled_p95_per_operational_runway"], 1.7 / 3)

    def test_runway_normalization_handles_zero_and_uncertain_counts(self):
        cases = {
            "no runways": [],
            "closed only": [{**OPEN, "closed": "1"}],
            "water and gravel only": [WATER, GRAVEL],
            "unknown status": [OPEN, {**OPEN, "closed": ""}],
            "unknown surface": [OPEN, {**OPEN, "surface": "PER"}],
        }
        for label, runways in cases.items():
            ops = build(runways=runways)["kpis"][OPERATIONAL_PRESSURE]
            self.assertIsNone(ops["signals"]["scheduled_p95_per_operational_runway"], label)
            self.assertIsNone(ops["context"]["operational_runways"], label)
            self.assertIsNotNone(ops["context"]["operational_runways_reason_if_missing"], label)
            self.assertAlmostEqual(ops["context"]["scheduled_p95_movements_per_hour"], 1.7)  # raw kept


class ScaleConnectivityKpiTests(NoNetwork):
    def test_signals(self):
        scale = build()["kpis"][SCALE_CONNECTIVITY]
        self.assertAlmostEqual(scale["signals"]["annual_passengers"], 1331)
        self.assertEqual(scale["signals"]["scheduled_departures"], 4)
        self.assertEqual(scale["signals"]["unique_destinations"], 3)          # SEA, ORD, XLH
        self.assertEqual(scale["context"]["destinations"], ["ORD", "SEA", "XLH"])
        # long-haul over all 6 rows touching ANC: only XLH (3,500 mi) >= 3,450
        self.assertEqual(scale["signals"]["long_haul_flights"], 1)
        self.assertAlmostEqual(scale["signals"]["long_haul_share"], 1 / 6)

    def test_coverage_metadata_preserved(self):
        record = build()
        scale = record["kpis"][SCALE_CONNECTIVITY]["context"]
        self.assertFalse(scale["connectivity_is_complete_network"])
        self.assertIn("international", scale["connectivity_source_scope"])
        self.assertEqual(scale["long_haul_threshold_miles"], 3450)
        self.assertEqual(scale["long_haul_distance_source"], {"bts": 6, "great_circle": 0})
        self.assertTrue(record["coverage"]["bts_ontime"]["available"])

    def test_unknown_distances_make_long_haul_missing_not_zero(self):
        rows = [flight(distance=""), flight(distance="")]
        scale = build(flights=rows)["kpis"][SCALE_CONNECTIVITY]
        self.assertIsNone(scale["signals"]["long_haul_flights"])
        self.assertIsNone(scale["signals"]["long_haul_share"])
        self.assertEqual(scale["signals"]["unique_destinations"], 1)


class NoOntimeDataTests(NoNetwork):
    def test_ontime_signals_are_none_not_zero(self):
        record = build(flights=[flight(origin="SEA", dest="PDX")])     # nothing touches ANC
        self.assertFalse(record["coverage"]["bts_ontime"]["available"])
        for signal in FAMILY_SIGNAL_WEIGHTS[OPERATIONAL_PRESSURE]:
            self.assertIsNone(record["kpis"][OPERATIONAL_PRESSURE]["signals"][signal], signal)
        for signal in ("scheduled_departures", "unique_destinations", "long_haul_flights", "long_haul_share"):
            self.assertIsNone(record["kpis"][SCALE_CONNECTIVITY]["signals"][signal], signal)
        self.assertAlmostEqual(record["kpis"][SCALE_CONNECTIVITY]["signals"]["annual_passengers"], 1331)


class CapacityPressureInputsTests(NoNetwork):
    def test_proxy_inputs_exposed_and_marked(self):
        capacity = build()["capacity_pressure_inputs"]
        self.assertTrue(capacity["is_proxy"])
        self.assertFalse(capacity["scored"])
        self.assertIn("PROXY", capacity["note"])
        self.assertAlmostEqual(capacity["passenger_cagr"], 0.10)
        self.assertEqual(capacity["seasonality"]["years"], [2022, 2023, 2024, 2025])
        self.assertEqual((capacity["seasonality"]["peak_month"], capacity["seasonality"]["trough_month"]), (7, 1))
        self.assertIn("scheduled_p95_per_operational_runway", capacity["operational_pressure"])
        for forbidden in ("terminal_capacity", "terminal_utilization", "unmet_demand"):
            self.assertNotIn(forbidden, capacity)

    def test_no_seasonality_without_trend_years(self):
        self.assertIsNone(build(t100_rows=[])["capacity_pressure_inputs"]["seasonality"])


# --- normalization and scoring -----------------------------------------------------------

def record(code, demand=0.0, ops=1.0, scale=1.0, **overrides):
    """Synthetic KPI record: every signal of a family gets the same raw 'level' (spacing is
    inverted so that a higher level always means MORE of the KPI). overrides: signal=value."""
    def value(signal, level):
        if signal in overrides:
            return overrides[signal]
        return -level if SIGNAL_DIRECTIONS[signal] == "lower" else level

    levels = {DEMAND_GROWTH: demand, OPERATIONAL_PRESSURE: ops, SCALE_CONNECTIVITY: scale}
    return {
        "airport": code,
        "kpis": {family: {"signals": {s: value(s, levels[family]) for s in weights}, "context": {}}
                 for family, weights in FAMILY_SIGNAL_WEIGHTS.items()},
        "capacity_pressure_inputs": {"is_proxy": True, "scored": False, "passenger_cagr": demand},
    }


def by_code(result):
    return {a["airport"]: a for a in result["airports"]}


class PercentileTests(NoNetwork):
    def test_distinct_values(self):
        self.assertEqual(percentile_scores({"A": 10, "B": 20, "C": 30}), {"A": 0, "B": 50, "C": 100})

    def test_ties_share_mid_rank(self):
        self.assertEqual(percentile_scores({"A": 10, "B": 20, "C": 20, "D": 30}),
                         {"A": 0, "B": 50, "C": 50, "D": 100})
        self.assertEqual(percentile_scores({"A": 5, "B": 5, "C": 5}), {"A": 50, "B": 50, "C": 50})

    def test_lower_direction(self):
        self.assertEqual(percentile_scores({"A": 5, "B": 15}, "lower"), {"A": 100, "B": 0})

    def test_missing_values_excluded_not_zero(self):
        self.assertEqual(percentile_scores({"A": 10, "B": None, "C": 30}), {"A": 0, "B": None, "C": 100})

    def test_fewer_than_two_values_cannot_be_ranked(self):
        self.assertEqual(percentile_scores({"A": 10, "B": None}), {"A": None, "B": None})
        self.assertEqual(percentile_scores({}), {})

    def test_invalid_values(self):
        for bad in (float("nan"), float("inf"), True, "10"):
            with self.assertRaises(ValueError):
                percentile_scores({"A": bad, "B": 1})
        with self.assertRaises(ValueError):
            percentile_scores({"A": 1, "B": 2}, "sideways")


class WeightConfigTests(NoNetwork):
    def test_investment_weights(self):
        self.assertEqual(INVESTMENT_SCORE_WEIGHTS,
                         {OPERATIONAL_PRESSURE: 0.40, DEMAND_GROWTH: 0.35, SCALE_CONNECTIVITY: 0.25})
        self.assertAlmostEqual(sum(INVESTMENT_SCORE_WEIGHTS.values()), 1.0)

    def test_family_weights_sum_to_one_and_have_directions(self):
        for family, weights in FAMILY_SIGNAL_WEIGHTS.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, msg=family)
            for signal in weights:
                self.assertIn(SIGNAL_DIRECTIONS[signal], ("higher", "lower"))
        ops = FAMILY_SIGNAL_WEIGHTS[OPERATIONAL_PRESSURE]
        self.assertLess(ops["arrival_delayed_15_rate"], ops["departure_delayed_15_rate"])

    def test_capacity_pressure_is_not_a_weighted_component(self):
        self.assertEqual(set(INVESTMENT_SCORE_WEIGHTS), {OPERATIONAL_PRESSURE, DEMAND_GROWTH, SCALE_CONNECTIVITY})
        self.assertEqual(set(FAMILY_SIGNAL_WEIGHTS), set(INVESTMENT_SCORE_WEIGHTS))
        self.assertFalse(any("capacity" in name for name in INVESTMENT_SCORE_WEIGHTS))


class SingleAirportTests(NoNetwork):
    def test_no_relative_score_is_fabricated(self):
        result = score_comparison_group([record("ANC")])
        self.assertEqual(result["status"], "comparison_group_required")
        self.assertIn("comparison group", result["message"])
        airport = result["airports"][0]
        self.assertEqual(airport["score_status"], "comparison_group_required")
        self.assertIsNone(airport["investment_score"])
        self.assertIsNone(airport["rank"])
        self.assertTrue(all(v is None for v in airport["family_scores"].values()))
        self.assertTrue(all(v is None for family in airport["normalized_signals"].values()
                            for v in family.values()))
        self.assertEqual(airport["raw_kpis"]["airport"], "ANC")                 # raw KPIs kept


class ScoringTests(NoNetwork):
    def test_weighted_score_and_ranking(self):
        # Family percentiles: X = (ops 0, demand 100, scale 50); Y = (50, 50, 100); Z = (100, 0, 0)
        result = score_comparison_group([
            record("X", ops=1, demand=3, scale=2),
            record("Y", ops=2, demand=2, scale=3),
            record("Z", ops=3, demand=1, scale=1),
        ])
        self.assertEqual(result["status"], "scored")
        airports = by_code(result)
        self.assertAlmostEqual(airports["X"]["investment_score"], 0.40 * 0 + 0.35 * 100 + 0.25 * 50)    # 47.5
        self.assertAlmostEqual(airports["Y"]["investment_score"], 0.40 * 50 + 0.35 * 50 + 0.25 * 100)   # 62.5
        self.assertAlmostEqual(airports["Z"]["investment_score"], 0.40 * 100 + 0.35 * 0 + 0.25 * 0)     # 40
        self.assertEqual([a["airport"] for a in result["airports"]], ["Y", "X", "Z"])
        self.assertEqual([a["rank"] for a in result["airports"]], [1, 2, 3])
        for airport in airports.values():
            self.assertEqual(airport["score_status"], "complete")
            self.assertEqual(airport["coverage"], 1.0)
            self.assertEqual(airport["effective_family_weights"], INVESTMENT_SCORE_WEIGHTS)
            for family, weights in FAMILY_SIGNAL_WEIGHTS.items():
                for signal, weight in weights.items():
                    self.assertAlmostEqual(airport["families"][family]["effective_weights"][signal], weight)
        self.assertEqual(result["non_discriminative_signals"], [])

    def test_lower_spacing_means_more_pressure(self):
        airports = by_code(score_comparison_group([record("A", ops=1), record("B", ops=2)]))
        self.assertEqual(airports["B"]["normalized_signals"][OPERATIONAL_PRESSURE]["scheduled_peak_spacing_minutes"], 100)
        self.assertLess(airports["B"]["raw_kpis"]["kpis"][OPERATIONAL_PRESSURE]["signals"]["scheduled_peak_spacing_minutes"],
                        airports["A"]["raw_kpis"]["kpis"][OPERATIONAL_PRESSURE]["signals"]["scheduled_peak_spacing_minutes"])

    def test_tied_airports_share_rank(self):
        # Only demand differs; ops and scale are identical for everyone (non-discriminative).
        result = score_comparison_group([record("A", demand=2), record("B", demand=2), record("C", demand=1)])
        airports = by_code(result)
        self.assertEqual(airports["A"]["investment_score"], 75)              # mid-rank of the tie
        self.assertEqual(airports["A"]["investment_score"], airports["B"]["investment_score"])
        self.assertEqual([(a["airport"], a["rank"]) for a in result["airports"]], [("A", 1), ("B", 1), ("C", 3)])

    def test_capacity_pressure_inputs_do_not_change_the_score(self):
        records = [record("A", demand=1, ops=2, scale=3), record("B", demand=3, ops=1, scale=2)]
        before = {a["airport"]: a["investment_score"] for a in score_comparison_group(records)["airports"]}
        changed = copy.deepcopy(records)
        changed[0]["capacity_pressure_inputs"] = {"is_proxy": True, "passenger_cagr": 99, "seasonality": {"peak_index": 9}}
        after = {a["airport"]: a["investment_score"] for a in score_comparison_group(changed)["airports"]}
        self.assertEqual(before, after)

    def test_raw_values_preserved_next_to_scores(self):
        records = [record("A"), record("B", ops=2)]
        result = score_comparison_group(records)
        self.assertIs(by_code(result)["A"]["raw_kpis"], records[0])
        self.assertEqual(result["weights"]["investment"], INVESTMENT_SCORE_WEIGHTS)
        self.assertIn("mid-rank", result["percentile_method"])

    def test_invalid_groups(self):
        with self.assertRaises(ValueError):
            score_comparison_group([])
        with self.assertRaises(ValueError):
            score_comparison_group([record("A"), record("A")])


class NonDiscriminativeTests(NoNetwork):
    def test_group_status(self):
        self.assertEqual(signal_group_status({"A": 0, "B": 0, "C": 0}), "non_discriminative")
        self.assertEqual(signal_group_status({"A": 7.5, "B": 7.5}), "non_discriminative")
        self.assertEqual(signal_group_status({"A": 0, "B": 0, "C": None}), "non_discriminative")
        self.assertEqual(signal_group_status({"A": 1, "B": 2}), "ranked")
        self.assertEqual(signal_group_status({"A": 1, "B": None}), "not_rankable")

    def test_all_zero_signals_are_excluded_and_weight_redistributed(self):
        # Long-haul = 0 everywhere (like ANC, FAI, JNU); the other scale signals vary.
        records = [record(code, scale=level, long_haul_flights=0, long_haul_share=0.0)
                   for code, level in (("A", 1), ("B", 2), ("C", 3))]
        result = score_comparison_group(records)
        # (only scale varies in these synthetic records, so demand/ops are non-discriminative too)
        self.assertEqual([s for s in result["non_discriminative_signals"] if s.startswith(SCALE_CONNECTIVITY)],
                         [f"{SCALE_CONNECTIVITY}.long_haul_flights", f"{SCALE_CONNECTIVITY}.long_haul_share"])
        a = by_code(result)["A"]
        scale = a["families"][SCALE_CONNECTIVITY]
        self.assertEqual(scale["non_discriminative_signals"], ["long_haul_flights", "long_haul_share"])
        self.assertEqual(scale["missing_signals"], [])
        self.assertIsNone(a["normalized_signals"][SCALE_CONNECTIVITY]["long_haul_share"])
        self.assertEqual(a["raw_kpis"]["kpis"][SCALE_CONNECTIVITY]["signals"]["long_haul_share"], 0.0)  # raw kept
        # 0.40 / 0.20 / 0.20 redistributed over 0.80 -> 0.50 / 0.25 / 0.25
        self.assertEqual(scale["effective_weights"],
                         {"annual_passengers": 0.5, "scheduled_departures": 0.25, "unique_destinations": 0.25})
        self.assertEqual(scale["score"], 0)       # lowest airport: no constant 50-point component
        self.assertEqual(by_code(result)["C"]["families"][SCALE_CONNECTIVITY]["score"], 100)
        self.assertEqual(scale["coverage"], 1.0)  # not missing data -> coverage unaffected
        self.assertEqual(FAMILY_SIGNAL_WEIGHTS[SCALE_CONNECTIVITY]["long_haul_share"], 0.10)  # config unchanged

    def test_identical_nonzero_signal_drops_whole_family(self):
        records = [record(code, ops=level, scale=level, passenger_cagr=0.05)
                   for code, level in (("A", 1), ("B", 2))]
        result = score_comparison_group(records)
        b = by_code(result)["B"]
        self.assertIsNone(b["family_scores"][DEMAND_GROWTH])
        self.assertIn("non_discriminative", b["missing_families"][DEMAND_GROWTH])
        self.assertAlmostEqual(b["effective_family_weights"][OPERATIONAL_PRESSURE], 0.40 / 0.65)
        self.assertAlmostEqual(b["effective_family_weights"][SCALE_CONNECTIVITY], 0.25 / 0.65)
        self.assertAlmostEqual(b["investment_score"], 100)
        self.assertEqual((b["coverage"], b["score_status"]), (1.0, "complete"))

    def test_mixture_of_discriminative_and_non_discriminative(self):
        # Ops: p95 per runway and spacing vary, the three delay rates are identical.
        records = [record(code, ops=level, departure_delayed_15_rate=0.1, arrival_delayed_15_rate=0.1,
                          departure_delayed_60_rate=0.02) for code, level in (("A", 1), ("B", 2), ("C", 3))]
        ops = by_code(score_comparison_group(records))["B"]["families"][OPERATIONAL_PRESSURE]
        self.assertEqual(ops["non_discriminative_signals"],
                         ["departure_delayed_15_rate", "arrival_delayed_15_rate", "departure_delayed_60_rate"])
        self.assertAlmostEqual(ops["effective_weights"]["scheduled_p95_per_operational_runway"], 0.50 / 0.70)
        self.assertAlmostEqual(ops["effective_weights"]["scheduled_peak_spacing_minutes"], 0.20 / 0.70)
        self.assertAlmostEqual(ops["score"], 50)


class MissingDataTests(NoNetwork):
    def test_missing_signal_redistributed_within_family(self):
        result = score_comparison_group([
            record("A", ops=1), record("B", ops=2),
            record("C", ops=3, scheduled_p95_per_operational_runway=None),   # e.g. runway count uncertain
        ])
        airports = by_code(result)
        c = airports["C"]
        ops = c["families"][OPERATIONAL_PRESSURE]
        self.assertEqual(ops["missing_signals"], ["scheduled_p95_per_operational_runway"])
        self.assertEqual(ops["non_discriminative_signals"], [])
        self.assertAlmostEqual(ops["coverage"], 0.50)
        # remaining 0.20 / 0.15 / 0.075 / 0.075 over 0.50
        self.assertAlmostEqual(ops["effective_weights"]["scheduled_peak_spacing_minutes"], 0.40)
        self.assertAlmostEqual(ops["effective_weights"]["departure_delayed_15_rate"], 0.30)
        self.assertNotIn("scheduled_p95_per_operational_runway", ops["effective_weights"])
        self.assertEqual(ops["score"], 100)
        self.assertAlmostEqual(c["coverage"], 0.40 * 0.50 + 0.35 + 0.25)       # 0.80
        self.assertEqual(c["score_status"], "partial")
        self.assertIsNotNone(c["investment_score"])
        self.assertIsNone(c["normalized_signals"][OPERATIONAL_PRESSURE]["scheduled_p95_per_operational_runway"])
        # C's missing value is not 0: the signal is ranked among A and B only.
        self.assertEqual(airports["A"]["normalized_signals"][OPERATIONAL_PRESSURE]["scheduled_p95_per_operational_runway"], 0)
        self.assertEqual(airports["B"]["normalized_signals"][OPERATIONAL_PRESSURE]["scheduled_p95_per_operational_runway"], 100)

    def test_missing_and_non_discriminative_tracked_separately(self):
        result = score_comparison_group([record("A", scale=1, long_haul_share=0.0),
                                         record("B", scale=2, long_haul_share=0.0),
                                         record("C", scale=3, long_haul_share=None)])
        self.assertEqual(result["signal_status"][SCALE_CONNECTIVITY]["long_haul_share"], "non_discriminative")
        a_scale = by_code(result)["A"]["families"][SCALE_CONNECTIVITY]
        c_scale = by_code(result)["C"]["families"][SCALE_CONNECTIVITY]
        self.assertEqual((a_scale["non_discriminative_signals"], a_scale["missing_signals"]), (["long_haul_share"], []))
        self.assertEqual((c_scale["non_discriminative_signals"], c_scale["missing_signals"]), ([], ["long_haul_share"]))
        self.assertEqual(a_scale["coverage"], 1.0)
        self.assertAlmostEqual(c_scale["coverage"], 0.90)

    def test_pair_where_one_airport_lacks_spacing_is_still_scored(self):
        result = score_comparison_group([record("A", ops=1, demand=1, scale=1),
                                         record("B", ops=2, demand=2, scale=2, scheduled_peak_spacing_minutes=None)])
        self.assertEqual(result["signal_status"][OPERATIONAL_PRESSURE]["scheduled_peak_spacing_minutes"], "not_rankable")
        a, b = by_code(result)["A"], by_code(result)["B"]
        self.assertEqual(a["families"][OPERATIONAL_PRESSURE]["not_rankable_signals"], ["scheduled_peak_spacing_minutes"])
        self.assertEqual(b["families"][OPERATIONAL_PRESSURE]["missing_signals"], ["scheduled_peak_spacing_minutes"])
        self.assertEqual((a["coverage"], a["score_status"]), (1.0, "complete"))
        self.assertAlmostEqual(b["coverage"], 1 - 0.40 * 0.20)                   # 0.92
        self.assertEqual(b["score_status"], "partial")
        self.assertEqual([x["rank"] for x in result["airports"]], [1, 2])

    def test_signal_with_only_one_value_is_not_rankable(self):
        result = score_comparison_group([record("A", ops=1, scale=1, passenger_cagr=0.05),
                                         record("B", ops=2, scale=2, passenger_cagr=None)])
        a, b = by_code(result)["A"], by_code(result)["B"]
        self.assertIsNone(a["normalized_signals"][DEMAND_GROWTH]["passenger_cagr"])
        self.assertEqual(a["families"][DEMAND_GROWTH]["not_rankable_signals"], ["passenger_cagr"])
        self.assertIsNone(a["family_scores"][DEMAND_GROWTH])
        self.assertAlmostEqual(b["coverage"], 0.65)
        self.assertIsNotNone(b["investment_score"])                            # 0.65 >= 0.60


class FamilyCoverageTests(NoNetwork):
    """MIN_FAMILY_COVERAGE = 0.50 on Operational Pressure (weights 0.50/0.20/0.15/0.075/0.075)."""

    def ops_family_of_c(self, **missing):
        result = score_comparison_group([record("A", ops=1, demand=1, scale=1),
                                         record("B", ops=2, demand=2, scale=2),
                                         record("C", ops=3, demand=3, scale=3, **missing)])
        return by_code(result)["C"]

    def test_threshold_value(self):
        self.assertEqual(MIN_FAMILY_COVERAGE, 0.50)

    def test_exactly_at_threshold_is_scored(self):
        c = self.ops_family_of_c(scheduled_p95_per_operational_runway=None)
        ops = c["families"][OPERATIONAL_PRESSURE]
        self.assertAlmostEqual(ops["coverage"], 0.50)          # 0.2 + 0.15 + 0.075 + 0.075 (float tolerance)
        self.assertFalse(ops["insufficient_family_coverage"])
        self.assertEqual(ops["score"], 100)

    def test_just_below_threshold_has_no_family_score(self):
        c = self.ops_family_of_c(scheduled_p95_per_operational_runway=None, arrival_delayed_15_rate=None)
        ops = c["families"][OPERATIONAL_PRESSURE]
        self.assertAlmostEqual(ops["coverage"], 0.425)
        self.assertTrue(ops["insufficient_family_coverage"])
        self.assertIsNone(ops["score"])
        self.assertEqual(ops["effective_weights"], {})        # no redistribution of the minor signals
        self.assertIn("insufficient_family_coverage", ops["reason_if_no_score"])
        self.assertIn("insufficient_family_coverage", c["missing_families"][OPERATIONAL_PRESSURE])
        # available signals stay visible, raw and normalized
        self.assertEqual(c["normalized_signals"][OPERATIONAL_PRESSURE]["departure_delayed_15_rate"], 100)
        self.assertEqual(c["raw_kpis"]["kpis"][OPERATIONAL_PRESSURE]["signals"]["departure_delayed_15_rate"], 3)
        # the other families carry the score: 0.35 / 0.60 and 0.25 / 0.60
        self.assertAlmostEqual(c["effective_family_weights"][DEMAND_GROWTH], 0.35 / 0.60)
        self.assertNotIn(OPERATIONAL_PRESSURE, c["effective_family_weights"])
        self.assertAlmostEqual(c["coverage"], 0.40 * 0.425 + 0.35 + 0.25)
        self.assertEqual(c["score_status"], "partial")

    def test_above_threshold_is_scored(self):
        c = self.ops_family_of_c(scheduled_peak_spacing_minutes=None)
        ops = c["families"][OPERATIONAL_PRESSURE]
        self.assertAlmostEqual(ops["coverage"], 0.80)
        self.assertFalse(ops["insufficient_family_coverage"])
        self.assertAlmostEqual(ops["effective_weights"]["scheduled_p95_per_operational_runway"], 0.50 / 0.80)
        self.assertEqual(ops["score"], 100)

    def test_minor_signal_alone_cannot_carry_the_family(self):
        only_arrivals = {s: None for s in FAMILY_SIGNAL_WEIGHTS[OPERATIONAL_PRESSURE] if s != "arrival_delayed_15_rate"}
        ops = self.ops_family_of_c(**only_arrivals)["families"][OPERATIONAL_PRESSURE]
        self.assertAlmostEqual(ops["coverage"], 0.075)
        self.assertIsNone(ops["score"])

    def test_non_discriminative_signals_do_not_reduce_family_coverage(self):
        # p95 and spacing identical for everyone (0.70 of the family weight) -> coverage stays 1.0
        records = [record(code, ops=level, scheduled_p95_per_operational_runway=2.0,
                          scheduled_peak_spacing_minutes=10.0) for code, level in (("A", 1), ("B", 2))]
        ops = by_code(score_comparison_group(records))["B"]["families"][OPERATIONAL_PRESSURE]
        self.assertEqual(ops["coverage"], 1.0)
        self.assertFalse(ops["insufficient_family_coverage"])
        self.assertEqual(ops["score"], 100)


class InvestmentCoverageTests(NoNetwork):
    ONTIME_SIGNALS = {s: None for s in FAMILY_SIGNAL_WEIGHTS[OPERATIONAL_PRESSURE]}

    def test_threshold_value(self):
        self.assertEqual(MIN_INVESTMENT_SCORE_COVERAGE, 0.60)

    def test_losing_one_whole_family_still_scored_at_threshold(self):
        # No operational data at all: coverage = 0.35 + 0.25 = 0.60 (exactly the minimum)
        result = score_comparison_group([record("A", ops=1, demand=1, scale=1),
                                         record("B", ops=2, demand=2, scale=2),
                                         record("C", demand=3, scale=3, **self.ONTIME_SIGNALS)])
        c = by_code(result)["C"]
        self.assertAlmostEqual(c["coverage"], 0.60)
        self.assertEqual(c["score_status"], "partial")
        self.assertIn(OPERATIONAL_PRESSURE, c["missing_families"])
        self.assertAlmostEqual(c["effective_family_weights"][DEMAND_GROWTH], 0.35 / 0.60)
        self.assertAlmostEqual(c["investment_score"], 100)
        self.assertEqual(c["rank"], 1)

    def test_no_ontime_data_is_insufficient(self):
        # Like an airport without BTS On-Time rows: no ops, and only annual passengers in scale.
        no_ontime = dict(self.ONTIME_SIGNALS, scheduled_departures=None, unique_destinations=None,
                         long_haul_flights=None, long_haul_share=None)
        result = score_comparison_group([record("A", ops=1, demand=1, scale=1),
                                         record("B", ops=2, demand=2, scale=2),
                                         record("C", demand=3, scale=3, **no_ontime)])
        c = by_code(result)["C"]
        self.assertAlmostEqual(c["coverage"], 0.35 + 0.25 * 0.40)             # 0.45
        self.assertEqual(c["score_status"], "insufficient_coverage")
        self.assertIsNone(c["investment_score"])
        self.assertIsNone(c["rank"])
        self.assertIsNotNone(c["family_scores"][DEMAND_GROWTH])                # partial results still visible
        self.assertEqual([a["airport"] for a in result["airports"]], ["B", "A", "C"])
        self.assertIn("2 of 3", result["message"])

    def test_just_below_threshold(self):
        # whole ops family + arrival delay... : remove ops (0.40) and long-haul share (0.25 * 0.10)
        result = score_comparison_group([record("A", ops=1, demand=1, scale=1),
                                         record("B", ops=2, demand=2, scale=2),
                                         record("C", demand=3, scale=3, long_haul_share=None, **self.ONTIME_SIGNALS)])
        c = by_code(result)["C"]
        self.assertAlmostEqual(c["coverage"], 0.575)
        self.assertEqual(c["score_status"], "insufficient_coverage")


class EndToEndTests(NoNetwork):
    def test_build_then_score_two_airports(self):
        # FAI needs a peak hour with two movements (08:00, 08:15) so it has a peak spacing.
        fai_flights = [flight(origin="FAI", dest="ANC", crs_dep="0800", crs_arr="0900", distance="261.00"),
                       flight(origin="FAI", dest="SEA", crs_dep="0815", crs_arr="1500", dep_delay="30.00"),
                       flight(origin="ANC", dest="FAI", crs_dep="1200", crs_arr="1300", arr_delay="5.00")]
        flights = ANC_FLIGHTS + fai_flights
        anc = build(flights=flights)
        fai = build(flights=flights, airport="FAI", runways=[OPEN, WATER, GRAVEL],
                    t100_rows=t100({2022: 500, 2023: 500, 2024: 520, 2025: 540}))
        self.assertEqual(fai["kpis"][OPERATIONAL_PRESSURE]["context"]["operational_runways"], 1)
        result = score_comparison_group([anc, fai])
        airports = by_code(result)
        self.assertEqual(result["status"], "scored")
        for code in ("ANC", "FAI"):
            self.assertEqual(airports[code]["score_status"], "complete", airports[code]["families"])
        # ANC has more passengers, departures, destinations and the only long-haul flight
        anc_scale = airports["ANC"]["normalized_signals"][SCALE_CONNECTIVITY]
        self.assertEqual(anc_scale, {s: 100 for s in FAMILY_SIGNAL_WEIGHTS[SCALE_CONNECTIVITY]})
        self.assertEqual(airports["ANC"]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
