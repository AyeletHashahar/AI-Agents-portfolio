"""Unit tests for metrics.passengers using small synthetic BTS T-100 rows (no network)."""

import unittest
from unittest import mock

import metrics.passengers as passengers_module
from metrics.passengers import (
    annual_passenger_totals,
    passenger_cagr,
    seasonality_index,
    yoy_monthly_growth,
)


def row(year, month, passengers):
    """Synthetic T-100 row; passengers=None leaves the field out (as data.bts.gov does for nulls)."""
    record = {"reporting_month": f"{year}-{month:02d}-01T00:00:00.000", "origin_airport_code": "ANC"}
    if passengers is not None:
        record["total_passengers"] = float(passengers)
    return record


def full_year(year, values):
    return [row(year, month, value) for month, value in zip(range(1, 13), values)]


class NoNetwork(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("requests.sessions.Session.send",
                             side_effect=AssertionError("Real HTTP request attempted"))
        patcher.start()
        self.addCleanup(patcher.stop)


class AnnualTotalsTests(NoNetwork):
    def test_complete_and_partial_years(self):
        rows = full_year(2024, [100] * 12) + [row(2025, 1, 10), row(2025, 2, 20), row(2025, 3, 30)]
        self.assertEqual(annual_passenger_totals(rows), {
            2024: {"passengers": 1200.0, "months": 12, "complete": True},
            2025: {"passengers": 60.0, "months": 3, "complete": False},
        })

    def test_missing_month_value_is_not_counted_as_zero(self):
        values = [100] * 12
        values[5] = None
        result = annual_passenger_totals(full_year(2024, values))
        self.assertEqual(result[2024], {"passengers": 1100.0, "months": 11, "complete": False})

    def test_year_with_only_missing_values(self):
        self.assertEqual(annual_passenger_totals([row(2024, 1, None)]),
                         {2024: {"passengers": None, "months": 0, "complete": False}})

    def test_zero_passengers_is_a_known_value(self):
        self.assertEqual(annual_passenger_totals([row(2024, 1, 0)])[2024],
                         {"passengers": 0.0, "months": 1, "complete": False})

    def test_empty_input(self):
        self.assertEqual(annual_passenger_totals([]), {})

    def test_duplicate_or_invalid_month_rejected(self):
        with self.assertRaises(ValueError):
            annual_passenger_totals([row(2024, 1, 10), row(2024, 1, 20)])
        with self.assertRaises(ValueError):
            annual_passenger_totals([{"total_passengers": 5.0}])


class YoyGrowthTests(NoNetwork):
    def test_normal_case(self):
        rows = [row(2024, 1, 100), row(2025, 1, 110), row(2025, 2, 50)]
        self.assertEqual(yoy_monthly_growth(rows), [
            {"month": "2024-01", "passengers": 100.0, "prior_year_passengers": None, "growth": None},
            {"month": "2025-01", "passengers": 110.0, "prior_year_passengers": 100.0,
             "growth": 0.1},
            {"month": "2025-02", "passengers": 50.0, "prior_year_passengers": None, "growth": None},
        ])

    def test_decline_is_negative(self):
        result = yoy_monthly_growth([row(2024, 3, 200), row(2025, 3, 150)])
        self.assertEqual(result[-1]["growth"], -0.25)

    def test_unknown_or_zero_values_give_none(self):
        rows = [row(2024, 1, 0), row(2025, 1, 50),       # prior 0 -> undefined
                row(2024, 2, None), row(2025, 2, 50),    # prior unknown
                row(2024, 3, 100), row(2025, 3, None)]   # current unknown
        growth = {r["month"]: r["growth"] for r in yoy_monthly_growth(rows)}
        self.assertIsNone(growth["2025-01"])
        self.assertIsNone(growth["2025-02"])
        self.assertIsNone(growth["2025-03"])

    def test_order_independent(self):
        rows = [row(2025, 1, 110), row(2024, 1, 100)]
        self.assertEqual(yoy_monthly_growth(rows)[-1]["growth"], 0.1)

    def test_empty_input(self):
        self.assertEqual(yoy_monthly_growth([]), [])


class SeasonalityIndexTests(NoNetwork):
    def test_known_pattern(self):
        pattern = [50] * 6 + [150] * 6   # average month = 100
        rows = full_year(2023, pattern) + full_year(2024, pattern)
        index = seasonality_index(rows, years=[2023, 2024])
        self.assertEqual(index[1], 0.5)
        self.assertEqual(index[12], 1.5)
        self.assertAlmostEqual(sum(index.values()) / 12, 1.0)

    def test_only_listed_years_are_used(self):
        rows = (full_year(2020, [1] * 11 + [10_000])        # e.g. a COVID year, excluded
                + full_year(2023, [100] * 12))
        self.assertEqual(seasonality_index(rows, years=[2023]), {m: 1.0 for m in range(1, 13)})

    def test_averages_across_years(self):
        rows = full_year(2023, [100] * 12) + full_year(2024, [300] + [200] * 11)
        # month 1: mean(100, 300) = 200; overall mean = (1200 + 2500) / 24
        self.assertAlmostEqual(seasonality_index(rows, years=[2023, 2024])[1], 200 / (3700 / 24))

    def test_incomplete_or_missing_year_rejected(self):
        values = [100] * 12
        values[3] = None
        with self.assertRaises(ValueError):
            seasonality_index(full_year(2024, values), years=[2024])
        with self.assertRaises(ValueError):
            seasonality_index(full_year(2024, [100] * 12), years=[2025])

    def test_empty_years_or_zero_traffic_rejected(self):
        with self.assertRaises(ValueError):
            seasonality_index(full_year(2024, [100] * 12), years=[])
        with self.assertRaises(ValueError):
            seasonality_index(full_year(2024, [0] * 12), years=[2024])


def years(totals):
    """Complete years with the given annual totals (each month = total / 12)."""
    rows = []
    for year, total in totals.items():
        rows += full_year(year, [total / 12] * 12)
    return rows


class PassengerCagrTests(NoNetwork):
    def test_complete_post_2022_years(self):
        rows = years({2022: 1000, 2023: 1100, 2024: 1210, 2025: 1331})
        result = passenger_cagr(rows)
        self.assertEqual((result["first_year"], result["last_year"], result["years"], result["periods"]),
                         (2022, 2025, 4, 3))
        self.assertAlmostEqual(result["first_passengers"], 1000)
        self.assertAlmostEqual(result["last_passengers"], 1331)
        self.assertAlmostEqual(result["cagr"], 0.10)   # (1331 / 1000) ** (1/3) - 1
        self.assertIsNone(result["reason"])

    def test_incomplete_latest_year_is_ignored(self):
        rows = years({2022: 1000, 2023: 1100}) + [row(2026, m, 999) for m in range(1, 5)]
        result = passenger_cagr(rows)
        self.assertEqual((result["first_year"], result["last_year"]), (2022, 2023))
        self.assertAlmostEqual(result["cagr"], 0.10)

    def test_incomplete_middle_year_breaks_the_sequence(self):
        rows = years({2022: 1000, 2024: 1000, 2025: 1200})
        rows += [row(2023, m, 100) for m in range(1, 12)]          # 2023 has only 11 months
        result = passenger_cagr(rows)
        self.assertEqual((result["first_year"], result["last_year"]), (2024, 2025))
        self.assertAlmostEqual(result["cagr"], 0.20)

    def test_covid_years_excluded_and_never_bridged(self):
        rows = years({2018: 900, 2019: 1000, 2020: 300, 2021: 600, 2022: 800, 2023: 1000})
        # Even with no lower bound, 2020-2021 are excluded and 2019 -> 2022 is not bridged.
        result = passenger_cagr(rows, earliest_year=None)
        self.assertEqual((result["first_year"], result["last_year"]), (2022, 2023))
        self.assertAlmostEqual(result["cagr"], 0.25)
        self.assertEqual(result["eligible_years"], [2018, 2019, 2022, 2023])

    def test_pre_covid_run_only_when_explicitly_allowed(self):
        rows = years({2018: 900, 2019: 1000, 2020: 300, 2021: 600, 2022: 800})
        self.assertIsNone(passenger_cagr(rows)["cagr"])            # default: 2022 onward only
        result = passenger_cagr(rows, earliest_year=None)
        self.assertEqual((result["first_year"], result["last_year"]), (2018, 2019))

    def test_decline_is_negative(self):
        self.assertAlmostEqual(passenger_cagr(years({2022: 1000, 2024: 810, 2023: 900}))["cagr"], -0.10)

    def test_insufficient_years(self):
        for rows in ([], years({2025: 1000}), years({2020: 1, 2021: 2})):
            result = passenger_cagr(rows)
            self.assertIsNone(result["cagr"])
            self.assertIsNone(result["first_year"])
            self.assertIn("Fewer than two", result["reason"])

    def test_zero_first_year(self):
        result = passenger_cagr(years({2022: 0, 2023: 100}))
        self.assertIsNone(result["cagr"])
        self.assertIn("undefined", result["reason"])


class PurityTests(unittest.TestCase):
    def test_module_does_not_use_network_or_clients(self):
        names = vars(passengers_module)
        self.assertNotIn("requests", names)
        self.assertFalse(any(name.endswith("Client") for name in names))


if __name__ == "__main__":
    unittest.main()
