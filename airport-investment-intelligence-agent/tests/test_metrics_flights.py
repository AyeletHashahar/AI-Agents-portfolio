"""Unit tests for metrics.flights using small synthetic BTS On-Time rows (no network)."""

import math
import unittest
from datetime import date, datetime
from unittest import mock

import metrics.flights as flights_module
from metrics.airports import EARTH_RADIUS_MILES
from metrics.flights import (
    DELAY_THRESHOLD_MINUTES,
    LONG_HAUL_THRESHOLD_MILES,
    busiest_hour,
    cancellation_rate,
    delay_metrics,
    flight_counts,
    hourly_movement_counts,
    long_haul_summary,
    movement_spacing_minutes,
    peak_hour_traffic,
    peak_period_spacing,
    percentile_linear,
    unique_destinations,
)

APRIL_1 = date(2026, 4, 1)
APRIL_2 = date(2026, 4, 2)
MILES_PER_DEGREE = 2 * math.pi * EARTH_RADIUS_MILES / 360   # along the equator


def flight(origin="ANC", dest="SEA", date="2026-04-01", crs_dep="0600", crs_arr="1000",
           dep_delay="0.00", arr_delay="0.00", cancelled="0.00", diverted="0.00",
           distance="1448.00"):
    """Synthetic On-Time row with only the fields the metrics use."""
    return {
        "FlightDate": date, "Origin": origin, "Dest": dest,
        "CRSDepTime": crs_dep, "CRSArrTime": crs_arr,
        "DepDelay": dep_delay, "ArrDelay": arr_delay,
        "Cancelled": cancelled, "Diverted": diverted, "Distance": distance,
    }


def departures_at(*times, date="2026-04-01"):
    """One ANC departure (scheduled, operated, no delay) at each 'HHMM' time."""
    return [flight(crs_dep=t, date=date) for t in times]


def dt(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


class NoNetwork(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("requests.sessions.Session.send",
                             side_effect=AssertionError("Real HTTP request attempted"))
        patcher.start()
        self.addCleanup(patcher.stop)


class FlightCountsTests(NoNetwork):
    def test_normal_case(self):
        rows = [
            flight(origin="ANC", dest="SEA"),                                   # departure operated
            flight(origin="ANC", dest="SEA", cancelled="1.00", dep_delay=""),   # departure cancelled
            flight(origin="ANC", dest="SEA", diverted="1.00"),                  # diverted: still departed
            flight(origin="SEA", dest="ANC"),                                   # arrival completed
            flight(origin="SEA", dest="ANC", diverted="1.00", arr_delay=""),    # arrival diverted
            flight(origin="SEA", dest="ANC", cancelled="1.00"),                 # arrival cancelled
            flight(origin="SEA", dest="PDX"),                                   # unrelated
        ]
        result = flight_counts(rows, "ANC")
        self.assertEqual(result["departures"],
                         {"scheduled": 3, "cancelled": 1, "operated": 2, "unknown": 0})
        self.assertEqual(result["arrivals"],
                         {"scheduled": 3, "cancelled": 1, "diverted": 1, "completed": 1, "unknown": 0})
        self.assertEqual(result["total"], {"scheduled": 6, "operated": 3})

    def test_missing_status_counts_as_unknown(self):
        rows = [flight(cancelled=""), flight(origin="SEA", dest="ANC", diverted=None)]
        result = flight_counts(rows, "ANC")
        self.assertEqual(result["departures"]["unknown"], 1)
        self.assertEqual(result["departures"]["operated"], 0)
        self.assertEqual(result["arrivals"]["unknown"], 1)
        self.assertEqual(result["total"], {"scheduled": 2, "operated": 0})

    def test_empty_input(self):
        result = flight_counts([], "ANC")
        self.assertEqual(result["total"], {"scheduled": 0, "operated": 0})

    def test_airport_code_normalized(self):
        self.assertEqual(flight_counts([flight()], " anc ")["departures"]["scheduled"], 1)


class CancellationRateTests(NoNetwork):
    def test_normal_case(self):
        rows = [flight(), flight(cancelled="1.00"), flight(origin="SEA", dest="ANC"), flight()]
        self.assertEqual(cancellation_rate(rows, "ANC"),
                         {"scheduled": 4, "cancelled": 1, "unknown": 0, "rate": 0.25})

    def test_no_cancellations_is_zero_not_none(self):
        self.assertEqual(cancellation_rate([flight(), flight()], "ANC")["rate"], 0.0)

    def test_unknown_status_excluded_from_both_counts(self):
        rows = [flight(cancelled="1.00"), flight(), flight(cancelled="")]
        self.assertEqual(cancellation_rate(rows, "ANC"),
                         {"scheduled": 2, "cancelled": 1, "unknown": 1, "rate": 0.5})

    def test_other_airports_ignored(self):
        rows = [flight(), flight(origin="SEA", dest="PDX", cancelled="1.00")]
        self.assertEqual(cancellation_rate(rows, "ANC")["rate"], 0.0)

    def test_empty_and_all_unknown_give_none(self):
        self.assertIsNone(cancellation_rate([], "ANC")["rate"])
        self.assertIsNone(cancellation_rate([flight(cancelled=None)], "ANC")["rate"])


class DelayMetricsTests(NoNetwork):
    def departures(self):
        return [
            flight(dep_delay="0.00"),                                    # on time
            flight(dep_delay="-5.00"),                                   # early
            flight(dep_delay="15.00"),                                   # exactly 15 -> delayed
            flight(dep_delay="14.00"),                                   # just below -> not delayed
            flight(dep_delay="60.00"),                                   # delayed
            flight(dep_delay="30.00", diverted="1.00", arr_delay=""),    # diverted: still departed
            flight(dep_delay=""),                                        # eligible, delay unknown
            flight(dep_delay="-6.00", cancelled="1.00"),                 # cancelled: not eligible
            flight(dep_delay="20.00", cancelled=""),                     # unknown status
        ]

    def arrivals(self):
        return [
            flight(origin="SEA", dest="ANC", arr_delay="10.00"),
            flight(origin="SEA", dest="ANC", arr_delay="-20.00"),
            flight(origin="SEA", dest="ANC", arr_delay="15.00"),
            flight(origin="SEA", dest="ANC", arr_delay="N/A"),                    # unknown delay
            flight(origin="SEA", dest="ANC", arr_delay="45.00", diverted="1.00"),  # did not land here
            flight(origin="SEA", dest="ANC", arr_delay="", cancelled="1.00"),
            flight(origin="SEA", dest="ANC", arr_delay="5.00", diverted=None),     # unknown status
        ]

    def test_threshold_is_15_minutes(self):
        self.assertEqual(DELAY_THRESHOLD_MINUTES, 15)

    def test_departures(self):
        dep = delay_metrics(self.departures(), "ANC")["departures"]
        self.assertEqual({k: dep[k] for k in ("scheduled", "cancelled", "unknown_status", "eligible",
                                              "known_delay", "unknown_delay", "delayed_15_plus")},
                         {"scheduled": 9, "cancelled": 1, "unknown_status": 1, "eligible": 7,
                          "known_delay": 6, "unknown_delay": 1, "delayed_15_plus": 3})
        # known delays: 0, -5, 15, 14, 60, 30
        self.assertAlmostEqual(dep["mean_delay_minutes"], 114 / 6)
        self.assertEqual(dep["median_delay_minutes"], 14.5)
        self.assertEqual(dep["delayed_15_plus_rate"], 0.5)

    def test_arrivals(self):
        arr = delay_metrics(self.arrivals(), "ANC")["arrivals"]
        self.assertEqual({k: arr[k] for k in ("scheduled", "cancelled", "diverted", "unknown_status",
                                              "eligible", "known_delay", "unknown_delay", "delayed_15_plus")},
                         {"scheduled": 7, "cancelled": 1, "diverted": 1, "unknown_status": 1,
                          "eligible": 4, "known_delay": 3, "unknown_delay": 1, "delayed_15_plus": 1})
        # known delays: 10, -20, 15 (the diverted 45 is not a completed arrival)
        self.assertAlmostEqual(arr["mean_delay_minutes"], 5 / 3)
        self.assertEqual(arr["median_delay_minutes"], 10)
        self.assertAlmostEqual(arr["delayed_15_plus_rate"], 1 / 3)

    def test_delayed_60_plus(self):
        rows = [flight(dep_delay="59.00"), flight(dep_delay="60.00"), flight(dep_delay="120.00"),
                flight(dep_delay="-3.00"), flight(dep_delay="90.00", cancelled="1.00")]
        result = delay_metrics(rows, "ANC")
        self.assertEqual(result["severe_threshold_minutes"], 60)
        dep = result["departures"]
        self.assertEqual((dep["delayed_60_plus"], dep["delayed_15_plus"], dep["known_delay"]), (2, 3, 4))
        self.assertEqual(dep["delayed_60_plus_rate"], 0.5)
        self.assertIsNone(delay_metrics([], "ANC")["departures"]["delayed_60_plus_rate"])

    def test_cancelled_flight_with_delay_value_is_not_delayed(self):
        result = delay_metrics([flight(dep_delay="90.00", cancelled="1.00")], "ANC")["departures"]
        self.assertEqual((result["eligible"], result["delayed_15_plus"]), (0, 0))
        self.assertIsNone(result["mean_delay_minutes"])

    def test_negative_delays_are_kept(self):
        dep = delay_metrics([flight(dep_delay="-10.00"), flight(dep_delay="-4.00")], "ANC")["departures"]
        self.assertEqual(dep["mean_delay_minutes"], -7)
        self.assertEqual(dep["median_delay_minutes"], -7)
        self.assertEqual(dep["delayed_15_plus_rate"], 0)

    def test_all_delays_unknown_gives_none_not_zero(self):
        dep = delay_metrics([flight(dep_delay=""), flight(dep_delay=None)], "ANC")["departures"]
        self.assertEqual((dep["eligible"], dep["known_delay"], dep["unknown_delay"]), (2, 0, 2))
        self.assertIsNone(dep["mean_delay_minutes"])
        self.assertIsNone(dep["median_delay_minutes"])
        self.assertIsNone(dep["delayed_15_plus_rate"])

    def test_both_directions_and_other_airports(self):
        rows = self.departures() + self.arrivals() + [flight(origin="SEA", dest="PDX", dep_delay="99")]
        result = delay_metrics(rows, " anc ")
        self.assertEqual(result["threshold_minutes"], 15)
        self.assertEqual((result["departures"]["scheduled"], result["arrivals"]["scheduled"]), (9, 7))

    def test_empty_input(self):
        result = delay_metrics([], "ANC")
        for direction in ("departures", "arrivals"):
            self.assertEqual(result[direction]["eligible"], 0)
            self.assertEqual(result[direction]["delayed_15_plus"], 0)
            self.assertIsNone(result[direction]["mean_delay_minutes"])
            self.assertIsNone(result[direction]["median_delay_minutes"])
            self.assertIsNone(result[direction]["delayed_15_plus_rate"])


class UniqueDestinationsTests(NoNetwork):
    def test_counts_distinct_destinations_of_departures(self):
        rows = [flight(dest="SEA"), flight(dest="SEA"), flight(dest="ORD"),
                flight(dest="FAI", cancelled="1.00"),           # scheduled network: counted
                flight(origin="JNU", dest="ANC"),               # arrival: not a destination
                flight(dest=""),                                # no destination code
                flight(origin="SEA", dest="PDX")]               # other airport
        self.assertEqual(unique_destinations(rows, " anc "),
                         {"count": 3, "destinations": ["FAI", "ORD", "SEA"], "departures": 5})

    def test_empty_input(self):
        self.assertEqual(unique_destinations([], "ANC"), {"count": 0, "destinations": [], "departures": 0})


class HourlyMovementCountsTests(NoNetwork):
    def test_scheduled_basis_counts_departures_and_arrivals(self):
        rows = [
            flight(crs_dep="0605"),
            flight(crs_dep="0659", cancelled="1.00", dep_delay=""),  # scheduled: still counted
            flight(crs_dep="0715"),
            flight(origin="SEA", dest="ANC", crs_dep="0300", crs_arr="0630"),
        ]
        result = hourly_movement_counts(rows, "ANC", time_basis="scheduled")
        self.assertEqual(result, {"counts": {dt("2026-04-01 06:00"): 3, dt("2026-04-01 07:00"): 1},
                                  "excluded": 0})

    def test_actual_basis_adds_delay_and_excludes_flights_without_actual_time(self):
        rows = [
            flight(crs_dep="0605", dep_delay="60.00"),                       # -> 07:05
            flight(crs_dep="0610", dep_delay="-5.00"),                       # early -> 06:05
            flight(crs_dep="0620", cancelled="1.00", dep_delay=""),          # no actual time
            flight(origin="SEA", dest="ANC", crs_arr="0630", diverted="1.00", arr_delay=""),
        ]
        result = hourly_movement_counts(rows, "ANC", time_basis="actual")
        self.assertEqual(result["counts"], {dt("2026-04-01 06:00"): 1, dt("2026-04-01 07:00"): 1})
        self.assertEqual(result["excluded"], 2)

    def test_actual_basis_excludes_cancelled_flight_that_has_a_delay_value(self):
        # Real BTS pattern (ANC-DEN, 2026-04-04): left the gate (DepDelay present), then cancelled.
        rows = [flight(crs_dep="2347", dep_delay="-6.00", cancelled="1.00", arr_delay="")]
        result = hourly_movement_counts(rows, "ANC", time_basis="actual")
        self.assertEqual(result, {"counts": {}, "excluded": 1})

    def test_actual_basis_excludes_diverted_arrival_even_with_delay_value(self):
        rows = [flight(origin="PHX", dest="ANC", diverted="1.00", arr_delay="12.00")]
        result = hourly_movement_counts(rows, "ANC", time_basis="actual")
        self.assertEqual(result, {"counts": {}, "excluded": 1})

    def test_actual_basis_keeps_departure_of_diverted_flight(self):
        rows = [flight(origin="ANC", dest="JNU", crs_dep="0715", dep_delay="-3.00",
                       diverted="1.00", arr_delay="")]
        result = hourly_movement_counts(rows, "ANC", time_basis="actual")
        self.assertEqual(result, {"counts": {dt("2026-04-01 07:00"): 1}, "excluded": 0})

    def test_actual_basis_excludes_unknown_status(self):
        rows = [flight(cancelled=""), flight(origin="SEA", dest="ANC", diverted=None)]
        result = hourly_movement_counts(rows, "ANC", time_basis="actual")
        self.assertEqual(result, {"counts": {}, "excluded": 2})

    def test_actual_movements_equal_operated_count(self):
        rows = [
            flight(), flight(cancelled="1.00", dep_delay="-6.00"), flight(diverted="1.00"),
            flight(origin="SEA", dest="ANC"), flight(origin="SEA", dest="ANC", diverted="1.00"),
            flight(origin="SEA", dest="ANC", cancelled="1.00", arr_delay=""),
        ]
        actual = hourly_movement_counts(rows, "ANC", time_basis="actual")
        self.assertEqual(sum(actual["counts"].values()), flight_counts(rows, "ANC")["total"]["operated"])
        self.assertEqual(sum(actual["counts"].values()) + actual["excluded"], len(rows))

    def test_actual_delay_crossing_midnight_moves_to_next_day(self):
        rows = [flight(crs_dep="2350", dep_delay="20.00")]
        result = hourly_movement_counts(rows, "ANC", time_basis="actual")
        self.assertEqual(result["counts"], {dt("2026-04-02 00:00"): 1})

    def test_overnight_scheduled_arrival_is_next_day(self):
        rows = [flight(origin="MSP", dest="ANC", crs_dep="2115", crs_arr="0538")]
        result = hourly_movement_counts(rows, "ANC", time_basis="scheduled")
        self.assertEqual(result["counts"], {dt("2026-04-02 05:00"): 1})

    def test_2400_means_midnight_at_end_of_day(self):
        rows = [flight(crs_dep="2400")]
        result = hourly_movement_counts(rows, "ANC", time_basis="scheduled")
        self.assertEqual(result["counts"], {dt("2026-04-02 00:00"): 1})

    def test_malformed_or_missing_times_are_excluded(self):
        rows = [flight(crs_dep="2430"), flight(crs_dep=""), flight(crs_dep="615"),
                flight(date=None), flight(crs_dep="0600")]
        result = hourly_movement_counts(rows, "ANC", time_basis="scheduled")
        self.assertEqual(result["counts"], {dt("2026-04-01 06:00"): 1})
        self.assertEqual(result["excluded"], 4)

    def test_movement_filter(self):
        rows = [flight(crs_dep="0600"), flight(origin="SEA", dest="ANC", crs_arr="0900")]
        departures = hourly_movement_counts(rows, "ANC", time_basis="scheduled", movement="departures")
        arrivals = hourly_movement_counts(rows, "ANC", time_basis="scheduled", movement="arrivals")
        self.assertEqual(departures["counts"], {dt("2026-04-01 06:00"): 1})
        self.assertEqual(arrivals["counts"], {dt("2026-04-01 09:00"): 1})

    def test_empty_input(self):
        self.assertEqual(hourly_movement_counts([], "ANC", time_basis="actual"),
                         {"counts": {}, "excluded": 0})

    def test_time_basis_is_required_and_validated(self):
        with self.assertRaises(TypeError):
            hourly_movement_counts([flight()], "ANC")
        with self.assertRaises(ValueError):
            hourly_movement_counts([flight()], "ANC", time_basis="estimated")
        with self.assertRaises(ValueError):
            hourly_movement_counts([flight()], "ANC", time_basis="scheduled", movement="landings")


class BusiestHourTests(NoNetwork):
    def test_normal_case(self):
        counts = {dt("2026-04-01 06:00"): 3, dt("2026-04-01 07:00"): 5, dt("2026-04-01 08:00"): 1}
        self.assertEqual(busiest_hour(counts),
                         {"hour_start": dt("2026-04-01 07:00"), "movements": 5, "tied_hours": 1})

    def test_tie_goes_to_earliest_hour(self):
        counts = {dt("2026-04-02 09:00"): 4, dt("2026-04-01 18:00"): 4, dt("2026-04-01 06:00"): 2}
        self.assertEqual(busiest_hour(counts),
                         {"hour_start": dt("2026-04-01 18:00"), "movements": 4, "tied_hours": 2})

    def test_empty_input(self):
        self.assertIsNone(busiest_hour({}))


class MovementSpacingTests(NoNetwork):
    def test_gaps_within_a_day(self):
        rows = [
            flight(crs_dep="0600"),
            flight(crs_dep="0610"),
            flight(origin="SEA", dest="ANC", crs_arr="0610"),    # same minute -> gap 0
            flight(crs_dep="0645"),
            flight(date="2026-04-02", crs_dep="0500"),           # next day: overnight gap ignored
            flight(date="2026-04-02", crs_dep="0530"),
        ]
        result = movement_spacing_minutes(rows, "ANC", time_basis="scheduled")
        self.assertEqual(result, {"gaps": [10.0, 0.0, 35.0, 30.0], "excluded": 0})

    def test_actual_basis_excludes_cancelled(self):
        rows = [flight(crs_dep="0600"), flight(crs_dep="0605", cancelled="1.00", dep_delay=""),
                flight(crs_dep="0610", dep_delay="5.00")]
        result = movement_spacing_minutes(rows, "ANC", time_basis="actual")
        self.assertEqual(result, {"gaps": [15.0], "excluded": 1})

    def test_empty_and_single_movement(self):
        self.assertEqual(movement_spacing_minutes([], "ANC", time_basis="scheduled")["gaps"], [])
        self.assertEqual(movement_spacing_minutes([flight()], "ANC", time_basis="scheduled")["gaps"], [])


class PercentileTests(NoNetwork):
    def test_linear_interpolation(self):
        # n=10: rank = 9 * 0.95 = 8.55 -> 9 + 0.55 * (10 - 9)
        self.assertAlmostEqual(percentile_linear(range(1, 11), 95), 9.55)

    def test_input_order_does_not_matter(self):
        self.assertAlmostEqual(percentile_linear([10, 1, 7, 3, 9, 2, 8, 4, 6, 5], 95), 9.55)

    def test_ties(self):
        self.assertEqual(percentile_linear([5, 5, 5, 5], 95), 5)
        # rank = 3 * 0.95 = 2.85 -> 0 + 0.85 * (1 - 0)
        self.assertAlmostEqual(percentile_linear([0, 0, 0, 1], 95), 0.85)
        # both neighbours tied at 2 -> exactly 2
        self.assertEqual(percentile_linear([0] * 21 + [2, 2, 3], 95), 2)

    def test_edges(self):
        self.assertIsNone(percentile_linear([], 95))
        self.assertEqual(percentile_linear([7], 95), 7)
        self.assertEqual(percentile_linear([3, 1, 2], 0), 1)
        self.assertEqual(percentile_linear([3, 1, 2], 100), 3)
        with self.assertRaises(ValueError):
            percentile_linear([1, 2], 101)


class PeakHourTrafficTests(NoNetwork):
    # One day = 24 hourly counts. 06h: 4, 07h: 2, 08h: 1, other 21 hours: 0.
    ROWS = departures_at("0600", "0610", "0620", "0630", "0700", "0730", "0800")

    def test_95th_percentile_includes_empty_hours(self):
        result = peak_hour_traffic(self.ROWS, "ANC", time_basis="scheduled",
                                   first_date=APRIL_1, last_date=APRIL_1)
        # sorted: 21 zeros, 1, 2, 4; rank = 23 * 0.95 = 21.85 -> 1 + 0.85 * (2 - 1)
        self.assertAlmostEqual(result["value"], 1.85)
        self.assertEqual((result["hours"], result["movements"], result["excluded"],
                          result["outside_period"]), (24, 7, 0, 0))
        self.assertEqual(result["percentile"], 95)

    def test_busiest_hour_is_kept_and_differs(self):
        counts = hourly_movement_counts(self.ROWS, "ANC", time_basis="scheduled")["counts"]
        self.assertEqual(busiest_hour(counts)["movements"], 4)

    def test_actual_basis_uses_actual_definition(self):
        rows = self.ROWS + [flight(crs_dep="0640", cancelled="1.00", dep_delay="")]
        scheduled = peak_hour_traffic(rows, "ANC", time_basis="scheduled",
                                      first_date=APRIL_1, last_date=APRIL_1)
        actual = peak_hour_traffic(rows, "ANC", time_basis="actual",
                                   first_date=APRIL_1, last_date=APRIL_1)
        self.assertEqual((scheduled["movements"], actual["movements"]), (8, 7))
        self.assertEqual(actual["excluded"], 1)

    def test_all_hours_equal(self):
        rows = departures_at(*[f"{h:02d}15" for h in range(24)])
        result = peak_hour_traffic(rows, "ANC", time_basis="scheduled",
                                   first_date=APRIL_1, last_date=APRIL_1)
        self.assertEqual(result["value"], 1)

    def test_movements_outside_period_reported(self):
        overnight = flight(origin="MSP", dest="ANC", crs_dep="2115", crs_arr="0538")  # lands Apr 2
        result = peak_hour_traffic(self.ROWS + [overnight], "ANC", time_basis="scheduled",
                                   first_date=APRIL_1, last_date=APRIL_1)
        self.assertEqual((result["movements"], result["outside_period"]), (7, 1))

    def test_no_movements_gives_none(self):
        for rows in ([], [flight(date="2026-05-10")]):
            result = peak_hour_traffic(rows, "ANC", time_basis="scheduled",
                                       first_date=APRIL_1, last_date=APRIL_1)
            self.assertIsNone(result["value"])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            peak_hour_traffic(self.ROWS, "ANC", time_basis="scheduled",
                              first_date=APRIL_2, last_date=APRIL_1)
        with self.assertRaises(TypeError):
            peak_hour_traffic(self.ROWS, "ANC", first_date=APRIL_1, last_date=APRIL_1)


class PeakPeriodSpacingTests(NoNetwork):
    def spacing(self, rows, first=APRIL_1, last=APRIL_1, basis="scheduled"):
        return peak_period_spacing(rows, "ANC", time_basis=basis, first_date=first, last_date=last)

    def test_median_within_consecutive_peak_hours(self):
        # 06h: 4, 07h: 2, 12h: 1 -> threshold 1.85; peak hours 06 + 07 (one period); 12 is not peak.
        rows = departures_at("0600", "0610", "0620", "0640", "0700", "0705", "1200")
        result = self.spacing(rows)
        self.assertAlmostEqual(result["peak_threshold"], 1.85)
        self.assertEqual((result["peak_hours"], result["peak_periods"], result["gaps"]), (2, 1, 5))
        self.assertEqual(result["median_spacing_minutes"], 10)   # gaps 10, 10, 20, 20, 5

    def test_hours_exactly_at_threshold_are_peak(self):
        # 06h: 3, 07h: 2, 09h: 2 -> sorted 21 zeros, 2, 2, 3 -> threshold exactly 2.0
        rows = departures_at("0600", "0620", "0640", "0700", "0730", "0900", "0915")
        result = self.spacing(rows)
        self.assertEqual(result["peak_threshold"], 2)
        self.assertEqual((result["peak_hours"], result["peak_periods"]), (3, 2))
        # period 06-07: 20, 20, 20, 30; period 09: 15; the 07:30 -> 09:00 gap crosses periods
        self.assertEqual(result["gaps"], 5)
        self.assertEqual(result["median_spacing_minutes"], 20)

    def test_peak_hours_with_single_movement_give_none(self):
        rows = departures_at("0600", "0900", "1200")   # threshold 1; isolated single-movement hours
        result = self.spacing(rows)
        self.assertEqual((result["peak_hours"], result["gaps"]), (3, 0))
        self.assertIsNone(result["median_spacing_minutes"])

    def test_peak_period_is_split_at_midnight(self):
        rows = (departures_at("2300", "2330", date="2026-04-01")
                + departures_at("0000", "0015", date="2026-04-02"))
        result = self.spacing(rows, first=APRIL_1, last=APRIL_2)
        # 48 hours, 46 empty -> threshold 0; peak = non-empty hours; 23:30 -> 00:00 not counted
        self.assertEqual(result["peak_threshold"], 0)
        self.assertEqual((result["peak_hours"], result["peak_periods"], result["gaps"]), (2, 2, 2))
        self.assertEqual(result["median_spacing_minutes"], 22.5)   # median of 30 and 15

    def test_simultaneous_movements_give_real_zero(self):
        rows = departures_at("0600", "0600", "0600")
        self.assertEqual(self.spacing(rows)["median_spacing_minutes"], 0)

    def test_actual_basis(self):
        rows = departures_at("0600", "0610") + [flight(crs_dep="0605", cancelled="1.00", dep_delay="")]
        self.assertEqual(self.spacing(rows, basis="actual")["median_spacing_minutes"], 10)
        self.assertEqual(self.spacing(rows, basis="scheduled")["median_spacing_minutes"], 5)

    def test_empty_data(self):
        result = self.spacing([])
        self.assertIsNone(result["peak_threshold"])
        self.assertIsNone(result["median_spacing_minutes"])
        self.assertEqual((result["peak_hours"], result["gaps"]), (0, 0))


class LongHaulTests(NoNetwork):
    def test_threshold_is_project_value(self):
        self.assertEqual(LONG_HAUL_THRESHOLD_MILES, 3450)

    def test_exactly_threshold_is_long_haul_and_just_below_is_not(self):
        rows = [flight(distance="3450.00"), flight(distance="3449.99")]
        result = long_haul_summary(rows, "ANC")
        self.assertEqual((result["long_haul"], result["not_long_haul"]), (1, 1))
        self.assertEqual(result["long_haul_share"], 0.5)

    def test_unknown_distance_excluded_from_denominator(self):
        rows = [flight(distance="4000.00"), flight(distance="1000.00"),
                flight(distance=""), flight(distance=None), flight(distance="0.00")]
        result = long_haul_summary(rows, "ANC")
        self.assertEqual((result["flights"], result["known_distance"], result["unknown_distance"]),
                         (5, 2, 3))
        self.assertEqual(result["long_haul_share"], 0.5)   # 1 / 2, not 1 / 5

    def test_great_circle_fallback(self):
        airports = {"ANC": {"latitude_deg": "0", "longitude_deg": "0"},
                    "FAR": {"latitude_deg": "0", "longitude_deg": "50"},    # ~3,454.7 mi
                    "NEAR": {"latitude_deg": "0", "longitude_deg": "49.9"}}  # ~3,447.8 mi
        self.assertGreater(50 * MILES_PER_DEGREE, 3450)
        self.assertLess(49.9 * MILES_PER_DEGREE, 3450)
        rows = [flight(dest="FAR", distance=""), flight(dest="NEAR", distance="")]
        result = long_haul_summary(rows, "ANC", airports)
        self.assertEqual((result["long_haul"], result["not_long_haul"]), (1, 1))
        self.assertEqual(result["distance_source"], {"bts": 0, "great_circle": 2})

    def test_bts_distance_has_priority_over_coordinates(self):
        airports = {"ANC": {"latitude_deg": "0", "longitude_deg": "0"},
                    "FAR": {"latitude_deg": "0", "longitude_deg": "60"}}
        result = long_haul_summary([flight(dest="FAR", distance="1000.00")], "ANC", airports)
        self.assertEqual((result["long_haul"], result["distance_source"]["bts"]), (0, 1))

    def test_missing_or_invalid_coordinates_stay_unknown(self):
        airports = {"ANC": {"latitude_deg": "0", "longitude_deg": "0"},
                    "BAD": {"latitude_deg": "95", "longitude_deg": "0"},
                    "NOC": {"latitude_deg": "", "longitude_deg": ""}}
        rows = [flight(dest="XYZ", distance=""), flight(dest="BAD", distance=""),
                flight(dest="NOC", distance="")]
        result = long_haul_summary(rows, "ANC", airports)
        self.assertEqual((result["unknown_distance"], result["not_long_haul"]), (3, 0))
        self.assertIsNone(result["long_haul_share"])

    def test_empty_input_and_other_airports(self):
        self.assertIsNone(long_haul_summary([], "ANC")["long_haul_share"])
        result = long_haul_summary([flight(origin="SEA", dest="PDX", distance="5000")], "ANC")
        self.assertEqual(result["flights"], 0)


class PurityTests(unittest.TestCase):
    def test_module_does_not_use_network_or_clients(self):
        names = vars(flights_module)
        self.assertNotIn("requests", names)
        self.assertFalse(any(name.endswith("Client") for name in names))


if __name__ == "__main__":
    unittest.main()
