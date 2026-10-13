"""Unit tests for metrics.airports using small synthetic OurAirports rows (no network)."""

import math
import unittest
from unittest import mock

import metrics.airports as airports_module
from metrics.airports import (
    EARTH_RADIUS_MILES,
    airport_distance_miles,
    great_circle_distance_miles,
    is_new_england,
    operational_runway_count,
    runway_count,
    runway_surface_category,
    us_state,
)


def airport(country="US", region="US-AK", lat="61.179004", lon="-149.992561"):
    return {"iso_country": country, "iso_region": region, "latitude_deg": lat, "longitude_deg": lon}


class NoNetwork(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("requests.sessions.Session.send",
                             side_effect=AssertionError("Real HTTP request attempted"))
        patcher.start()
        self.addCleanup(patcher.stop)


class StateTests(NoNetwork):
    def test_us_state(self):
        self.assertEqual(us_state(airport()), "AK")
        self.assertEqual(us_state(airport(region="US-MA")), "MA")

    def test_non_us_territory_or_missing_region(self):
        self.assertIsNone(us_state(airport(country="CA", region="CA-BC")))
        self.assertIsNone(us_state(airport(country="PR", region="PR-U-A")))
        self.assertIsNone(us_state(airport(region="US-U-A")))
        self.assertIsNone(us_state(airport(region=None)))
        self.assertIsNone(us_state({}))

    def test_is_new_england(self):
        for state in ["CT", "ME", "MA", "NH", "RI", "VT"]:
            self.assertTrue(is_new_england(airport(region=f"US-{state}")), state)
        self.assertFalse(is_new_england(airport(region="US-NY")))
        self.assertFalse(is_new_england(airport()))
        self.assertIsNone(is_new_england(airport(country="CA", region="CA-QC")))


class RunwayCountTests(NoNetwork):
    def test_mixed_statuses(self):
        runways = [{"closed": "0"}, {"closed": "0"}, {"closed": "1"}, {"closed": ""}, {}]
        self.assertEqual(runway_count(runways), {"open": 2, "closed": 1, "unknown": 2})

    def test_anchorage_like_input(self):
        self.assertEqual(runway_count([{"closed": "0"}] * 3)["open"], 3)

    def test_empty_input(self):
        self.assertEqual(runway_count([]), {"open": 0, "closed": 0, "unknown": 0})


class RunwaySurfaceTests(NoNetwork):
    def test_hard_surfaces(self):
        for surface in ["ASP", "asp", "ASPH-G", "ASPHALT", "CON", "CONC-E", "CONCRETE-GROOVED",
                        "CONCRETE/GROOVED", "CONC-GRVD", "ASPH-CONC-F", "ASPHALT/CONCRETE", "ASPH/ CONC", "PEM", "BIT"]:
            self.assertEqual(runway_surface_category(surface), "hard", surface)

    def test_unpaved_surfaces(self):
        for surface in ["GRVL-G", "GVL", "GRAVEL-P", "GRE", "TURF", "TURF-GRVL - F", "GRS", "GRASS",
                        "DIRT", "GRVL-DIRT-E", "PSP", "SNOW"]:
            self.assertEqual(runway_surface_category(surface), "unpaved", surface)

    def test_water_surfaces(self):
        for surface in ["WATER", "WAT", "WATER-E"]:
            self.assertEqual(runway_surface_category(surface), "water", surface)

    def test_unknown_and_ambiguous_surfaces(self):
        # unlisted codes, qualifiers on their own, empty values and partly-paved mixes
        for surface in ["PER", "UNK", "GRV", "G", "X", "", None, "ASPH-TURF", "CONC-GRVL", "TURF-WATER"]:
            self.assertEqual(runway_surface_category(surface), "unknown", surface)


class OperationalRunwayCountTests(NoNetwork):
    def runway(self, surface="ASP", closed="0", le_ident="07"):
        return {"surface": surface, "closed": closed, "le_ident": le_ident}

    def test_only_open_hard_non_helipad_runways_count(self):
        runways = [self.runway("ASP"), self.runway("CONC-G", le_ident="15"),
                   self.runway("WATER", le_ident="07W"), self.runway("GRVL-G", le_ident="02"),
                   self.runway("ASP", closed="1", le_ident="10"),          # closed
                   self.runway("CON", le_ident="H1")]                       # helipad
        result = operational_runway_count(runways)
        self.assertEqual(result["operational"], 2)
        self.assertEqual(result["open_by_surface"], {"hard": 2, "unpaved": 1, "water": 1, "unknown": 0})
        self.assertEqual((result["helipads"], result["closed"], result["unknown_status"]), (1, 1, 0))
        self.assertTrue(result["count_is_certain"])

    def test_unknown_surface_or_status_makes_count_uncertain(self):
        unknown_surface = operational_runway_count([self.runway("ASP"), self.runway("PER", le_ident="13")])
        self.assertEqual(unknown_surface["operational"], 1)
        self.assertFalse(unknown_surface["count_is_certain"])
        unknown_status = operational_runway_count([self.runway("ASP"), self.runway("ASP", closed="")])
        self.assertEqual(unknown_status["unknown_status"], 1)
        self.assertFalse(unknown_status["count_is_certain"])
        # an unknown surface on a CLOSED runway does not matter
        self.assertTrue(operational_runway_count([self.runway("ASP"), self.runway("UNK", closed="1")])["count_is_certain"])

    def test_water_and_gravel_only_airport(self):
        result = operational_runway_count([self.runway("WATER"), self.runway("GRVL")])
        self.assertEqual(result["operational"], 0)
        self.assertTrue(result["count_is_certain"])

    def test_empty_input(self):
        result = operational_runway_count([])
        self.assertEqual((result["operational"], result["count_is_certain"]), (0, True))


class DistanceTests(NoNetwork):
    def test_same_point_is_zero(self):
        self.assertEqual(great_circle_distance_miles(61.18, -149.99, 61.18, -149.99), 0.0)

    def test_one_degree_along_equator(self):
        expected = 2 * math.pi * EARTH_RADIUS_MILES / 360   # ~69.09 miles
        self.assertAlmostEqual(great_circle_distance_miles(0, 0, 0, 1), expected, places=6)

    def test_pole_to_pole_is_half_circumference(self):
        self.assertAlmostEqual(great_circle_distance_miles(90, 0, -90, 0),
                               math.pi * EARTH_RADIUS_MILES, places=6)

    def test_across_the_antimeridian(self):
        expected = 2 * 2 * math.pi * EARTH_RADIUS_MILES / 360   # 2 degrees, not 358
        self.assertAlmostEqual(great_circle_distance_miles(0, 179, 0, -179), expected, places=6)

    def test_symmetric_and_accepts_strings(self):
        a = great_circle_distance_miles("61.179004", "-149.992561", "47.449", "-122.309")
        b = great_circle_distance_miles(47.449, -122.309, 61.179004, -149.992561)
        self.assertAlmostEqual(a, b)

    def test_missing_coordinate_gives_none(self):
        self.assertIsNone(great_circle_distance_miles(None, 0, 0, 0))
        self.assertIsNone(great_circle_distance_miles("", 0, 0, 0))
        self.assertIsNone(great_circle_distance_miles("abc", 0, 0, 0))

    def test_invalid_latitude_rejected(self):
        with self.assertRaises(ValueError):
            great_circle_distance_miles(91, 0, 0, 0)

    def test_airport_distance_from_rows(self):
        a, b = airport(lat="0", lon="0"), airport(lat="0", lon="1")
        self.assertAlmostEqual(airport_distance_miles(a, b), 2 * math.pi * EARTH_RADIUS_MILES / 360)
        self.assertIsNone(airport_distance_miles(a, {"latitude_deg": "0"}))


class PurityTests(unittest.TestCase):
    def test_module_does_not_use_network_or_clients(self):
        names = vars(airports_module)
        self.assertNotIn("requests", names)
        self.assertFalse(any(name.endswith("Client") for name in names))


if __name__ == "__main__":
    unittest.main()
