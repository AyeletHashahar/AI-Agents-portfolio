"""Offline tests for OurAirportsClient.find_airports, using a synthetic cached airports.csv."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_sources.ourairports import DISCOVERY_FIELDS, OurAirportsClient

AIRPORT_COLUMNS = [
    "id", "ident", "type", "name", "latitude_deg", "longitude_deg", "elevation_ft", "continent",
    "iso_country", "iso_region", "municipality", "scheduled_service", "icao_code", "iata_code",
    "gps_code", "local_code", "home_link", "wikipedia_link", "keywords",
]

# (ident, type, name, iso_country, iso_region, iata_code, scheduled_service)
SYNTHETIC_AIRPORTS = [
    ("KAAA", "large_airport", "Alpha International", "US", "US-CA", "AAA", "yes"),
    ("KBBB", "medium_airport", "Bravo Regional", "US", "US-CA", "BBB", "no"),
    ("1CA1", "small_airport", "Charlie Field", "US", "US-CA", "", "no"),          # no IATA
    ("4CA4", "seaplane_base", "Delta Seaplane Base", "US", "US-CA", "", "no"),
    ("2CA2", "heliport", "Echo Heliport", "US", "US-CA", "HHH", "yes"),          # excluded type
    ("3CA3", "closed", "Foxtrot (closed)", "US", "US-CA", "", "no"),             # excluded type
    ("5CA5", "balloonport", "Golf Balloonport", "US", "US-CA", "", "no"),        # excluded type
    ("KMMM", "medium_airport", "Mike Municipal", "US", "US-MA", "MMM", "yes"),
    ("KCCC", "small_airport", "Connecticut Strip", "US", "US-CT", "CCC", "no"),
    ("5AK5", "seaplane_base", "Kilo Harbor Seaplane Base", "US", "US-AK", "SSS", "yes"),
    ("KNNN", "large_airport", "November International", "US", "US-NY", "NNN", "yes"),
    ("CYYY", "large_airport", "Yankee International", "CA", "CA-BC", "YYY", "yes"),
]
REGIONS = ["US-CA", "US-MA", "US-CT", "US-AK", "US-NY", "US-VT", "CA-BC"]


class FindAirportsTests(unittest.TestCase):
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
            for i, (ident, typ, name, country, region, iata, scheduled) in enumerate(SYNTHETIC_AIRPORTS):
                writer.writerow({
                    "id": i, "ident": ident, "type": typ, "name": name,
                    "latitude_deg": f"{40 + i}.5", "longitude_deg": f"-{100 + i}.25",
                    "iso_country": country, "iso_region": region, "municipality": f"Town {i}",
                    "scheduled_service": scheduled, "icao_code": ident if ident.startswith(("K", "C")) else "",
                    "iata_code": iata, "gps_code": ident, "local_code": ident[-3:],
                })
        with open(cache / "regions.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "code", "name", "iso_country"])
            writer.writeheader()
            for i, code in enumerate(REGIONS):
                writer.writerow({"id": i, "code": code, "name": code, "iso_country": code[:2]})
        self.client = OurAirportsClient(cache_dir=cache)

    def idents(self, airports):
        return [a["ident"] for a in airports]

    def test_one_state_excludes_irrelevant_facility_types(self):
        found = self.client.find_airports(country="US", states=["CA"])
        # heliport, closed and balloonport are excluded; order: large, medium, small, seaplane
        self.assertEqual(self.idents(found), ["KAAA", "KBBB", "1CA1", "4CA4"])

    def test_multiple_states(self):
        found = self.client.find_airports(country="US", states=["MA", "CT"])
        self.assertEqual(self.idents(found), ["KCCC", "KMMM"])   # sorted by region: US-CT, US-MA

    def test_state_code_forms_accepted(self):
        for states in (["us-ma"], ["US-MA"], " ma ", "MA"):
            self.assertEqual(self.idents(self.client.find_airports("us", states=states)), ["KMMM"])

    def test_country_filter(self):
        us = self.idents(self.client.find_airports(country="US"))
        self.assertNotIn("CYYY", us)
        self.assertEqual(len(us), 8)   # all US rows except heliport, closed, balloonport
        self.assertEqual(self.idents(self.client.find_airports(country="CA")), ["CYYY"])
        self.assertEqual(self.idents(self.client.find_airports(country="CA", states=["BC"])), ["CYYY"])

    def test_missing_iata_code_kept_by_default_and_optional_filter(self):
        found = self.client.find_airports(country="US", states=["CA"])
        charlie = next(a for a in found if a["ident"] == "1CA1")
        self.assertEqual(charlie["iata_code"], "")
        with_iata = self.client.find_airports(country="US", states=["CA"], require_iata=True)
        self.assertEqual(self.idents(with_iata), ["KAAA", "KBBB"])

    def test_explicit_types_override_default_exclusion(self):
        found = self.client.find_airports(country="US", states=["CA"], types=["heliport"])
        self.assertEqual(self.idents(found), ["2CA2"])
        found = self.client.find_airports(country="US", types=["large_airport", "medium_airport"])
        self.assertEqual(self.idents(found), ["KAAA", "KBBB", "KMMM", "KNNN"])

    def test_scheduled_service_filter(self):
        found = self.client.find_airports(country="US", scheduled_service_only=True)
        self.assertEqual(self.idents(found), ["5AK5", "KAAA", "KMMM", "KNNN"])

    def test_no_matching_airports(self):
        self.assertEqual(self.client.find_airports(country="US", states=["VT"]), [])
        self.assertEqual(self.client.find_airports(country="US", states=["NY"], types=["seaplane_base"]), [])

    def test_returned_metadata(self):
        airport = self.client.find_airports(country="US", states=["MA"])[0]
        self.assertEqual(list(airport), DISCOVERY_FIELDS)
        self.assertEqual((airport["iata_code"], airport["icao_code"], airport["name"], airport["type"],
                          airport["municipality"], airport["iso_country"], airport["iso_region"]),
                         ("MMM", "KMMM", "Mike Municipal", "medium_airport", "Town 7", "US", "US-MA"))
        self.assertEqual((airport["latitude_deg"], airport["longitude_deg"]), ("47.5", "-107.25"))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.client.find_airports(country="US", states=["XX"])        # unknown region
        with self.assertRaises(ValueError):
            self.client.find_airports(country="US", states=[])            # empty list
        with self.assertRaises(ValueError):
            self.client.find_airports(country="USA")                      # not a 2-letter code
        with self.assertRaises(ValueError):
            self.client.find_airports(country="")


if __name__ == "__main__":
    unittest.main()
