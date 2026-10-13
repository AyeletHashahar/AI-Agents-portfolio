"""Offline tests for OurAirports -> BTS airport-code resolution (data_sources.identifiers)."""

import unittest
from unittest import mock

from data_sources.identifiers import bts_code_candidates, resolve_bts_code


def airport(ident, iata="", local="", keywords=""):
    return {"ident": ident, "iata_code": iata, "local_code": local, "keywords": keywords}


# OurAirports code owners (iata_code / local_code -> idents), modelled on the real data.
OWNERS = {
    "ANC": {"PANC"},
    "DJT": {"KPBI"},                      # Palm Beach: every structured code is DJT
    "BLD": {"K61B"}, "BVU": {"K61B", "PABG"},   # Boulder City local BVU = Beluga's IATA
    "AZA": {"KIWA"}, "IWA": {"KIWA"}, "CHD": {"KCHD"},
    "YUM": {"KNYL"}, "NYL": {"KNYL"},
    "AAA": {"KAAA"}, "BBB": {"KAAA"},
}


class NoNetwork(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("requests.sessions.Session.send",
                             side_effect=AssertionError("Real HTTP request attempted"))
        patcher.start()
        self.addCleanup(patcher.stop)


class CandidateTests(NoNetwork):
    def test_order_and_sources(self):
        row = airport("KPBI", iata="DJT", local="DJT", keywords="PBI, KPBI, MFW, South Florida")
        self.assertEqual(bts_code_candidates(row), [("DJT", "iata_code"), ("PBI", "keywords"), ("MFW", "keywords")])

    def test_ordinary_words_and_malformed_codes_are_not_candidates(self):
        row = airport("X", iata="ab", local="12345", keywords="Air Bay, PHX-area, KXYZ, XY")
        self.assertEqual(bts_code_candidates(row), [("PHX", "keywords")])


class ResolutionTests(NoNetwork):
    def test_exact_iata_match(self):
        r = resolve_bts_code(airport("PANC", iata="ANC", local="ANC"), {"ANC"}, OWNERS)
        self.assertEqual((r["bts_code"], r["source"], r["status"]), ("ANC", "iata_code", "exact"))
        self.assertEqual(r["ourairports_iata"], "ANC")

    def test_exact_match_preferred_even_if_alternate_also_in_bts(self):
        # Yuma: BTS has both YUM and NYL; the exact IATA code wins, no merge.
        r = resolve_bts_code(airport("KNYL", iata="YUM", local="NYL"), {"YUM", "NYL"}, OWNERS)
        self.assertEqual((r["bts_code"], r["status"]), ("YUM", "exact"))

    def test_palm_beach_resolved_through_keyword_code(self):
        row = airport("KPBI", iata="DJT", local="DJT", keywords="PBI, KPBI, MFW, South Florida")
        r = resolve_bts_code(row, {"PBI", "ANC"}, OWNERS)
        self.assertEqual((r["bts_code"], r["source"], r["status"]), ("PBI", "keywords", "alternate"))
        self.assertEqual(r["ourairports_iata"], "DJT")          # canonical OurAirports code kept
        self.assertIn("DJT", r["detail"])

    def test_alternate_used_by_another_airport_is_rejected(self):
        # Boulder City: IATA BLD not in BTS (in this test); local BVU belongs to Beluga too.
        r = resolve_bts_code(airport("K61B", iata="BLD", local="BVU"), {"BVU"}, OWNERS)
        self.assertIsNone(r["bts_code"])
        self.assertEqual(r["status"], "not_found_in_bts")
        self.assertIn("PABG", r["detail"])
        self.assertIn("Not evidence of zero activity", r["detail"])

    def test_keyword_code_of_another_airport_is_rejected(self):
        # Mesa Gateway keywords mention CHD = Chandler Municipal, a different airport.
        r = resolve_bts_code(airport("KIWA", iata="AZA", local="IWA", keywords="CHD, Air"), {"CHD"}, OWNERS)
        self.assertIsNone(r["bts_code"])

    def test_local_code_alternate_when_unambiguous(self):
        r = resolve_bts_code(airport("KZZZ", iata="ZZ1", local="ZZL"), {"ZZL"}, OWNERS)
        self.assertEqual((r["bts_code"], r["source"], r["status"]), ("ZZL", "local_code", "alternate"))

    def test_several_eligible_alternates_are_ambiguous(self):
        row = airport("KQQQ", iata="QQ1", local="QQL", keywords="QQK")
        r = resolve_bts_code(row, {"QQL", "QQK"}, OWNERS)
        self.assertIsNone(r["bts_code"])
        self.assertEqual(r["status"], "ambiguous")
        self.assertIn("QQL", r["detail"])

    def test_not_found(self):
        r = resolve_bts_code(airport("KNNN", iata="NNN", local="NNN"), {"ANC"}, OWNERS)
        self.assertEqual((r["bts_code"], r["status"]), (None, "not_found_in_bts"))
        self.assertEqual(r["candidates"], ["NNN"])

    def test_airport_without_iata_code(self):
        r = resolve_bts_code(airport("5XX", local="5XX"), {"5XX"}, OWNERS)
        self.assertEqual((r["bts_code"], r["source"], r["ourairports_iata"]), ("5XX", "local_code", None))


if __name__ == "__main__":
    unittest.main()
