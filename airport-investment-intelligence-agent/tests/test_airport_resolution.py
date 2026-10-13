"""Offline tests for airport-reference resolution: OurAirportsClient.search_airports (data
layer), the resolve_airports agent tool, its registration, the name/municipality fields of
compare/discover output, and the Agent instructions. Synthetic CSVs; no network.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import SYSTEM_INSTRUCTION, TOOLS, AgentSession, execute_tool, tool_declarations
from agent_tools import AgentToolContext, compare_airports, discover_airports, resolve_airports
from data_sources.ourairports import OurAirportsClient, normalize_reference
from tests.test_agent_loop import AgentLoopTestCase, FakeGemini, call_response, text_response
from tests.test_agent_tools import AIRPORT_COLUMNS, ToolTestCase

# ident, type, name, municipality, region, scheduled, iata, local, keywords
AIRPORTS = [
    ("PANC", "large_airport", "Ted Stevens Anchorage International Airport", "Anchorage", "US-AK", "yes", "ANC", "ANC", ""),
    ("PAMR", "medium_airport", "Merrill Field", "Anchorage", "US-AK", "yes", "MRI", "MRI", ""),
    ("PALH", "seaplane_base", "Lake Hood Seaplane Base", "Anchorage", "US-AK", "no", "LHD", "LHD", ""),
    ("PAFA", "medium_airport", "Fairbanks International Airport", "Fairbanks", "US-AK", "yes", "FAI", "FAI", ""),
    ("PAFB", "medium_airport", "Ladd Army Airfield", "Fairbanks", "US-AK", "no", "FBK", "FBK", ""),
    ("PAJN", "medium_airport", "Juneau International Airport", "Juneau", "US-AK", "yes", "JNU", "JNU", ""),
    ("2AK2", "heliport", "Juneau Heliport", "Juneau", "US-AK", "yes", "", "", ""),
    ("KORD", "large_airport", "Chicago O'Hare International Airport", "Chicago", "US-IL", "yes", "ORD", "ORD", ""),
    ("KMDW", "large_airport", "Chicago Midway International Airport", "Chicago", "US-IL", "yes", "MDW", "MDW", ""),
    ("KRFD", "medium_airport", "Chicago Rockford International Airport", "Chicago/Rockford", "US-IL", "yes",
     "RFD", "RFD", ""),
    ("KSPI", "medium_airport", "Abraham Lincoln Capital Airport", "Springfield", "US-IL", "yes", "SPI", "SPI", ""),
    ("KSGF", "medium_airport", "Springfield Branson National Airport", "Springfield", "US-MO", "yes", "SGF", "SGF", ""),
    ("KPDX", "large_airport", "Portland International Airport", "Portland", "US-OR", "yes", "PDX", "PDX", ""),
    ("KPWM", "large_airport", "Portland International Jetport", "Portland", "US-ME", "yes", "PWM", "PWM", ""),
    ("PHNL", "large_airport", "Daniel K. Inouye International Airport", "Honolulu, Oahu", "US-HI", "yes",
     "HNL", "HNL", ""),
    ("PHIK", "medium_airport", "Hickam Air Force Base", "Honolulu", "US-HI", "no", "", "", ""),
    ("KINT", "small_airport", "Smith Reynolds Airport", "Winston-Salem", "US-NC", "no", "INT", "INT", ""),
    ("KSLE", "medium_airport", "Salem Municipal Airport", "Salem", "US-OR", "yes", "SLE", "SLE", ""),
    ("KSTL", "large_airport", "St. Louis Lambert International Airport", "St. Louis", "US-MO", "yes",
     "STL", "STL", ""),
    ("1QV1", "small_airport", "Quietville Strip", "Quietville", "US-KS", "no", "", "1QV1", ""),
    ("2QV2", "small_airport", "North Quietville Field", "Quietville", "US-KS", "no", "", "2QV2", ""),
    ("3LN3", "small_airport", "Lonely Field", "Lonelytown", "US-KS", "no", "", "3LN3", ""),
    ("KTWA", "medium_airport", "Twin Name Airport", "Westtown", "US-KS", "yes", "TWA", "TWA", ""),
    ("KTWB", "small_airport", "Twin Name Airport", "Easttown", "US-NE", "no", "TWB", "TWB", ""),
    ("KDBA", "medium_airport", "Double Field", "Northtown", "US-KS", "yes", "DBA", "DBA", ""),
    ("KDBB", "medium_airport", "Double Field", "Southtown", "US-NE", "yes", "DBB", "DBB", ""),
    ("KALF", "small_airport", "Alpha Field", "Alphaville", "US-KS", "no", "", "ZZZ", ""),
    ("KBTF", "small_airport", "Beta Field", "Betaville", "US-NE", "no", "", "ZZZ", ""),
    ("KVUO", "small_airport", "Pearson Field", "Vancouver", "US-WA", "no", "", "VUO", ""),
    ("CYVR", "large_airport", "Vancouver International Airport", "Vancouver", "CA-BC", "yes", "YVR", "", ""),
    ("KOLD", "closed", "Old Closed Airport", "Ghosttown", "US-KS", "no", "", "", ""),
    ("KPBI", "large_airport", "Palm Beach International Airport", "West Palm Beach", "US-FL", "yes",
     "DJT", "DJT", "PBI, KPBI"),
    ("KSNA", "large_airport", "John Wayne Orange County International Airport", "Santa Ana", "US-CA", "yes",
     "SNA", "SNA", ""),
    ("KLAX", "large_airport", "Los Angeles International Airport", "Los Angeles", "US-CA", "yes", "LAX", "LAX", ""),
    ("KBOS", "large_airport", "Boston Logan International Airport", "Boston", "US-MA", "yes", "BOS", "BOS", ""),
    ("KBIL", "medium_airport", "Billings Logan International Airport", "Billings", "US-MT", "yes", "BIL", "BIL", ""),
]
REGIONS = {"US-AK": "Alaska", "US-IL": "Illinois", "US-MO": "Missouri", "US-OR": "Oregon", "US-ME": "Maine",
           "US-HI": "Hawaii", "US-NC": "North Carolina", "US-KS": "Kansas", "US-NE": "Nebraska",
           "US-FL": "Florida", "US-WA": "Washington", "US-CA": "California", "US-MA": "Massachusetts",
           "US-MT": "Montana", "CA-BC": "British Columbia"}


def write_fixture(cache):
    with open(cache / "airports.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AIRPORT_COLUMNS)
        writer.writeheader()
        for i, (ident, typ, name, city, region, sched, iata, local, kw) in enumerate(AIRPORTS):
            writer.writerow({"id": i, "ident": ident, "type": typ, "name": name, "iso_country": region[:2],
                             "iso_region": region, "municipality": city, "scheduled_service": sched,
                             "icao_code": ident if ident[0] in "PKC" and ident.isalpha() else "", "iata_code": iata,
                             "gps_code": ident, "local_code": local, "keywords": kw})
    with open(cache / "regions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "code", "name", "iso_country"])
        writer.writeheader()
        for i, (code, name) in enumerate(REGIONS.items()):
            writer.writerow({"id": i, "code": code, "name": name, "iso_country": code[:2]})


class ResolutionTestCase(unittest.TestCase):
    def setUp(self):
        for target in ("requests.sessions.Session.send", "requests.get"):
            patcher = mock.patch(target, side_effect=AssertionError("Real HTTP request attempted"))
            patcher.start()
            self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        write_fixture(Path(tmp.name))
        self.client = OurAirportsClient(cache_dir=tmp.name)
        # BTS clients must never be touched by resolution.
        self.ctx = AgentToolContext(ourairports=self.client,
                                    t100=mock.Mock(side_effect=AssertionError("BTS used")),
                                    ontime=mock.Mock(side_effect=AssertionError("BTS used")))

    def search(self, text):
        return self.client.search_airports(text)

    def code(self, text):
        found = self.search(text)
        return found["status"], found["airport"]["iata_code"] if found["airport"] else None

    def candidates(self, text):
        return [row["ident"] for row in self.search(text)["candidates"]]


# --- data layer --------------------------------------------------------------------------------

class SearchAirportsTests(ResolutionTestCase):
    def test_exact_code(self):
        for code, iata in (("ANC", "ANC"), ("PANC", "ANC"), ("FAI", "FAI"), ("PBI", "DJT")):
            found = self.search(code)
            self.assertEqual((found["status"], found["matched_on"], found["airport"]["iata_code"]),
                             ("found", "code", iata), code)

    def test_exact_full_name(self):
        found = self.search("Ted Stevens Anchorage International Airport")
        self.assertEqual((found["status"], found["matched_on"], found["selection"], found["airport"]["ident"]),
                         ("found", "name", "unique_match", "PANC"))
        self.assertEqual(self.code("Fairbanks International Airport"), ("found", "FAI"))
        self.assertEqual(self.code("Juneau International Airport"), ("found", "JNU"))

    def test_case_whitespace_punctuation_and_abbreviations(self):
        for text in ("ted stevens anchorage international airport", "  TED STEVENS ANCHORAGE INTERNATIONAL AIRPORT ",
                     "Ted-Stevens, Anchorage International Airport.", "Ted Stevens Anchorage Intl Airport",
                     "ted stevens anchorage intl"):
            self.assertEqual(self.code(text), ("found", "ANC"), text)
        for text in ("O'Hare", "OHare", "o’hare", "Chicago O'Hare"):
            self.assertEqual(self.code(text), ("found", "ORD"), text)
        self.assertEqual(self.code("Saint Louis"), ("found", "STL"))
        self.assertEqual(normalize_reference("  Café  O’Hare-Intl. "), "cafe ohare international")

    def test_city_resolution_from_data(self):
        self.assertEqual(self.code("Fairbanks"), ("found", "FAI"))
        self.assertEqual(self.search("Fairbanks")["selection"], "only_scheduled_airport")
        self.assertEqual(self.code("Juneau"), ("found", "JNU"))                 # heliport never matched
        self.assertEqual(self.code("Anchorage airport"), ("found", "ANC"))
        self.assertEqual(self.code("the Fairbanks airport"), ("found", "FAI"))

    def test_one_scheduled_airport_selected_by_largest_facility_type(self):
        found = self.search("Anchorage")
        self.assertEqual((found["status"], found["matched_on"], found["selection"], found["airport"]["ident"]),
                         ("found", "place", "largest_scheduled_airport", "PANC"))
        self.assertEqual([row["ident"] for row in found["candidates"]], ["PANC", "PAMR", "PALH"])

    def test_several_scheduled_airports_are_ambiguous(self):
        for text, idents in (("Chicago", ["KMDW", "KORD", "KRFD"]), ("Portland", ["KPDX", "KPWM"]),
                             ("Springfield", ["KSPI", "KSGF"])):
            found = self.search(text)
            self.assertEqual((found["status"], found["reason"], found["airport"]), ("ambiguous",
                                                                                   "several_scheduled_airports", None))
            self.assertEqual([row["ident"] for row in found["candidates"]], idents, text)

    def test_state_qualifier_narrows(self):
        for text, iata in (("Portland, OR", "PDX"), ("Portland, Oregon", "PDX"), ("Portland, ME", "PWM"),
                           ("Portland, maine", "PWM"), ("Springfield, IL", "SPI"), ("Springfield, Missouri", "SGF")):
            self.assertEqual(self.code(text), ("found", iata), text)
        self.assertEqual(self.search("Portland, OR")["region"], "US-OR")
        found = self.search("Portland, Atlantis")        # not a state: a place qualifier no airport matches
        self.assertEqual((found["status"], found["reason"], [r["ident"] for r in found["candidates"]]),
                         ("ambiguous", "qualifier_mismatch", ["KPDX", "KPWM"]))

    def test_airport_in_city_qualifier(self):
        for text, iata in (("John Wayne Airport in Santa Ana", "SNA"), ("John Wayne Airport, Santa Ana", "SNA"),
                           ("john wayne airport IN santa ana", "SNA"), ("John Wayne Airport in Santa Ana, CA", "SNA"),
                           ("Midway in Chicago", "MDW"), ("Springfield in Illinois", "SPI"),
                           ("Springfield in Missouri", "SGF"), ("Portland International Airport in Maine", "PWM"),
                           ("ORD in Chicago", "ORD")):
            found = self.search(text)
            self.assertEqual((found["status"], found["airport"]["iata_code"]), ("found", iata), text)
        self.assertEqual(self.search("John Wayne Airport in Santa Ana")["qualifier"], "Santa Ana")

    def test_qualifier_disambiguates_with_data(self):
        self.assertEqual(self.code("Logan Airport"), ("found", "BOS"))                 # largest scheduled
        self.assertEqual(self.code("Logan Airport in Billings"), ("found", "BIL"))      # the city decides
        self.assertEqual(self.code("Logan in Boston"), ("found", "BOS"))
        found = self.search("Chicago in Illinois")                                    # still two large ones
        self.assertEqual((found["status"], found["reason"]), ("ambiguous", "several_scheduled_airports"))

    def test_generic_airport_in_city_is_the_city(self):
        for text in ("the airport in Anchorage", "airports in Anchorage", "an airport in Anchorage"):
            self.assertEqual(self.code(text), ("found", "ANC"), text)
        self.assertEqual(self.search("the airports in Chicago")["status"], "ambiguous")

    def test_qualifier_never_overrides_the_data(self):
        for text, idents in (("John Wayne Airport in Chicago", ["KSNA"]), ("Juneau International Airport in Anchorage",
                                                                            ["PAJN"]), ("SNA in Boston", ["KSNA"])):
            found = self.search(text)
            self.assertEqual((found["status"], found["reason"], found["airport"]),
                             ("ambiguous", "qualifier_mismatch", None), text)
            self.assertEqual([r["ident"] for r in found["candidates"]], idents, text)
        for text in ("Atlantis in Santa Ana", "John Wayne Airport in", "in Santa Ana", "Anchorag in Alaska"):
            self.assertEqual(self.search(text)["status"], "not_found", text)

    def test_whole_reference_is_tried_before_the_qualifier(self):
        found = self.search("Fairbanks International Airport")
        self.assertIsNone(found["qualifier"])
        self.assertEqual(self.search("Los Angeles International Airport")["qualifier"], None)

    def test_municipality_lists_but_not_hyphenated_names(self):
        self.assertEqual(self.code("Honolulu"), ("found", "HNL"))                 # "Honolulu, Oahu"
        self.assertEqual(self.code("Oahu"), ("found", "HNL"))
        self.assertIn("KRFD", self.candidates("Rockford"))                         # "Chicago/Rockford"
        self.assertEqual(self.code("Salem"), ("found", "SLE"))                    # not Winston-Salem

    def test_place_never_auto_resolves_to_unscheduled_airport(self):
        for text, idents in (("Lonelytown", ["3LN3"]), ("Quietville", ["2QV2", "1QV1"]), ("Vancouver", ["KVUO"])):
            found = self.search(text)
            self.assertEqual((found["status"], found["reason"]), ("ambiguous", "no_scheduled_airport"), text)
            self.assertEqual([row["ident"] for row in found["candidates"]], idents, text)

    def test_exact_name_resolves_even_without_scheduled_service(self):
        self.assertEqual(self.search("Merrill Field")["airport"]["ident"], "PAMR")
        self.assertEqual(self.search("Lonely Field")["airport"]["ident"], "3LN3")

    def test_duplicate_exact_names(self):
        found = self.search("Twin Name Airport")
        self.assertEqual((found["status"], found["selection"], found["airport"]["ident"]),
                         ("found", "only_scheduled_airport", "KTWA"))
        found = self.search("Double Field")
        self.assertEqual((found["status"], found["reason"], [r["ident"] for r in found["candidates"]]),
                         ("ambiguous", "several_scheduled_airports", ["KDBA", "KDBB"]))

    def test_ambiguous_code_is_never_narrowed(self):
        found = self.search("ZZZ")
        self.assertEqual((found["status"], found["matched_on"], found["reason"]),
                         ("ambiguous", "code", "several_airports_for_code"))
        self.assertEqual([r["ident"] for r in found["candidates"]], ["KALF", "KBTF"])

    def test_unknown_generic_and_out_of_scope_references_are_not_found(self):
        for text in ("Atlantis", "International Airport", "Airport", "the airport", "!!!", "YVR", "Old Closed Airport",
                     "Ghosttown", "anc"):
            self.assertEqual(self.search(text)["status"], "not_found", text)
        self.assertEqual(self.client.search_airports("YVR", country="CA")["airport"]["ident"], "CYVR")

    def test_no_fuzzy_guessing(self):
        for text in ("Anchorag", "Fairbank", "Ankorage", "Chicgo", "Juneau Internatinal", "Anchorage Fairbanks",
                     "Fairbanks Juneau International"):
            self.assertEqual(self.search(text)["status"], "not_found", text)

    def test_deterministic(self):
        first = [self.search(t) for t in ("Chicago", "Anchorage", "Quietville")]
        reloaded = OurAirportsClient(cache_dir=self.client.cache_dir)
        self.assertEqual([reloaded.search_airports(t) for t in ("Chicago", "Anchorage", "Quietville")], first)


# --- agent tool --------------------------------------------------------------------------------

class ResolveAirportsToolTests(ResolutionTestCase):
    def resolve(self, *references):
        result = resolve_airports(list(references), context=self.ctx)
        json.dumps(result)
        return result

    def test_full_name(self):
        result = self.resolve("Ted Stevens Anchorage International Airport")
        self.assertEqual((result["tool"], result["status"], result["errors"], result["warnings"]),
                         ("resolve_airports", "ok", [], []))
        (item,) = result["results"]
        self.assertEqual((item["reference"], item["status"], item["matched_on"], item["selection"]),
                         ("Ted Stevens Anchorage International Airport", "found", "name", "unique_match"))
        self.assertEqual(item["airport"], {"code": "ANC", "name": "Ted Stevens Anchorage International Airport",
                                           "municipality": "Anchorage", "state": "AK", "airport_type": "large_airport",
                                           "scheduled_service": True, "ourairports_ident": "PANC"})

    def test_city_reports_other_scheduled_candidates(self):
        (item,) = self.resolve("Anchorage")["results"]
        self.assertEqual((item["status"], item["selection"], item["airport"]["code"]),
                         ("found", "largest_scheduled_airport", "ANC"))
        self.assertEqual([c["code"] for c in item["other_scheduled_candidates"]], ["MRI"])
        (item,) = self.resolve("Fairbanks")["results"]
        self.assertNotIn("other_scheduled_candidates", item)

    def test_several_references_mixed_code_name_city(self):
        result = self.resolve("ANC", "Fairbanks International Airport", "Juneau", "Portland, OR")
        self.assertEqual(result["status"], "ok")
        self.assertEqual([r["airport"]["code"] for r in result["results"]], ["ANC", "FAI", "JNU", "PDX"])
        self.assertEqual([r["matched_on"] for r in result["results"]], ["code", "name", "place", "place"])
        self.assertEqual(result["results"][3]["state_filter"], "OR")
        self.assertEqual(result["counts"], {"found": 4, "ambiguous": 0, "not_found": 0})

    def test_found_ambiguous_not_found_together(self):
        result = self.resolve("Fairbanks", "Chicago", "Atlantis")
        self.assertEqual(result["status"], "partial")
        found, ambiguous, missing = result["results"]
        self.assertEqual(found["airport"]["code"], "FAI")
        self.assertEqual((ambiguous["status"], ambiguous["reason"], ambiguous["candidate_count"]),
                         ("ambiguous", "several_scheduled_airports", 3))
        self.assertEqual([(c["code"], c["name"]) for c in ambiguous["candidates"]],
                         [("MDW", "Chicago Midway International Airport"),
                          ("ORD", "Chicago O'Hare International Airport"),
                          ("RFD", "Chicago Rockford International Airport")])
        self.assertNotIn("airport", ambiguous)
        self.assertEqual(missing, {"reference": "Atlantis", "status": "not_found", "matched_on": None})
        self.assertEqual([(e["code"], e["airport"]) for e in result["errors"]],
                         [("airport_reference_ambiguous", "Chicago"), ("airport_not_found", "Atlantis")])
        self.assertEqual(result["counts"], {"found": 1, "ambiguous": 1, "not_found": 1})

    def test_case_02_references_as_written_by_the_user(self):
        # Evaluation Case 02: "Compare congestion at Los Angeles International Airport and John Wayne
        # Airport in Santa Ana." - the references a model would pass, unchanged.
        result = self.resolve("Los Angeles International Airport", "John Wayne Airport in Santa Ana")
        self.assertEqual(result["status"], "ok")
        self.assertEqual([(r["airport"]["code"], r["airport"]["name"]) for r in result["results"]],
                         [("LAX", "Los Angeles International Airport"),
                          ("SNA", "John Wayne Orange County International Airport")])
        self.assertEqual([r["matched_on"] for r in result["results"]], ["name", "place"])

    def test_nothing_found_is_error_status_with_results(self):
        result = self.resolve("Atlantis", "Quietville")
        self.assertEqual(result["status"], "error")
        self.assertEqual([r["status"] for r in result["results"]], ["not_found", "ambiguous"])
        self.assertEqual(result["results"][1]["reason"], "no_scheduled_airport")

    def test_analysis_code_round_trips_through_lookup(self):
        result = self.resolve("PBI", "Pearson Field", "Quietville Strip")
        codes = [r["airport"]["code"] for r in result["results"]]
        self.assertEqual(codes, ["DJT", "KVUO", "1QV1"])                # IATA when unique, else the ident
        for result_item, code in zip(result["results"], codes):
            found = self.client.lookup_code(code)
            self.assertEqual(found["airport"]["ident"], result_item["airport"]["ourairports_ident"])

    def test_invalid_arguments(self):
        for bad in (None, [], "Anchorage", [""], ["  "], [42], ["x"] * 31, ["x" * 201]):
            result = resolve_airports(bad, context=self.ctx)
            self.assertEqual((result["status"], result["errors"][0]["code"]), ("error", "invalid_argument"), bad)

    def test_no_bts_or_network_used(self):
        self.resolve("Anchorage", "Chicago", "ANC", "Atlantis")
        self.ctx.t100.assert_not_called()
        self.ctx.ontime.assert_not_called()
        self.assertEqual(self.ctx.t100.method_calls, [])
        self.assertEqual(self.ctx.ontime.method_calls, [])

    def test_candidate_lists_are_bounded(self):
        rows = [("K%03d" % i, "small_airport", f"Field {i}", "Bigtown", "US-KS", "no", "", f"B{i:02d}", "")
                for i in range(12)]
        with mock.patch(f"{__name__}.AIRPORTS", AIRPORTS + rows):
            write_fixture(self.client.cache_dir)
        client = OurAirportsClient(cache_dir=self.client.cache_dir)
        (item,) = resolve_airports(["Bigtown"], context=AgentToolContext(ourairports=client, t100=mock.Mock(),
                                                                        ontime=mock.Mock()))["results"]
        self.assertEqual((item["candidate_count"], len(item["candidates"])), (12, 5))


# --- registry, dispatcher, instructions ---------------------------------------------------------

class RegistryAndInstructionTests(ResolutionTestCase):
    def test_four_business_tools(self):
        self.assertEqual(sorted(TOOLS), ["analyze_airport", "compare_airports", "discover_airports",
                                         "resolve_airports"])

    def test_schema(self):
        declaration = TOOLS["resolve_airports"].declaration
        schema = declaration.parameters_json_schema
        self.assertEqual(schema["required"], ["references"])
        self.assertEqual(set(schema["properties"]), {"references"})
        self.assertEqual(schema["properties"]["references"]["type"], "array")
        self.assertEqual(schema["properties"]["references"]["items"], {"type": "string"})
        text = json.dumps([d.model_dump(exclude_none=True) for d in tool_declarations()])
        self.assertNotIn('"context":', text)
        for policy in ("BEFORE analyze_airport", "ambiguous", "memory", "discover_airports"):
            self.assertIn(policy, declaration.description)

    def test_dispatcher_executes_it_with_the_session_context(self):
        result = execute_tool("resolve_airports", {"references": ["Juneau"]}, self.ctx)
        self.assertEqual(result["results"][0]["airport"]["code"], "JNU")
        result = execute_tool("resolve_airports", {"references": ["Juneau"], "country": "CA"}, self.ctx)
        self.assertEqual(result["errors"][0]["code"], "invalid_tool_arguments")

    def test_instruction_policy(self):
        lowered = SYSTEM_INSTRUCTION.lower()
        for policy in ("resolve_airports", "memory", "ambiguous", "discover_airports", "(code)", "full name"):
            self.assertIn(policy, lowered)


# --- output shapes of the existing tools (shared offline fixture) --------------------------------

class CanonicalNameOutputTests(ToolTestCase):
    def test_compare_rows_carry_names(self):
        result = compare_airports(["ANC", "FAI", "JNU", "SEA", "DJT", "XXA"], context=self.ctx)
        names = {row["code"]: row["name"] for row in result["ranking"] + result["not_scored"]}
        self.assertEqual(names["ANC"], "Anchorage Intl")
        self.assertEqual(names["XXA"], "Unverified Commuter Field")
        self.assertEqual(names["PBI"], "President Donald J. Trump Intl")          # BTS code, OurAirports name
        for row in result["ranking"]:
            self.assertEqual(set(row), {"code", "name", "rank", "investment_score", "coverage"})

    def test_discovery_rows_carry_municipality(self):
        result = discover_airports("US", ["AK"], context=self.ctx)
        self.assertEqual(result["columns"][:4], ["code", "name", "municipality", "state"])
        rows = {row[0]: dict(zip(result["columns"], row)) for row in result["airports"]}
        self.assertEqual(rows["ANC"]["municipality"], "Anchorage Intl City")
        for row in result["airports"]:
            self.assertEqual(len(row), len(result["columns"]))

    def test_analyze_keeps_name_and_municipality(self):
        from agent_tools import analyze_airport
        airport = analyze_airport("ANC", context=self.ctx)["airport"]
        self.assertEqual((airport["name"], airport["municipality"]), ("Anchorage Intl", "Anchorage Intl City"))


# --- end to end through the agent loop (fake Gemini) --------------------------------------------

class AgentResolutionFlowTests(AgentLoopTestCase):
    def test_city_names_resolved_then_compared(self):
        gemini = FakeGemini(call_response(("resolve_airports", {"references": ["Anchorage", "Fairbanks"]}, "r1")),
                            call_response(("compare_airports", {"airport_codes": ["ANC", "FAI"]}, "c1")),
                            text_response("Anchorage Intl (ANC) ranks first."))
        session = AgentSession(client=gemini, context=self.ctx)
        self.assertEqual(session.send("Compare the airports in Anchorage and Fairbanks."),
                         "Anchorage Intl (ANC) ranks first.")
        resolved = gemini.requests[1]["contents"][-1].parts[0].function_response.response
        self.assertEqual([r["airport"]["code"] for r in resolved["results"]], ["ANC", "FAI"])
        self.assertEqual([c.args[0] for c in self.dispatch.call_args_list], ["resolve_airports", "compare_airports"])
        compared = gemini.requests[2]["contents"][-1].parts[0].function_response.response
        self.assertEqual({r["name"] for r in compared["ranking"]}, {"Anchorage Intl", "Fairbanks Intl"})
        self.assertEqual([(t.name, t.category, t.status) for t in session.last_trace.tool_calls],
                         [("resolve_airports", "business", "ok"), ("compare_airports", "business", "ok")])


if __name__ == "__main__":
    unittest.main()
