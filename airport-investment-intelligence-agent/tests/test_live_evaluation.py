"""Offline tests for the live-evaluation infrastructure (evaluation/).

No real Gemini: scripted FakeGemini responses drive the real AgentSession on the synthetic
tool fixtures, or execute_case is replaced by canned records. sleep is always faked, HTTP is
blocked, and results go to temporary directories.
"""

import dataclasses
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from google.genai import errors as genai_errors
from google.genai import types

import agent.loop as loop
import agent.tools as tools_module
import evaluation.evaluator as evaluator
import evaluation.run_live_evaluation as runner
import evaluation.storage as storage
from agent import MEMORY_TOOL_NAME, AgentError, RegisteredTool
from evaluation.cases import CASE_IDS, CASES, EVALUATION_VERSION
from evaluation.evaluator import FAIL, NA, PASS, REVIEW, evaluate_case
from evaluation.run_live_evaluation import (DEFAULT_RETRY_WAIT_S, FATAL, TRANSIENT, Attempt, classify,
                                            execute_case, main, retry_wait_seconds)
from evaluation.storage import CaseAlreadyCompleted, RunStore, RunStoreError, atomic_write_json
from tests.test_agent_loop import AgentLoopTestCase, FakeGemini, call_response, text_response

CASE = {case.id: case for case in CASES}


def api_error(code, status, retry_delay=None, cls=None):
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}] if retry_delay else []
    cls = cls or (genai_errors.ClientError if code < 500 else genai_errors.ServerError)
    return cls(code, {"error": {"code": code, "message": "provider says no", "status": status, "details": details}})


def resolve(*references, call_id="r1"):
    return call_response(("resolve_airports", {"references": list(references)}, call_id))


def compare(*codes, call_id="c1", **extra):
    return call_response(("compare_airports", {"airport_codes": list(codes), **extra}, call_id))


def analyze(code, call_id="a1"):
    return call_response(("analyze_airport", {"airport_code": code}, call_id))


def memory(query, call_id="m1"):
    return call_response((MEMORY_TOOL_NAME, {"query": query}, call_id))


def discover(states, call_id="d1"):
    return call_response(("discover_airports", {"country": "US", "states": list(states)}, call_id))


MEMORY_SETUP_SCRIPT = (resolve("Anchorage", "Fairbanks", "Juneau", call_id="s1"),
                       compare("ANC", "FAI", "JNU", call_id="s2"),
                       text_response("Anchorage Intl ranks first; Fairbanks Intl grows fastest."),
                       text_response("CAGR is a smoothed annual growth rate."),
                       text_response("A percentile rank is the share of the group below a value."))


class FailingAtRequest(FakeGemini):
    def __init__(self, fail_at, error, *responses):
        super().__init__(*responses)
        self.fail_at, self.error = fail_at, error

    def generate_content(self, *, model, contents, config):
        if len(self.requests) + 1 == self.fail_at:
            self.requests.append({"model": model, "contents": list(contents), "config": config})
            raise self.error
        return super().generate_content(model=model, contents=contents, config=config)


class LiveEvalTestCase(AgentLoopTestCase):
    """Offline tool fixtures (ANC/FAI/JNU/SEA/...), HTTP blocked, real Gemini creation blocked."""

    def run_case(self, case, *responses, client=None):
        attempt = Attempt(case, client or FakeGemini(*responses), lambda: self.ctx)
        return execute_case(case, attempt), attempt

    def evaluate(self, case, *responses):
        record, attempt = self.run_case(case, *responses)
        return evaluate_case(case, record), record, attempt


# --- deterministic evaluation on real sessions (fake Gemini) -------------------------------------

class SingleAirportEvaluationTests(LiveEvalTestCase):
    def test_resolve_then_analyze_passes_mechanical_dimensions(self):
        result, record, _ = self.evaluate(CASE["01"], resolve("Anchorage"), analyze("ANC"),
                                          text_response("Anchorage Intl: 12% long-haul."))
        self.assertEqual(result["dimensions"], {"tool_selection": PASS, "tool_arguments": PASS,
                                                "entity_resolution": PASS, "memory_behavior": PASS,
                                                "grounding": REVIEW, "task_success": REVIEW})
        self.assertEqual(record["trace"]["total_llm_calls"], 3)
        self.assertIn("ANC", result["tool_values"])
        names = {c["name"]: c["result"] for c in result["checks"]}
        self.assertFalse(names["canonical_name_in_answer:Ted Stevens Anchorage International Airport"])
        self.assertTrue(any("canonical name not used" in f for f in result["review_flags"]))

    def test_memorized_code_without_resolution_fails_entity_resolution(self):
        result, _, _ = self.evaluate(CASE["01"], analyze("ANC"), text_response("12%."))
        self.assertEqual(result["dimensions"]["tool_selection"], FAIL)
        self.assertEqual(result["dimensions"]["entity_resolution"], FAIL)
        self.assertIn("codes_come_from_tool_results", " ".join(result["dimension_reasons"]["entity_resolution"]))

    def test_analysis_before_resolution_fails(self):
        result, _, _ = self.evaluate(CASE["01"], analyze("ANC"), resolve("Anchorage", call_id="r2"),
                                     text_response("12%."))
        self.assertEqual(result["dimensions"]["entity_resolution"], FAIL)

    def test_wrong_airport_fails_tool_arguments(self):
        result, _, _ = self.evaluate(CASE["01"], resolve("Anchorage", "Seattle"), analyze("SEA"),
                                     text_response("..."))
        self.assertEqual(result["dimensions"]["tool_arguments"], FAIL)
        self.assertEqual(result["dimensions"]["tool_selection"], FAIL)        # ANC never analysed

    def test_unnecessary_memory_and_discovery_fail(self):
        result, _, _ = self.evaluate(CASE["01"], memory("Anchorage"), discover(["AK"]), resolve("Anchorage"),
                                     analyze("ANC"), text_response("..."))
        self.assertEqual(result["dimensions"]["memory_behavior"], FAIL)
        self.assertEqual(result["dimensions"]["tool_selection"], FAIL)

    def test_optional_analysis_case(self):
        case = dataclasses.replace(CASE["05"], expected_codes=("SEA",), expected_names=())
        result, _, _ = self.evaluate(case, resolve("Seattle"), text_response("Unmet demand is not measured."))
        self.assertEqual((result["dimensions"]["tool_selection"], result["dimensions"]["tool_arguments"],
                          result["dimensions"]["entity_resolution"]), (PASS, NA, PASS))
        result, _, _ = self.evaluate(case, text_response("Unmet demand is not measured."))
        self.assertEqual(result["dimensions"]["tool_selection"], FAIL)        # airport never resolved

    def test_flag_terms_are_review_flags_not_failures(self):
        result, _, _ = self.evaluate(CASE["08"], resolve("Fairbanks"), analyze("FAI"),
                                     text_response("Delays are caused by weather."))
        self.assertEqual(result["dimensions"]["grounding"], REVIEW)
        self.assertTrue(any("weather" in flag for flag in result["review_flags"]))

    def test_agent_error_is_a_case_result(self):
        empty = types.GenerateContentResponse(candidates=[])
        result, record, _ = self.evaluate(CASE["01"], empty)
        self.assertEqual(record["outcome"], "agent_error")
        self.assertEqual(record["error"]["type"], "EmptyModelResponse")
        self.assertEqual(result["dimensions"]["task_success"], FAIL)


class ComparisonEvaluationTests(LiveEvalTestCase):
    def test_resolved_comparison_passes_and_reports_growth_winner(self):
        result, _, _ = self.evaluate(CASE["03"], resolve("Anchorage", "Fairbanks", "Juneau"),
                                     compare("ANC", "FAI", "JNU"), text_response("Fairbanks Intl grew fastest."))
        self.assertEqual([result["dimensions"][d] for d in ("tool_selection", "tool_arguments", "entity_resolution")],
                         [PASS, PASS, PASS])
        winner = result["tool_values"]["_highest_passenger_cagr"]
        self.assertEqual(winner["code"], "FAI")                                 # from THIS run's tool result
        self.assertTrue({c["name"]: c["result"] for c in result["checks"]}["highest_cagr_airport_mentioned"])

    def test_analyze_each_airport_is_an_accepted_trajectory(self):
        result, _, _ = self.evaluate(CASE["03"], resolve("Anchorage", "Fairbanks", "Juneau"),
                                     call_response(("analyze_airport", {"airport_code": "ANC"}, "a1"),
                                                   ("analyze_airport", {"airport_code": "FAI"}, "a2"),
                                                   ("analyze_airport", {"airport_code": "JNU"}, "a3")),
                                     text_response("..."))
        self.assertEqual(result["dimensions"]["tool_selection"], PASS)

    def test_missing_airport_and_extra_airport(self):
        result, _, _ = self.evaluate(CASE["03"], resolve("Anchorage", "Fairbanks", "Juneau"), compare("ANC", "FAI"),
                                     text_response("..."))
        self.assertEqual(result["dimensions"]["tool_selection"], FAIL)
        result, _, _ = self.evaluate(CASE["03"], resolve("Anchorage", "Fairbanks", "Juneau", "Seattle"),
                                     compare("ANC", "FAI", "JNU", "SEA"), text_response("..."))
        self.assertEqual(result["dimensions"]["tool_arguments"], FAIL)


class GeographicEvaluationTests(LiveEvalTestCase):
    case = dataclasses.replace(CASE["07"], discover_states=("AK",))      # fixture has Alaska only

    def test_within_limit_compare_of_discovered_group_passes(self):
        result, _, _ = self.evaluate(self.case, discover(["AK"]), compare("ANC", "FAI", "JNU", "XXA", "HOA", "CGO"),
                                     text_response("..."))
        self.assertEqual((result["dimensions"]["tool_selection"], result["dimensions"]["tool_arguments"],
                          result["dimensions"]["entity_resolution"]), (PASS, PASS, NA))
        self.assertIn("premise not met: New England group was within the automatic limit", result["review_flags"])

    def test_wrong_states_fail_arguments(self):
        result, _, _ = self.evaluate(self.case, discover(["AK", "WA"]), text_response("..."))
        self.assertEqual(result["dimensions"]["tool_arguments"], FAIL)

    def test_above_limit_asking_the_user_passes(self):
        with mock.patch.object(evaluator, "AUTOMATIC_ANALYSIS_LIMIT", 3):
            result, _, _ = self.evaluate(self.case, discover(["AK"]), text_response("6 airports: narrow or authorise?"))
        self.assertEqual((result["dimensions"]["tool_selection"], result["dimensions"]["task_success"]), (PASS, REVIEW))

    def test_above_limit_guardrail_violations_fail(self):
        scripts = {
            "self-chosen subset": (discover(["AK"]), compare("ANC", "FAI", "JNU"), text_response("...")),
            "allow_large_group": (discover(["AK"]), compare("ANC", "FAI", "JNU", "XXA", "HOA", "CGO",
                                                            allow_large_group=True), text_response("...")),
            "split groups": (discover(["AK"]), compare("ANC", "FAI", call_id="c1"), compare("JNU", "HOA", call_id="c2"),
                             text_response("...")),
        }
        for label, script in scripts.items():
            with self.subTest(label), mock.patch.object(evaluator, "AUTOMATIC_ANALYSIS_LIMIT", 3):
                result, _, _ = self.evaluate(self.case, *script)
                self.assertEqual(result["dimensions"]["tool_selection"], FAIL)
                self.assertEqual(result["dimensions"]["task_success"], FAIL)

    def test_split_groups_fail_arguments_and_invented_codes_fail(self):
        result, _, _ = self.evaluate(self.case, discover(["AK"]), compare("ANC", "FAI", call_id="c1"),
                                     compare("JNU", "HOA", call_id="c2"), text_response("..."))
        self.assertEqual(result["dimensions"]["tool_arguments"], FAIL)
        result, _, _ = self.evaluate(self.case, discover(["AK"]), compare("ANC", "SEA"), text_response("..."))
        self.assertEqual(result["dimensions"]["tool_arguments"], FAIL)         # SEA was not discovered

    def test_resolve_not_required(self):
        result, _, _ = self.evaluate(self.case, discover(["AK"]), compare("ANC", "FAI", "JNU", "XXA", "HOA", "CGO"),
                                     text_response("..."))
        self.assertEqual(result["dimensions"]["tool_selection"], PASS)
        self.assertFalse({c["name"]: c["result"] for c in result["checks"]}["resolve_airports_used"])


class InvestmentComparisonEvaluationTests(LiveEvalTestCase):
    """Case 06: natural-language resolution + ONE comparison group + recommendation."""

    def test_definition(self):
        case = CASE["06"]
        self.assertEqual((case.kind, case.expected_codes, case.discover_states, case.setup),
                         ("investment_comparison", ("ANC", "FAI", "JNU"), (), ()))
        self.assertIn("Between the airports in Anchorage, Fairbanks and Juneau", case.prompt)

    def test_resolve_then_compare_passes(self):
        result, _, _ = self.evaluate(CASE["06"], resolve("Anchorage", "Fairbanks", "Juneau"),
                                     compare("ANC", "FAI", "JNU"),
                                     text_response("Anchorage Intl, Fairbanks Intl and Juneau Intl compared."))
        self.assertEqual([result["dimensions"][d] for d in ("tool_selection", "tool_arguments", "entity_resolution",
                                                             "memory_behavior", "grounding", "task_success")],
                         [PASS, PASS, PASS, PASS, REVIEW, REVIEW])
        rank_1 = {c["name"]: c for c in result["checks"]}["rank_1_airport_mentioned"]
        self.assertTrue(rank_1["result"])                                  # rank #1 read from THIS run's result
        self.assertEqual(rank_1["detail"]["name"], result["tool_values"][
            min((k for k in result["tool_values"] if not k.startswith("_")),
                key=lambda k: result["tool_values"][k]["rank"])]["name"])

    def test_city_phrasing_resolves(self):
        result, _, _ = self.evaluate(CASE["06"], resolve("the airport in Anchorage", "Fairbanks", "Juneau"),
                                     compare("ANC", "FAI", "JNU"), text_response("..."))
        self.assertEqual(result["dimensions"]["entity_resolution"], PASS)

    def test_analyzing_each_airport_is_not_enough(self):
        result, _, _ = self.evaluate(CASE["06"], resolve("Anchorage", "Fairbanks", "Juneau"),
                                     call_response(("analyze_airport", {"airport_code": "ANC"}, "a1"),
                                                   ("analyze_airport", {"airport_code": "FAI"}, "a2"),
                                                   ("analyze_airport", {"airport_code": "JNU"}, "a3")),
                                     text_response("..."))
        self.assertEqual(result["dimensions"]["tool_selection"], FAIL)     # no comparison group, no score

    def test_geographic_discovery_and_memorized_codes_fail(self):
        result, _, _ = self.evaluate(CASE["06"], discover(["AK"]), compare("ANC", "FAI", "JNU"), text_response("..."))
        self.assertEqual(result["dimensions"]["tool_selection"], FAIL)     # discovery, no resolve
        self.assertEqual(result["dimensions"]["entity_resolution"], FAIL)
        result, _, _ = self.evaluate(CASE["06"], compare("ANC", "FAI", "JNU"), text_response("..."))
        self.assertEqual(result["dimensions"]["entity_resolution"], FAIL)   # codes from model memory

    def test_split_or_extra_airports_fail(self):
        result, _, _ = self.evaluate(CASE["06"], resolve("Anchorage", "Fairbanks", "Juneau"),
                                     compare("ANC", "FAI", call_id="c1"), compare("JNU", call_id="c2"),
                                     text_response("..."))
        self.assertEqual(result["dimensions"]["tool_selection"], FAIL)
        result, _, _ = self.evaluate(CASE["06"], resolve("Anchorage", "Fairbanks", "Juneau", "Seattle"),
                                     compare("ANC", "FAI", "JNU", "SEA"), text_response("..."))
        self.assertEqual(result["dimensions"]["tool_arguments"], FAIL)


class NoToolsEvaluationTests(LiveEvalTestCase):
    """Case 10: knowing when NOT to use tools."""

    def test_definition(self):
        case = CASE["10"]
        self.assertEqual((case.kind, case.expected_codes, case.setup), ("no_tools", (), ()))
        self.assertIn("comparison groups", case.prompt)
        self.assertTrue(any("comparison-relative" in focus for focus in case.review_focus))
        self.assertTrue(any("Cross-group" in focus for focus in case.review_focus))

    def test_answer_without_tools_passes(self):
        result, record, _ = self.evaluate(CASE["10"], text_response(
            "The Investment Score is a percentile rank within its comparison group, so scores from different "
            "groups are not on one scale."))
        self.assertEqual(result["dimensions"], {"tool_selection": PASS, "tool_arguments": NA,
                                                "entity_resolution": NA, "memory_behavior": PASS,
                                                "grounding": REVIEW, "task_success": REVIEW})
        self.assertEqual(record["trace"]["total_llm_calls"], 1)
        self.assertEqual(result["review_flags"], [])

    def test_any_business_tool_fails_tool_selection(self):
        scripts = {"resolve": resolve("Anchorage"), "discover": discover(["AK"]), "analyze": analyze("ANC"),
                   "compare": compare("ANC", "FAI")}
        for label, call in scripts.items():
            with self.subTest(label):
                result, _, _ = self.evaluate(CASE["10"], call, text_response("..."))
                self.assertEqual(result["dimensions"]["tool_selection"], FAIL)
                self.assertEqual(result["dimensions"]["memory_behavior"], PASS)
                self.assertEqual(result["dimensions"]["tool_arguments"], NA)

    def test_memory_lookup_fails_memory_and_tool_selection(self):
        result, _, _ = self.evaluate(CASE["10"], memory("Investment Score"), text_response("..."))
        self.assertEqual((result["dimensions"]["memory_behavior"], result["dimensions"]["tool_selection"]),
                         (FAIL, FAIL))

    def test_invented_numbers_or_codes_are_review_flags(self):
        result, _, _ = self.evaluate(CASE["10"], text_response("For example ANC scored 87 and SEA 42."))
        self.assertEqual(result["dimensions"]["grounding"], REVIEW)
        self.assertEqual(len(result["review_flags"]), 2)


class MemoryEvaluationTests(LiveEvalTestCase):
    def test_memory_recall_passes_and_old_turn_not_in_initial_context(self):
        result, record, attempt = self.evaluate(CASE["09"], *MEMORY_SETUP_SCRIPT,
                                                memory("Anchorage Fairbanks Juneau passenger growth comparison"),
                                                text_response("Fairbanks Intl had the highest growth."))
        self.assertEqual(result["dimensions"]["memory_behavior"], PASS)
        self.assertEqual(result["dimensions"]["tool_selection"], PASS)
        self.assertEqual(record["stored_turns_before_evaluation"], 3)
        self.assertEqual(record["visible_turns_at_evaluation"], 2)
        self.assertFalse(record["evaluated_requests"][0]["contains_setup_turn_1"])
        self.assertTrue(any(r["contains_setup_turn_1"] for r in attempt.observer.requests if r["phase"] == "setup_1"))
        self.assertEqual(result["tool_values"]["_setup_turn_1"]["_highest_passenger_cagr"]["code"], "FAI")
        self.assertTrue({c["name"]: c["result"] for c in result["checks"]}["highest_cagr_airport_mentioned"])

    def test_no_memory_call_fails(self):
        result, _, _ = self.evaluate(CASE["09"], *MEMORY_SETUP_SCRIPT, text_response("I don't know which three."))
        self.assertEqual((result["dimensions"]["memory_behavior"], result["dimensions"]["tool_selection"]),
                         (FAIL, FAIL))

    def test_irrelevant_memory_query_fails_retrieval(self):
        result, _, _ = self.evaluate(CASE["09"], *MEMORY_SETUP_SCRIPT, memory("weather"), text_response("..."))
        self.assertEqual(result["dimensions"]["memory_behavior"], FAIL)

    def test_invalid_setup_skips_the_evaluated_turn(self):
        gemini = FakeGemini(text_response("Which airports do you mean?"), text_response("x"), text_response("y"))
        record, _ = self.run_case(CASE["09"], client=gemini)
        self.assertEqual(record["outcome"], "setup_invalid")
        self.assertEqual(len(gemini.requests), 3)                           # no evaluated-turn request
        result = evaluate_case(CASE["09"], record)
        self.assertEqual(result["dimensions"]["task_success"], FAIL)
        self.assertEqual(result["dimensions"]["memory_behavior"], NA)


class ReviewDimensionTests(LiveEvalTestCase):
    def test_grounding_always_needs_review_and_task_success_never_auto_passes(self):
        for case_id, script in (("01", (resolve("Anchorage"), analyze("ANC"), text_response("x"))),
                                ("04", (resolve("Ted Stevens Anchorage International Airport"), analyze("ANC"),
                                        text_response("Capacity Pressure is a proxy."))),
                                ("08", (resolve("Fairbanks"), analyze("FAI"), text_response("x")))):
            result, _, _ = self.evaluate(CASE[case_id], *script)
            self.assertEqual(result["dimensions"]["grounding"], REVIEW, case_id)
            self.assertEqual(result["dimensions"]["task_success"], REVIEW, case_id)


# --- error classification and retry delay ---------------------------------------------------------

class ClassificationTests(LiveEvalTestCase):
    def test_transient_and_fatal(self):
        import httpx
        import requests
        self.assertEqual(classify(api_error(429, "RESOURCE_EXHAUSTED")), TRANSIENT)
        self.assertEqual(classify(api_error(503, "UNAVAILABLE")), TRANSIENT)
        self.assertEqual(classify(api_error(504, "DEADLINE_EXCEEDED")), TRANSIENT)
        self.assertEqual(classify(httpx.ConnectError("down")), TRANSIENT)
        self.assertEqual(classify(requests.ConnectionError("down")), TRANSIENT)
        for exc in (api_error(400, "INVALID_ARGUMENT"), api_error(401, "UNAUTHENTICATED"),
                    api_error(403, "PERMISSION_DENIED"), api_error(404, "NOT_FOUND"), ValueError("bug"),
                    TypeError("bug"), KeyError("x"), AgentError("GEMINI_API_KEY was not found in .env"),
                    RunStoreError("corrupt")):
            self.assertEqual(classify(exc), FATAL, exc)

    def test_retry_wait(self):
        self.assertEqual(retry_wait_seconds(api_error(429, "RESOURCE_EXHAUSTED")), DEFAULT_RETRY_WAIT_S)
        self.assertEqual(retry_wait_seconds(api_error(429, "RESOURCE_EXHAUSTED", "40s")), DEFAULT_RETRY_WAIT_S)
        self.assertEqual(retry_wait_seconds(api_error(429, "RESOURCE_EXHAUSTED", "7200s")), 7200)
        self.assertEqual(retry_wait_seconds(api_error(503, "UNAVAILABLE", "1900.5s")), 1900.5)
        error = api_error(503, "UNAVAILABLE")
        error.response = mock.Mock(headers={"retry-after": "3600"})
        self.assertEqual(retry_wait_seconds(error), 3600)
        error.response = mock.Mock(headers={"retry-after": "not a number or date"})
        self.assertEqual(retry_wait_seconds(error), DEFAULT_RETRY_WAIT_S)


# --- runner, storage, resume ----------------------------------------------------------------------

def canned_record(calls=2):
    trace = {"status": "success", "total_llm_calls": calls, "tool_rounds": 0, "finalization_used": False,
             "tool_calls": [], "llm_calls": []}
    return {"outcome": "completed", "final_answer": "answer", "trace": trace, "turn": None, "setup": [],
            "setup_checks": [], "error": None, "evaluated_requests": []}


class Clock:
    def __init__(self):
        self.value = datetime(2026, 10, 12, 14, 0)

    def __call__(self):
        self.value += timedelta(seconds=1)
        return self.value


class RunnerTestCase(LiveEvalTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.results = Path(tmp.name)
        self.sleeps, self.lines, self.clock = [], [], Clock()
        self.client_factory = mock.Mock(return_value=FakeGemini())

    def main(self, *argv, execute=None):
        patcher = mock.patch.object(runner, "execute_case", side_effect=execute) if execute else mock.MagicMock()
        with patcher:
            return main(list(argv), client_factory=self.client_factory, sleep=self.sleeps.append, now=self.clock,
                        out=lambda line, **kw: self.lines.append(line), results_dir=self.results,
                        context_factory=lambda: self.ctx)

    def run_dir(self):
        (directory,) = [p for p in self.results.iterdir() if p.is_dir()]
        return directory

    def read(self, name):
        return json.loads((self.run_dir() / name).read_text(encoding="utf-8"))


class RunnerTests(RunnerTestCase):
    def test_full_run_persists_each_case_and_report_immediately(self):
        seen = []

        def execute(case, attempt):
            if seen:                                  # case N-1 must already be on disk with the report
                report = self.read("report.json")
                self.assertEqual(report["completed_cases"], seen)
                self.assertTrue((self.run_dir() / f"case_{seen[-1]}.json").is_file())
                self.assertEqual(report["pending_case"]["case_id"], case.id)
            seen.append(case.id)
            return canned_record(calls=int(case.id))

        self.assertEqual(self.main(execute=execute), 0)
        self.assertEqual(seen, list(CASE_IDS))
        report = self.read("report.json")
        self.assertEqual((report["status"], report["remaining_cases"], report["pending_case"]), ("completed", [], None))
        self.assertEqual(report["total_gemini_calls_completed_cases"], sum(range(1, 11)))
        self.assertEqual(report["gemini_calls_per_completed_case"]["03"], 3)
        case = self.read("case_03.json")
        for key in ("case_id", "case_name", "prompt", "setup_description", "final_answer", "trace", "checks",
                    "dimensions", "duration_s", "gemini_calls", "tool_rounds", "retry_count", "completed_at",
                    "failed_attempts", "evaluation_version", "run_id"):
            self.assertIn(key, case)
        self.assertEqual(case["gemini_calls"]["total"], 3)                 # from the trace
        self.assertEqual(case["model"], "gemini-3.5-flash-lite")           # the session's actual model
        self.assertEqual(self.client_factory.call_count, 1)
        self.assertTrue(any("FINISHED" in line for line in self.lines))
        self.assertEqual(self.sleeps, [])

    def test_transient_429_retries_same_case_after_30_minutes(self):
        failures = {"02": [api_error(429, "RESOURCE_EXHAUSTED")], "05": [api_error(503, "UNAVAILABLE", "7200s")]}
        order = []

        def execute(case, attempt):
            order.append(case.id)
            if failures.get(case.id):
                raise failures[case.id].pop(0)
            return canned_record()

        self.assertEqual(self.main(execute=execute), 0)
        self.assertEqual(order, ["01", "02", "02", "03", "04", "05", "05", "06", "07", "08", "09", "10"])
        self.assertEqual(sum(self.sleeps[:6]), DEFAULT_RETRY_WAIT_S)           # 6 x 5-minute heartbeats
        self.assertEqual(sum(self.sleeps[6:]), 7200)                            # longer provider hint honoured
        case = self.read("case_02.json")
        self.assertEqual(case["retry_count"], 1)
        (failure,) = case["failed_attempts"]
        self.assertEqual((failure["status_code"], failure["provider_status"], failure["classification"],
                          failure["wait_s"]), (429, "RESOURCE_EXHAUSTED", "transient", DEFAULT_RETRY_WAIT_S))
        self.assertEqual(self.read("report.json")["retry_counts"]["05"], 1)
        text = "\n".join(self.lines)
        for marker in ("RESOURCE_EXHAUSTED", "Progress saved", "Waiting — retrying at approximately",
                       "Still waiting", "Retrying case 02"):
            self.assertIn(marker, text)

    def test_fatal_error_persists_and_stops(self):
        def execute(case, attempt):
            if case.id == "03":
                raise ValueError("programming bug")
            return canned_record()

        with mock.patch("sys.stderr"):
            self.assertEqual(self.main(execute=execute), 1)
        run, report = self.read("run.json"), self.read("report.json")
        self.assertEqual((run["status"], report["status"]), ("stopped_fatal", "stopped_fatal"))
        self.assertEqual(report["completed_cases"], ["01", "02"])
        self.assertEqual(report["pending_case"]["case_id"], "03")
        self.assertIn("ValueError: programming bug", run["fatal_error"]["traceback"])
        self.assertEqual(self.sleeps, [])
        self.assertTrue(any("STOPPED" in line for line in self.lines))

    def test_config_and_auth_errors_are_not_retried(self):
        for error in (api_error(401, "UNAUTHENTICATED"), api_error(400, "INVALID_ARGUMENT")):
            with self.subTest(error.code):
                calls = []

                def execute(case, attempt):
                    calls.append(case.id)
                    raise error

                with mock.patch("sys.stderr"):
                    self.assertEqual(self.main(execute=execute), 1)
                self.assertEqual((calls, self.sleeps), (["01"], []))
                for directory in list(self.results.iterdir()):
                    for f in directory.iterdir():
                        f.unlink()
                    directory.rmdir()

    def test_missing_api_key_stops_before_any_case(self):
        self.client_factory.side_effect = AgentError("GEMINI_API_KEY was not found in .env")
        execute = mock.Mock()
        with mock.patch("sys.stderr"):
            self.assertEqual(self.main(execute=execute), 1)
        execute.assert_not_called()
        self.assertEqual(self.read("run.json")["fatal_error"]["phase"], "client_creation")

    def test_completed_case_survives_later_crash_and_resume_skips_it(self):
        def crash_at_04(case, attempt):
            if case.id == "04":
                raise KeyboardInterrupt                 # e.g. process killed mid-case
            return canned_record()

        self.assertEqual(self.main(execute=crash_at_04), 130)
        before = {name: (self.run_dir() / name).read_bytes() for name in ("case_01.json", "case_02.json", "case_03.json")}
        run = self.read("run.json")
        self.assertEqual((run["pending_case"]["case_id"], run["pending_case"]["attempts_started"]), ("04", 1))

        executed = []
        self.assertEqual(self.main("--resume", str(self.run_dir()),
                                   execute=lambda case, attempt: executed.append(case.id) or canned_record()), 0)
        self.assertEqual(executed, ["04", "05", "06", "07", "08", "09", "10"])
        for name, content in before.items():
            self.assertEqual((self.run_dir() / name).read_bytes(), content)
        case = self.read("case_04.json")
        self.assertEqual((case["retry_count"], case["interrupted_attempts"]), (1, 1))
        self.assertEqual(len(self.read("run.json")["resumed_at"]), 1)

    def test_resume_after_fatal_continues(self):
        def execute(case, attempt):
            if case.id == "02":
                raise ValueError("bug")
            return canned_record()

        with mock.patch("sys.stderr"):
            self.main(execute=execute)
        self.assertEqual(self.main("--resume", str(self.run_dir()), execute=lambda c, a: canned_record()), 0)
        report = self.read("report.json")
        self.assertEqual((report["status"], len(report["completed_cases"])), ("completed", 10))
        self.assertEqual(self.read("case_02.json")["failed_attempts"][0]["error_type"], "ValueError")


class ResumeValidationTests(RunnerTestCase):
    def started_run(self):
        def execute(case, attempt):
            if case.id == "02":
                raise KeyboardInterrupt
            return canned_record()
        self.main(execute=execute)
        return self.run_dir()

    def assert_rejected(self, directory):
        snapshot = {p.name: p.read_bytes() for p in directory.iterdir()}
        execute = mock.Mock()
        self.assertEqual(self.main("--resume", str(directory), execute=execute), 2)
        execute.assert_not_called()
        self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, snapshot)    # untouched
        self.assertTrue(any("Cannot use the run directory" in line for line in self.lines))

    def test_version_mismatch_rejected(self):
        directory = self.started_run()
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        run["evaluation_version"] = "live-eval-v0"
        atomic_write_json(directory / "run.json", run)
        self.assert_rejected(directory)

    def test_run_from_previous_version_rejected(self):
        directory = self.started_run()
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        run["evaluation_version"] = "live-eval-v1"
        atomic_write_json(directory / "run.json", run)
        self.assert_rejected(directory)

    def assert_contract_change_rejected(self, patcher, component):
        directory = self.started_run()
        with patcher:
            self.assert_rejected(directory)
        self.assertTrue(any(component in line and "Agent contract changed" in line for line in self.lines),
                        self.lines[-1])
        executed = []                                          # contract restored -> the same run resumes
        self.assertEqual(self.main("--resume", str(directory),
                                   execute=lambda case, attempt: executed.append(case.id) or canned_record()), 0)
        self.assertEqual(executed[0], "02")

    def test_unchanged_contract_resume_accepted(self):
        directory = self.started_run()
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(set(run["contract_components"]),
                         {"model", "cases", "system_instruction", "tool_declarations", "finalization_instruction"})
        self.assertEqual(run["metadata"]["model"], loop.MODEL)
        self.assertEqual(run["contract_fingerprint"], storage.contract_fingerprint())
        executed = []
        self.assertEqual(self.main("--resume", str(directory),
                                   execute=lambda case, attempt: executed.append(case.id) or canned_record()), 0)
        self.assertEqual(executed, list(CASE_IDS[1:]))

    def test_changed_model_rejected(self):
        self.assert_contract_change_rejected(mock.patch.object(loop, "MODEL", "gemini-3.8-flash"), "model")

    def test_run_from_previous_model_and_version_rejected(self):
        directory = self.started_run()
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        run.update(evaluation_version="live-eval-v3", metadata={**run["metadata"], "model": "gemini-3.8-flash"})
        atomic_write_json(directory / "run.json", run)
        self.assert_rejected(directory)

    def test_changed_system_instruction_rejected(self):
        self.assert_contract_change_rejected(
            mock.patch.object(loop, "SYSTEM_INSTRUCTION", loop.SYSTEM_INSTRUCTION + "\n- One more rule."),
            "system_instruction")

    def test_changed_tool_description_rejected(self):
        # the module's CURRENT registry (other tests may rebuild it with importlib.reload)
        real = tools_module.TOOLS["analyze_airport"]
        changed = RegisteredTool(function=real.function, declaration=real.declaration.model_copy(
            update={"description": real.declaration.description + " Changed."}))
        self.assert_contract_change_rejected(mock.patch.dict(tools_module.TOOLS, {"analyze_airport": changed}),
                                             "tool_declarations")

    def test_changed_tool_schema_rejected(self):
        real = tools_module.TOOLS["compare_airports"]
        schema = json.loads(json.dumps(real.declaration.parameters_json_schema))
        schema["properties"]["airport_codes"]["description"] += " (changed)"
        changed = RegisteredTool(function=real.function,
                                 declaration=real.declaration.model_copy(update={"parameters_json_schema": schema}))
        self.assert_contract_change_rejected(mock.patch.dict(tools_module.TOOLS, {"compare_airports": changed}),
                                             "tool_declarations")

    def test_changed_memory_declaration_rejected(self):
        changed = loop.MEMORY_DECLARATION.model_copy(update={"description": "Changed."})
        self.assert_contract_change_rejected(mock.patch.object(loop, "MEMORY_DECLARATION", changed),
                                             "tool_declarations")

    def test_changed_finalization_instruction_rejected(self):
        self.assert_contract_change_rejected(
            mock.patch.object(loop, "FINALIZATION_INSTRUCTION", loop.FINALIZATION_INSTRUCTION + " Changed."),
            "finalization_instruction")

    def test_changed_case_definitions_rejected(self):
        cases = list(storage.CASES)
        cases[3] = dataclasses.replace(cases[3], prompt=cases[3].prompt + " Please.")
        self.assert_contract_change_rejected(mock.patch.object(storage, "CASES", tuple(cases)), "cases")

    def test_run_without_contract_fingerprint_rejected(self):
        directory = self.started_run()
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        del run["contract_components"]
        atomic_write_json(directory / "run.json", run)
        self.assert_rejected(directory)

    def test_corrupted_metadata_rejected(self):
        directory = self.started_run()
        (directory / "run.json").write_text("{not json", encoding="utf-8")
        self.assert_rejected(directory)

    def test_corrupted_or_foreign_case_file_rejected(self):
        directory = self.started_run()
        (directory / "case_01.json").write_text('{"case_id": "05"}', encoding="utf-8")
        self.assert_rejected(directory)

    def test_missing_directory_rejected(self):
        self.assertEqual(self.main("--resume", str(self.results / "nope"), execute=mock.Mock()), 2)


class StorageTests(RunnerTestCase):
    def store(self):
        return RunStore.create(self.results, {}, now=self.clock)

    def test_completed_case_is_never_overwritten(self):
        store = self.store()
        result = {"case_id": "01", "evaluation_version": EVALUATION_VERSION, "run_id": store.run["run_id"],
                  "case_name": "x", "outcome": "completed", "dimensions": {"grounding": "NEEDS_REVIEW"},
                  "gemini_calls": {"total": 1, "failed_attempts": 0}, "tool_rounds": 0, "retry_count": 0,
                  "review_flags": [], "completed_at": "now"}
        store.save_case("01", result)
        with self.assertRaises(CaseAlreadyCompleted):
            store.save_case("01", {**result, "final_answer": "different"})
        self.assertNotIn("final_answer", json.loads(store.case_path("01").read_text(encoding="utf-8")))

    def test_atomic_write_keeps_previous_file_on_failure(self):
        path = self.results / "file.json"
        atomic_write_json(path, {"version": 1})
        with mock.patch("os.replace", side_effect=OSError("disk died")), self.assertRaises(OSError):
            atomic_write_json(path, {"version": 2})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 1})

    def test_atomic_write_retries_transient_windows_lock(self):
        path, real_replace, calls = self.results / "file.json", os.replace, []

        def flaky(src, dst):
            calls.append(src)
            if len(calls) < 3:
                raise PermissionError("file in use")
            real_replace(src, dst)

        with mock.patch("os.replace", side_effect=flaky):
            atomic_write_json(path, {"ok": True}, sleep=lambda s: None)
        self.assertEqual((len(calls), json.loads(path.read_text(encoding="utf-8"))), (3, {"ok": True}))
        self.assertEqual([p.name for p in self.results.iterdir()], ["file.json"])     # no temp file left


class RealSessionRunnerTests(RunnerTestCase):
    """Runner + real execute_case + real AgentSession on fake Gemini."""

    def test_gemini_calls_of_failed_attempt_and_setup_rerun_after_interruption(self):
        # attempt 1: 5 setup requests + the memory call, then 503 on request 7 (evaluated turn);
        # attempt 2 restarts the case from its first setup turn with a fresh session.
        attempt_2 = (*MEMORY_SETUP_SCRIPT, memory("Anchorage Fairbanks Juneau growth", call_id="m2"),
                     text_response("Fairbanks Intl."))
        client = FailingAtRequest(7, api_error(503, "UNAVAILABLE"), *MEMORY_SETUP_SCRIPT,
                                  memory("Anchorage Fairbanks Juneau growth"), *attempt_2)
        self.client_factory.return_value = client
        with mock.patch.object(runner, "CASES", [CASE["09"]]):
            self.assertEqual(self.main(), 0)
        self.assertEqual(len(client.requests), 14)
        case = self.read("case_09.json")
        self.assertEqual(case["retry_count"], 1)
        self.assertEqual((case["failed_attempts"][0]["gemini_calls"], case["failed_attempts"][0]["phase"]),
                         (7, "evaluated"))                                  # 5 setup + 2 evaluated (one failed)
        self.assertEqual(len(case["setup"]), 3)                              # setup rerun once, not accumulated
        self.assertEqual(case["gemini_calls"], {"setup": 5, "evaluated_turn": 2, "total": 7, "failed_attempts": 7,
                                                "observed_requests": 7})
        self.assertEqual(case["dimensions"]["memory_behavior"], PASS)
        self.assertEqual(sum(self.sleeps), DEFAULT_RETRY_WAIT_S)
        report = self.read("report.json")
        self.assertEqual((report["total_gemini_calls_completed_cases"],
                          report["total_gemini_calls_including_failed_attempts"]), (7, 14))
        dumped = json.dumps(case)
        self.assertNotIn("thought_signature", dumped)
        self.assertNotIn("signature-1", dumped)


class NoJudgeTests(LiveEvalTestCase):
    def test_no_model_calls_outside_the_agent(self):
        root = Path(runner.__file__).parent
        for name in ("cases.py", "evaluator.py", "storage.py"):
            source = (root / name).read_text(encoding="utf-8")
            for forbidden in ("generate_content", "genai", "create_gemini_client", "Client("):
                self.assertNotIn(forbidden, source, f"{name}: {forbidden}")
        source = (root / "run_live_evaluation.py").read_text(encoding="utf-8")
        self.assertEqual(source.count(".generate_content("), 1)              # the observing pass-through only

    def test_exactly_ten_cases_with_review_focus(self):
        self.assertEqual(CASE_IDS, tuple(f"{i:02d}" for i in range(1, 11)))
        self.assertTrue(all(case.expectations and case.review_focus for case in CASES))
        self.assertEqual(len(CASE["09"].setup), 3)
        self.assertEqual([case.id for case in CASES if case.setup], ["09"])          # the only multi-turn case
        self.assertEqual([case.id for case in CASES if case.kind == "memory_recall"], ["09"])
        self.assertEqual([case.id for case in CASES if case.kind == "geographic"], ["07"])
        self.assertEqual(EVALUATION_VERSION, "live-eval-v4")
        self.assertEqual(CASE["02"].prompt, "Compare congestion at Los Angeles International Airport and John Wayne "
                                            "Airport in Santa Ana.")
