"""Offline tests for the execution trace (agent/trace.py, AgentSession.last_trace).

Scripted fake Gemini, real dispatcher and business tools on synthetic fixtures. No network.
Timing values are only checked to exist and be non-negative.
"""

import dataclasses
import json

from google.genai import types

from agent import MEMORY_TOOL_NAME, AgentSession, EmptyModelResponse, FinalizationFailed, TurnTrace, run_agent_turn
from agent.trace import MAX_TRACE_STRING_CHARS
from tests.test_agent_loop import AgentLoopTestCase, FakeGemini, analyze, call_response, text_response
from tests.test_agent_session import FailingAt


def memory_call(query, call_id="m1"):
    return call_response((MEMORY_TOOL_NAME, {"query": query}, call_id))


def thinking_text_response(thought, text):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        role="model", parts=[types.Part(text=thought, thought=True), types.Part(text=text)]), finish_reason="STOP")])


def thinking_call_response(thought, name, args, call_id):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[
        types.Part(text=thought, thought=True),
        types.Part(function_call=types.FunctionCall(name=name, args=args, id=call_id))]), finish_reason="STOP")])


class FailOnFinalization(FakeGemini):
    def generate_content(self, *, model, contents, config):
        if config.tool_config.function_calling_config.mode == "NONE":
            self.requests.append({"model": model, "contents": list(contents), "config": config})
            raise ConnectionError("finalization unreachable")
        return super().generate_content(model=model, contents=contents, config=config)


def outcomes(trace):
    return [(c.phase, c.outcome, c.requested_functions) for c in trace.llm_calls]


def tools(trace):
    return [(t.round, t.name, t.category, t.status) for t in trace.tool_calls]


class TraceTestCase(AgentLoopTestCase):
    def session(self, gemini, **kw):
        return AgentSession(client=gemini, context=self.ctx, **kw)

    def seeded(self, *responses, **kw):
        """Session with 3 committed turns (turn 1 = FAI delays), then the scripted responses.
        With the default 2 recent turns, only turn 1 is searchable by memory."""
        gemini = FakeGemini(text_response("FAI shows elevated departure delays."), text_response("CAGR explained."),
                            text_response("Runways explained."), *responses)
        session = self.session(gemini, **kw)
        for question in ("Why are FAI delays high?", "What is CAGR?", "Runways?"):
            session.send(question)
        return session, gemini

    def assert_timings(self, trace):
        for value in [trace.latency_s, *(c.latency_s for c in trace.llm_calls),
                      *(t.latency_s for t in trace.tool_calls)]:
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)

    def assert_success(self, trace, llm_calls, tool_rounds, finalization=False):
        self.assertIsInstance(trace, TurnTrace)
        self.assertEqual((trace.status, trace.error_type, trace.error_message), ("success", None, None))
        self.assertEqual((trace.total_llm_calls, trace.tool_rounds, trace.finalization_used),
                         (llm_calls, tool_rounds, finalization))
        self.assert_timings(trace)
        json.dumps(trace.to_dict())                                   # plain, serialisable data


class SuccessfulTurnTraceTests(TraceTestCase):
    def test_1_direct_answer(self):
        session = self.session(FakeGemini(text_response("Hello.")))
        self.assertIsNone(session.last_trace)
        session.send("Hi")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=1, tool_rounds=0)
        self.assertEqual(outcomes(trace), [("normal", "final_text", ())])
        self.assertEqual((trace.tool_calls, trace.unexecuted_calls), ((), ()))

    def test_2_business_tool_then_answer(self):
        session = self.session(FakeGemini(analyze("ANC"), text_response("ANC is strong.")))
        session.send("Analyze ANC")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=2, tool_rounds=1)
        self.assertEqual(outcomes(trace), [("normal", "function_calls", ("analyze_airport",)),
                                           ("normal", "final_text", ())])
        (call,) = trace.tool_calls
        self.assertEqual((call.round, call.call_id, call.name, call.category, call.arguments, call.status),
                         (1, "a1", "analyze_airport", "business", {"airport_code": "ANC"}, "ok"))
        self.assertEqual((call.error_codes, call.memory_match_count, call.memory_matched_turns), ((), None, None))

    def test_3_memory_then_answer(self):
        session, _ = self.seeded(memory_call("FAI delays"), text_response("We said FAI delays are elevated."))
        session.send("What did we say about FAI delays?")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=2, tool_rounds=1)
        (call,) = trace.tool_calls
        self.assertEqual((call.name, call.category, call.arguments, call.status),
                         (MEMORY_TOOL_NAME, "memory", {"query": "FAI delays"}, "ok"))
        self.assertEqual((call.memory_match_count, call.memory_matched_turns), (1, (1,)))

    def test_memory_without_match_records_zero_matches(self):
        session, _ = self.seeded(memory_call("Boston"), text_response("Nothing found."))
        session.send("What about Boston?")
        (call,) = session.last_trace.tool_calls
        self.assertEqual((call.memory_match_count, call.memory_matched_turns), (0, ()))

    def test_4_memory_then_business_tool_then_answer(self):
        session, _ = self.seeded(memory_call("FAI delays"), analyze("FAI"), text_response("FAI still delayed."))
        session.send("Re-check the FAI delays we discussed.")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=3, tool_rounds=2)
        self.assertEqual(tools(trace), [(1, MEMORY_TOOL_NAME, "memory", "ok"), (2, "analyze_airport", "business", "ok")])

    def test_5_memory_and_business_in_one_response(self):
        session, _ = self.seeded(call_response((MEMORY_TOOL_NAME, {"query": "FAI delays"}, "m1"),
                                               ("analyze_airport", {"airport_code": "FAI"}, "a1")),
                                 text_response("Both done."))
        session.send("Recall FAI delays and analyze FAI.")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=2, tool_rounds=1)
        self.assertEqual(outcomes(trace)[0], ("normal", "function_calls", (MEMORY_TOOL_NAME, "analyze_airport")))
        self.assertEqual(tools(trace), [(1, MEMORY_TOOL_NAME, "memory", "ok"), (1, "analyze_airport", "business", "ok")])
        self.assertEqual([t.call_id for t in trace.tool_calls], ["m1", "a1"])

    def test_6_multiple_business_calls_in_one_response(self):
        session = self.session(FakeGemini(call_response(("analyze_airport", {"airport_code": "ANC"}, "a1"),
                                                        ("analyze_airport", {"airport_code": "FAI"}, "a2"),
                                                        ("analyze_airport", {"airport_code": "ZZZZ"}, "a3")),
                                          text_response("Done.")))
        session.send("Analyze ANC, FAI and ZZZZ.")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=2, tool_rounds=1)
        self.assertEqual([(t.round, t.call_id, t.arguments["airport_code"]) for t in trace.tool_calls],
                         [(1, "a1", "ANC"), (1, "a2", "FAI"), (1, "a3", "ZZZZ")])
        self.assertEqual(trace.tool_calls[2].status, "error")          # structured tool error is traced
        self.assertTrue(trace.tool_calls[2].error_codes)

    def test_7_multiple_tool_rounds(self):
        session = self.session(FakeGemini(analyze("ANC", "a1"), analyze("FAI", "a2"), analyze("JNU", "a3"),
                                          text_response("Done.")))
        session.send("Analyze three airports one by one.")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=4, tool_rounds=3)
        self.assertEqual([(t.round, t.call_id) for t in trace.tool_calls], [(1, "a1"), (2, "a2"), (3, "a3")])
        self.assertEqual(trace.normal_llm_calls, 4)

    def test_8_budget_exhaustion_forces_finalization(self):
        session = self.session(FakeGemini(analyze("ANC", "a1"), analyze("FAI", "a2"),
                                          call_response(("analyze_airport", {"airport_code": "JNU"}, "a3"),
                                                        (MEMORY_TOOL_NAME, {"query": "x"}, "m1")),
                                          text_response("Best effort.")), max_tool_rounds=2)
        self.assertEqual(session.send("Analyze everything."), "Best effort.")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=4, tool_rounds=2, finalization=True)
        self.assertEqual(trace.normal_llm_calls, 3)
        self.assertEqual(outcomes(trace), [("normal", "function_calls", ("analyze_airport",)),
                                           ("normal", "function_calls", ("analyze_airport",)),
                                           ("normal", "function_calls", ("analyze_airport", MEMORY_TOOL_NAME)),
                                           ("finalization", "finalization", ())])
        self.assertEqual([t.call_id for t in trace.tool_calls], ["a1", "a2"])
        self.assertEqual([(c.name, c.arguments) for c in trace.unexecuted_calls],
                         [("analyze_airport", {"airport_code": "JNU"}), (MEMORY_TOOL_NAME, {"query": "x"})])

    def test_llm_call_budget_finalization(self):
        session = self.session(FakeGemini(analyze("ANC", "a1"), analyze("FAI", "a2"), text_response("Final.")),
                               max_llm_calls=2)
        session.send("Analyze.")
        trace = session.last_trace
        self.assert_success(trace, llm_calls=3, tool_rounds=1, finalization=True)
        self.assertEqual(trace.normal_llm_calls, 2)
        self.assertEqual([c.name for c in trace.unexecuted_calls], ["analyze_airport"])


class FailedTurnTraceTests(TraceTestCase):
    def assert_failed(self, session, committed, error_type, llm_outcomes, tool_names, tool_rounds):
        trace = session.last_trace
        self.assertEqual(session.history, committed)                  # 13. history stays atomic
        self.assertEqual((trace.status, trace.error_type), ("error", error_type))
        self.assertTrue(trace.error_message)
        self.assertEqual(outcomes(trace), llm_outcomes)
        self.assertEqual([t.name for t in trace.tool_calls], tool_names)
        self.assertEqual(trace.tool_rounds, tool_rounds)
        self.assert_timings(trace)
        return trace

    def test_9_failure_on_first_gemini_call(self):
        session, _ = self.seeded()
        committed = session.history
        session._client = FailingAt(1)
        with self.assertRaises(ConnectionError):
            session.send("Analyze ANC.")
        self.assert_failed(session, committed, "ConnectionError", [("normal", "error", ())], [], 0)

    def test_10_failure_after_memory(self):
        session, _ = self.seeded()
        committed = session.history
        session._client = FailingAt(2, memory_call("FAI delays"))
        with self.assertRaises(ConnectionError):
            session.send("What did we say about FAI delays?")
        trace = self.assert_failed(session, committed, "ConnectionError",
                                   [("normal", "function_calls", (MEMORY_TOOL_NAME,)), ("normal", "error", ())],
                                   [MEMORY_TOOL_NAME], 1)
        self.assertEqual(trace.tool_calls[0].memory_matched_turns, (1,))

    def test_11_failure_after_business_tool(self):
        session, _ = self.seeded()
        committed = session.history
        session._client = FailingAt(2, analyze("FAI"))
        with self.assertRaises(ConnectionError):
            session.send("Analyze FAI.")
        trace = self.assert_failed(session, committed, "ConnectionError",
                                   [("normal", "function_calls", ("analyze_airport",)), ("normal", "error", ())],
                                   ["analyze_airport"], 1)
        self.assertEqual(trace.tool_calls[0].status, "ok")

    def test_12_failure_during_finalization(self):
        session, _ = self.seeded(max_tool_rounds=1)
        committed = session.history
        session._client = FailOnFinalization(analyze("ANC", "a1"), analyze("FAI", "a2"))
        with self.assertRaises(ConnectionError):
            session.send("Analyze both.")
        trace = self.assert_failed(session, committed, "ConnectionError",
                                   [("normal", "function_calls", ("analyze_airport",)),
                                    ("normal", "function_calls", ("analyze_airport",)),
                                    ("finalization", "error", ())], ["analyze_airport"], 1)
        self.assertTrue(trace.finalization_used)
        self.assertEqual([c.arguments for c in trace.unexecuted_calls], [{"airport_code": "FAI"}])

    def test_unusable_responses_are_error_calls(self):
        empty = types.GenerateContentResponse(candidates=[])
        session = self.session(FakeGemini(empty))
        with self.assertRaises(EmptyModelResponse):
            session.send("Hi")
        self.assert_failed(session, (), "EmptyModelResponse", [("normal", "error", ())], [], 0)

        session = self.session(FakeGemini(analyze("ANC"), analyze("FAI", "a2")), max_tool_rounds=0)
        with self.assertRaises(FinalizationFailed):
            session.send("Analyze.")
        self.assert_failed(session, (), "FinalizationFailed",
                           [("normal", "function_calls", ("analyze_airport",)),
                            ("finalization", "error", ("analyze_airport",))], [], 0)

    def test_tool_exception_is_traced_and_propagates(self):
        session = self.session(FakeGemini(analyze("ANC")))
        self.dispatch.side_effect = RuntimeError("tool crashed")
        with self.assertRaises(RuntimeError):
            session.send("Analyze ANC.")
        trace = self.assert_failed(session, (), "RuntimeError",
                                   [("normal", "function_calls", ("analyze_airport",))], ["analyze_airport"], 0)
        self.assertEqual(trace.tool_calls[0].status, "exception")

    def test_client_creation_failure_is_traced(self):
        session = AgentSession(context=self.ctx)                      # create_gemini_client is patched to fail
        with self.assertRaises(AssertionError):
            session.send("Hi")
        self.assertEqual((session.last_trace.status, session.last_trace.llm_calls), ("error", ()))

    def test_13_success_after_failure_commits_only_the_successful_turn(self):
        session, _ = self.seeded()
        committed = session.history
        session._client = FailingAt(2, analyze("FAI"))
        with self.assertRaises(ConnectionError):
            session.send("FAILED ATTEMPT")
        session._client = FakeGemini(text_response("Recovered."))
        session.send("Try again")
        self.assertEqual(session.last_trace.status, "success")
        self.assertEqual(len(session.history), len(committed) + 2)
        self.assertNotIn("FAILED ATTEMPT", [p.text for c in session.history for p in c.parts])


class TraceContractTests(TraceTestCase):
    def test_14_each_send_replaces_last_trace(self):
        session = self.session(FakeGemini(analyze("ANC"), text_response("one"), text_response("two")))
        session.send("first")
        first = session.last_trace
        session.send("second")
        second = session.last_trace
        self.assertIsNot(first, second)
        self.assertEqual((first.total_llm_calls, second.total_llm_calls), (2, 1))
        self.assertEqual((len(first.tool_calls), len(second.tool_calls)), (1, 0))
        self.assertFalse(hasattr(session, "traces"))

    def test_last_trace_is_read_only(self):
        session = self.session(FakeGemini(analyze("ANC"), text_response("ok")))
        session.send("Analyze ANC")
        trace = session.last_trace
        with self.assertRaises(AttributeError):
            session.last_trace = None
        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.status = "error"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.tool_calls[0].name = "other"
        self.assertIsInstance(trace.llm_calls, tuple)
        self.assertIsInstance(trace.tool_calls, tuple)

    def test_recorded_arguments_are_copies(self):
        args = {"airport_codes": ["ANC", "FAI"]}
        gemini = FakeGemini(call_response(("compare_airports", args, "c1")), text_response("ok"))
        session = self.session(gemini)
        session.send("Compare")
        recorded = session.last_trace.tool_calls[0].arguments
        recorded["airport_codes"].append("JNU")
        sent = gemini.requests[1]["contents"][1].parts[0].function_call.args
        self.assertEqual(sent, {"airport_codes": ["ANC", "FAI"]})

    def test_15_no_thought_text_in_trace(self):
        session = self.session(FakeGemini(thinking_call_response("SECRET-THOUGHT-1", "analyze_airport",
                                                                  {"airport_code": "ANC"}, "a1"),
                                          thinking_text_response("SECRET-THOUGHT-2", "VISIBLE-ANSWER")))
        self.assertEqual(session.send("Analyze ANC"), "VISIBLE-ANSWER")
        dumped = json.dumps(session.last_trace.to_dict()) + repr(session.last_trace)
        for secret in ("SECRET-THOUGHT", "VISIBLE-ANSWER", "Analyze ANC"):  # no thoughts, no model or user text
            self.assertNotIn(secret, dumped)

    def test_16_full_results_are_not_copied(self):
        session, gemini = self.seeded(call_response((MEMORY_TOOL_NAME, {"query": "FAI delays"}, "m1"),
                                                    ("analyze_airport", {"airport_code": "ANC"}, "a1")),
                                      text_response("ok"))
        session.send("Recall and analyze.")
        memory_result, business_result = [p.function_response.response for p in gemini.requests[-1]["contents"][-1].parts]
        dumped = json.dumps(session.last_trace.to_dict())
        self.assertNotIn("FAI shows elevated departure delays", dumped)       # memory answer text
        self.assertIn("FAI shows elevated departure delays", json.dumps(memory_result))
        for key in set(business_result) - {"tool", "status", "errors", "warnings"}:
            self.assertNotIn(f'"{key}"', dumped, key)                          # no business payload sections
        self.assertLess(len(dumped), len(json.dumps(business_result, default=str)))

    def test_long_arguments_are_shortened(self):
        session = self.session(FakeGemini(memory_call("FAI " * 10_000), text_response("ok")))
        session.send("x")
        query = session.last_trace.tool_calls[0].arguments["query"]
        self.assertLessEqual(len(query), MAX_TRACE_STRING_CHARS + 20)

    def test_17_timing_fields_exist_and_are_non_negative(self):
        session, _ = self.seeded(memory_call("FAI"), analyze("FAI"), text_response("ok"))
        session.send("x")
        trace = session.last_trace
        self.assertEqual(len(trace.llm_calls), 3)
        self.assertEqual(len(trace.tool_calls), 2)
        self.assert_timings(trace)

    def test_18_tracing_does_not_change_agent_behavior(self):
        script = lambda: [call_response((MEMORY_TOOL_NAME, {"query": "x"}, "m1"),
                                        ("analyze_airport", {"airport_code": "ANC"}, "a1")),
                          analyze("FAI", "a2"), analyze("JNU", "a3"), text_response("Final.")]
        untraced, traced = FakeGemini(*script()), FakeGemini(*script())
        untraced_answer = run_agent_turn("Q", self.ctx, client=untraced, max_tool_rounds=2)
        untraced_dispatch = self.dispatch.call_args_list[:]
        self.dispatch.reset_mock()
        traced_answer = self.session(traced, max_tool_rounds=2).send("Q")
        self.assertEqual(traced_answer, untraced_answer)
        self.assertEqual(self.dispatch.call_args_list, untraced_dispatch)
        self.assertEqual(traced.modes(), untraced.modes())
        dump = lambda gemini: [[c.model_dump(mode="json") for c in r["contents"]] for r in gemini.requests]
        self.assertEqual(dump(traced), dump(untraced))
