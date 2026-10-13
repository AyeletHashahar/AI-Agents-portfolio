"""Offline tests for the single-turn agent loop (agent/loop.py).

Gemini is replaced by a scripted fake returning real google-genai response objects. The
real dispatcher and real business tools run on the synthetic offline fixtures. No network.
"""

import unittest
from unittest import mock

from google.genai import types

import agent.loop as loop
from agent import (MAX_LLM_CALLS, MAX_TOOL_ROUNDS, SYSTEM_INSTRUCTION, EmptyModelResponse, FinalizationFailed,
                   execute_tool, run_agent_turn)
from tests.test_agent_tools import ToolTestCase


def text_response(text):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason="STOP")])


def call_response(*calls):
    """Model response requesting tools; calls are (name, args, id). The first part carries a
    thought signature, as Gemini attaches to function calls, to check it is sent back."""
    parts = [types.Part(function_call=types.FunctionCall(name=name, args=args, id=call_id))
             for name, args, call_id in calls]
    parts[0].thought_signature = b"signature-1"
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=parts), finish_reason="STOP")])


def analyze(code="ANC", call_id="a1"):
    return call_response(("analyze_airport", {"airport_code": code}, call_id))


class FakeGemini:
    """Stands in for google-genai's Client: returns scripted responses, records requests."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        return self.responses.pop(0)

    def modes(self):
        """Function-calling mode of every request: AUTO = normal call, NONE = finalization."""
        return [r["config"].tool_config.function_calling_config.mode for r in self.requests]


def function_responses(content):
    return [part.function_response for part in content.parts]


class AgentLoopTestCase(ToolTestCase):
    def setUp(self):
        super().setUp()
        # Every tool execution goes through the real dispatcher; the spy records the calls.
        spy = mock.patch.object(loop, "execute_tool", wraps=execute_tool)
        self.dispatch = spy.start()
        self.addCleanup(spy.stop)
        never = mock.patch.object(loop, "create_gemini_client",
                                  side_effect=AssertionError("real Gemini client created"))
        never.start()
        self.addCleanup(never.stop)

    def run_turn(self, gemini, message="Compare ANC, FAI and JNU as airport investment opportunities.", **kw):
        return run_agent_turn(message, self.ctx, client=gemini, **kw)


class AgentLoopTests(AgentLoopTestCase):
    def test_direct_answer_without_tools(self):
        gemini = FakeGemini(text_response("Hello! I analyse U.S. airports."))
        self.assertEqual(self.run_turn(gemini, "Hi"), "Hello! I analyse U.S. airports.")
        self.assertEqual(len(gemini.requests), 1)
        self.dispatch.assert_not_called()

    def test_one_tool_round(self):
        tool_request = call_response(("compare_airports", {"airport_codes": ["ANC", "FAI", "JNU"]}, "c1"))
        gemini = FakeGemini(tool_request, text_response("ANC ranks first."))
        self.assertEqual(self.run_turn(gemini), "ANC ranks first.")
        self.assertIs(gemini.requests[1]["contents"][1], tool_request.candidates[0].content)  # model turn kept as is
        self.dispatch.assert_called_once_with("compare_airports", {"airport_codes": ["ANC", "FAI", "JNU"]}, self.ctx)

        user, model_turn, results = gemini.requests[1]["contents"]
        self.assertEqual((user.role, user.parts[0].text),
                         ("user", "Compare ANC, FAI and JNU as airport investment opportunities."))
        self.assertEqual(model_turn.role, "model")
        self.assertEqual(model_turn.parts[0].function_call.name, "compare_airports")
        self.assertEqual(model_turn.parts[0].thought_signature, b"signature-1")
        self.assertEqual(results.role, "user")         # Gemini-native: function responses in a user turn
        (response,) = function_responses(results)
        self.assertEqual((response.name, response.id), ("compare_airports", "c1"))
        self.assertEqual(response.response["status"], "ok")              # structured, not prose
        self.assertEqual(response.response["metadata"]["analyzed_count"], 3)
        self.assertEqual(gemini.modes(), ["AUTO", "AUTO"])               # no finalization needed

    def test_two_sequential_tool_rounds(self):
        gemini = FakeGemini(call_response(("discover_airports", {"country": "US", "states": ["AK"]}, "d1")),
                            call_response(("compare_airports", {"airport_codes": ["ANC", "FAI", "JNU"]}, "c1")),
                            text_response("Final ranking: ANC, FAI, JNU."))
        self.assertEqual(self.run_turn(gemini, "Which Alaska airports are the best candidates?"),
                         "Final ranking: ANC, FAI, JNU.")
        self.assertEqual([c.args[0] for c in self.dispatch.call_args_list], ["discover_airports", "compare_airports"])
        self.assertEqual([len(r["contents"]) for r in gemini.requests], [1, 3, 5])
        final_contents = gemini.requests[2]["contents"]
        self.assertEqual(function_responses(final_contents[2])[0].response["candidate_count"], 6)
        self.assertEqual(function_responses(final_contents[4])[0].name, "compare_airports")

    def test_multiple_tool_calls_in_one_response(self):
        gemini = FakeGemini(call_response(("analyze_airport", {"airport_code": "FAI"}, "a1"),
                                          ("analyze_airport", {"airport_code": "JNU"}, "a2")),
                            text_response("FAI vs JNU explained."))
        self.assertEqual(self.run_turn(gemini, "Why did FAI rank above JNU?"), "FAI vs JNU explained.")
        self.assertEqual(self.dispatch.call_count, 2)
        model_turn, results = gemini.requests[1]["contents"][1:]
        self.assertEqual(len(model_turn.parts), 2)
        self.assertEqual([(r.name, r.id, r.response["airport"]["code"]) for r in function_responses(results)],
                         [("analyze_airport", "a1", "FAI"), ("analyze_airport", "a2", "JNU")])

    def test_same_context_reaches_every_tool_execution(self):
        gemini = FakeGemini(call_response(("discover_airports", {"country": "US", "states": ["AK"]}, "d1"),
                                          ("analyze_airport", {"airport_code": "ANC"}, "a1")),
                            call_response(("compare_airports", {"airport_codes": ["ANC", "FAI"]}, "c1")),
                            text_response("done"))
        self.run_turn(gemini)
        self.assertEqual(self.dispatch.call_count, 3)
        for call in self.dispatch.call_args_list:
            self.assertIs(call.args[2], self.ctx)

    def test_structured_tool_error_is_returned_to_the_model(self):
        gemini = FakeGemini(analyze("ZZZ"), text_response("I could not find airport ZZZ."))
        self.assertEqual(self.run_turn(gemini, "Analyze ZZZ"), "I could not find airport ZZZ.")
        (response,) = function_responses(gemini.requests[1]["contents"][2])
        self.assertEqual(response.response["status"], "error")
        self.assertEqual(response.response["errors"][0]["code"], "airport_not_found")

    def test_unknown_tool_goes_through_the_dispatcher(self):
        gemini = FakeGemini(call_response(("delete_everything", {"now": True}, "x1")),
                            text_response("That tool does not exist."))
        self.run_turn(gemini)
        self.dispatch.assert_called_once_with("delete_everything", {"now": True}, self.ctx)
        (response,) = function_responses(gemini.requests[1]["contents"][2])
        self.assertEqual(response.response["errors"][0]["code"], "unknown_tool")

    def test_empty_final_response_is_explicit(self):
        thought_only = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text="thinking...", thought=True)]))])
        cases = [text_response("   "), thought_only, types.GenerateContentResponse(candidates=[]),
                 types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="SAFETY")])]
        for response in cases:
            with self.assertRaises(EmptyModelResponse):
                self.run_turn(FakeGemini(response))

    def test_config_uses_registry_and_disables_automatic_calling(self):
        gemini = FakeGemini(text_response("hi"))
        self.run_turn(gemini, "hi")
        request = gemini.requests[0]
        self.assertEqual(request["model"], loop.MODEL)
        config = request["config"]
        self.assertTrue(config.automatic_function_calling.disable)
        self.assertEqual(config.tool_config.function_calling_config.mode, "AUTO")
        # the registered business tools plus the conversation-memory lookup, in one Gemini Tool
        self.assertEqual({d.name for d in config.tools[0].function_declarations},
                         {"resolve_airports", "discover_airports", "compare_airports", "analyze_airport",
                          "retrieve_conversation_context"})

    def test_infrastructure_and_programming_errors_surface(self):
        class FailingGemini(FakeGemini):
            def generate_content(self, **kwargs):
                raise ConnectionError("Gemini unreachable")

        with self.assertRaises(ConnectionError):
            self.run_turn(FailingGemini())
        self.dispatch.side_effect = ZeroDivisionError("bug inside a tool")
        with self.assertRaises(ZeroDivisionError):
            self.run_turn(FakeGemini(analyze()))

    def test_client_created_lazily_only_when_not_supplied(self):
        gemini = FakeGemini(text_response("hi"))
        with mock.patch.object(loop, "create_gemini_client", return_value=gemini) as create:
            self.assertEqual(run_agent_turn("hi", self.ctx), "hi")
        create.assert_called_once_with()

    def test_falsey_injected_client_is_not_replaced(self):
        class FalseyGemini(FakeGemini):
            def __bool__(self):
                return False

        gemini = FalseyGemini(text_response("still the injected client"))
        self.assertEqual(self.run_turn(gemini, "hi"), "still the injected client")   # no real client created
        self.assertEqual(len(gemini.requests), 1)

    def test_invalid_budgets_rejected(self):
        for kwargs in ({"max_llm_calls": 0}, {"max_tool_rounds": -1}):
            with self.assertRaises(ValueError):
                self.run_turn(FakeGemini(text_response("x")), **kwargs)


class BudgetTests(AgentLoopTestCase):
    def test_default_budget_constants(self):
        self.assertEqual(MAX_TOOL_ROUNDS, 6)
        self.assertEqual(MAX_LLM_CALLS, MAX_TOOL_ROUNDS + 1)

    def test_tool_round_budget_finalizes_without_executing_the_new_request(self):
        gemini = FakeGemini(analyze("ANC", "a1"), analyze("FAI", "a2"),
                            analyze("JNU", "a3"),                         # over budget: must NOT run
                            text_response("Best effort: ANC and FAI analysed; JNU not analysed."))
        answer = self.run_turn(gemini, max_tool_rounds=2)
        self.assertEqual(answer, "Best effort: ANC and FAI analysed; JNU not analysed.")
        self.assertEqual([c.args[1]["airport_code"] for c in self.dispatch.call_args_list], ["ANC", "FAI"])
        self.assertEqual(gemini.modes(), ["AUTO", "AUTO", "AUTO", "NONE"])  # 3 normal + 1 finalization

    def test_finalization_request_contents_and_config(self):
        gemini = FakeGemini(analyze("ANC", "a1"), analyze("JNU", "a2"), text_response("final"))
        self.run_turn(gemini, max_tool_rounds=1)
        final = gemini.requests[-1]
        config = final["config"]
        self.assertEqual(config.tool_config.function_calling_config.mode, "NONE")    # calling disabled
        self.assertTrue(config.automatic_function_calling.disable)
        user, model_turn, results, instruction = final["contents"]
        # evidence already collected is available: the executed call and its real result
        self.assertEqual(model_turn.parts[0].function_call.args, {"airport_code": "ANC"})
        self.assertEqual(function_responses(results)[0].response["airport"]["code"], "ANC")
        # the unexecuted request is only NAMED in the instruction; no call turn, no fake result
        self.assertEqual((instruction.role, len(instruction.parts)), ("user", 1))
        text = instruction.parts[0].text
        self.assertIn("EXECUTION BUDGET REACHED", text)
        self.assertIn('analyze_airport({"airport_code": "JNU"})', text)
        self.assertIn("NOT executed", text)
        every_part = [p for c in final["contents"] for p in c.parts]
        self.assertEqual([p.function_response.id for p in every_part if p.function_response], ["a1"])
        self.assertEqual([p.function_call.id for p in every_part if p.function_call], ["a1"])
        self.dispatch.assert_called_once()

    def test_llm_call_budget_limits_normal_calls_independently(self):
        # Tool-round budget is generous (6); the LLM-call budget (3) is the binding one.
        gemini = FakeGemini(analyze("ANC", "a1"), analyze("FAI", "a2"), analyze("JNU", "a3"),
                            text_response("best effort"))
        self.assertEqual(self.run_turn(gemini, max_llm_calls=3, max_tool_rounds=6), "best effort")
        self.assertEqual(gemini.modes(), ["AUTO", "AUTO", "AUTO", "NONE"])
        # the 3rd normal call's request is not executed: no normal call would be left to use it
        self.assertEqual(self.dispatch.call_count, 2)

    def test_exact_worst_case_with_default_budgets(self):
        endless = [analyze("ANC", f"a{i}") for i in range(MAX_LLM_CALLS)]
        gemini = FakeGemini(*endless, text_response("best effort after the budget"))
        self.assertEqual(self.run_turn(gemini), "best effort after the budget")
        self.assertEqual(len(gemini.requests), MAX_LLM_CALLS + 1)                    # 7 normal + 1 final
        self.assertEqual(gemini.modes(), ["AUTO"] * MAX_LLM_CALLS + ["NONE"])
        self.assertEqual(self.dispatch.call_count, MAX_TOOL_ROUNDS)                  # 6 executed rounds

    def test_answer_on_the_last_normal_call_needs_no_finalization(self):
        gemini = FakeGemini(analyze(), text_response("ok"))
        self.assertEqual(self.run_turn(gemini, max_tool_rounds=1, max_llm_calls=2), "ok")
        self.assertEqual(gemini.modes(), ["AUTO", "AUTO"])

    def test_zero_tool_rounds_finalizes_immediately(self):
        gemini = FakeGemini(analyze(), text_response("answered without tools"))
        self.assertEqual(self.run_turn(gemini, max_tool_rounds=0), "answered without tools")
        self.dispatch.assert_not_called()
        self.assertEqual(gemini.modes(), ["AUTO", "NONE"])

    def test_multiple_calls_in_an_over_budget_response_are_all_skipped(self):
        gemini = FakeGemini(call_response(("analyze_airport", {"airport_code": "FAI"}, "a1"),
                                          ("analyze_airport", {"airport_code": "JNU"}, "a2")),
                            text_response("best effort"))
        self.run_turn(gemini, max_tool_rounds=0)
        self.dispatch.assert_not_called()
        text = gemini.requests[-1]["contents"][-1].parts[0].text
        self.assertIn('{"airport_code": "FAI"}', text)
        self.assertIn('{"airport_code": "JNU"}', text)

    def test_finalization_is_one_terminal_call(self):
        # Even if the model asks for tools again during finalization, nothing loops or executes.
        gemini = FakeGemini(analyze("ANC", "a1"), analyze("JNU", "a2"),
                            analyze("SEA", "a3"))                     # reply to the finalization request
        with self.assertRaises(FinalizationFailed):
            self.run_turn(gemini, max_tool_rounds=1)
        self.assertEqual(gemini.modes(), ["AUTO", "AUTO", "NONE"])
        self.assertEqual(self.dispatch.call_count, 1)

    def test_empty_finalization_raises(self):
        for empty in (text_response("  "), types.GenerateContentResponse(candidates=[])):
            gemini = FakeGemini(analyze(), empty)
            with self.assertRaises(FinalizationFailed):
                self.run_turn(gemini, max_tool_rounds=0)

    def test_finalization_infrastructure_error_surfaces(self):
        class FailOnFinal(FakeGemini):
            def generate_content(self, *, model, contents, config):
                if config.tool_config.function_calling_config.mode == "NONE":
                    raise ConnectionError("Gemini unreachable")
                return super().generate_content(model=model, contents=contents, config=config)

        with self.assertRaises(ConnectionError):
            self.run_turn(FailOnFinal(analyze()), max_tool_rounds=0)

    def test_structured_errors_still_flow_within_budget(self):
        gemini = FakeGemini(analyze("ZZZ", "a1"), text_response("ZZZ not found"))
        self.assertEqual(self.run_turn(gemini, max_tool_rounds=1, max_llm_calls=2), "ZZZ not found")
        self.assertEqual(function_responses(gemini.requests[1]["contents"][2])[0].response["errors"][0]["code"],
                         "airport_not_found")


class FunctionResponseSerializationTests(AgentLoopTestCase):
    """Regression: the API rejects role "tool" (400 INVALID_ARGUMENT). Content.role must be
    "user" or "model"; function responses travel in a "user" turn (as the SDK's own automatic
    function calling does), right after the model turn that requested them."""

    def test_no_turn_uses_the_unsupported_tool_role(self):
        gemini = FakeGemini(call_response(("analyze_airport", {"airport_code": "FAI"}, "a1"),
                                          ("analyze_airport", {"airport_code": "JNU"}, "a2")),
                            call_response(("compare_airports", {"airport_codes": ["ANC", "FAI"]}, "c1")),
                            analyze("SEA", "a3"),                     # over budget -> finalization
                            text_response("done"))
        self.run_turn(gemini, max_tool_rounds=2)
        for request in gemini.requests:                              # normal requests + finalization
            for content in request["contents"]:
                self.assertIn(content.role, ("user", "model"))
                self.assertNotEqual(content.role, "tool")

    def test_function_responses_follow_their_call_turn_with_matching_ids(self):
        gemini = FakeGemini(call_response(("analyze_airport", {"airport_code": "FAI"}, "a1"),
                                          ("analyze_airport", {"airport_code": "JNU"}, "a2")),
                            text_response("done"))
        self.run_turn(gemini)
        contents = gemini.requests[1]["contents"]
        response_turns = [i for i, c in enumerate(contents) if any(p.function_response for p in c.parts)]
        self.assertEqual(response_turns, [2])
        call_turn, response_turn = contents[1], contents[2]
        self.assertEqual(call_turn.role, "model")
        self.assertEqual(response_turn.role, "user")
        self.assertTrue(all(p.function_response for p in response_turn.parts))   # only responses in that turn
        self.assertEqual([(p.function_call.id, p.function_call.name) for p in call_turn.parts],
                         [(p.function_response.id, p.function_response.name) for p in response_turn.parts])
        for part in response_turn.parts:
            self.assertIsInstance(part.function_response.response, dict)          # structured result

    def test_serialized_request_has_no_tool_role(self):
        gemini = FakeGemini(analyze("ANC", "a1"), text_response("done"))
        self.run_turn(gemini)
        payload = [c.model_dump(mode="json", exclude_none=True) for c in gemini.requests[1]["contents"]]
        self.assertEqual([c["role"] for c in payload], ["user", "model", "user"])
        self.assertIn("function_response", payload[2]["parts"][0])


class SystemInstructionTests(AgentLoopTestCase):
    """The grounding policy is sent with every request (not whether Gemini obeys it - that is
    checked separately with a real smoke test)."""

    def test_normal_requests_carry_the_system_instruction(self):
        gemini = FakeGemini(analyze(), text_response("done"))
        self.run_turn(gemini)
        for request in gemini.requests:
            config = request["config"]
            self.assertIs(config.system_instruction, SYSTEM_INSTRUCTION)
            self.assertEqual(config.tool_config.function_calling_config.mode, "AUTO")
            self.assertTrue(config.automatic_function_calling.disable)

    def test_instruction_is_native_not_a_fake_user_message(self):
        gemini = FakeGemini(text_response("hi"))
        self.run_turn(gemini, "hi")
        (user,) = gemini.requests[0]["contents"]
        self.assertEqual([p.text for p in user.parts], ["hi"])
        self.assertNotIn("EVIDENCE LEVELS", user.parts[0].text)

    def test_finalization_keeps_the_policy_and_disables_tools(self):
        gemini = FakeGemini(analyze(), text_response("best effort"))
        self.run_turn(gemini, max_tool_rounds=0)
        final = gemini.requests[-1]["config"]
        self.assertIs(final.system_instruction, SYSTEM_INSTRUCTION)
        self.assertEqual(final.tool_config.function_calling_config.mode, "NONE")
        self.assertTrue(final.automatic_function_calling.disable)
        self.dispatch.assert_not_called()

    def test_instruction_covers_the_critical_grounding_concepts(self):
        text = SYSTEM_INSTRUCTION
        required = {
            "tool-result grounding": "Base factual claims about the analysed airports on tool results",
            "no unobserved metrics": "Never state that a metric was observed or calculated unless a tool returned it",
            "evidence level": "1. EVIDENCE",
            "interpretation level": "2. INTERPRETATION",
            "hypothesis level": "3. HYPOTHESIS",
            "no silent upgrade": "never present level 2 or 3 as level 1",
            "capacity pressure proxy": "Capacity Pressure is a PROXY",
            "relative score": "Investment Opportunity Score is RELATIVE to the comparison group",
            "BTS coverage": "domestic scheduled passenger flights of reporting carriers",
            "no unsupported causes": "Do not attribute causes to observed outcomes unless the evidence establishes",
            "no plausible-sounding interventions": "Do not recommend a specific infrastructure intervention",
            "labelled background knowledge": "General background knowledge may be used only as clearly labelled context",
            "no invented research": "Do not invent research, sources or citations",
            "no fabricated tool results": "Never fabricate tool results",
        }
        for concept, phrase in required.items():
            self.assertIn(phrase, text, concept)
        for unsupported in ("terminal capacity", "unmet passenger demand", "investment returns",
                            "construction", "financing attractiveness"):
            self.assertIn(unsupported, text)

    def test_instruction_is_defined_once(self):
        self.assertIs(loop.build_config().system_instruction, loop.build_finalization_config().system_instruction)
        self.assertLess(len(SYSTEM_INSTRUCTION), 5000)          # concise: sent with every request


if __name__ == "__main__":
    unittest.main()
