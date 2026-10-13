"""Offline tests for bounded model context + on-demand conversation memory.

Gemini is the scripted FakeGemini; the real dispatcher and business tools run on the
synthetic offline fixtures. No network.
"""

import json
from unittest import mock

from google.genai import types

import agent.loop as loop
from agent import (MAX_MEMORY_RESULTS, MEMORY_TOOL_NAME, RECENT_TURNS, SYSTEM_INSTRUCTION, AgentSession,
                   ConversationMemory, run_agent_turn)
from agent.memory import MAX_ANSWER_CHARS, _terms, summarize_turn
from tests.test_agent_loop import AgentLoopTestCase, FakeGemini, analyze, call_response, text_response
from tests.test_agent_session import FailingAt, compare_call, roles, texts


def memory_call(query, call_id="m1"):
    return call_response((MEMORY_TOOL_NAME, {"query": query}, call_id))


def turn(user, answer):
    """A committed text-only turn (user message, final model answer)."""
    return (types.Content(role="user", parts=[types.Part(text=user)]),
            types.Content(role="model", parts=[types.Part(text=answer)]))


def memory_responses(request):
    return [p.function_response.response for c in request["contents"] for p in c.parts
            if p.function_response and p.function_response.name == MEMORY_TOOL_NAME]


EARLY = ("Compare ANC, FAI and JNU as airport investment opportunities.",
         "ANC ranks first. FAI shows elevated departure delays despite low runway pressure.")
FILLER = [("What is passenger CAGR?", "CAGR is the compound annual growth rate."),
          ("What does P95 mean?", "The 95th percentile of hourly movements."),
          ("Explain coverage.", "Coverage is the share of configured signal weight with data."),
          ("What is a percentile rank?", "A position within the comparison group.")]


class MemoryTestCase(AgentLoopTestCase):
    def setUp(self):
        super().setUp()
        spy = mock.patch.object(ConversationMemory, "retrieve", autospec=True,
                                side_effect=ConversationMemory.retrieve)
        self.retrieve = spy.start()
        self.addCleanup(spy.stop)

    def session_with(self, turns, *responses, **kw):
        """Session whose stored conversation already holds `turns` (built through real send())."""
        script = [text_response(answer) for _, answer in turns] + list(responses)
        gemini = FakeGemini(*script)
        session = AgentSession(client=gemini, context=self.ctx, **kw)
        for user, _ in turns:
            session.send(user)
        return session, gemini


class BoundedContextTests(MemoryTestCase):
    def test_long_history_unrelated_question_sends_only_recent_turns(self):
        old = [(f"Question {i}", f"Answer {i}") for i in range(50)]
        session, gemini = self.session_with(old, text_response("CAGR is compound annual growth."))
        self.assertEqual(session.send("What is passenger CAGR?"), "CAGR is compound annual growth.")
        sent = gemini.requests[-1]["contents"]
        self.assertEqual(len(sent), RECENT_TURNS * 2 + 1)                  # 2 recent turns + the question
        self.assertEqual(texts(sent), ["Question 48", "Answer 48", "Question 49", "Answer 49",
                                       "What is passenger CAGR?"])
        self.assertEqual(len(session.history), 51 * 2)                     # everything is still stored
        self.retrieve.assert_not_called()

    def test_context_stays_bounded_as_stored_history_grows(self):
        session, gemini = self.session_with([(f"Q{i}", f"A{i}") for i in range(20)])
        sent_sizes = [len(r["contents"]) for r in gemini.requests]
        stored_sizes = [2 * (i + 1) for i in range(20)]
        self.assertEqual(sent_sizes[:3], [1, 3, 5])
        self.assertEqual(max(sent_sizes), RECENT_TURNS * 2 + 1)            # never grows past 5
        self.assertEqual(len(session.history), stored_sizes[-1])           # stored grows to 40

    def test_recent_tool_turns_are_sent_whole(self):
        gemini = FakeGemini(compare_call(), text_response("ANC first."), text_response("FAI has more growth."))
        session = AgentSession(client=gemini, context=self.ctx)
        session.send(EARLY[0])
        session.send("Which of those has the highest passenger growth?")
        self.assertEqual(roles(gemini.requests[-1]["contents"]), ["user", "model", "user", "model", "user"])
        self.retrieve.assert_not_called()

    def test_recent_window_cuts_only_at_turn_boundaries(self):
        gemini = FakeGemini(compare_call("c1"), text_response("t1"), analyze("FAI", "a1"), text_response("t2"),
                            analyze("JNU", "a2"), text_response("t3"), text_response("t4"))
        session = AgentSession(client=gemini, context=self.ctx, recent_turns=2)
        for message in ("one", "two", "three", "four"):
            session.send(message)
        sent = gemini.requests[-1]["contents"]
        self.assertEqual(texts(sent)[0], "two")                            # starts at a user message
        calls = [p.function_call.id for c in sent for p in c.parts if p.function_call]
        responses = [p.function_response.id for c in sent for p in c.parts if p.function_response]
        self.assertEqual(calls, responses)                                 # no orphan call or response
        self.assertEqual(calls, ["a1", "a2"])                              # turn "one" (c1) not sent

    def test_zero_recent_turns_sends_only_the_question(self):
        session, gemini = self.session_with(FILLER[:2], text_response("ok"), recent_turns=0)
        session.send("Hi")
        self.assertEqual(texts(gemini.requests[-1]["contents"]), ["Hi"])


class RetrievalFlowTests(MemoryTestCase):
    def test_older_reference_is_retrieved_then_answered(self):
        session, gemini = self.session_with([EARLY] + FILLER, memory_call("the previous answer about FAI delays"),
                                            text_response("Earlier I said FAI had elevated delays."))
        answer = session.send("Summarize the answer you gave me earlier about FAI delays.")
        self.assertEqual(answer, "Earlier I said FAI had elevated delays.")
        (result,) = memory_responses(gemini.requests[-1])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["searchable_earlier_turns"], 3)             # 5 stored - 2 visible
        first = result["matches"][0]
        self.assertEqual((first["turn"], first["turns_ago"]), (1, 5))
        self.assertEqual(first["user_message"], EARLY[0])
        self.assertIn("elevated departure delays", first["assistant_answer"])
        self.assertIn("fai", first["match"]["matched_terms"])
        self.dispatch.assert_not_called()                                   # memory is not a business tool

    def test_memory_then_business_tool_in_one_turn(self):
        session, gemini = self.session_with(
            [EARLY] + FILLER,
            memory_call("the earlier Alaska comparison of ANC, FAI and JNU"),
            call_response(("compare_airports", {"airport_codes": ["FAI", "SEA"]}, "c9")),
            text_response("FAI vs SEA: SEA is larger."))
        self.assertEqual(session.send("Compare the second airport from that earlier Alaska comparison with SEA."),
                         "FAI vs SEA: SEA is larger.")
        self.dispatch.assert_called_once_with("compare_airports", {"airport_codes": ["FAI", "SEA"]}, self.ctx)
        final = gemini.requests[-1]["contents"]
        self.assertEqual(roles(final[-5:]), ["user", "model", "user", "model", "user"])
        self.assertEqual(final[-3].parts[0].function_response.name, MEMORY_TOOL_NAME)
        self.assertEqual(final[-1].parts[0].function_response.name, "compare_airports")
        # the committed turn keeps the whole valid sequence: question, memory call/result,
        # business call/result, final answer
        last_turn = session._turns[-1]
        self.assertEqual(roles(last_turn), ["user", "model", "user", "model", "user", "model"])
        self.assertEqual(last_turn[-1].parts[0].text, "FAI vs SEA: SEA is larger.")

    def test_business_tool_then_memory_in_one_turn(self):
        session, gemini = self.session_with([EARLY] + FILLER, analyze("SEA", "a1"),
                                            memory_call("earlier comparison ANC FAI JNU"),
                                            text_response("Compared with the earlier result."))
        session.send("Analyze SEA and relate it to what you said before about Alaska.")
        names = [p.function_response.name for c in gemini.requests[-1]["contents"] for p in c.parts
                 if p.function_response]
        self.assertEqual(names, ["analyze_airport", MEMORY_TOOL_NAME])

    def test_memory_and_business_call_in_the_same_response(self):
        session, gemini = self.session_with(
            [EARLY] + FILLER,
            call_response((MEMORY_TOOL_NAME, {"query": "FAI delays"}, "m1"),
                          ("analyze_airport", {"airport_code": "FAI"}, "a1")),
            text_response("done"))
        session.send("Remind me about FAI delays and analyse FAI now.")
        results = gemini.requests[-1]["contents"][-1].parts
        self.assertEqual([(p.function_response.name, p.function_response.id) for p in results],
                         [(MEMORY_TOOL_NAME, "m1"), ("analyze_airport", "a1")])
        self.assertEqual(self.dispatch.call_count, 1)

    def test_memory_counts_as_a_tool_round(self):
        session, gemini = self.session_with(
            [EARLY] + FILLER, memory_call("FAI delays"),
            call_response(("compare_airports", {"airport_codes": ["FAI", "SEA"]}, "c1")),   # over budget
            text_response("Best effort from memory only."), max_tool_rounds=1)
        self.assertEqual(session.send("Compare that earlier airport with SEA."), "Best effort from memory only.")
        self.dispatch.assert_not_called()
        self.assertEqual(gemini.modes()[-3:], ["AUTO", "AUTO", "NONE"])

    def test_current_turn_contents_are_never_trimmed(self):
        session, gemini = self.session_with(
            [(f"Q{i}", f"A{i}") for i in range(10)],
            memory_call("Q3", "m1"), analyze("ANC", "a1"), analyze("FAI", "a2"), text_response("done"),
            recent_turns=1)
        session.send("Long multi-step question.")
        final = gemini.requests[-1]["contents"]
        self.assertEqual(texts(final)[:3], ["Q9", "A9", "Long multi-step question."])
        self.assertEqual(roles(final), ["user", "model", "user"] + ["model", "user"] * 3)
        self.assertEqual([p.function_response.id for c in final for p in c.parts if p.function_response],
                         ["m1", "a1", "a2"])


class RetrievalBoundsTests(MemoryTestCase):
    def test_results_bounded_and_broad_query_cannot_dump_history(self):
        turns = [turn(f"Airport question {i}", f"Airport answer {i}") for i in range(30)]
        memory = ConversationMemory(turns, visible_turns=2)
        result = memory.retrieve({"query": "every airport question and airport answer we ever discussed"})
        self.assertEqual(result["searchable_earlier_turns"], 28)
        self.assertEqual(len(result["matches"]), MAX_MEMORY_RESULTS)
        self.assertEqual([m["turn"] for m in result["matches"]], [28, 27, 26])   # tie -> most recent first
        self.assertLess(len(json.dumps(result)), 5000)

    def test_long_answers_are_truncated(self):
        memory = ConversationMemory([turn("FAI delays?", "x" * (MAX_ANSWER_CHARS + 500))], visible_turns=0)
        match = memory.retrieve({"query": "FAI delays"})["matches"][0]
        self.assertTrue(match["truncated"])
        self.assertLessEqual(len(match["assistant_answer"]), MAX_ANSWER_CHARS + 20)

    def test_visible_turns_are_not_searched(self):
        memory = ConversationMemory([turn("FAI delays?", "old"), turn("FAI again", "visible")], visible_turns=1)
        result = memory.retrieve({"query": "FAI"})
        self.assertEqual([m["turn"] for m in result["matches"]], [1])

    def test_no_lexical_match_returns_no_turns(self):
        memory = ConversationMemory([turn("Alpha?", "one"), turn("Beta?", "two"), turn("Gamma?", "three")],
                                    visible_turns=1)
        result = memory.retrieve({"query": "the answer immediately before"})
        self.assertEqual((result["status"], result["matches"], result["searchable_earlier_turns"]), ("ok", [], 2))

    def test_lexical_normalization_is_generic(self):
        self.assertEqual(_terms("comparison of Delays") & _terms("compare the delay"), {"compa", "delay"})
        self.assertNotIn("the", _terms("the earlier answer"))

    def test_invalid_query_is_a_structured_error(self):
        memory = ConversationMemory([turn("a", "b")])
        for bad in (None, {}, {"query": ""}, {"query": 3}, {"query": "x", "limit": 1000}, ["query"]):
            result = memory.retrieve(bad)
            self.assertEqual((result["status"], result["errors"][0]["code"]), ("error", "invalid_tool_arguments"))


class TurnBoundaryTests(MemoryTestCase):
    def test_retrieved_tool_turn_is_a_plain_completed_turn_record(self):
        gemini = FakeGemini(compare_call("c1"), text_response("ANC ranks first; FAI second."),
                            *[text_response(a) for _, a in FILLER[:2]],
                            memory_call("the comparison of ANC FAI JNU"), text_response("FAI was second."))
        session = AgentSession(client=gemini, context=self.ctx)
        session.send(EARLY[0])
        for user, _ in FILLER[:2]:
            session.send(user)
        session.send("Which airport was second in that comparison?")
        (result,) = memory_responses(gemini.requests[-1])
        (match,) = [m for m in result["matches"] if m["turn"] == 1]
        self.assertEqual(match["user_message"], EARLY[0])
        self.assertEqual(match["assistant_answer"], "ANC ranks first; FAI second.")
        self.assertEqual(match["tools_used"], [{"name": "compare_airports",
                                                "args": {"airport_codes": ["ANC", "FAI", "JNU"]}}])
        text = json.dumps(result)                                   # plain JSON data only
        for provider_field in ("function_call", "function_response", "thought_signature", '"role"'):
            self.assertNotIn(provider_field, text)

    def test_summarize_turn_ignores_intermediate_and_memory_contents(self):
        call = types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name=MEMORY_TOOL_NAME, args={"query": "x"}, id="m1")),
            types.Part(function_call=types.FunctionCall(name="analyze_airport", args={"airport_code": "ANC"}, id="a1"))])
        response = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id="a1", name="analyze_airport", response={"status": "ok"}))])
        completed = (types.Content(role="user", parts=[types.Part(text="Analyze ANC")]), call, response,
                     types.Content(role="model", parts=[types.Part(text="ANC summary")]))
        record = summarize_turn(completed, 4)
        self.assertEqual(record, {"turn": 4, "user_message": "Analyze ANC", "assistant_answer": "ANC summary",
                                  "tools_used": [{"name": "analyze_airport", "args": {"airport_code": "ANC"}}]})


class SeparationAndAtomicityTests(MemoryTestCase):
    def test_same_tool_context_and_memory_not_in_it(self):
        session, gemini = self.session_with([EARLY] + FILLER, memory_call("FAI delays"), analyze("FAI", "a1"),
                                            text_response("done"))
        before = set(vars(self.ctx))
        session.send("What did you say about FAI delays? Re-check FAI.")
        self.assertIs(session.context, self.ctx)
        self.assertIs(self.dispatch.call_args.args[2], self.ctx)
        self.assertEqual(set(vars(self.ctx)), before)                       # no conversation state added
        self.assertNotIn("conversation", json.dumps(self.ctx.cache.stats()).lower())

    def test_failed_turn_after_memory_retrieval_keeps_history(self):
        session, _ = self.session_with([EARLY] + FILLER)
        committed = session.history
        session._client = FailingAt(2, memory_call("FAI delays"))          # request after retrieval fails
        with self.assertRaises(ConnectionError):
            session.send("What did you say about FAI delays?")
        self.assertEqual(session.history, committed)
        self.retrieve.assert_called_once()

    def test_failed_turn_after_memory_and_business_tool_keeps_history(self):
        session, _ = self.session_with([EARLY] + FILLER)
        committed = session.history
        session._client = FailingAt(3, memory_call("FAI delays"), analyze("FAI", "a1"))
        with self.assertRaises(ConnectionError):
            session.send("Re-check the airport with delays.")
        self.assertEqual(session.history, committed)
        self.assertEqual(self.dispatch.call_count, 1)

    def test_separate_sessions_do_not_share_memory(self):
        a, _ = self.session_with([EARLY] + FILLER)
        gemini_b = FakeGemini(memory_call("FAI delays"), text_response("Nothing earlier."))
        b = AgentSession(client=gemini_b)
        b.send("What did you say about FAI delays?")
        (result,) = memory_responses(gemini_b.requests[-1])
        self.assertEqual((result["matches"], result["searchable_earlier_turns"]), ([], 0))
        self.assertIsNot(a.context, b.context)

    def test_memory_lookup_does_not_mutate_committed_history(self):
        session, gemini = self.session_with([EARLY] + FILLER, memory_call("FAI"), text_response("ok"))
        stored = session._turns
        snapshot = list(stored)
        session.send("FAI?")
        self.assertEqual(stored, snapshot)                                  # old list object untouched
        self.assertEqual(len(session._turns), len(snapshot) + 1)


class StandaloneAndConfigTests(MemoryTestCase):
    def test_standalone_memory_request_is_safe(self):
        gemini = FakeGemini(memory_call("anything earlier"), text_response("There is no earlier conversation."))
        self.assertEqual(run_agent_turn("What did we discuss?", self.ctx, client=gemini),
                         "There is no earlier conversation.")
        (result,) = memory_responses(gemini.requests[-1])
        self.assertEqual((result["status"], result["matches"]), ("ok", []))
        self.assertIn("No earlier conversation", result["note"])

    def test_config_native_instruction_grounding_and_memory_declaration(self):
        session, gemini = self.session_with([EARLY] + FILLER, memory_call("FAI"), text_response("ok"))
        session.send("FAI?")
        for request in gemini.requests:
            config = request["config"]
            self.assertIs(config.system_instruction, SYSTEM_INSTRUCTION)
            self.assertIn(MEMORY_TOOL_NAME, {d.name for d in config.tools[0].function_declarations})
            self.assertTrue(config.automatic_function_calling.disable)
        self.assertIn("Capacity Pressure is a PROXY", SYSTEM_INSTRUCTION)            # grounding kept
        self.assertIn("call retrieve_conversation_context", SYSTEM_INSTRUCTION)
        self.assertIn("never claim to remember details you have not retrieved", SYSTEM_INSTRUCTION)
        for text in texts(session.history):
            self.assertNotIn("CONVERSATION CONTEXT", text)

    def test_memory_declaration_hides_internals(self):
        declaration = next(d for d in loop.agent_tools()[0].function_declarations if d.name == MEMORY_TOOL_NAME)
        self.assertEqual(set(declaration.parameters_json_schema["properties"]), {"query"})
        text = json.dumps(declaration.model_dump(exclude_none=True))
        for internal in ("AgentSession", "AgentToolContext", "history", "index"):
            self.assertNotIn(internal, text)

    def test_no_tool_role_anywhere(self):
        session, gemini = self.session_with([EARLY] + FILLER, memory_call("FAI"), analyze("FAI", "a1"),
                                            text_response("ok"))
        session.send("FAI?")
        for request in gemini.requests:
            for content in request["contents"]:
                self.assertIn(content.role, ("user", "model"))
