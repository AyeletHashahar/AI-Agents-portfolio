"""Offline tests for multi-turn conversations (agent/session.py, run_conversation_turn).

Gemini is the scripted FakeGemini from test_agent_loop; the real dispatcher and business
tools run on the synthetic offline fixtures. No network.
"""

from unittest import mock

from google.genai import types

import agent.loop as loop
from agent import AgentSession, EmptyModelResponse, FinalizationFailed, SYSTEM_INSTRUCTION, run_conversation_turn
from agent_tools import AgentToolContext
from tests.test_agent_loop import AgentLoopTestCase, FakeGemini, analyze, call_response, text_response

COMPARE = "Compare ANC, FAI and JNU as airport investment opportunities."
FOLLOW_UP = "Which of those has the highest passenger growth?"


def compare_call(call_id="c1"):
    return call_response(("compare_airports", {"airport_codes": ["ANC", "FAI", "JNU"]}, call_id))


def roles(contents):
    return [c.role for c in contents]


def texts(contents):
    return [p.text for c in contents for p in c.parts if p.text]


class FailingAt(FakeGemini):
    """Raises on the given 1-based request number (simulated provider/network failure)."""

    def __init__(self, fail_at, *responses):
        super().__init__(*responses)
        self.fail_at = fail_at

    def generate_content(self, *, model, contents, config):
        if len(self.requests) + 1 == self.fail_at:
            self.requests.append({"model": model, "contents": list(contents), "config": config})
            raise ConnectionError("Gemini unreachable")
        return super().generate_content(model=model, contents=contents, config=config)


class SessionTestCase(AgentLoopTestCase):
    def session(self, gemini, **kw):
        return AgentSession(client=gemini, context=self.ctx, **kw)


class ConversationTests(SessionTestCase):
    def test_two_direct_turns_carry_the_conversation(self):
        first = text_response("I compare U.S. airports.")
        gemini = FakeGemini(first, text_response("ANC, FAI and JNU."))
        session = self.session(gemini)
        self.assertEqual(session.send("What can you do?"), "I compare U.S. airports.")
        self.assertEqual(session.send("Which airports did we discuss?"), "ANC, FAI and JNU.")
        second = gemini.requests[1]["contents"]
        self.assertEqual(roles(second), ["user", "model", "user"])
        self.assertEqual(texts(second), ["What can you do?", "I compare U.S. airports.", "Which airports did we discuss?"])
        self.assertIs(second[1], first.candidates[0].content)              # original model content kept
        self.assertEqual(roles(session.history), ["user", "model", "user", "model"])

    def test_tool_turn_is_preserved_for_the_follow_up(self):
        tool_request = compare_call()
        gemini = FakeGemini(tool_request, text_response("ANC ranks first."), text_response("FAI has the highest growth."))
        session = self.session(gemini)
        session.send(COMPARE)
        session.send(FOLLOW_UP)
        second_turn = gemini.requests[2]["contents"]
        self.assertEqual(roles(second_turn), ["user", "model", "user", "model", "user"])
        user, call_turn, response_turn, answer, follow_up = second_turn
        self.assertEqual(user.parts[0].text, COMPARE)
        self.assertIs(call_turn, tool_request.candidates[0].content)       # function call, not rebuilt
        self.assertEqual(call_turn.parts[0].thought_signature, b"signature-1")
        self.assertEqual(response_turn.parts[0].function_response.name, "compare_airports")
        self.assertEqual(response_turn.parts[0].function_response.response["metadata"]["analyzed_count"], 3)
        self.assertEqual(answer.parts[0].text, "ANC ranks first.")
        self.assertEqual(follow_up.parts[0].text, FOLLOW_UP)

    def test_follow_up_may_call_another_tool(self):
        gemini = FakeGemini(analyze("ANC", "a1"), text_response("ANC grows 2%."),
                            call_response(("compare_airports", {"airport_codes": ["ANC", "FAI"]}, "c1")),
                            text_response("FAI grows faster than ANC."))
        session = self.session(gemini)
        session.send("Analyze ANC.")
        self.assertEqual(session.send("How does its growth compare with FAI?"), "FAI grows faster than ANC.")
        self.assertEqual([c.args[0] for c in self.dispatch.call_args_list], ["analyze_airport", "compare_airports"])
        self.assertEqual(roles(session.history), ["user", "model", "user", "model"] * 2)

    def test_multiple_calls_and_rounds_stay_in_valid_order(self):
        gemini = FakeGemini(call_response(("analyze_airport", {"airport_code": "FAI"}, "a1"),
                                          ("analyze_airport", {"airport_code": "JNU"}, "a2")),
                            compare_call(), text_response("done"), text_response("follow-up"))
        session = self.session(gemini)
        session.send(COMPARE)
        session.send(FOLLOW_UP)
        history = session.history
        self.assertEqual(roles(history), ["user", "model", "user", "model", "user", "model", "user", "model"])
        self.assertEqual([p.function_response.id for p in history[2].parts], ["a1", "a2"])
        self.assertEqual(history[4].parts[0].function_response.id, "c1")
        for content in history:
            self.assertIn(content.role, ("user", "model"))                 # never "tool"


class StateSeparationTests(SessionTestCase):
    def test_same_tool_context_reused_across_turns(self):
        gemini = FakeGemini(analyze("ANC", "a1"), text_response("one"), analyze("FAI", "a2"), text_response("two"))
        session = self.session(gemini)
        session.send("Analyze ANC.")
        session.send("And FAI?")
        self.assertIs(session.context, self.ctx)
        self.assertEqual(self.dispatch.call_count, 2)
        for call in self.dispatch.call_args_list:
            self.assertIs(call.args[2], self.ctx)

    def test_tool_context_never_enters_the_history(self):
        gemini = FakeGemini(analyze("ANC", "a1"), text_response("done"))
        session = self.session(gemini)
        session.send("Analyze ANC.")
        for content in session.history:
            self.assertIsInstance(content, types.Content)
        dumped = str([c.model_dump(mode="json", exclude_none=True) for c in session.history])
        self.assertNotIn("AgentToolContext", dumped)
        self.assertNotIn("SessionCache", dumped)

    def test_separate_sessions_share_nothing(self):
        gemini_a, gemini_b = FakeGemini(text_response("a")), FakeGemini(text_response("b"))
        a, b = AgentSession(client=gemini_a), AgentSession(client=gemini_b)
        a.send("Hello from A")
        b.send("Hello from B")
        self.assertIsNot(a.context, b.context)
        self.assertIsInstance(a.context, AgentToolContext)
        self.assertEqual(texts(a.history), ["Hello from A", "a"])
        self.assertEqual(texts(b.history), ["Hello from B", "b"])
        self.assertEqual(texts(gemini_b.requests[0]["contents"]), ["Hello from B"])

    def test_system_instruction_is_config_not_history(self):
        gemini = FakeGemini(text_response("one"), compare_call(), text_response("two"))
        session = self.session(gemini)
        session.send("Hi")
        session.send(COMPARE)
        for request in gemini.requests:
            self.assertIs(request["config"].system_instruction, SYSTEM_INSTRUCTION)
        for text in texts(session.history):
            self.assertNotIn("EVIDENCE LEVELS", text)
            self.assertNotIn(SYSTEM_INSTRUCTION, text)

    def test_history_is_read_only(self):
        session = self.session(FakeGemini(text_response("hi")))
        session.send("Hi")
        self.assertIsInstance(session.history, tuple)
        with self.assertRaises(AttributeError):
            session.history = ()

    def test_client_created_once_and_reused(self):
        gemini = FakeGemini(text_response("one"), text_response("two"))
        with mock.patch.object(loop, "create_gemini_client", return_value=gemini) as create:
            session = AgentSession(context=self.ctx)
            session.send("first")
            session.send("second")
        create.assert_called_once_with()

    def test_context_type_checked(self):
        with self.assertRaises(TypeError):
            AgentSession(client=FakeGemini(), context={"not": "a context"})


class BudgetPerTurnTests(SessionTestCase):
    def test_budgets_reset_for_each_user_turn(self):
        # One tool round allowed per turn: both turns may use it, no finalization needed.
        gemini = FakeGemini(analyze("ANC", "a1"), text_response("one"), analyze("FAI", "a2"), text_response("two"))
        session = self.session(gemini, max_tool_rounds=1, max_llm_calls=2)
        self.assertEqual(session.send("Analyze ANC."), "one")
        self.assertEqual(session.send("Analyze FAI."), "two")
        self.assertEqual(gemini.modes(), ["AUTO", "AUTO", "AUTO", "AUTO"])
        self.assertEqual(self.dispatch.call_count, 2)

    def test_successful_finalization_commits_a_valid_turn(self):
        gemini = FakeGemini(analyze("ANC", "a1"), analyze("JNU", "a2"),        # 2nd request over budget
                            text_response("Best effort from ANC only."),        # finalization answer
                            text_response("Next turn works."))
        session = self.session(gemini, max_tool_rounds=1)
        self.assertEqual(session.send(COMPARE), "Best effort from ANC only.")
        history = session.history
        self.assertEqual(roles(history), ["user", "model", "user", "user", "model"])
        self.assertIn("EXECUTION BUDGET REACHED", history[3].parts[0].text)     # what the model was told
        self.assertEqual(history[-1].parts[0].text, "Best effort from ANC only.")
        calls = [p.function_call.id for c in history for p in c.parts if p.function_call]
        self.assertEqual(calls, ["a1"])                                          # unexecuted a2 never stored
        self.assertEqual(session.send(FOLLOW_UP), "Next turn works.")
        self.assertEqual(gemini.modes(), ["AUTO", "AUTO", "NONE", "AUTO"])


class AtomicityTests(SessionTestCase):
    def test_failed_first_turn_leaves_history_empty(self):
        session = self.session(FailingAt(1))
        with self.assertRaises(ConnectionError):
            session.send(COMPARE)
        self.assertEqual(session.history, ())

    def test_failed_first_turn_after_tool_execution_leaves_history_empty(self):
        session = self.session(FailingAt(2, compare_call()))              # tool ran, then Gemini failed
        with self.assertRaises(ConnectionError):
            session.send(COMPARE)
        self.assertEqual(session.history, ())
        self.assertEqual(self.dispatch.call_count, 1)

    def test_failed_second_turn_keeps_the_committed_history(self):
        gemini = FailingAt(4, compare_call(), text_response("ANC ranks first."), analyze("FAI", "a1"))
        session = self.session(gemini)
        session.send(COMPARE)
        committed = session.history
        with self.assertRaises(ConnectionError):
            session.send(FOLLOW_UP)
        self.assertEqual(session.history, committed)
        self.assertEqual(roles(session.history), ["user", "model", "user", "model"])
        self.assertNotIn(FOLLOW_UP, texts(session.history))

    def test_agent_errors_are_not_committed_either(self):
        empty = types.GenerateContentResponse(candidates=[])
        session = self.session(FakeGemini(text_response("ok"), empty))
        session.send("Hi")
        committed = session.history
        with self.assertRaises(EmptyModelResponse):
            session.send("Again")
        self.assertEqual(session.history, committed)

        session = self.session(FakeGemini(analyze(), empty), max_tool_rounds=0)
        with self.assertRaises(FinalizationFailed):
            session.send("Analyze ANC.")
        self.assertEqual(session.history, ())

    def test_session_recovers_after_a_failed_turn(self):
        gemini = FailingAt(2, text_response("first"), text_response("third"))
        session = self.session(gemini)
        session.send("one")
        with self.assertRaises(ConnectionError):
            session.send("two")
        self.assertEqual(session.send("three"), "third")
        self.assertEqual(texts(gemini.requests[2]["contents"]), ["one", "first", "three"])


class RunConversationTurnTests(SessionTestCase):
    def test_does_not_modify_the_given_turns(self):
        prior = [(types.Content(role="user", parts=[types.Part(text="earlier")]),
                  types.Content(role="model", parts=[types.Part(text="earlier answer")]))]
        snapshot = list(prior)
        gemini = FakeGemini(text_response("ok"))
        text, turn = run_conversation_turn(prior, "now", self.ctx, client=gemini)
        self.assertEqual(text, "ok")
        self.assertEqual(prior, snapshot)
        self.assertEqual(roles(turn), ["user", "model"])                         # only the new turn
        self.assertEqual(roles(gemini.requests[0]["contents"]), ["user", "model", "user"])

    def test_standalone_run_agent_turn_unchanged(self):
        gemini = FakeGemini(compare_call(), text_response("ANC ranks first."))
        self.assertEqual(loop.run_agent_turn(COMPARE, self.ctx, client=gemini), "ANC ranks first.")
        self.assertEqual(roles(gemini.requests[0]["contents"]), ["user"])
