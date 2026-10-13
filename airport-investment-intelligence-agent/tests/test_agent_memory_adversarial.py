"""Adversarial / characterization tests for on-demand conversation memory. Offline only.

Two kinds of tests:
- correctness tests: assert the intended behavior;
- CHARACTERIZATION tests (test_characterize_*): record what the V1 lexical retrieval
  actually does, including known weaknesses, so they are visible and deliberate.
"""

import copy
import importlib.util
import json
import pathlib
import random
from unittest import mock

from google.genai import types

import agent.loop as loop
from agent import (MAX_MEMORY_RESULTS, MEMORY_TOOL_NAME, SYSTEM_INSTRUCTION, AgentSession, ConversationMemory,
                   run_agent_turn, run_conversation_turn)
from agent.memory import MAX_ANSWER_CHARS, MAX_USER_MESSAGE_CHARS, MEMORY_DECLARATION, _terms, summarize_turn
from agent.tools import TOOLS
from tests.test_agent_loop import AgentLoopTestCase, FakeGemini, analyze, call_response, text_response
from tests.test_agent_session import FailingAt, roles, texts


# --- builders ------------------------------------------------------------------------------

def user(text):
    return types.Content(role="user", parts=[types.Part(text=text)])


def model(text):
    return types.Content(role="model", parts=[types.Part(text=text)])


def text_turn(question, answer):
    return (user(question), model(answer))


def tool_turn(question, rounds, answer):
    """Completed turn: question, then per round a model call turn and a user-role response
    turn, then the final answer. rounds = [[(name, args), ...], ...]."""
    contents, n = [user(question)], 0
    for calls in rounds:
        ids = [f"id{n + k}" for k in range(len(calls))]
        n += len(calls)
        contents.append(types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name=name, args=args, id=i))
            for (name, args), i in zip(calls, ids)]))
        contents.append(types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(name=name, id=i, response={"status": "ok"}))
            for (name, _), i in zip(calls, ids)]))
    contents.append(model(answer))
    return tuple(contents)


def search(turns, query, visible=0):
    return ConversationMemory(list(turns), visible_turns=visible).retrieve({"query": query})


def ranked(result):
    return [m["turn"] for m in result["matches"]]


def memory_call(query, call_id="m1"):
    return call_response((MEMORY_TOOL_NAME, {"query": query}, call_id))


def memory_results(request):
    return [p.function_response.response for c in request["contents"] for p in c.parts
            if p.function_response and p.function_response.name == MEMORY_TOOL_NAME]


COMPARISON = text_turn("Compare ANC, FAI and JNU as airport investment opportunities.",
                       "ANC ranks first, FAI second and JNU third.")
CAGR = text_turn("What is passenger CAGR?", "CAGR is the compound annual growth rate of passengers.")
RUNWAY = text_turn("How many runways does PANC have?", "PANC has 3 operational runways.")
WEATHER = text_turn("Does the data include weather?", "No weather data is available in our sources.")


class NoNetworkCase(AgentLoopTestCase):
    """Offline fixtures + network blocked + dispatcher spy (from AgentLoopTestCase)."""


# --- 1. ConversationMemory directly ------------------------------------------------------------

class DirectRetrievalTests(NoNetworkCase):
    def test_a_exact_topic_ranks_first(self):
        result = search([COMPARISON, CAGR, RUNWAY], "earlier comparison of ANC FAI JNU")
        self.assertEqual(ranked(result)[0], 1)
        self.assertEqual(set(result["matches"][0]["match"]["matched_terms"]), {"compa", "anc", "fai", "jnu"})

    def test_b_specific_overlap_beats_generic_overlap(self):
        turns = [text_turn("FAI passenger volume?", "FAI had 563,572 passengers."),
                 text_turn("FAI runways?", "FAI has 2 operational runways."),
                 text_turn("Why are FAI delays high?", "FAI shows elevated departure delays."),
                 text_turn("ANC runways?", "ANC has 3.")]
        self.assertEqual(ranked(search(turns, "FAI delays")), [3, 2, 1])   # 2 terms, then 1-term ties by recency

    def test_c_equal_scores_prefer_the_more_recent_turn(self):
        turns = [text_turn("ANC growth?", "x"), text_turn("unrelated", "y"), text_turn("ANC growth again?", "z")]
        self.assertEqual(ranked(search(turns, "ANC growth")), [3, 1])

    def test_d_conversational_stopwords_add_no_score(self):
        result = search([COMPARISON, CAGR], "what did you say earlier about the previous answer about ANC")
        self.assertEqual(ranked(result), [1])
        self.assertEqual(result["matches"][0]["match"]["matched_terms"], ["anc"])

    def test_d_conversational_verbs_are_stopwords(self):
        for query in ("what did you tell me earlier", "what did you say before", "you mentioned it",
                      "what we talked about", "recall that again", "remember what you said ago",
                      "back to what you told me"):
            self.assertEqual(_terms(query), set(), query)
        turns = [text_turn("Tell me about runways", "PANC has 3."), text_turn("CAGR?", "growth rate")]
        self.assertEqual(search(turns, "what did you tell me earlier")["matches"], [])
        self.assertEqual(ranked(search(turns, "what did you tell me about runways")), [1])

    def test_e_case_insensitive(self):
        rankings = {q: ranked(search([CAGR, COMPARISON, RUNWAY], q)) for q in ("ANC", "anc", "Anc", "aNc")}
        self.assertEqual(len({tuple(r) for r in rankings.values()}), 1)
        self.assertEqual(rankings["ANC"][0], 2)

    def test_f_punctuation_is_ignored(self):
        clean = search([CAGR, COMPARISON], "ANC FAI JNU")
        noisy = search([CAGR, COMPARISON], "ANC, FAI & JNU?!... (ANC)")
        self.assertEqual(ranked(clean), ranked(noisy))
        self.assertEqual(clean["matches"][0]["match"]["matched_terms"], noisy["matches"][0]["match"]["matched_terms"])

    def test_g_numbers_are_terms(self):
        turns = [text_turn("2024 passengers at ANC?", "2.7 million in 2024."),
                 text_turn("2025 passengers at ANC?", "2.71 million in 2025."),
                 text_turn("passenger definition?", "enplaned passengers")]
        self.assertEqual(ranked(search(turns, "the 2025 passenger comparison"))[0], 2)

    def test_h_invalid_arguments_are_structured_errors(self):
        memory = ConversationMemory([COMPARISON])
        bad_inputs = [None, {}, {"other": "x"}, {"query": "x", "extra": 1}, {"query": None}, {"query": ""},
                      {"query": "   \n\t "}, {"query": 42}, {"query": ["ANC"]}, "ANC", 7, ["query"]]
        for bad in bad_inputs:
            result = memory.retrieve(bad)
            self.assertEqual(result["status"], "error", bad)
            self.assertEqual(result["errors"][0]["code"], "invalid_tool_arguments", bad)
            json.dumps(result)

    def test_i_no_searchable_history(self):
        for memory in (ConversationMemory(), ConversationMemory([COMPARISON, CAGR], visible_turns=2)):
            result = memory.retrieve({"query": "ANC"})
            self.assertEqual((result["status"], result["matches"], result["searchable_earlier_turns"]), ("ok", [], 0))
            self.assertIn("No earlier conversation", result["note"])

    def test_j_result_limit(self):
        turns = [text_turn(f"ANC question {i}", f"ANC answer {i}") for i in range(40)]
        result = search(turns, "ANC question answer")
        self.assertEqual(len(result["matches"]), MAX_MEMORY_RESULTS)
        self.assertEqual(ranked(result), [40, 39, 38])

    def test_k_truncation_boundaries(self):
        cases = [("q" * MAX_USER_MESSAGE_CHARS, "a", False), ("q" * (MAX_USER_MESSAGE_CHARS + 1), "a", True),
                 ("ANC", "a" * MAX_ANSWER_CHARS, False), ("ANC", "a" * (MAX_ANSWER_CHARS + 1), True),
                 ("ANC " + "q" * 5000, "a" * 9000, True)]
        for question, answer, truncated in cases:
            match = search([text_turn(question, answer)], "ANC qqqqq")["matches"][0]
            self.assertEqual(match["truncated"], truncated, (len(question), len(answer)))
            self.assertLessEqual(len(match["user_message"]), MAX_USER_MESSAGE_CHARS + 20)
            self.assertLessEqual(len(match["assistant_answer"]), MAX_ANSWER_CHARS + 20)

    def test_l_retrieval_does_not_modify_stored_turns(self):
        turns = [tool_turn("Compare ANC FAI", [[("compare_airports", {"airport_codes": ["ANC", "FAI"]})]], "ANC"),
                 COMPARISON, CAGR]
        before = copy.deepcopy([[c.model_dump() for c in t] for t in turns])
        identities = [[id(c) for c in t] for t in turns]
        stored = list(turns)
        ConversationMemory(stored, visible_turns=1).retrieve({"query": "ANC FAI compare"})
        self.assertEqual([[c.model_dump() for c in t] for t in turns], before)
        self.assertEqual([[id(c) for c in t] for t in turns], identities)
        self.assertEqual(stored, turns)


# --- 2. Lexical behavior: true positives, false positives, false negatives ---------------------

class LexicalCharacterizationTests(NoNetworkCase):
    def test_stemming_true_positives(self):
        for a, b in [("compare", "comparison"), ("delay", "delays"), ("delay", "delayed"),
                     ("passenger", "passengers"), ("airport", "airports"), ("runway", "runways")]:
            self.assertTrue(_terms(a) & _terms(b), (a, b))

    def test_characterize_stemming_false_negatives(self):
        # V1 limitation: these related words share NO term, so they never match each other.
        for a, b in [("growth", "growing"), ("grew", "growth"), ("busy", "busiest"), ("fast", "fastest")]:
            self.assertFalse(_terms(a) & _terms(b), (a, b))

    def test_characterize_stemming_false_positives(self):
        # V1 limitation: the first-5-letters rule merges unrelated words.
        for a, b in [("compare", "company"), ("passenger", "passes"), ("international", "internal"),
                     ("terminal", "terminate"), ("capacity", "capacitor")]:
            self.assertTrue(_terms(a) & _terms(b), (a, b))

    def test_characterize_false_positive_retrieval(self):
        turns = [text_turn("Which company runs the terminal?", "No operator data is available."),
                 text_turn("What is a percentile?", "A rank within the group.")]
        result = search(turns, "the earlier airport comparison")
        self.assertEqual(ranked(result), [1])                               # matched only via "compa"
        self.assertEqual(result["matches"][0]["match"]["matched_terms"], ["compa"])

    def test_characterize_semantic_false_negative(self):
        # Related meaning, different vocabulary: lexical retrieval finds nothing (and returns nothing).
        turns = [text_turn("Which airport grew most?", "FAI had the highest passenger CAGR."),
                 text_turn("Weather?", "Not available.")]
        self.assertEqual(search(turns, "Which was growing fastest?")["matches"], [])


# --- 3. No lexical match: empty result, never an unrelated turn -----------------------------------

class NoMatchTests(NoNetworkCase):
    def test_characterize_single_generic_term_matches_wrong_airport(self):
        # "Boston" was never discussed, but "investment" alone matches the ANC/FAI/JNU comparison.
        result = search([COMPARISON, RUNWAY, WEATHER], "Which investment did we discuss regarding Boston?")
        self.assertEqual(ranked(result), [1])
        self.assertEqual(result["matches"][0]["match"], {"type": "lexical", "matched_terms": ["inves"]})

    def test_no_match_returns_empty_result(self):
        turns = [text_turn("Compare ANC, FAI and JNU.", "ANC ranks first."), RUNWAY, WEATHER]
        result = search(turns, "Which investment did we discuss regarding Boston?")
        self.assertEqual((result["status"], result["matches"], result["searchable_earlier_turns"]), ("ok", [], 3))
        self.assertIn("No earlier turn shares any search term", result["note"])
        self.assertIn("Do not guess", result["note"])
        self.assertNotIn("weather", json.dumps(result).lower())             # no unrelated turn leaks in

    def test_queries_without_usable_terms_return_nothing(self):
        for query in ("what did we discuss before?", "מה אמרנו על העיכובים", "?!...", "🚀✈️"):
            self.assertEqual(search([COMPARISON, CAGR, WEATHER], query)["matches"], [], query)

    def test_no_fallback_remains_anywhere(self):
        source = pathlib.Path(importlib.util.find_spec("agent.memory").origin).read_text(encoding="utf-8")
        self.assertNotIn("fallback", source)
        rng = random.Random(7)
        vocabulary = ["ANC", "FAI", "JNU", "delays", "runway", "CAGR", "weather", "zzz", "Boston", "what"]
        turns = [text_turn(" ".join(rng.sample(vocabulary, 3)), "x") for _ in range(12)]
        for _ in range(300):
            query = " ".join(rng.sample(vocabulary, rng.randint(1, 4)))
            result = search(turns, query)
            if not _terms(query) & set().union(*(_terms(" ".join(texts(t))) for t in turns)):
                self.assertEqual(result["matches"], [], query)
            for match in result["matches"]:
                self.assertEqual(match["match"]["type"], "lexical", query)
                self.assertTrue(match["match"]["matched_terms"], query)


# --- 4. Completed-turn structure ---------------------------------------------------------------

class TurnStructureTests(NoNetworkCase):
    def test_turn_shapes_are_single_retrieval_units(self):
        a = text_turn("Plain question about ANC?", "Plain answer.")
        b = tool_turn("Analyze ANC", [[("analyze_airport", {"airport_code": "ANC"})]], "ANC summary.")
        c = tool_turn("FAI and JNU?", [[("analyze_airport", {"airport_code": "FAI"}),
                                        ("analyze_airport", {"airport_code": "JNU"})]], "FAI vs JNU.")
        d = tool_turn("Alaska best?", [[(MEMORY_TOOL_NAME, {"query": "x"})],
                                       [("discover_airports", {"country": "US", "states": ["AK"]})],
                                       [("compare_airports", {"airport_codes": ["ANC", "FAI"]})]], "ANC best.")
        records = [summarize_turn(t, i + 1) for i, t in enumerate([a, b, c, d])]
        self.assertEqual([r["user_message"] for r in records],
                         ["Plain question about ANC?", "Analyze ANC", "FAI and JNU?", "Alaska best?"])
        self.assertEqual([r["assistant_answer"] for r in records],
                         ["Plain answer.", "ANC summary.", "FAI vs JNU.", "ANC best."])
        self.assertEqual(records[2]["tools_used"], [{"name": "analyze_airport", "args": {"airport_code": "FAI"}},
                                                    {"name": "analyze_airport", "args": {"airport_code": "JNU"}}])
        self.assertEqual([t["name"] for t in records[3]["tools_used"]], ["discover_airports", "compare_airports"])
        result = search([a, b, c, d], "ANC FAI JNU Alaska")
        self.assertEqual(result["searchable_earlier_turns"], 4)            # 4 turns, not 4+ contents
        self.assertTrue(all(1 <= t <= 4 for t in ranked(result)))

    def test_tool_arguments_contribute_but_json_keys_do_not(self):
        turns = [tool_turn("hello", [[("compare_airports", {"airport_codes": ["BOS", "PVD"]})]], "done"),
                 text_turn("unrelated", "chat")]
        self.assertEqual(ranked(search(turns, "BOS PVD")), [1])            # argument VALUES are searchable
        self.assertEqual(search(turns, "name args codes")["matches"], [])   # structural JSON keys are not


# --- 5. Strange / defensive content ------------------------------------------------------------

class StrangeContentTests(NoNetworkCase):
    def test_unusual_parts_do_not_crash(self):
        multi = (user("ANC?"), types.Content(role="model", parts=[types.Part(text="ANC is "),
                                                                  types.Part(text="first.")]))
        thoughts = (user("FAI?"), types.Content(role="model", parts=[types.Part(text="secret plan", thought=True),
                                                                     types.Part(text="FAI is second.")]))
        empty_parts = (types.Content(role="user", parts=[]), types.Content(role="model", parts=[]))
        none_parts = (types.Content(role="user"), types.Content(role="model"))
        no_args = tool_turn("Run it", [[("analyze_airport", {})]], "ok")
        no_visible = (user("JNU?"), types.Content(role="model", parts=[types.Part(text="only thinking", thought=True)]))
        records = [summarize_turn(t, i + 1) for i, t in enumerate([multi, thoughts, empty_parts, none_parts,
                                                                   no_args, no_visible])]
        self.assertEqual(records[0]["assistant_answer"], "ANC is first.")
        self.assertEqual(records[1]["assistant_answer"], "FAI is second.")   # thought text excluded
        self.assertEqual((records[2]["user_message"], records[2]["assistant_answer"]), ("", ""))
        self.assertEqual((records[3]["user_message"], records[3]["assistant_answer"]), ("", ""))
        self.assertEqual(records[4]["tools_used"], [{"name": "analyze_airport", "args": {}}])
        self.assertEqual(records[5]["assistant_answer"], "")
        result = search([multi, thoughts, empty_parts, none_parts, no_args, no_visible], "secret plan FAI")
        self.assertNotIn("secret", json.dumps(result["matches"]))          # thoughts never leak into memory
        self.assertEqual(ranked(result)[0], 2)

    def test_unicode_and_hebrew_content(self):
        turns = [text_turn("מה עם העיכובים ב-FAI?", "FAI shows elevated delays — מעניין."),
                 text_turn("Café ☕ émojis 🚀", "ok ✈️")]
        result = search(turns, "מה אמרנו על FAI delays?")                  # mixed Hebrew/English
        self.assertEqual(ranked(result)[0], 1)
        self.assertEqual(result["matches"][0]["match"]["matched_terms"], ["delay", "fai"])
        json.dumps(result, ensure_ascii=False)

    def test_characterize_hebrew_only_query_has_no_terms(self):
        # V1 design: retrieval is English-only ([a-z0-9]+). Gemini must send English terms;
        # a Hebrew-only query has no terms and finds nothing.
        self.assertEqual(_terms("מה אמרנו על העיכובים"), set())
        turns = [text_turn("מה עם העיכובים?", "עיכובים גבוהים"), text_turn("Weather?", "n/a")]
        self.assertEqual(search(turns, "מה אמרנו על העיכובים")["matches"], [])

    def test_characterize_hebrew_only_history_is_unfindable_even_with_english_query(self):
        # Documented V1 limitation: a stored turn with no English terms or identifiers cannot be
        # found, even by a correct English query.
        turns = [text_turn("מה עם העיכובים בשדה?", "יש עיכובים גבוהים בחורף."), text_turn("Weather?", "n/a")]
        self.assertEqual(search(turns, "delays winter")["matches"], [])


# --- 6. Recent-window partitioning --------------------------------------------------------------

class WindowPartitionTests(NoNetworkCase):
    def partition(self, stored, recent_turns):
        """Return (visible turn numbers, highest searchable turn number, searchable count)."""
        turns = [text_turn(f"topic question {i + 1}", f"topic answer {i + 1}") for i in range(stored)]
        gemini = FakeGemini(memory_call("topic"), text_response("ok"))
        run_conversation_turn(turns, "new question", self.ctx, client=gemini, recent_turns=recent_turns)
        first = gemini.requests[0]["contents"]
        visible = [int(t.split()[-1]) for t in texts(first) if t.startswith("topic question")]
        (result,) = memory_results(gemini.requests[-1])
        top = ranked(result)[0] if result["matches"] else None
        return visible, top, result["searchable_earlier_turns"]

    def test_every_size_partitions_exactly(self):
        for stored in (0, 1, 2, 3, 50):
            for recent in (0, 1, 2, stored + 3):
                with self.subTest(stored=stored, recent=recent):
                    visible, top, searchable = self.partition(stored, recent)
                    expected_visible = list(range(max(stored - recent, 0) + 1, stored + 1)) if recent else []
                    self.assertEqual(visible, expected_visible)                       # A. visible
                    self.assertEqual(searchable, stored - len(expected_visible))     # B. searchable
                    self.assertEqual(top, (stored - len(expected_visible)) or None)  # newest hidden turn
                    self.assertEqual(searchable + len(visible), stored)              # none lost, none twice


# --- 7. Bounded model context with tool-bearing history ----------------------------------------

class BoundedContextTests(NoNetworkCase):
    def test_tool_history_does_not_leak_into_initial_context(self):
        script = []
        for i in range(15):
            script += [analyze("ANC", f"a{i}"), text_response(f"answer {i}")]
        gemini = FakeGemini(*script, memory_call("answer 3", "m1"), analyze("FAI", "x1"), text_response("final"))
        session = AgentSession(client=gemini, context=self.ctx)
        for i in range(15):
            session.send(f"question {i}")
        initial_sizes = [len(r["contents"]) for r in gemini.requests[0::2][:15]]
        self.assertEqual(max(initial_sizes), 2 * 4 + 1)                     # 2 tool turns (4 contents) + question
        session.send("What did you say in answer 3, and re-check FAI?")
        new_turn_requests = gemini.requests[30:]
        first_ids = [p.function_call.id for c in new_turn_requests[0]["contents"] for p in c.parts if p.function_call]
        self.assertEqual(first_ids, ["a13", "a14"])                         # only the 2 recent turns' calls
        self.assertEqual([len(r["contents"]) for r in new_turn_requests], [9, 11, 13])  # current turn grows
        self.assertEqual(len(session.history), 15 * 4 + 6)


# --- 8. Fake-LLM decision scenarios not covered elsewhere ---------------------------------------

class ScenarioTests(NoNetworkCase):
    def session(self, gemini, **kw):
        session = AgentSession(client=gemini, context=self.ctx, **kw)
        return session

    def seeded(self, *extra, **kw):
        gemini = FakeGemini(*[text_response(a) for a in ("ANC ranks first, FAI second.", "CAGR explained.",
                                                         "Runways explained.", "Weather n/a.")], *extra)
        session = self.session(gemini, **kw)
        for q in ("Compare ANC FAI JNU", "CAGR?", "Runways?", "Weather?"):
            session.send(q)
        return session, gemini

    def test_independent_question_never_retrieves(self):
        with mock.patch.object(ConversationMemory, "retrieve") as retrieve:
            session, _ = self.seeded(text_response("CAGR is compound annual growth."))
            session.send("What is CAGR?")
        retrieve.assert_not_called()

    def test_multiple_memory_lookups_in_one_turn(self):
        session, gemini = self.seeded(
            call_response((MEMORY_TOOL_NAME, {"query": "ANC FAI comparison"}, "m1"),
                          (MEMORY_TOOL_NAME, {"query": "runways"}, "m2")),
            memory_call("CAGR", "m3"), text_response("Combined answer."))
        self.assertEqual(session.send("Combine what you said about the comparison, runways and CAGR."),
                         "Combined answer.")
        results = memory_results(gemini.requests[-1])
        self.assertEqual(len(results), 3)
        self.assertEqual([r["query"] for r in results], ["ANC FAI comparison", "runways", "CAGR"])
        self.assertEqual(roles(session._turns[-1]), ["user", "model", "user", "model", "user", "model"])
        self.dispatch.assert_not_called()

    def test_repeated_memory_requests_end_in_finalization(self):
        endless = [memory_call(f"query {i}", f"m{i}") for i in range(10)]
        session, gemini = self.seeded(*endless[:3], text_response("Best effort."), max_tool_rounds=2,
                                      max_llm_calls=5)
        self.assertEqual(session.send("Keep looking."), "Best effort.")
        self.assertEqual(gemini.modes()[4:], ["AUTO", "AUTO", "AUTO", "NONE"])     # 2 rounds, 1 skipped, final
        self.assertEqual(len(memory_results(gemini.requests[-1])), 2)

    def test_memory_without_older_history_in_a_session(self):
        gemini = FakeGemini(memory_call("anything"), text_response("Nothing earlier."))
        session = self.session(gemini)
        self.assertEqual(session.send("What did we discuss?"), "Nothing earlier.")
        (result,) = memory_results(gemini.requests[-1])
        self.assertEqual((result["status"], result["matches"]), ("ok", []))


# --- 9. Defensive query inputs ------------------------------------------------------------------

class DefensiveQueryTests(NoNetworkCase):
    def test_queries_are_data(self):
        memory = ConversationMemory([COMPARISON, CAGR])
        queries = ["x" * 200_000, "\n" * 5000 + "ANC" + "\n" * 5000, '{"query": "ANC", "$where": "1=1"}',
                   "ANC'; DROP TABLE turns; --", "__import__('os').system('echo pwned')", "(.*)+[a-z]{2,}\\d?$",
                   "✈️🚀 ANC 🛫", "Ⅷ ﬁ ǅ ẞ 𝔸𝔹ℂ ANC", "\x00\x01 ANC"]
        with mock.patch("os.system", side_effect=AssertionError("query executed")):
            for query in queries:
                result = memory.retrieve({"query": query})
                self.assertEqual(result["status"], "ok")
                json.dumps(result)
                self.assertLess(len(json.dumps(result)), 10_000, "result must stay bounded")

    def test_long_query_echo_is_bounded(self):
        result = ConversationMemory([COMPARISON]).retrieve({"query": "ANC " + "x" * 100_000})
        self.assertLessEqual(len(result["query"]), MAX_USER_MESSAGE_CHARS + 20)
        self.assertEqual(ranked(result), [1])                                # still searched in full


# --- 10. Atomicity under failure ----------------------------------------------------------------

class AtomicityTests(NoNetworkCase):
    def committed_session(self):
        gemini = FakeGemini(text_response("ANC first."), text_response("CAGR explained."), text_response("Runways."))
        session = AgentSession(client=gemini, context=self.ctx)
        for q in ("Compare ANC FAI JNU", "CAGR?", "Runways?"):
            session.send(q)
        return session

    def test_failure_at_every_point_keeps_committed_turns(self):
        class FailOnFinal(FakeGemini):
            def generate_content(self, *, model, contents, config):
                if config.tool_config.function_calling_config.mode == "NONE":
                    raise ConnectionError("final failed")
                return super().generate_content(model=model, contents=contents, config=config)

        failures = {
            "first call": FailingAt(1),
            "after memory": FailingAt(2, memory_call("ANC")),
            "after business tool": FailingAt(2, analyze("FAI", "a1")),
            "after memory and business": FailingAt(3, memory_call("ANC"), analyze("FAI", "a1")),
            "during finalization": FailOnFinal(memory_call("ANC"), analyze("FAI", "a1")),
        }
        for label, client in failures.items():
            with self.subTest(label):
                session = self.committed_session()
                committed, turns = session.history, list(session._turns)
                session._client = client
                if label == "during finalization":
                    session.max_tool_rounds = 1
                with self.assertRaises(ConnectionError):
                    session.send("FAILED TURN MARKER question")
                self.assertEqual(session.history, committed)
                self.assertEqual(session._turns, turns)

    def test_failed_turn_never_becomes_searchable(self):
        session = self.committed_session()
        session._client = FailingAt(2, memory_call("ANC"))
        with self.assertRaises(ConnectionError):
            session.send("FAILED TURN MARKER question")
        session._client = gemini = FakeGemini(memory_call("FAILED TURN MARKER"), text_response("ok"))
        session.recent_turns = 0
        session.send("Did I ask about a failed marker?")
        (result,) = memory_results(gemini.requests[-1])
        self.assertNotIn("FAILED TURN MARKER", json.dumps([m["user_message"] for m in result["matches"]]))
        self.assertEqual(result["searchable_earlier_turns"], 3)


# --- 11. Session isolation ------------------------------------------------------------------------

class IsolationTests(NoNetworkCase):
    def test_sessions_sharing_one_fake_client_keep_separate_memory(self):
        shared = FakeGemini(text_response("ANC answer"), text_response("BOS answer"),
                            text_response("a2"), text_response("b2"), text_response("a3"), text_response("b3"),
                            memory_call("BOS"), text_response("a-final"),
                            memory_call("ANC"), text_response("b-final"))
        a, b = AgentSession(client=shared), AgentSession(client=shared)
        for q_a, q_b in (("ANC only", "BOS only"), ("filler a2", "filler b2"), ("filler a3", "filler b3")):
            a.send(q_a)
            b.send(q_b)
        a.send("What about BOS?")
        b.send("What about ANC?")
        result_a, result_b = memory_results(shared.requests[7])[0], memory_results(shared.requests[9])[0]
        self.assertNotIn("BOS only", json.dumps(result_a))
        self.assertNotIn("ANC only", json.dumps(result_b))
        self.assertIsNot(a.context, b.context)

    def test_shared_context_only_when_explicitly_injected(self):
        a = AgentSession(client=FakeGemini(), context=self.ctx)
        b = AgentSession(client=FakeGemini(), context=self.ctx)
        self.assertIs(a.context, b.context)
        self.assertIsNot(AgentSession(client=FakeGemini()).context, AgentSession(client=FakeGemini()).context)


# --- 12. Routing / tool contract --------------------------------------------------------------

class RoutingTests(NoNetworkCase):
    def run_one(self, *responses):
        gemini = FakeGemini(*responses)
        with mock.patch.object(ConversationMemory, "retrieve", autospec=True,
                               side_effect=ConversationMemory.retrieve) as retrieve:
            run_agent_turn("q", self.ctx, client=gemini)
        return gemini, retrieve

    def test_memory_never_reaches_dispatcher_and_business_never_reaches_memory(self):
        _, retrieve = self.run_one(memory_call("x"), text_response("ok"))
        retrieve.assert_called_once()
        self.dispatch.assert_not_called()
        _, retrieve = self.run_one(analyze("ANC"), text_response("ok"))
        retrieve.assert_not_called()
        self.dispatch.assert_called_once()

    def test_invalid_arguments_stay_on_their_own_side(self):
        gemini, retrieve = self.run_one(call_response((MEMORY_TOOL_NAME, {"q": 1}, "m1")), text_response("ok"))
        self.assertEqual(memory_results(gemini.requests[-1])[0]["errors"][0]["code"], "invalid_tool_arguments")
        self.dispatch.assert_not_called()
        gemini, retrieve = self.run_one(call_response(("analyze_airport", {"code": "ANC"}, "a1")), text_response("ok"))
        response = gemini.requests[-1]["contents"][-1].parts[0].function_response.response
        self.assertEqual(response["errors"][0]["code"], "invalid_tool_arguments")
        retrieve.assert_not_called()

    def test_unknown_tool_goes_to_dispatcher(self):
        gemini, retrieve = self.run_one(call_response(("retrieve_everything", {}, "x1")), text_response("ok"))
        response = gemini.requests[-1]["contents"][-1].parts[0].function_response.response
        self.assertEqual(response["errors"][0]["code"], "unknown_tool")
        retrieve.assert_not_called()

    def test_each_function_declared_exactly_once(self):
        names = [d.name for d in loop.agent_tools()[0].function_declarations]
        self.assertEqual(sorted(names), sorted(set(names)))
        self.assertEqual(names.count(MEMORY_TOOL_NAME), 1)
        self.assertEqual(sorted(n for n in names if n != MEMORY_TOOL_NAME), sorted(TOOLS))

    def test_name_collision_is_rejected_at_import(self):
        # Executed in a throwaway namespace: reloading agent.loop would replace its exception
        # classes for every other test module.
        source = compile(pathlib.Path(loop.__file__).read_text(encoding="utf-8"), loop.__file__, "exec")
        with mock.patch.dict(TOOLS, {MEMORY_TOOL_NAME: next(iter(TOOLS.values()))}):
            with self.assertRaises(RuntimeError):
                exec(source, {"__name__": "isolated_loop"})
        exec(source, {"__name__": "isolated_loop"})                         # no collision: imports fine


# --- 13. Provider message contract ------------------------------------------------------------

class ProviderContractTests(NoNetworkCase):
    def test_memory_is_function_response_data_not_spliced_history(self):
        gemini = FakeGemini(*[text_response(f"A{i}") for i in range(6)],
                            call_response((MEMORY_TOOL_NAME, {"query": "Q1"}, "m1"),
                                          ("analyze_airport", {"airport_code": "ANC"}, "a1")),
                            text_response("final"))
        session = AgentSession(client=gemini, context=self.ctx)
        for i in range(6):
            session.send(f"Q{i}")
        call_turn = gemini.responses[0].candidates[0].content
        session.send("What was Q1 about? Also analyze ANC.")
        final = gemini.requests[-1]["contents"]
        self.assertEqual(len(final), 2 * 2 + 3)                             # 2 recent turns + question, call, results
        self.assertIs(final[-2], call_turn)                                 # original call content, not rebuilt
        self.assertEqual(final[-2].parts[0].thought_signature, b"signature-1")
        self.assertEqual(final[-1].role, "user")
        self.assertEqual([(p.function_response.id, p.function_response.name) for p in final[-1].parts],
                         [("m1", MEMORY_TOOL_NAME), ("a1", "analyze_airport")])
        self.assertNotIn("Q1", texts(final[:-2]))                           # old turn not spliced as Content
        self.assertIn("Q1", json.dumps(final[-1].parts[0].function_response.response))
        for request in gemini.requests:
            for content in request["contents"]:
                self.assertIn(content.role, ("user", "model"))


# --- 14. Turn numbering ---------------------------------------------------------------------

class TurnNumberingTests(NoNetworkCase):
    def test_turn_numbers_are_absolute_and_turns_ago_counts_from_the_current_turn(self):
        turns = [text_turn(f"topic {i}", f"answer {i}") for i in range(1, 6)]   # 5 stored, 2 visible
        result = search(turns, "topic", visible=2)
        self.assertEqual([(m["turn"], m["turns_ago"]) for m in result["matches"]], [(3, 3), (2, 4), (1, 5)])
        self.assertTrue(all(m["user_message"] == f"topic {m['turn']}" for m in result["matches"]))


# --- 15. Multilingual requests: Gemini writes the memory query in English ------------------------

HEBREW_REQUEST = "מה אמרנו קודם על העיכובים ב-FAI?"


class MultilingualQueryTests(NoNetworkCase):
    def test_hebrew_request_with_english_memory_query_finds_the_turn(self):
        history = [text_turn("Why are FAI delays high?", "FAI shows elevated departure delays in winter."),
                   text_turn("מה עם ANC?", "ANC ranks first in passenger growth."),          # mixed-language turn
                   CAGR, RUNWAY, WEATHER]                                                  # last 2 visible
        gemini = FakeGemini(memory_call("FAI delays"), text_response("אמרנו ש-FAI מציג עיכובים גבוהים."))
        session = AgentSession(client=gemini, context=self.ctx)
        session._turns = list(history)
        self.assertEqual(session.send(HEBREW_REQUEST), "אמרנו ש-FAI מציג עיכובים גבוהים.")
        self.assertEqual(texts(gemini.requests[0]["contents"])[-1], HEBREW_REQUEST)         # Hebrew sent as is
        (result,) = memory_results(gemini.requests[-1])
        self.assertEqual(result["query"], "FAI delays")
        self.assertEqual([(m["turn"], m["match"]["matched_terms"]) for m in result["matches"]],
                         [(1, ["delay", "fai"])])
        self.assertEqual(len(gemini.requests), 2)                                           # no extra LLM call
        self.dispatch.assert_not_called()

    def test_english_query_finds_mixed_language_turn(self):
        history = [text_turn("מה עם העיכובים ב-FAI?", "ב-FAI יש delays גבוהים יחסית."),
                   text_turn("Compare ANC and JNU", "ANC first."), RUNWAY]
        gemini = FakeGemini(memory_call("FAI delays"), text_response("ok"))
        run_conversation_turn(history, HEBREW_REQUEST, self.ctx, client=gemini, recent_turns=1)
        (result,) = memory_results(gemini.requests[-1])
        self.assertEqual(ranked(result), [1])
        self.assertIn("ב-FAI", result["matches"][0]["user_message"])                       # original text kept

    def test_english_query_with_no_match_returns_empty_through_the_loop(self):
        gemini = FakeGemini(memory_call("Boston investment"), text_response("לא מצאתי."))
        run_conversation_turn([CAGR, RUNWAY, WEATHER], "מה אמרנו על בוסטון?", self.ctx, client=gemini,
                              recent_turns=1)
        (result,) = memory_results(gemini.requests[-1])
        self.assertEqual(result["matches"], [])

    def test_model_is_told_to_query_in_english(self):
        self.assertIn("ENGLISH", SYSTEM_INSTRUCTION)
        self.assertIn("ENGLISH", MEMORY_DECLARATION.description)
        self.assertIn("FAI delays", MEMORY_DECLARATION.description)
        self.assertIs(loop.build_config().system_instruction, SYSTEM_INSTRUCTION)
