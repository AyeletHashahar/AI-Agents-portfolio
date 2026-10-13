"""Offline checks of the instruction CONTRACT given to Gemini: tool-selection guidance between
compare_airports and analyze_airport, and grounding of airport-specific facts.

These tests verify what the model is told (system instruction and tool declarations), not
that the real model obeys it; that can only be observed in live evaluation.
"""

import unittest

import agent.loop as loop
from agent import SYSTEM_INSTRUCTION, TOOLS, tool_declarations
from agent_tools import TOP_K


def description(name):
    return TOOLS[name].declaration.description


class RedundantAnalysisGuidanceTests(unittest.TestCase):
    def test_compare_description_says_its_result_already_answers_comparisons(self):
        text = description("compare_airports")
        self.assertIn("Use it for questions that compare airports", text)
        self.assertIn("already contains the comparison-level metrics needed to answer them", text)
        self.assertIn(f"top {TOP_K} airports", text)            # honest about what is (not) per-airport detail

    def test_analyze_description_forbids_re_fetching_compared_metrics(self):
        text = description("analyze_airport")
        self.assertIn("After compare_airports, do NOT call it for an airport included in that comparison merely to "
                      "retrieve information already present in the comparison result", text)

    def test_analyze_description_keeps_the_two_legitimate_uses(self):
        text = description("analyze_airport")
        self.assertIn("an airport that was not in the previous comparison", text)
        self.assertIn("requires airport-level information the comparison result did not provide", text)

    def test_system_instruction_states_the_general_rule_once(self):
        self.assertIn("Do not re-fetch evidence you already have", SYSTEM_INSTRUCTION)
        self.assertIn("analyze_airport after compare_airports", SYSTEM_INSTRUCTION)

    def test_rule_is_guidance_not_a_dispatcher_restriction(self):
        # analyze_airport stays callable after compare_airports: no execution-time block exists.
        self.assertNotIn("compare_airports", TOOLS["analyze_airport"].function.__code__.co_names)
        self.assertEqual(TOOLS["analyze_airport"].required, {"airport_code"})


class AirportFactGroundingTests(unittest.TestCase):
    def test_airport_specific_facts_must_come_from_tools(self):
        text = SYSTEM_INSTRUCTION
        self.assertIn("AIRPORT-SPECIFIC FACTUAL CLAIMS MUST BE GROUNDED IN TOOL RESULTS", text)
        for example in ("runway use or restrictions", "terminals", "route distances", "airline operations",
                        "infrastructure", "operational rules", "congestion causes",
                        "historical or geographic facts used as evidence"):
            self.assertIn(example, text, example)

    def test_unsupported_facts_are_omitted_or_labelled_never_numbers(self):
        text = SYSTEM_INSTRUCTION
        self.assertIn("omit it if unnecessary", text)
        self.assertIn("label it an unverified hypothesis, not evidence", text)
        self.assertIn("never invent a number", text)

    def test_background_knowledge_only_for_concepts_and_methodology(self):
        self.assertIn("clearly labelled context for general concepts and methodology", SYSTEM_INSTRUCTION)
        self.assertIn("never for airport-specific facts", SYSTEM_INSTRUCTION)

    def test_existing_grounding_rules_preserved(self):
        for rule in ("domestic scheduled passenger flights of reporting carriers",
                     "Capacity Pressure is a PROXY",
                     "Do not claim actual terminal capacity or utilisation",
                     "unmet passenger demand", "investment returns", "financing attractiveness",
                     "Base factual claims about the analysed airports on tool results",
                     "Never state that a metric was observed or calculated unless a tool returned it"):
            self.assertIn(rule, SYSTEM_INSTRUCTION, rule)

    def test_finalization_uses_the_same_instruction(self):
        self.assertIs(loop.build_finalization_config().system_instruction, SYSTEM_INSTRUCTION)


class ModelTests(unittest.TestCase):
    def test_production_model(self):
        from agent import AgentSession
        self.assertEqual(loop.MODEL, "gemini-3.5-flash-lite")
        self.assertEqual(AgentSession(client=object()).model, loop.MODEL)     # sessions default to it


class ToolSurfaceTests(unittest.TestCase):
    def test_exactly_the_four_business_tools(self):
        expected = {"discover_airports", "resolve_airports", "analyze_airport", "compare_airports"}
        self.assertEqual(set(TOOLS), expected)
        self.assertEqual({d.name for d in tool_declarations()}, expected)
        gemini = {d.name for d in loop.agent_tools()[0].function_declarations}
        self.assertEqual(gemini, expected | {"retrieve_conversation_context"})


if __name__ == "__main__":
    unittest.main()
