"""Offline tests for the agent tool registry and dispatcher (agent/tools.py). No network."""

import importlib
import inspect
import json
import unittest
from unittest import mock

import agent.tools as agent_tools_module
import agent_tools
from agent import TOOLS, RegisteredTool, execute_tool, gemini_tools, tool_declarations
from agent_tools import AUTOMATIC_ANALYSIS_LIMIT, AgentToolContext
from tests.test_agent_tools import ToolTestCase

EXPECTED = {"resolve_airports", "discover_airports", "compare_airports", "analyze_airport"}


class NoNetwork(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("requests.sessions.Session.send",
                             side_effect=AssertionError("Real HTTP request attempted"))
        patcher.start()
        self.addCleanup(patcher.stop)
        # A real AgentToolContext whose clients are never used by the fakes below.
        self.context = AgentToolContext(ourairports=mock.Mock(), t100=mock.Mock(), ontime=mock.Mock())


def fake_registry_entry(name):
    """Replace a registered tool's function by an autospec fake (same signature)."""
    real = TOOLS[name]
    fake = mock.create_autospec(real.function, return_value={"tool": name, "status": "ok"})
    return mock.patch.dict(TOOLS, {name: RegisteredTool(function=fake, declaration=real.declaration)}), fake


class RegistryTests(NoNetwork):
    def test_four_business_tools_registered(self):
        self.assertEqual(set(TOOLS), EXPECTED)
        self.assertEqual({d.name for d in tool_declarations()}, EXPECTED)
        self.assertEqual({TOOLS[name].function for name in EXPECTED},
                         {agent_tools.resolve_airports, agent_tools.discover_airports, agent_tools.compare_airports,
                          agent_tools.analyze_airport})

    def test_name_schema_and_callable_agree(self):
        for name, tool in TOOLS.items():
            self.assertEqual((tool.name, tool.function.__name__), (name, name))
            signature = inspect.signature(tool.function)
            self.assertTrue(tool.parameters <= set(signature.parameters))
            self.assertEqual(signature.parameters["context"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_context_is_never_llm_visible(self):
        for declaration in tool_declarations():
            schema = declaration.parameters_json_schema
            self.assertNotIn("context", schema["properties"])
            self.assertNotIn("context", schema["required"])
        text = json.dumps([d.model_dump(exclude_none=True) for d in tool_declarations()])
        self.assertNotIn('"context":', text)          # no schema property of that name anywhere
        self.assertNotIn("AgentToolContext", text)    # internal type never mentioned to the LLM

    def test_schema_arguments_and_required(self):
        expected = {"resolve_airports": ({"references"}, {"references"}),
                    "discover_airports": ({"country", "states"}, {"country"}),
                    "compare_airports": ({"airport_codes", "year", "month", "allow_large_group"}, {"airport_codes"}),
                    "analyze_airport": ({"airport_code", "year", "month"}, {"airport_code"})}
        for name, (parameters, required) in expected.items():
            self.assertEqual((TOOLS[name].parameters, TOOLS[name].required), (parameters, required))

    def test_compare_description_uses_project_limit_constant(self):
        description = TOOLS["compare_airports"].declaration.description
        self.assertIn(str(AUTOMATIC_ANALYSIS_LIMIT), description)
        self.assertIn("allow_large_group=true ONLY after the user has explicitly authorised", description)
        try:   # rebuild the registry with a different limit: the description must follow the constant
            with mock.patch.object(agent_tools, "AUTOMATIC_ANALYSIS_LIMIT", 7):
                reloaded = importlib.reload(agent_tools_module)
                description = reloaded.TOOLS["compare_airports"].declaration.description
                self.assertIn("Up to 7 airports", description)
                self.assertNotIn(str(AUTOMATIC_ANALYSIS_LIMIT), description)
        finally:
            importlib.reload(agent_tools_module)

    def test_analyze_and_discover_descriptions_state_their_limits(self):
        self.assertIn("does NOT return an Investment Opportunity Score or rank",
                      TOOLS["analyze_airport"].declaration.description)
        self.assertIn("does NOT rank", TOOLS["discover_airports"].declaration.description)
        self.assertIn("New England", TOOLS["discover_airports"].declaration.description)

    def test_gemini_tool_wrapper(self):
        (tool,) = gemini_tools()
        self.assertEqual({d.name for d in tool.function_declarations}, EXPECTED)

    def test_contract_check_rejects_drifting_schemas(self):
        def no_context(airport_code):
            return {}

        def with_context(airport_code, *, context=None):
            return {}

        with self.assertRaises(TypeError):
            agent_tools_module.register(no_context, "x", {"airport_code": {"type": "string"}})
        with self.assertRaises(TypeError):
            agent_tools_module.register(with_context, "x", {"context": {"type": "string"}})
        with self.assertRaises(TypeError):
            agent_tools_module.register(with_context, "x", {"not_a_parameter": {"type": "string"}})


class DispatchTests(NoNetwork):
    def test_arguments_forwarded_and_same_context_injected(self):
        patcher, fake = fake_registry_entry("compare_airports")
        with patcher:
            result = execute_tool("compare_airports", {"airport_codes": ["ANC", "FAI"], "year": 2026, "month": 4},
                                  self.context)
        self.assertEqual(result, {"tool": "compare_airports", "status": "ok"})
        fake.assert_called_once_with(airport_codes=["ANC", "FAI"], year=2026, month=4, context=self.context)
        self.assertIs(fake.call_args.kwargs["context"], self.context)       # the supplied object, not a new one

    def test_each_registered_tool_is_dispatched(self):
        calls = {"resolve_airports": {"references": ["Anchorage"]},
                 "discover_airports": {"country": "US", "states": ["AK"]},
                 "compare_airports": {"airport_codes": ["ANC"], "allow_large_group": False},
                 "analyze_airport": {"airport_code": "ANC"}}
        for name, arguments in calls.items():
            patcher, fake = fake_registry_entry(name)
            with patcher:
                execute_tool(name, arguments, self.context)
            fake.assert_called_once_with(**arguments, context=self.context)

    def test_context_is_required(self):
        for bad in (None, {}, "context"):
            with self.assertRaises(TypeError):
                execute_tool("analyze_airport", {"airport_code": "ANC"}, bad)

    def test_unknown_tool(self):
        for name in ("drop_tables", "os.system", "__import__", "analyze_airports", None, 42):
            result = execute_tool(name, {}, self.context)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["errors"][0]["code"], "unknown_tool")
            self.assertEqual(result["errors"][0]["details"]["available_tools"], sorted(EXPECTED))

    def test_malformed_arguments(self):
        patcher, fake = fake_registry_entry("analyze_airport")
        with patcher:
            for bad in (["ANC"], "ANC", 42, ("airport_code", "ANC")):
                result = execute_tool("analyze_airport", bad, self.context)
                self.assertEqual(result["errors"][0]["code"], "invalid_tool_arguments")
                self.assertEqual(result["errors"][0]["details"]["received_type"], type(bad).__name__)
        fake.assert_not_called()

    def test_unexpected_and_missing_arguments(self):
        patcher, fake = fake_registry_entry("compare_airports")
        with patcher:
            cases = [
                ({"airport_codes": ["ANC"], "context": "attacker"}, ["context"], []),
                ({"airport_codes": ["ANC"], "__class__": 1, "eval": "x"}, ["__class__", "eval"], []),
                ({"airport_codes": ["ANC"], 1: "non-string key"}, ["1"], []),
                ({"year": 2026}, [], ["airport_codes"]),
                (None, [], ["airport_codes"]),
            ]
            for arguments, unexpected, missing in cases:
                result = execute_tool("compare_airports", arguments, self.context)
                self.assertEqual(result["status"], "error")
                error = result["errors"][0]
                self.assertEqual(error["code"], "invalid_tool_arguments")
                self.assertEqual((error["details"]["unexpected"], error["details"]["missing"]), (unexpected, missing))
        fake.assert_not_called()                                           # nothing executed

    def test_tool_exceptions_are_not_hidden(self):
        patcher, fake = fake_registry_entry("analyze_airport")
        fake.side_effect = ZeroDivisionError("bug inside a tool")
        with patcher, self.assertRaises(ZeroDivisionError):
            execute_tool("analyze_airport", {"airport_code": "ANC"}, self.context)


class DispatchIntegrationTests(ToolTestCase):
    """The real business tools via the dispatcher, on the synthetic offline fixtures."""

    def test_real_tools_run_with_the_injected_session_context(self):
        result = execute_tool("compare_airports", {"airport_codes": ["ANC", "FAI", "JNU"]}, self.ctx)
        self.assertEqual((result["tool"], result["status"]), ("compare_airports", "ok"))
        self.assertEqual(result["metadata"]["analyzed_count"], 3)
        detail = execute_tool("analyze_airport", {"airport_code": "FAI"}, self.ctx)
        self.assertEqual(detail["airport"]["code"], "FAI")
        discovered = execute_tool("discover_airports", {"country": "US", "states": ["AK"]}, self.ctx)
        self.assertEqual(discovered["candidate_count"], 6)
        self.assertEqual(len(self.ontime.calls), 2)          # the session's own (fake) clients were used

    def test_tool_level_validation_still_applies(self):
        result = execute_tool("compare_airports", {"airport_codes": "ANC"}, self.ctx)   # wrong type
        self.assertEqual(result["errors"][0]["code"], "invalid_argument")


if __name__ == "__main__":
    unittest.main()
