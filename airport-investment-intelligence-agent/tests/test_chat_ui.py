"""Offline tests for the Chainlit chat layer (chat_support.py, chat_app.py).

No browser, no Chainlit server, no Gemini: friendly error mapping, the safe execution summary,
and the chat handlers driven with a fake AgentSession and a fake Chainlit session store.
"""

import asyncio
import importlib
import socket
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
import requests
from google.genai import errors as genai_errors

import chat_support
from agent import AgentError, EmptyModelResponse, FinalizationFailed
from agent.trace import TurnRecorder
from chat_support import (BUSY, NO_ANSWER, NOT_CONFIGURED, TOOL_LABELS, UNAVAILABLE, UNEXPECTED, execution_summary,
                          friendly_error)

SECRET = "SECRET-DETAIL api_key=AIza-not-real /internal/path"


def api_error(code, status):
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "status": status, "message": SECRET}})


class FriendlyErrorTests(unittest.TestCase):
    def test_mapping(self):
        cases = [
            (api_error(429, "RESOURCE_EXHAUSTED"), BUSY),
            (api_error(503, "UNAVAILABLE"), UNAVAILABLE),
            (api_error(500, "INTERNAL"), UNAVAILABLE),
            (api_error(502, "BAD_GATEWAY"), UNAVAILABLE),
            (api_error(504, "DEADLINE_EXCEEDED"), UNAVAILABLE),
            (api_error(408, "REQUEST_TIMEOUT"), UNAVAILABLE),
            (httpx.ConnectError(SECRET), UNAVAILABLE),
            (requests.ConnectionError(SECRET), UNAVAILABLE),
            (TimeoutError(SECRET), UNAVAILABLE),
            (EmptyModelResponse(SECRET), NO_ANSWER),
            (FinalizationFailed(SECRET), NO_ANSWER),
            (AgentError("GEMINI_API_KEY was not found in .env"), NOT_CONFIGURED),
            (api_error(400, "INVALID_ARGUMENT"), NOT_CONFIGURED),
            (api_error(401, "UNAUTHENTICATED"), NOT_CONFIGURED),
            (api_error(403, "PERMISSION_DENIED"), NOT_CONFIGURED),
            (api_error(404, "NOT_FOUND"), NOT_CONFIGURED),
            (api_error(418, "UNKNOWN"), UNEXPECTED),
            (ValueError(SECRET), UNEXPECTED),
            (KeyError(SECRET), UNEXPECTED),
            (RuntimeError(SECRET), UNEXPECTED),
        ]
        for exc, expected in cases:
            self.assertEqual(friendly_error(exc), expected, repr(exc))

    def test_never_returns_exception_text(self):
        messages = {BUSY, UNAVAILABLE, NO_ANSWER, NOT_CONFIGURED, UNEXPECTED}
        for exc in (api_error(429, "RESOURCE_EXHAUSTED"), api_error(403, "PERMISSION_DENIED"), ValueError(SECRET),
                    EmptyModelResponse(SECRET), AgentError(SECRET)):
            text = friendly_error(exc)
            self.assertIn(text, messages)
            self.assertNotIn("SECRET", text)
            self.assertNotIn("AIza", text)
            self.assertNotIn(str(exc), text)


def trace(*calls):
    """A real TurnTrace with the given (name, arguments, status) tool calls."""
    recorder = TurnRecorder()
    for name, arguments, status in calls:
        call = SimpleNamespace(name=name, args=arguments, id="x")
        recorder.tool_called(call, 0.01, {"status": status, "errors": []} if status != "exception" else None,
                             exception=RuntimeError(SECRET) if status == "exception" else None)
    return recorder.finish()


class ExecutionSummaryTests(unittest.TestCase):
    def test_labels_in_order_with_counts(self):
        lines = execution_summary(trace(("resolve_airports", {"references": ["Anchorage"]}, "ok"),
                                        ("compare_airports", {"airport_codes": ["ANC", "FAI"]}, "ok"),
                                        ("analyze_airport", {"airport_code": "ANC"}, "ok"),
                                        ("analyze_airport", {"airport_code": "FAI"}, "ok")))
        self.assertEqual(lines, ["Resolved airport", "Compared airports", "Analyzed airport (2x)"])

    def test_every_label(self):
        self.assertEqual(TOOL_LABELS, {"resolve_airports": "Resolved airport", "compare_airports": "Compared airports",
                                       "analyze_airport": "Analyzed airport",
                                       "discover_airports": "Found airports in the region",
                                       "retrieve_conversation_context": "Checked earlier conversation"})
        for name, label in TOOL_LABELS.items():
            self.assertEqual(execution_summary(trace((name, {}, "ok"))), [label])

    def test_nothing_sensitive_is_exposed(self):
        lines = execution_summary(trace(
            ("retrieve_conversation_context", {"query": "SECRET memory query"}, "ok"),
            ("resolve_airports", {"references": ["SECRET reference"]}, "error"),
            ("analyze_airport", {"airport_code": "SECRET"}, "exception"),
            ("drop_tables", {"sql": "SECRET"}, "error")))
        self.assertEqual(lines, ["Checked earlier conversation", "Resolved airport - not completed",
                                 "Analyzed airport - not completed"])
        self.assertNotIn("SECRET", " ".join(lines))

    def test_no_tools_or_no_trace(self):
        self.assertEqual(execution_summary(None), [])
        self.assertEqual(execution_summary(trace()), [])


class FakeSession:
    def __init__(self, answer="Answer.", error=None):
        self.answer, self.error, self.messages = answer, error, []
        self.last_trace = trace(("resolve_airports", {}, "ok"), ("analyze_airport", {}, "ok"))

    def send(self, message):
        self.messages.append(message)
        if self.error:
            raise self.error
        return self.answer


class ChatHandlerTests(unittest.TestCase):
    """chat_app handlers with Chainlit's session store, steps and messages replaced by fakes."""

    @classmethod
    def setUpClass(cls):
        with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network used")):
            cls.app = importlib.import_module("chat_app")

    def setUp(self):
        self.store, self.sent, self.steps = {}, [], []
        app, test = self.app, self

        class Step:
            def __init__(self, name, **kw):
                self.name, self.kw, self.output, self.removed = name, kw, None, False
                test.steps.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def remove(self):
                self.removed = True

        class Message:
            def __init__(self, content):
                self.content = content

            async def send(self):
                test.sent.append(self.content)

        user_session = SimpleNamespace(set=self.store.__setitem__, get=self.store.get)
        for name, value in (("user_session", user_session), ("Step", Step), ("Message", Message),
                            ("make_async", lambda func: (lambda *a: asyncio.to_thread(func, *a)))):
            patcher = mock.patch.object(app.cl, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_async(self, coro):
        return asyncio.run(coro)

    def test_chat_start_creates_one_session_and_lock(self):
        with mock.patch.object(self.app, "AgentSession") as factory:
            self.run_async(self.app.start_chat())
        factory.assert_called_once_with()
        self.assertIs(self.store["agent_session"], factory.return_value)
        self.assertIsInstance(self.store["agent_lock"], asyncio.Lock)

    def test_separate_chats_get_separate_sessions(self):
        self.run_async(self.app.start_chat())
        first = self.store["agent_session"]
        self.store.clear()
        self.run_async(self.app.start_chat())
        self.assertIsNot(self.store["agent_session"], first)

    def test_message_goes_to_the_same_session_and_answer_is_sent(self):
        session = FakeSession("ANC ranks first.")
        self.store.update(agent_session=session, agent_lock=asyncio.Lock())
        for text in ("Compare Anchorage and Fairbanks", "And Juneau?"):
            self.run_async(self.app.on_message(SimpleNamespace(content=text)))
        self.assertEqual(session.messages, ["Compare Anchorage and Fairbanks", "And Juneau?"])
        self.assertEqual(self.sent, ["ANC ranks first.", "ANC ranks first."])
        summary = [s for s in self.steps if s.name == "Execution summary"][0]
        self.assertEqual(summary.output, "- Resolved airport\n- Analyzed airport")
        self.assertFalse(summary.kw["default_open"])

    def test_working_step_shown_while_running_and_removed_after_success(self):
        self.store.update(agent_session=FakeSession("Answer."), agent_lock=asyncio.Lock())
        self.run_async(self.app.on_message(SimpleNamespace(content="Hi")))
        working, summary = self.steps
        self.assertEqual(working.name, "Working on your question")       # the in-progress indicator
        self.assertTrue(working.removed)                                  # gone once the answer is there
        self.assertIsNone(working.output)                                 # no "Done." text
        self.assertEqual(summary.name, "Execution summary")
        self.assertFalse(summary.removed)                                 # stays, collapsed
        self.assertEqual(self.sent, ["Answer."])

    def test_error_shows_only_friendly_text(self):
        session = FakeSession(error=api_error(429, "RESOURCE_EXHAUSTED"))
        self.store.update(agent_session=session, agent_lock=asyncio.Lock())
        with self.assertLogs("airport_agent.chat", level="ERROR") as logs:
            self.run_async(self.app.on_message(SimpleNamespace(content="Hi")))
        self.assertEqual(self.sent, [BUSY])
        visible = " ".join(self.sent + [str(s.output) for s in self.steps])
        self.assertNotIn("SECRET", visible)
        self.assertIn("SECRET", "\n".join(logs.output))                 # full detail stays server-side
        self.assertEqual([s.name for s in self.steps], ["Working on your question"])   # no summary on failure
        self.assertFalse(self.steps[0].removed)                           # error behaviour unchanged
        self.assertEqual(self.steps[0].output, "Stopped.")

    def test_unexpected_error(self):
        self.store.update(agent_session=FakeSession(error=ValueError(SECRET)), agent_lock=asyncio.Lock())
        with self.assertLogs("airport_agent.chat", level="ERROR"):
            self.run_async(self.app.on_message(SimpleNamespace(content="Hi")))
        self.assertEqual(self.sent, [UNEXPECTED])

    def test_stop_keeps_the_chat_locked_until_the_running_turn_ends(self):
        import threading
        release, started = threading.Event(), threading.Event()

        class SlowSession(FakeSession):
            def send(self, message):
                started.set()
                release.wait(5)
                return super().send(message)

        session, lock = SlowSession("late answer"), asyncio.Lock()
        self.store.update(agent_session=session, agent_lock=lock)

        async def scenario():
            task = asyncio.ensure_future(self.app.on_message(SimpleNamespace(content="first")))
            await asyncio.to_thread(started.wait, 5)
            task.cancel()                                   # the user pressed Stop
            await asyncio.sleep(0.05)
            locked_while_running = lock.locked()
            release.set()                                   # the worker thread finishes its turn
            with self.assertRaises(asyncio.CancelledError):
                await task
            return locked_while_running, lock.locked()

        locked_while_running, locked_after = self.run_async(scenario())
        self.assertTrue(locked_while_running)               # a second message would have waited
        self.assertFalse(locked_after)
        self.assertEqual(self.sent, [])                     # the cancelled turn's answer is not posted

    def test_app_uses_the_existing_agent_session(self):
        from agent import AgentSession
        self.assertIs(self.app.AgentSession, AgentSession)


class StaticConfigTests(unittest.TestCase):
    """Presentation configuration: title, language, branding assets, copy and starters."""

    @classmethod
    def setUpClass(cls):
        import json
        import tomllib
        from pathlib import Path
        cls.root = Path(chat_support.__file__).parent
        cls.config = tomllib.loads((cls.root / ".chainlit" / "config.toml").read_text(encoding="utf-8"))
        cls.english = json.loads((cls.root / ".chainlit" / "translations" / "en-US.json").read_text(encoding="utf-8"))

    def test_title_features_and_english(self):
        ui = self.config["UI"]
        self.assertEqual(ui["name"], "Airport Investment Intelligence Agent")
        self.assertEqual(ui["language"], "en-US")                      # not the browser locale
        self.assertNotIn("default_theme", ui)                           # dark/light toggle kept
        self.assertNotIn("custom_css", ui)
        self.assertNotIn("custom_js", ui)
        self.assertFalse(self.config["features"]["spontaneous_file_upload"]["enabled"])
        self.assertFalse(self.config["features"]["unsafe_allow_html"])
        self.assertFalse(self.config["features"]["edit_message"])

    def test_placeholder_and_header_label(self):
        self.assertEqual(self.english["chat"]["input"]["placeholder"], "Ask about an airport or region...")
        self.assertEqual(self.english["navigation"]["header"]["readme"], "About")

    def test_branding_assets_replace_chainlit_defaults(self):
        import xml.etree.ElementTree as ET
        public = self.root / "public"
        # file names Chainlit looks up: /logo -> public/logo_<theme>.*, /favicon -> public/favicon.*,
        # assistant avatar -> public/avatars/<ui name, lower case, spaces -> "_">.*
        avatar = self.config["UI"]["name"].strip().lower().replace(" ", "_").replace(".", "_")
        for name in ("logo_light.svg", "logo_dark.svg", "favicon.svg", f"avatars/{avatar}.svg"):
            svg = ET.parse(public / name).getroot()
            self.assertTrue(svg.tag.endswith("svg"), name)
        for theme in ("light", "dark"):
            logo = ET.parse(public / f"logo_{theme}.svg").getroot()
            self.assertEqual(logo.get("aria-label"), "Airport Investment Intelligence Agent")
        self.assertNotEqual((public / "logo_light.svg").read_text(encoding="utf-8"),
                            (public / "logo_dark.svg").read_text(encoding="utf-8"))

    def test_welcome_copy(self):
        text = (self.root / "chainlit.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Airport Investment Intelligence Agent"))
        self.assertIn("AI-powered analysis of U.S. airports using aviation data.", text)
        for topic in ("passenger growth", "operational pressure", "connectivity", "relative"):
            self.assertIn(topic, text)
        self.assertLess(len(text.split()), 80)                          # short

    def test_starters_are_natural_questions(self):
        self.assertEqual([message for _, message in chat_support.STARTERS],
                         ["Compare Anchorage, Fairbanks and Juneau for airport investment.",
                          "Which New England airports show the strongest investment signals?",
                          "What percentage of flights from Anchorage are long-haul?"])
        for label, message in chat_support.STARTERS:
            self.assertEqual(label, message)

    def test_starter_handler_returns_them(self):
        app = importlib.import_module("chat_app")
        starters = asyncio.run(app.starters())
        self.assertEqual([(s.label, s.message) for s in starters], list(chat_support.STARTERS))


if __name__ == "__main__":
    unittest.main()
