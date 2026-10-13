"""Offline tests for validate_sources.py command-line behavior. No network requests are made.

Run:  .venv/Scripts/python.exe -m unittest discover -s tests -t . -v
"""

import contextlib
import io
import os
import unittest
from unittest import mock

import validate_sources

SOURCE_TESTS = [
    "test_ourairports_airport",
    "test_ourairports_runways",
    "test_bts_t100",
    "test_bts_ontime",
]


class ValidateSourcesCliTests(unittest.TestCase):
    def setUp(self):
        # Safety net: any real network send fails the test.
        mock.patch(
            "requests.sessions.Session.send",
            side_effect=AssertionError("Real HTTP request attempted"),
        ).start()
        # Replace every validation with a stub, so only main()'s choices are tested.
        self.source_stubs = [
            mock.patch.object(validate_sources, name, return_value=True).start()
            for name in SOURCE_TESTS
        ]
        self.opensky_test = mock.patch.object(validate_sources, "test_opensky", return_value=True).start()
        self.opensky_client = mock.patch.object(validate_sources, "OpenSkyClient").start()
        mock.patch.object(validate_sources, "load_dotenv").start()  # never read the real .env
        self.addCleanup(mock.patch.stopall)

    def run_main(self, argv, env):
        with mock.patch.dict(os.environ, env, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            results = validate_sources.main(argv)
        return dict(results), out.getvalue()

    # --- argument parsing ---
    def test_parse_args_default(self):
        self.assertFalse(validate_sources.parse_args([]).opensky)

    def test_parse_args_opensky_flag(self):
        self.assertTrue(validate_sources.parse_args(["--opensky"]).opensky)

    def test_parse_args_rejects_unknown_argument(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            validate_sources.parse_args(["--bogus"])

    # --- default: no OpenSky, even when credentials exist ---
    def test_default_skips_opensky_even_with_credentials(self):
        results, output = self.run_main([], {"OPENSKY_CLIENT_ID": "id", "OPENSKY_CLIENT_SECRET": "secret"})
        for stub in self.source_stubs:
            stub.assert_called_once()
        self.opensky_test.assert_not_called()
        self.opensky_client.assert_not_called()
        self.assertEqual(results["OpenSky flights"], "SKIPPED (use --opensky to validate)")
        self.assertIn("use --opensky to validate", output)

    # --- --opensky ---
    def test_opensky_flag_without_credentials_skips(self):
        results, output = self.run_main(["--opensky"], {})
        self.opensky_test.assert_not_called()
        self.assertEqual(results["OpenSky flights"], "SKIPPED (credentials missing)")
        self.assertIn("not set in .env", output)

    def test_opensky_flag_with_credentials_runs_opensky_validation(self):
        results, _ = self.run_main(["--opensky"], {"OPENSKY_CLIENT_ID": "id", "OPENSKY_CLIENT_SECRET": "secret"})
        for stub in self.source_stubs:
            stub.assert_called_once()
        self.opensky_test.assert_called_once()
        self.assertEqual(results["OpenSky flights"], "PASS")

    def test_results_do_not_accumulate_between_runs(self):
        env = {"OPENSKY_CLIENT_ID": "id", "OPENSKY_CLIENT_SECRET": "secret"}
        self.run_main([], env)
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
            results = validate_sources.main([])
        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
