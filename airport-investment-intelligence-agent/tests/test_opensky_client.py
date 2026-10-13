"""Offline tests for OpenSkyClient. All HTTP is mocked: these tests use zero OpenSky credits.

Run:  .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

import json
import os
import unittest
from datetime import datetime, timezone
from unittest import mock

import requests

import data_sources.opensky as opensky
from data_sources.opensky import (
    OpenSkyClient,
    OpenSkyCooldownError,
    OpenSkyCreditGuardError,
    OpenSkyRateLimitError,
)


def ts(text):
    """Unix timestamp for a UTC 'YYYY-MM-DD HH:MM:SS' string."""
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


NOW = ts("2026-10-01 12:00:00")  # frozen "current time" for all tests (mocked clock)
DAY_BEGIN = ts("2026-09-26 00:00:00")
DAY_END = ts("2026-09-26 23:59:59")


def fake_response(status=200, body=None, headers=None):
    response = requests.Response()
    response.status_code = status
    response._content = b"" if body is None else json.dumps(body).encode()
    response.headers.update(headers or {})
    response.url = opensky.API_URL
    return response


TOKEN_RESPONSE = fake_response(200, {"access_token": "test-token", "expires_in": 1800})
FLIGHT = {"icao24": "abc123", "firstSeen": DAY_BEGIN + 60, "estDepartureAirport": "PANC"}


class OpenSkyClientTests(unittest.TestCase):
    def setUp(self):
        # Safety net: any real network send anywhere in `requests` fails the test.
        self.no_network = mock.patch(
            "requests.sessions.Session.send",
            side_effect=AssertionError("Real HTTP request attempted"),
        )
        self.no_network.start()
        # Fake credentials; never read the real .env file.
        mock.patch.object(opensky, "load_dotenv").start()
        mock.patch.dict(os.environ, {"OPENSKY_CLIENT_ID": "id", "OPENSKY_CLIENT_SECRET": "secret"}).start()
        self.post = mock.patch.object(opensky.requests, "post", return_value=TOKEN_RESPONSE).start()
        self.get = mock.patch.object(opensky.requests, "get").start()
        # Mocked clock: opensky's `time` module is replaced; set self.now to move time.
        self.now = NOW
        self.clock = mock.patch.object(opensky, "time").start()
        self.clock.time.side_effect = lambda: self.now
        self.addCleanup(mock.patch.stopall)

    # --- valid request construction ---
    def test_valid_request_construction(self):
        self.get.return_value = fake_response(200, [FLIGHT], {"X-Rate-Limit-Remaining": "3910"})
        flights = OpenSkyClient().get_departures(" panc ", DAY_BEGIN, DAY_END)

        self.assertEqual(flights, [FLIGHT])
        self.get.assert_called_once_with(
            "https://opensky-network.org/api/flights/departure",
            params={"airport": "PANC", "begin": DAY_BEGIN, "end": DAY_END},
            headers={"Authorization": "Bearer test-token"},
            timeout=60,
        )
        token_call = self.post.call_args
        self.assertEqual(token_call.args[0], opensky.TOKEN_URL)
        self.assertEqual(token_call.kwargs["data"]["grant_type"], "client_credentials")

    def test_arrivals_use_arrival_endpoint(self):
        self.get.return_value = fake_response(200, [])
        OpenSkyClient().get_arrivals("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(self.get.call_args.args[0], "https://opensky-network.org/api/flights/arrival")

    # --- invalid input rejected before any HTTP ---
    def assert_rejected_without_http(self, exception, airport, begin, end):
        client = OpenSkyClient()
        with self.assertRaises(exception):
            client.get_departures(airport, begin, end)
        with self.assertRaises(exception):
            client.get_arrivals(airport, begin, end)
        self.get.assert_not_called()
        self.post.assert_not_called()  # not even a token request

    def test_begin_equal_end_rejected(self):
        self.assert_rejected_without_http(ValueError, "PANC", DAY_BEGIN, DAY_BEGIN)

    def test_begin_after_end_rejected(self):
        self.assert_rejected_without_http(ValueError, "PANC", DAY_END, DAY_BEGIN)

    def test_non_integer_times_rejected(self):
        self.assert_rejected_without_http(TypeError, "PANC", "1790380800", DAY_END)
        self.assert_rejected_without_http(TypeError, "PANC", DAY_BEGIN, float(DAY_END))

    def test_empty_airport_rejected(self):
        self.assert_rejected_without_http(ValueError, "   ", DAY_BEGIN, DAY_END)
        self.assert_rejected_without_http(ValueError, None, DAY_BEGIN, DAY_END)

    # --- interval limit: at most one UTC day boundary ---
    def test_interval_spanning_two_day_boundaries_rejected(self):
        # 23:00 on day 1 to 00:30 on day 3: only 25.5 hours, but crosses 2 UTC midnights.
        self.assert_rejected_without_http(
            ValueError, "PANC", ts("2026-09-24 23:00:00"), ts("2026-09-26 00:30:00"))

    def test_interval_longer_than_two_days_rejected(self):
        self.assert_rejected_without_http(
            ValueError, "PANC", ts("2026-09-20 00:00:00"), ts("2026-09-26 00:00:00"))

    def test_interval_with_one_day_boundary_allowed(self):
        self.get.return_value = fake_response(200, [])
        client = OpenSkyClient()
        client.get_departures("PANC", ts("2026-09-25 23:00:00"), ts("2026-09-26 01:00:00"))
        client.get_departures("PANC", ts("2026-09-25 00:00:00"), ts("2026-09-26 23:59:59"))
        self.assertEqual(self.get.call_count, 2)

    # --- token reuse ---
    def test_token_is_reused_across_requests(self):
        self.get.return_value = fake_response(200, [])
        client = OpenSkyClient()
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        client.get_arrivals("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(self.post.call_count, 1)
        self.assertEqual(self.get.call_count, 2)

    def test_token_is_refreshed_when_expired(self):
        self.get.return_value = fake_response(200, [])
        client = OpenSkyClient()
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        client._token_expires_at = 0  # simulate expiry
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(self.post.call_count, 2)

    # --- 404 means "no flights" ---
    def test_404_returns_empty_list(self):
        self.get.return_value = fake_response(404, None, {"X-Rate-Limit-Remaining": "3000"})
        client = OpenSkyClient()
        self.assertEqual(client.get_departures("PANC", DAY_BEGIN, DAY_END), [])
        self.assertEqual(client.last_rate_limit_remaining, 3000)

    def test_other_http_errors_are_raised(self):
        self.get.return_value = fake_response(500, None)
        with self.assertRaises(requests.HTTPError):
            OpenSkyClient().get_departures("PANC", DAY_BEGIN, DAY_END)

    # --- 429 ---
    def test_429_raises_rate_limit_error_with_retry_after(self):
        self.get.return_value = fake_response(429, None, {"X-Rate-Limit-Retry-After-Seconds": "3600"})
        client = OpenSkyClient()
        with self.assertRaises(OpenSkyRateLimitError) as ctx:
            client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(ctx.exception.retry_after_seconds, 3600)
        self.assertIn("3600 seconds", str(ctx.exception))
        self.assertEqual(client.last_retry_after_seconds, 3600)
        self.assertEqual(self.get.call_count, 1)  # no automatic retry

    def test_429_without_retry_header(self):
        self.get.return_value = fake_response(429, None)
        with self.assertRaises(OpenSkyRateLimitError) as ctx:
            OpenSkyClient().get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertIsNone(ctx.exception.retry_after_seconds)
        self.assertIn("did not say", str(ctx.exception))

    # --- remaining-credit parsing ---
    def test_remaining_credits_parsed_as_int(self):
        self.get.return_value = fake_response(200, [], {"X-Rate-Limit-Remaining": "3880"})
        client = OpenSkyClient()
        self.assertIsNone(client.last_rate_limit_remaining)
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(client.last_rate_limit_remaining, 3880)

    def test_missing_or_invalid_header_keeps_last_known_value(self):
        client = OpenSkyClient()
        self.get.return_value = fake_response(200, [], {"X-Rate-Limit-Remaining": "500"})
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.get.return_value = fake_response(200, [])
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(client.last_rate_limit_remaining, 500)
        self.get.return_value = fake_response(200, [], {"X-Rate-Limit-Remaining": "n/a"})
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(client.last_rate_limit_remaining, 500)

    # --- credit guard ---
    def test_first_request_allowed_when_balance_unknown(self):
        self.get.return_value = fake_response(200, [])
        client = OpenSkyClient(min_remaining_credits=10_000)
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(self.get.call_count, 1)

    def test_credit_guard_blocks_subsequent_request(self):
        self.get.return_value = fake_response(200, [], {"X-Rate-Limit-Remaining": "30"})
        client = OpenSkyClient()  # default threshold
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        with self.assertRaises(OpenSkyCreditGuardError):
            client.get_arrivals("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(self.get.call_count, 1)  # blocked before HTTP

    def test_credit_guard_allows_when_above_threshold(self):
        self.get.return_value = fake_response(200, [], {"X-Rate-Limit-Remaining": "31"})
        client = OpenSkyClient()
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        client.get_arrivals("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(self.get.call_count, 2)

    def test_credit_guard_explicit_override(self):
        self.get.return_value = fake_response(200, [], {"X-Rate-Limit-Remaining": "5"})
        client = OpenSkyClient(min_remaining_credits=100)
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        client.get_arrivals("PANC", DAY_BEGIN, DAY_END, ignore_credit_guard=True)
        self.assertEqual(self.get.call_count, 2)

    def test_credit_guard_can_be_disabled(self):
        self.get.return_value = fake_response(200, [], {"X-Rate-Limit-Remaining": "0"})
        client = OpenSkyClient(min_remaining_credits=None)
        client.get_departures("PANC", DAY_BEGIN, DAY_END)
        client.get_arrivals("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(self.get.call_count, 2)

    # --- data availability: only the previous UTC day or earlier ---
    def test_yesterday_is_allowed(self):
        self.get.return_value = fake_response(200, [])
        OpenSkyClient().get_departures("PANC", ts("2026-09-30 00:00:00"), ts("2026-09-30 23:59:59"))
        self.assertEqual(self.get.call_count, 1)

    def test_today_is_rejected(self):
        self.assert_rejected_without_http(
            ValueError, "PANC", ts("2026-10-01 00:00:00"), ts("2026-10-01 06:00:00"))

    def test_interval_ending_in_today_is_rejected(self):
        # Yesterday evening into today: begin is available, end is not.
        self.assert_rejected_without_http(
            ValueError, "PANC", ts("2026-09-30 20:00:00"), ts("2026-10-01 01:00:00"))

    def test_end_exactly_at_today_midnight_is_rejected(self):
        # [begin, end] includes `end`, and 00:00:00 already belongs to today.
        self.assert_rejected_without_http(
            ValueError, "PANC", ts("2026-09-30 00:00:00"), ts("2026-10-01 00:00:00"))

    def test_future_is_rejected(self):
        self.assert_rejected_without_http(
            ValueError, "PANC", ts("2026-10-05 00:00:00"), ts("2026-10-05 23:59:59"))

    def test_utc_boundary_just_after_midnight(self):
        # 30 s after UTC midnight, the day that just ended is now "yesterday".
        self.now = ts("2026-10-01 00:00:30")
        self.get.return_value = fake_response(200, [])
        OpenSkyClient().get_departures("PANC", ts("2026-09-30 00:00:00"), ts("2026-09-30 23:59:59"))
        self.assertEqual(self.get.call_count, 1)

    def test_utc_boundary_just_before_midnight(self):
        # 1 s before UTC midnight, that same day is still "today".
        self.now = ts("2026-09-30 23:59:59")
        self.assert_rejected_without_http(
            ValueError, "PANC", ts("2026-09-30 00:00:00"), ts("2026-09-30 12:00:00"))

    # --- 429 cooldown ---
    def test_429_with_retry_after_blocks_locally_until_time_passes(self):
        client = OpenSkyClient()
        self.get.return_value = fake_response(429, None, {"X-Rate-Limit-Retry-After-Seconds": "3600"})
        with self.assertRaises(OpenSkyRateLimitError):
            client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(client.blocked_until, NOW + 3600)
        self.assertEqual((self.get.call_count, self.post.call_count), (1, 1))

        # 10 seconds later: blocked locally, no token or flight request.
        self.now = NOW + 10
        client._token = None  # would force a token request if the cooldown let it through
        with self.assertRaises(OpenSkyCooldownError) as ctx:
            client.get_arrivals("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(ctx.exception.retry_after_seconds, 3590)
        self.assertIn("blocked locally", str(ctx.exception))
        self.assertIsInstance(ctx.exception, OpenSkyRateLimitError)
        self.assertEqual((self.get.call_count, self.post.call_count), (1, 1))

        # Credit-guard override does not bypass the cooldown.
        with self.assertRaises(OpenSkyCooldownError):
            client.get_arrivals("PANC", DAY_BEGIN, DAY_END, ignore_credit_guard=True)
        self.assertEqual(self.get.call_count, 1)

        # After the wait: the request is sent again and the block is cleared.
        self.now = NOW + 3600
        self.get.return_value = fake_response(200, [])
        self.assertEqual(client.get_arrivals("PANC", DAY_BEGIN, DAY_END), [])
        self.assertEqual(self.get.call_count, 2)
        self.assertIsNone(client.blocked_until)

    def test_429_without_retry_after_sets_no_cooldown(self):
        client = OpenSkyClient()
        self.get.return_value = fake_response(429, None)
        with self.assertRaises(OpenSkyRateLimitError) as ctx:
            client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertNotIsInstance(ctx.exception, OpenSkyCooldownError)
        self.assertIsNone(client.blocked_until)
        # Next request is sent (no invented wait period) and gets the server's 429 again.
        with self.assertRaises(OpenSkyRateLimitError):
            client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.assertEqual(self.get.call_count, 2)

    def test_no_sleep_or_automatic_retry(self):
        client = OpenSkyClient()
        self.get.return_value = fake_response(429, None, {"X-Rate-Limit-Retry-After-Seconds": "60"})
        with self.assertRaises(OpenSkyRateLimitError):
            client.get_departures("PANC", DAY_BEGIN, DAY_END)
        with self.assertRaises(OpenSkyCooldownError):
            client.get_departures("PANC", DAY_BEGIN, DAY_END)
        self.clock.sleep.assert_not_called()
        self.assertEqual(self.get.call_count, 1)

    def test_network_safety_net_is_active(self):
        with self.assertRaises(AssertionError):
            requests.Session().get("https://opensky-network.org/api/flights/departure")


if __name__ == "__main__":
    unittest.main()
