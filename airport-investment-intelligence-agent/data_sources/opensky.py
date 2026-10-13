"""OpenSky Network: flights observed departing from / arriving at an airport.

Docs: https://openskynetwork.github.io/opensky-api/rest.html
Official Python client: https://github.com/openskynetwork/opensky-api (python/opensky_api.py)

- Authentication: OAuth2 client credentials (required; anonymous /flights requests got 403).
  Set OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET in a local .env file (never commit it).
- Airports are identified by ICAO code (e.g. 'PANC', not 'ANC').
- Flight data is built by a nightly batch: only the previous day or earlier is available.
  Requests whose end is at or after 00:00 UTC today are rejected locally. (The docs do not
  say at what time the nightly batch finishes, so very recent data may still be missing.)
- Interval limit (official Python client, same rule for arrivals and departures):
  begin < end, and begin/end fall on the same UTC calendar day or span at most one
  UTC day boundary. Invalid requests are rejected locally, before any HTTP call.
- Credits (REST docs, "API Credits"): /flights/* has its own credit bucket. The response
  header X-Rate-Limit-Remaining shows the remaining balance; when credits are exhausted
  the API returns 429 and X-Rate-Limit-Retry-After-Seconds says how long to wait.
  After a 429 with that header, this client blocks further flight requests locally until
  the wait is over (no automatic sleep or retry).
- Times (firstSeen / lastSeen) are when the network first/last saw the aircraft,
  NOT scheduled or gate times. Airports are estimates and can be null.
"""

import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

API_URL = "https://opensky-network.org/api"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)

# Default credit guard threshold. The documented /flights cost table charges 30 credits
# for "1 - 2" day partitions; any interval allowed by _validate_request touches at most
# two UTC calendar days, so 30 is the highest documented cost of one allowed request.
DEFAULT_MIN_REMAINING_CREDITS = 30


class OpenSkyRateLimitError(Exception):
    """OpenSky returned 429: the credit quota / rate limit for /flights is exhausted."""

    def __init__(self, retry_after_seconds=None):
        self.retry_after_seconds = retry_after_seconds
        message = "OpenSky quota/rate limit reached (HTTP 429)."
        if retry_after_seconds is not None:
            message += f" Another request may be attempted in {retry_after_seconds} seconds."
        else:
            message += " OpenSky did not say when another request may be attempted."
        super().__init__(message)


class OpenSkyCooldownError(OpenSkyRateLimitError):
    """Request blocked locally because an earlier 429 asked us to wait (not sent to OpenSky)."""

    def __init__(self, retry_after_seconds):
        super().__init__(retry_after_seconds)
        self.args = (
            "OpenSky request blocked locally: an earlier response was HTTP 429 (quota/rate "
            f"limit reached). Another request may be attempted in {retry_after_seconds} seconds.",
        )


class OpenSkyCreditGuardError(Exception):
    """Request blocked locally because the remaining credits reached the configured threshold."""


def _utc_days_spanned(begin, end):
    """Number of UTC day boundaries between begin and end (same as the official client)."""
    begin_date = datetime.fromtimestamp(begin, tz=timezone.utc).date()
    end_date = datetime.fromtimestamp(end, tz=timezone.utc).date()
    return (end_date - begin_date).days


def _parse_int_header(response, name):
    """Return an integer response header, or None if it is missing or not a number."""
    try:
        return int(response.headers[name])
    except (KeyError, ValueError):
        return None


class OpenSkyClient:
    def __init__(self, min_remaining_credits=DEFAULT_MIN_REMAINING_CREDITS):
        """`min_remaining_credits`: once a response reports this many credits or fewer,
        further flight requests from this instance are blocked. None disables the guard."""
        load_dotenv()
        self.client_id = os.getenv("OPENSKY_CLIENT_ID")
        self.client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "OpenSky credentials missing: set OPENSKY_CLIENT_ID and "
                "OPENSKY_CLIENT_SECRET in your .env file."
            )
        self.min_remaining_credits = min_remaining_credits
        self._token = None
        self._token_expires_at = 0
        # Remaining credits reported by the last response (X-Rate-Limit-Remaining), as int.
        self.last_rate_limit_remaining = None
        # Seconds to wait reported by the last response (X-Rate-Limit-Retry-After-Seconds).
        self.last_retry_after_seconds = None
        # Unix time until which flight requests are blocked after a 429 (None = not blocked).
        self.blocked_until = None

    def _get_token(self):
        """Return a valid access token, requesting a new one shortly before expiry."""
        if self._token and time.time() < self._token_expires_at - 30:
            return self._token

        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.time() + payload.get("expires_in", 1800)
        return self._token

    @staticmethod
    def _validate_request(airport_icao, begin, end):
        """Check the request locally so invalid requests never reach OpenSky. Returns the ICAO code."""
        if not isinstance(airport_icao, str) or not airport_icao.strip():
            raise ValueError("airport_icao must be a non-empty ICAO code, e.g. 'PANC'.")
        for name, value in (("begin", begin), ("end", end)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer Unix timestamp (seconds).")
        if begin >= end:
            raise ValueError("end must be greater than begin.")
        if _utc_days_spanned(begin, end) > 1:
            raise ValueError(
                "The time interval must not span more than 1 UTC day boundary "
                "(OpenSky limit for /flights/arrival and /flights/departure)."
            )
        # Only the previous UTC day or earlier is available: end must be before 00:00 UTC today.
        today_start = datetime.fromtimestamp(time.time(), tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0)
        if end >= today_start.timestamp():
            raise ValueError(
                f"OpenSky flight data is only available up to the end of the previous UTC day: "
                f"end must be before {today_start.isoformat()}."
            )
        return airport_icao.strip().upper()

    def _check_cooldown(self):
        if self.blocked_until is None:
            return
        seconds_left = self.blocked_until - time.time()
        if seconds_left > 0:
            raise OpenSkyCooldownError(int(seconds_left + 0.999))  # round up to whole seconds
        self.blocked_until = None

    def _check_credit_guard(self):
        if self.min_remaining_credits is None or self.last_rate_limit_remaining is None:
            return
        if self.last_rate_limit_remaining <= self.min_remaining_credits:
            raise OpenSkyCreditGuardError(
                f"OpenSky request blocked locally: the last response reported "
                f"{self.last_rate_limit_remaining} remaining credits, at or below the "
                f"threshold of {self.min_remaining_credits}. Pass ignore_credit_guard=True "
                f"to send it anyway."
            )

    def _get_flights(self, path, airport_icao, begin, end, ignore_credit_guard):
        airport = self._validate_request(airport_icao, begin, end)
        self._check_cooldown()
        if not ignore_credit_guard:
            self._check_credit_guard()

        response = requests.get(
            f"{API_URL}{path}",
            params={"airport": airport, "begin": begin, "end": end},
            headers={"Authorization": f"Bearer {self._get_token()}"},
            timeout=60,
        )
        remaining = _parse_int_header(response, "X-Rate-Limit-Remaining")
        if remaining is not None:
            self.last_rate_limit_remaining = remaining
        self.last_retry_after_seconds = _parse_int_header(
            response, "X-Rate-Limit-Retry-After-Seconds"
        )

        if response.status_code == 429:
            if self.last_retry_after_seconds is not None:
                self.blocked_until = time.time() + self.last_retry_after_seconds
            raise OpenSkyRateLimitError(self.last_retry_after_seconds)
        # OpenSky documents 404 with an empty body as "no flights found".
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json()

    def get_departures(self, airport_icao, begin, end, ignore_credit_guard=False):
        """Flights that departed `airport_icao` between Unix times begin and end (list of dicts)."""
        return self._get_flights("/flights/departure", airport_icao, begin, end, ignore_credit_guard)

    def get_arrivals(self, airport_icao, begin, end, ignore_credit_guard=False):
        """Flights that arrived at `airport_icao` between Unix times begin and end (list of dicts)."""
        return self._get_flights("/flights/arrival", airport_icao, begin, end, ignore_credit_guard)
