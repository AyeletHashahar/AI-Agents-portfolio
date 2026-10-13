"""BTS T-100 Segment Summary by Origin Airport: monthly passengers, seats, departures, freight.

Source: U.S. DOT Bureau of Transportation Statistics, dataset "AFF - T100 Segment Summary
By Origin Airport" on data.bts.gov (id r495-tyji), queried through the Socrata SODA API.
Dataset page: https://data.bts.gov/d/r495-tyji

Validated meaning (see Data Coverage Matrix notes):
- One row per airport per month, keyed by ORIGIN airport.
- total_passengers = domestic_passengers + outbound_international_1, i.e. passengers
  departing (enplaned at) this airport. Arriving domestic passengers are not included.
- Several API field names are truncated by the portal; FIELD_LABELS gives their real names.

No authentication (anonymous requests are throttled, which is fine for small queries).
"""

import re

import requests

URL = "https://data.bts.gov/resource/r495-tyji.json"

# Official column labels for the API field names that are not self-explanatory.
FIELD_LABELS = {
    "reporting_month": "Date (first day of the month)",
    "outbound_international": "Outbound International Departures",
    "outbound_international_1": "Outbound International Passengers",
    "outbound_international_seats": "Outbound International Seats",
    "inbound_international": "Inbound International Departures",
    "inbound_international_1": "Inbound International Passengers",
    "inbound_international_seats": "Inbound International Seats",
}

TEXT_FIELDS = {
    "origin_airport_id", "year", "reporting_month", "origin_airport_code",
    "origin_city_name", "origin_airport_name",
}


# Fields the passenger metrics need (metrics/passengers.py); used by the batched query.
PASSENGER_FIELDS = ["origin_airport_code", "reporting_month", "total_passengers"]
_BATCH_CODES = 200          # airport codes per request (keeps the URL short)
_PAGE_SIZE = 50000          # rows per request; further pages are fetched with $offset
_CODE = re.compile(r"[A-Z0-9]{3}")


class BtsT100Client:
    def __init__(self, cache=None):
        """cache: optional data_sources.session_cache.SessionCache. When given, each airport
        code's passenger rows (including "no rows") are fetched once per session."""
        self.cache = cache

    def latest_reporting_month(self):
        """Latest month in the whole T-100 dataset, as 'YYYY-MM' (one small request)."""
        if self.cache is not None and self.cache.contains("t100_latest_month", None):
            return self.cache.get("t100_latest_month", None)
        response = requests.get(URL, params={"$select": "max(reporting_month) AS latest"}, timeout=60)
        response.raise_for_status()
        rows = response.json()
        latest = rows[0].get("latest") if rows else None
        latest = latest[:7] if latest else None
        if self.cache is not None:
            self.cache.put("t100_latest_month", None, latest)
        return latest

    def get_monthly_passengers_for_airports(self, airports):
        """Batched retrieval for a comparison group: {code: [monthly rows], ...}.

        One request per 200 codes (instead of one per airport), selecting only
        PASSENGER_FIELDS. Rows have the same format as get_monthly_airport_stats
        (total_passengers converted to float, omitted when null) and are ordered by month,
        so the passenger metrics give identical results. Every requested code is a key;
        codes without data map to []. With a session cache, only codes not fetched earlier
        in the session are requested, and results are cached only after the whole fetch
        succeeded.
        """
        codes = sorted({code.strip().upper() for code in airports})
        bad = [code for code in codes if not _CODE.fullmatch(code)]
        if bad:
            raise ValueError(f"Invalid airport code(s): {bad}")

        cached = {}
        if self.cache is not None:
            cached = {code: self.cache.get("t100_passengers", code)
                      for code in codes if self.cache.contains("t100_passengers", code)}
        fetched = self._fetch_passengers([code for code in codes if code not in cached])
        if self.cache is not None:
            for code, rows in fetched.items():
                self.cache.put("t100_passengers", code, rows)
        merged = {**cached, **fetched}
        return {code: list(merged[code]) for code in codes}

    def _fetch_passengers(self, codes):
        result = {code: [] for code in codes}
        for start in range(0, len(codes), _BATCH_CODES):
            chunk = ",".join(f"'{code}'" for code in codes[start:start + _BATCH_CODES])
            offset = 0
            while True:
                params = {
                    "$select": ",".join(PASSENGER_FIELDS),
                    "$where": f"origin_airport_code in ({chunk})",
                    "$order": "origin_airport_code ASC, reporting_month ASC",
                    "$limit": _PAGE_SIZE,
                    "$offset": offset,
                }
                response = requests.get(URL, params=params, timeout=120)
                response.raise_for_status()
                page = response.json()
                for row in page:
                    result[row["origin_airport_code"]].append({
                        key: value if key in TEXT_FIELDS else float(value)
                        for key, value in row.items()
                    })
                if len(page) < _PAGE_SIZE:
                    break
                offset += _PAGE_SIZE
        return result

    def get_monthly_airport_stats(self, airport, start_month=None, end_month=None):
        """Return one dict per month for `airport` (BTS 3-letter code, e.g. 'ANC'), oldest first.

        start_month / end_month are optional 'YYYY-MM' strings (inclusive).
        Numeric fields are converted from the API's strings to float.
        """
        where = [f"origin_airport_code = '{airport.strip().upper()}'"]
        if start_month:
            where.append(f"reporting_month >= '{start_month}-01T00:00:00'")
        if end_month:
            where.append(f"reporting_month <= '{end_month}-01T00:00:00'")

        params = {
            "$where": " AND ".join(where),
            "$order": "reporting_month ASC",
            "$limit": 5000,
        }
        response = requests.get(URL, params=params, timeout=60)
        response.raise_for_status()

        return [
            {
                key: value if key in TEXT_FIELDS else float(value)
                for key, value in row.items()
            }
            for row in response.json()
        ]
