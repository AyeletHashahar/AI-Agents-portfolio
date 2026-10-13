"""BTS Reporting Carrier On-Time Performance: one row per scheduled domestic flight.

Source: U.S. DOT Bureau of Transportation Statistics (TranStats).

Coverage (important): only U.S. carriers that are required to report on-time data,
domestic scheduled passenger flights only. Cargo flights and international flights
are NOT included.

How data is retrieved (no authentication, nothing is stored on disk):

1. Primary: the official TranStats download form (DL_SelectFields.aspx). The smallest
   server-side filter it offers is a STATE, plus year, month and the columns we select.
   We download one small ZIP per state/month into memory, then keep only flights whose
   Origin or Dest is one of the requested airports. The form is a web page, not a
   published API, so it may change without notice.
2. Optional fallback (allow_bulk_fallback=True): the full monthly PREZIP file
   (~30 MB, all states, all columns), streamed into a temporary file that is deleted
   automatically. Never used unless explicitly allowed.

Both paths return the same schema: dicts with the FIELDS below, values as raw strings,
using the PREZIP column names. Field definitions: readme.html inside each PREZIP file.
"""

import csv
import io
import re
import tempfile
import warnings
import zipfile
from datetime import datetime

import requests

from data_sources.ourairports import OurAirportsClient

FORM_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr"
BULK_URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
USER_AGENT = "Mozilla/5.0 (Airport Investment Intelligence Agent - learning project)"

# Our schema (PREZIP column name) -> column name in the TranStats download form.
# Times are LOCAL time as hhmm strings (per the BTS readme).
FORM_COLUMNS = {
    "FlightDate": "FL_DATE",
    "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",
    "Tail_Number": "TAIL_NUM",
    "Origin": "ORIGIN",
    "OriginCityName": "ORIGIN_CITY_NAME",
    "OriginState": "ORIGIN_STATE_ABR",
    "Dest": "DEST",
    "DestCityName": "DEST_CITY_NAME",
    "DestState": "DEST_STATE_ABR",
    "CRSDepTime": "CRS_DEP_TIME",
    "DepTime": "DEP_TIME",
    "DepDelay": "DEP_DELAY",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrTime": "ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "Cancelled": "CANCELLED",
    "CancellationCode": "CANCELLATION_CODE",
    "Diverted": "DIVERTED",
    "Distance": "DISTANCE",
}
FIELDS = list(FORM_COLUMNS)

# Form "Geography" option for U.S. territories (OurAirports iso_country -> form option).
# Verified against BTS OriginStateName values for SJU, STT, GUM and SPN (Apr-2026).
TERRITORY_GEOGRAPHY = {
    "PR": "Puerto Rico",
    "VI": "U.S. Virgin Islands",
    "GU": "U.S. Pacific Trust Territories and Possessions",
    "MP": "U.S. Pacific Trust Territories and Possessions",
    "AS": "U.S. Pacific Trust Territories and Possessions",
}


class TranStatsFormError(Exception):
    """The TranStats form did not return the expected ZIP file."""


class BtsOnTimeClient:
    def __init__(self, airports_client=None, cache=None):
        """cache: optional data_sources.session_cache.SessionCache. When given, the raw
        TranStats form rows of each (state, year, month) are downloaded once per session."""
        self.airports = airports_client or OurAirportsClient()
        self.cache = cache
        # What the last get_flights() call used: (method, geography, year, month), where
        # method is "form" (downloaded), "form_cached" (reused from the session cache) or "bulk".
        self.last_requests = []
        self._latest_month = None

    def latest_available_month(self):
        """(year, month) of the latest On-Time month BTS has published, read from the
        TranStats download page ("Latest Available Data: July 2026"). One request, then kept
        in memory for this client. Raises TranStatsFormError if the text is not found."""
        if self._latest_month is None:
            page = requests.get(FORM_URL, timeout=60, headers={"User-Agent": USER_AGENT})
            page.raise_for_status()
            match = re.search(r"Latest Available Data:\s*(?:<[^>]*>\s*)*([A-Za-z]+)\s+(\d{4})", page.text)
            if not match:
                raise TranStatsFormError("'Latest Available Data' not found on the TranStats page.")
            month = datetime.strptime(match.group(1)[:3], "%b").month
            self._latest_month = (int(match.group(2)), month)
        return self._latest_month

    def get_flights(self, airports, year, month, allow_bulk_fallback=False, airport_rows=None):
        """Return all flights departing from or arriving at the given airport(s) in one month.

        `airports` is one BTS 3-letter code ('ANC') or a list of codes (['ANC', 'SEA']).
        `airport_rows`: optional {BTS code: OurAirports airports.csv row} already known (e.g.
        from discovery); their iso_country / iso_region give the state without another
        lookup. Needed when the BTS code differs from the OurAirports codes (e.g. PBI).
        Each flight is returned once, as a dict with the FIELDS above.
        """
        codes = [airports] if isinstance(airports, str) else list(airports)
        codes = list(dict.fromkeys(code.strip().upper() for code in codes))
        airport_rows = airport_rows or {}
        geography = {code: self._form_geography(code, airport_rows.get(code)) for code in codes}
        self.last_requests = []

        try:
            return self._get_flights_from_form(geography, year, month)
        except (requests.RequestException, TranStatsFormError) as exc:
            if not allow_bulk_fallback:
                raise
            warnings.warn(f"TranStats form failed ({exc!r}); using the bulk monthly file instead.")
            self.last_requests = []
            return self._get_flights_from_bulk(set(codes), year, month)

    def _form_geography(self, code, airport=None):
        """Return the TranStats form 'Geography' option (a state name) for an airport code.
        `airport`: its OurAirports row if already known (otherwise looked up by code)."""
        if airport is None:
            airport = self.airports.get_airport(code)
        if airport is None:
            raise ValueError(f"Airport {code!r} not found in OurAirports.")

        country, region = airport["iso_country"], airport["iso_region"]
        if country == "US":
            # The form has no 'District of Columbia'; BTS files DCA under Virginia.
            if region == "US-DC":
                return "Virginia"
            return self.airports.get_region_name(region)
        if country in TERRITORY_GEOGRAPHY:
            return TERRITORY_GEOGRAPHY[country]
        raise ValueError(f"Airport {code!r} is not a U.S. airport (country {country}).")

    def _get_flights_from_form(self, geography, year, month):
        flights = []
        for state in sorted(set(geography.values())):
            rows, method = self._form_rows(state, year, month)
            self.last_requests.append((method, state, year, month))
            for row in rows:
                flight = self._normalize_form_row(row)
                # A flight between requested airports in two states appears in both
                # downloads. Keep it only from the state of its first requested endpoint
                # (origin if requested, otherwise destination), so it is returned once.
                owner = geography.get(flight["Origin"]) or geography.get(flight["Dest"])
                if owner == state:
                    flights.append(flight)
        return flights

    def _form_rows(self, state, year, month):
        """Raw form rows for one state/month: from the session cache if present, otherwise
        downloaded (and then cached). A failed download raises and caches nothing."""
        key = (state, year, month)
        if self.cache is not None and self.cache.contains("ontime_form", key):
            return self.cache.get("ontime_form", key), "form_cached"
        rows = self._download_form_rows(state, year, month)
        if self.cache is not None:
            self.cache.put("ontime_form", key, rows)
        return rows, "form"

    def _download_form_rows(self, state, year, month):
        """Submit the TranStats form for one state/month and return the CSV rows (in memory)."""
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT

        page = session.get(FORM_URL, timeout=60)
        page.raise_for_status()
        # ASP.NET pages require their hidden fields (__VIEWSTATE etc.) to be sent back.
        form = dict(re.findall(r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"', page.text))
        form.update({
            "cboGeography": state,
            "cboYear": str(year),
            "cboPeriod": str(month),
            "btnDownload": "Download",
        })
        form.update({column: "on" for column in FORM_COLUMNS.values()})

        response = session.post(FORM_URL, data=form, timeout=300)
        response.raise_for_status()
        if not response.content.startswith(b"PK"):  # every ZIP file starts with "PK"
            raise TranStatsFormError(f"No ZIP returned for {state} {year}-{month:02d}.")

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            text = z.read(csv_name).decode("latin-1")
        return list(csv.DictReader(io.StringIO(text)))

    @staticmethod
    def _normalize_form_row(row):
        """Rename form columns to our schema and convert FL_DATE to 'YYYY-MM-DD'."""
        flight = {field: row[column] for field, column in FORM_COLUMNS.items()}
        # The form returns e.g. '4/1/2026 12:00:00 AM'; the PREZIP file uses '2026-04-01'.
        flight["FlightDate"] = datetime.strptime(
            flight["FlightDate"], "%m/%d/%Y %I:%M:%S %p"
        ).date().isoformat()
        return flight

    def _get_flights_from_bulk(self, codes, year, month):
        """Fallback: stream the full monthly PREZIP into a self-deleting temp file and filter it."""
        url = BULK_URL_TEMPLATE.format(year=year, month=month)
        flights = []
        with tempfile.TemporaryFile() as tmp:  # deleted automatically when closed
            with requests.get(url, stream=True, timeout=600) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1 << 20):
                    tmp.write(chunk)
            tmp.seek(0)

            with zipfile.ZipFile(tmp) as z:
                csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
                with z.open(csv_name) as f:
                    for row in csv.DictReader(io.TextIOWrapper(f, encoding="latin-1")):
                        if row["Origin"] in codes or row["Dest"] in codes:
                            flights.append({field: row[field] for field in FIELDS})

        self.last_requests.append(("bulk", "All", year, month))
        return flights
