"""Validate each data source with a small request for Anchorage (ANC / PANC).

For every source: make the request, report success/failure, and check that the
fields mapped in data_requirements.xlsx really exist in the returned data.
Prints raw samples only; no metrics are calculated.

Run:  .venv/Scripts/python.exe validate_sources.py             (no OpenSky requests)
      .venv/Scripts/python.exe validate_sources.py --opensky  (also real OpenSky requests; uses credits)
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from data_sources import BtsOnTimeClient, BtsT100Client, OpenSkyClient, OurAirportsClient

IATA = "ANC"
ICAO = "PANC"

# Fields referenced in the "API Field(s)" column of the Data Coverage Matrix.
EXPECTED = {
    "OurAirports airports.csv": [
        "ident", "iata_code", "icao_code", "name", "municipality", "iso_region",
        "iso_country", "latitude_deg", "longitude_deg",
    ],
    "OurAirports runways.csv": [
        "airport_ident", "length_ft", "width_ft", "surface", "closed", "le_ident", "he_ident",
    ],
    "BTS On-Time": [
        "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
        "OriginState", "DestState", "CRSDepTime", "DepTime", "DepDelay", "CRSArrTime",
        "ArrTime", "ArrDelay", "Cancelled", "CancellationCode", "Distance",
    ],
    "BTS T-100": [
        "reporting_month", "origin_airport_code", "total_passengers", "domestic_passengers",
        "outbound_international_1", "inbound_international_1", "total_seats",
        "total_load_factor", "total_departures", "domestic_departures",
        "outbound_international", "inbound_international",
    ],
    "OpenSky flights": [
        "icao24", "callsign", "firstSeen", "lastSeen", "estDepartureAirport",
        "estArrivalAirport", "departureAirportCandidatesCount",
        "arrivalAirportCandidatesCount",
    ],
}

results = []


def check_fields(source, record):
    missing = [f for f in EXPECTED[source] if f not in record]
    status = "OK" if not missing else f"MISSING {missing}"
    print(f"  field check: {status}")
    return not missing


def run(source, test):
    print(f"\n=== {source}")
    try:
        ok = test()
        results.append((source, "PASS" if ok else "FAIL (fields)"))
    except Exception as exc:  # report and continue with the other sources
        print(f"  ERROR: {exc!r}")
        results.append((source, f"FAIL ({type(exc).__name__})"))


def test_ourairports_airport():
    airport = OurAirportsClient().get_airport(IATA)
    print("  ", {k: airport[k] for k in EXPECTED["OurAirports airports.csv"]})
    return check_fields("OurAirports airports.csv", airport)


def test_ourairports_runways():
    runways = OurAirportsClient().get_runways(ICAO)
    print(f"  {len(runways)} runways")
    for r in runways:
        print("  ", {k: r[k] for k in EXPECTED["OurAirports runways.csv"]})
    return bool(runways) and check_fields("OurAirports runways.csv", runways[0])


def test_bts_ontime():
    # April 2026 = latest month also available in T-100, so the two can be compared.
    flights = BtsOnTimeClient().get_flights(IATA, 2026, 4)
    departures = sum(f["Origin"] == IATA for f in flights)
    cancelled = sum(f["Cancelled"] == "1.00" for f in flights)
    airlines = sorted({f["Reporting_Airline"] for f in flights})
    print(f"  {len(flights)} rows ({departures} departing {IATA}), "
          f"{cancelled} with Cancelled=1, carriers={airlines}")
    print("  sample:", flights[0])
    return bool(flights) and check_fields("BTS On-Time", flights[0])


def test_bts_t100():
    rows = BtsT100Client().get_monthly_airport_stats(IATA)
    print(f"  {len(rows)} months: {rows[0]['reporting_month'][:7]} .. {rows[-1]['reporting_month'][:7]}")
    latest = rows[-1]
    print("  latest:", {k: latest.get(k) for k in EXPECTED["BTS T-100"]})
    return bool(rows) and check_fields("BTS T-100", latest)


def test_opensky():
    # One full UTC day, a few days ago (data is built by a nightly batch).
    day = (datetime.now(timezone.utc) - timedelta(days=5)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    begin = int(day.timestamp())
    end = begin + 86400 - 1
    client = OpenSkyClient()
    departures = client.get_departures(ICAO, begin, end)
    print(f"  {day.date()} departures: {len(departures)} "
          f"(credits remaining: {client.last_rate_limit_remaining})")
    arrivals = client.get_arrivals(ICAO, begin, end)
    print(f"  {day.date()} arrivals: {len(arrivals)} "
          f"(credits remaining: {client.last_rate_limit_remaining})")
    sample = departures[0] if departures else arrivals[0]
    print("  sample:", sample)
    no_dest = sum(f["estArrivalAirport"] is None for f in departures)
    print(f"  departures with estArrivalAirport = null: {no_dest}")
    return check_fields("OpenSky flights", sample)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate each data source with a small request for Anchorage (ANC / PANC)."
    )
    parser.add_argument(
        "--opensky",
        action="store_true",
        help="also validate OpenSky with real API requests (consumes OpenSky credits)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results.clear()

    run("OurAirports airports.csv", test_ourairports_airport)
    run("OurAirports runways.csv", test_ourairports_runways)
    run("BTS T-100", test_bts_t100)
    run("BTS On-Time", test_bts_ontime)

    if not args.opensky:
        print("\n=== OpenSky flights\n  SKIPPED: OpenSky requests use API credits; use --opensky to validate")
        results.append(("OpenSky flights", "SKIPPED (use --opensky to validate)"))
    else:
        load_dotenv()
        if os.getenv("OPENSKY_CLIENT_ID") and os.getenv("OPENSKY_CLIENT_SECRET"):
            run("OpenSky flights", test_opensky)
        else:
            print("\n=== OpenSky flights\n  SKIPPED: OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET not set in .env")
            results.append(("OpenSky flights", "SKIPPED (credentials missing)"))

    print("\n=== Summary")
    for source, outcome in results:
        print(f"  {source:28} {outcome}")
    return results


if __name__ == "__main__":
    main()
