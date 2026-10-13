# Data Sources: Data Acquisition Layer

Status as of 2026-10-01. Test airport: Anchorage, ANC (IATA) / PANC (ICAO).
The row-by-row mapping is in `data_requirements.xlsx`; this note explains how the sources were chosen.

## The four sources

| Source | Client | Auth | What it contributes | Validated |
|---|---|---|---|---|
| **OurAirports** `airports.csv`, `regions.csv`, `runways.csv` | `OurAirportsClient` | none | Airport discovery by country/state (`find_airports`), IATA↔ICAO crosswalk, name, location, coordinates, runways | ✅ |
| **BTS On-Time Performance** (TranStats download form, filtered by state) | `BtsOnTimeClient` | none | Individual flights: scheduled + actual times, delays, cancellations, origin/destination, distance | ✅ |
| **BTS T-100 Segment Summary by Origin Airport** (data.bts.gov `r495-tyji`) | `BtsT100Client` | none | Monthly passengers (enplaned), seats, departures incl. cargo, domestic/international split, 2014→ | ✅ |
| **OpenSky Network** `/flights/departure`, `/flights/arrival` | `OpenSkyClient` | OAuth2 | Every receiver-observed movement, incl. cargo and international | ⏳ needs credentials |

### Why each one is needed

- **OurAirports**: BTS uses 3-letter codes and OpenSky uses ICAO codes, so we need a crosswalk. It also provides coordinates, which let us compute route distances for flights that BTS doesn't cover. It is also the source for **discovering which airports exist** in a country or state (`find_airports`). Which of them have relevant commercial activity is confirmed from BTS, not from this list (see `docs/METRICS.md`).
- **BTS On-Time**: the only free source of *scheduled* times, which means it's the only source for delays and cancellations.
- **BTS T-100**: the only free source of passenger counts by airport and month, with a long history (needed for growth).
- **OpenSky**: BTS On-Time misses most of ANC's traffic. In April 2026 it had 1,333 ANC departures, against 5,412 total departures in T-100 (~25%). OpenSky is the only free source of *individual* cargo and international flights.

## How BTS On-Time data is retrieved

There is no official API that filters on-time flights by airport. The data.bts.gov and
data.transportation.gov portals have no flight-level on-time dataset. `BtsOnTimeClient` therefore uses:

1. **Primary: the TranStats download form** (`DL_SelectFields.aspx`). The smallest server-side
   filter is a **state**, plus year, month and the selected columns.
   - The airport's state comes from OurAirports (`iso_region` → `regions.csv` name). Exceptions:
     DC airports → "Virginia" (where BTS files DCA); PR, VI, GU, MP and AS → their TranStats territory option.
   - Requested airports are grouped by state: **one request per state per month**. The ZIP is
     read in memory, and only flights whose `Origin` or `Dest` is a requested airport are kept.
   - A flight between requested airports in two states appears in both downloads. It is kept only
     from the state of its first requested endpoint (origin first), so it is returned once.
   - Form columns are renamed to the PREZIP names, and `FL_DATE` is converted to `YYYY-MM-DD`, so
     callers see the same schema either way.
   - Each request takes about 1–10 s (BTS server time) and returns ~50 KB for Alaska.
   - The form is a web page, not a published API, so it may change without notice.
2. **Optional fallback: the monthly PREZIP file** (~30 MB, about 60 s). Used only with
   `get_flights(..., allow_bulk_fallback=True)`. It is streamed into a temporary file that is
   deleted automatically.

Nothing from On-Time is stored on disk. Validated 2026-10-01 for ANC, April 2026: the form path
returns exactly the same 2,667 flights (1,333 departures), with all 20 fields identical, as the PREZIP file.

## Comparison-group pipeline (`analysis.analyze_comparison_group`)

This is the deterministic backend for a future `compare_airports` tool. It takes the discovered airport
rows and a month, and runs: data acquisition → KPIs → scoring. It is optimized to avoid repeated work:

- **T-100:** one batched request for the whole group (`BtsT100Client.get_monthly_passengers_for_airports`),
  selecting only `origin_airport_code`, `reporting_month` and `total_passengers`. It gives the same passenger
  metrics as the per-airport method, which is kept for single-airport use.
- **On-Time:** one request per state and month. The state comes from the discovery rows (`airport_rows=`), so
  airports aren't looked up again.
- **OurAirports:** each CSV is parsed once per client and kept in memory, and runways are indexed by airport.
  There is no extra disk cache.
- **Flights:** indexed by airport once, so each airport's KPIs see only its own rows. The results are identical.

**Benchmark (April 2026, 29 Pacific Northwest airports):**
- Total: 113 s → **41 s**.
- T-100: 29 requests / 44 s → **1 request / 1.3 s**.
- OurAirports: 19.8 s → 0.8 s.
- The On-Time state downloads (about 34 s) are now the main cost.

### Airport identifiers: OurAirports → BTS (`data_sources.identifiers`)

BTS codes usually equal the OurAirports `iata_code`, but not always. Palm Beach is listed in OurAirports with
every structured code (IATA, ICAO, GPS, local) set to **DJT**, while BTS reports it as **PBI**; the old code
only survives in OurAirports `keywords`.

The resolution rule is conservative and never matches airport names:
1. **Exact:** use `iata_code` if that code exists in BTS. Alternates are then not considered.
2. **Alternate:** otherwise consider `local_code`, then upper-case 3-character codes in `keywords`. A candidate
   is eligible only if it exists in BTS **and** is not the IATA or local code of any *other* OurAirports
   airport. For example, Boulder City's local code BVU is Beluga's IATA code, and Mesa Gateway's keyword CHD is
   Chandler Municipal. Exactly one eligible candidate means it is resolved; more than one means `ambiguous`.
3. **Otherwise `not_found_in_bts`:** the airport is listed with no BTS data (`insufficient_coverage`) and an
   identifier issue. This is **not** treated as zero activity.

Both the OurAirports identifiers and the BTS code used are kept in every record (`identifiers`). Checking
whether a code exists in BTS uses the same batched T-100 query, so it costs no extra request.

On real data, 28 of 29 relevant airports whose local code differs from their IATA code are listed by BTS under
their IATA code. Palm Beach (DJT → PBI) is the only alternate resolution found in the 61-airport Southeast group.

## What each source does NOT give us

- **On-Time**: domestic scheduled passenger flights of large reporting carriers only. Times are local hhmm.
- **T-100**: monthly totals only, keyed by *origin* airport. `total_passengers` counts departing passengers only (= `domestic_passengers + outbound_international_1`). Data lags ~4–5 months.
- **OpenSky**: no schedules, so no delays or cancellations. `firstSeen`/`lastSeen` are when receivers saw the aircraft, not gate or runway times. Airports are *estimates* and can be null. Only data from the previous day or earlier is available.
- **OurAirports**: community-maintained. Runways are physical layout only, not runway capacity.

## Not covered by any free source

- Terminal capacity
- Number of gates (OpenStreetMap has partial, crowd-sourced data; not adopted)
- Runway capacity (movements per hour)
- Unmet demand

## Where we will rely on proxies (to decide together)

| Metric | Candidate proxy | Why it is only a proxy |
|---|---|---|
| Capacity utilization | T-100 `total_load_factor` | Measures how full *airline seats* are, not the airport |
| Demand vs capacity | passengers vs seats; movements vs runways | No real capacity data |
| Unmet demand | load factor, delays, cancellations, peak-hour concentration, growth | Unmet demand is not directly observable |
| Long-haul share | On-Time `Distance` + threshold | Domestic only unless OpenSky is added; threshold undefined |

## Considered and left out

- **FAA CY enplanement reports**: official, but annual only and redundant with T-100. Useful later as a cross-check.
- **OpenStreetMap Overpass (gates)**: completeness unknown, and requests were rate-limited.
- **BTS T-100 detailed segment files**: no stable download URL. TranStats generates these per request.

## Running the validation

```
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe validate_sources.py             # no OpenSky requests
.venv/Scripts/python.exe validate_sources.py --opensky  # also real OpenSky requests (uses credits)
```

OurAirports CSV files are cached in `data/raw/ourairports/` (git-ignored). BTS data is never written
to disk. OpenSky validation is opt-in: without `--opensky` it is always skipped, even when credentials
exist. With `--opensky` it runs only if `.env` contains `OPENSKY_CLIENT_ID` and
`OPENSKY_CLIENT_SECRET` (see `.env.example`); otherwise it is reported as skipped.

Offline unit tests (all HTTP mocked, zero OpenSky credits):

```
.venv/Scripts/python.exe -m unittest discover -s tests -t . -v
```
