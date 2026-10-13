# Metrics: Derived Metrics Layer

Status as of 2026-10-01. Code: `metrics/` (pure functions, no API calls, no LLM).
Row numbers refer to `data_requirements.xlsx`, the source of truth for which metrics are Direct or Derived.

The metrics layer only **calculates**. Data is retrieved first by `data_sources/`, and the
retrieved records are passed in, e.g. `cancellation_rate(flights, "ANC")`.

## General rules

- **Unknown is not zero.** Missing or unreadable values are counted as "unknown" or excluded,
  and the function reports how many. A rate or index that cannot be calculated is `None`.
- **Coverage.** Flight metrics use BTS On-Time, which covers only reporting carriers' domestic
  scheduled passenger flights (about 25% of ANC departures). Each result describes that subset only.
- **Time basis** (time-based flight metrics require it explicitly):
  - `scheduled`: the scheduled times (`CRSDepTime` / `CRSArrTime`) of every scheduled flight,
    including flights that were later cancelled.
  - `actual`: only movements that happened at the airport: departures with `Cancelled = 0`,
    and arrivals with `Cancelled = 0` and `Diverted = 0`. Each is placed at scheduled time + `DepDelay` / `ArrDelay` minutes.
- **Local time.** All On-Time times are the airport's local clock time. Hours are naive clock hours,
  24 per day, so on a daylight-saving change day one hour is missing or doubled.
- **Arrival date (accepted limitation).** BTS On-Time does not give an explicit arrival date in
  our selected fields; `FlightDate` is the scheduled departure date. For overnight flights, the
  arrival is assumed to be the next day when the arrival clock time is earlier than the departure
  clock time. No time-zone data is added for the rare westward-after-midnight exception.

## Flight activity: `metrics/flights.py` (BTS On-Time rows)

| Row | Function | Definition |
|---|---|---|
| 6–8 | `flight_counts` | Departures: scheduled / cancelled / operated (`Cancelled = 0`). Arrivals: scheduled / cancelled / diverted / completed. Total scheduled = departures + arrivals; total operated = operated departures + completed arrivals. |
| 24 | `cancellation_rate` | `count(Cancelled = 1) / count(rows with a known Cancelled value)`, over rows touching the airport. |
| **19, 22** | `delay_metrics` | Departure and arrival delay summary; definition below. |
| — | `unique_destinations` | Distinct `Dest` codes of scheduled departures (Origin = airport, including cancelled ones; the same population as long-haul). Domestic reporting-carrier network only. Used by the KPI layer (`docs/KPIS.md`). |
| 9 | `hourly_movement_counts` | Movements per local clock hour (only hours with movements listed). |
| 10 (descriptive) | `busiest_hour` | Maximum hourly count; a tie goes to the earliest hour. |
| **10 (comparison)** | `peak_hour_traffic` | **95th percentile of hourly movement counts** over every clock hour from `first_date` 00:00 to `last_date` 23:00; hours with no movements count as 0. Percentile method below. |
| 11 (raw) | `movement_spacing_minutes` | Every gap between consecutive movements on the same local date. |
| **11** | `peak_period_spacing` | **Median minutes between consecutive movements during peak periods** (definition below). |
| **15–16** | `long_haul_summary` | **Long-haul = distance ≥ 3,450 statute miles**; share = long-haul / flights with known distance. |

**Delay metrics** (`delay_metrics`). A flight is **delayed when its delay is ≥ 15 minutes**,
the standard U.S. DOT threshold, also used by the BTS `DepDel15` / `ArrDel15` indicators. Exactly 15 minutes counts as delayed.
- **Departures:** rows with `Origin` = airport. Eligible = `Cancelled = 0`. A diverted flight still
  departed, so its departure delay counts. Delay = `DepDelay`.
- **Arrivals:** rows with `Dest` = airport. Eligible = `Cancelled = 0` and `Diverted = 0`, i.e. completed
  arrivals that landed here. Delay = `ArrDelay`.
- **Not eligible:**
  - Cancelled flights are never delayed flights, even when BTS still shows a delay value.
  - Diverted arrivals are not completed arrivals.
  - Rows with an unknown cancelled or diverted status are counted as `unknown_status`.
- **Per direction:**
  - `known_delay` / `unknown_delay`: eligible flights with or without a readable delay. Unknown is never treated as 0.
  - `mean_delay_minutes` and `median_delay_minutes` are calculated over known delays. **Negative delays (early) are kept as they are.**
  - `delayed_15_plus` = known delays ≥ 15; `delayed_15_plus_rate` = `delayed_15_plus / known_delay`.
  - `delayed_60_plus` and `delayed_60_plus_rate`: the same for delays ≥ 60 minutes, an extreme-disruption indicator used by the KPI layer.
  - Means, medians and rates are `None` when no delay is known.
- ANC, April 2026 (On-Time only):
  - Departures: mean +1.7 min, median −5 min, 9.8% delayed.
  - Arrivals: mean −2.4 min, median −9 min, 14.0% delayed.

**Percentile method** (`percentile_linear`): linear interpolation between closest ranks
(Hyndman & Fan type 7, the same as Excel `PERCENTILE.INC` and NumPy's default). Sort the n values;
rank = (n − 1) × p / 100; result = x[⌊rank⌋] + (rank − ⌊rank⌋) × (x[⌊rank⌋+1] − x[⌊rank⌋]).

**Peak periods**, from the same hourly distribution as `peak_hour_traffic`:
1. Threshold = the 95th-percentile value.
2. Peak hours = hours with count ≥ threshold (hours exactly at the threshold are included) and at least 1 movement.
3. Peak periods = consecutive peak hours on the same local date. A period is split at local midnight.
4. Gaps are measured only between consecutive movements inside the same peak period. Spacing is `None`
   when no peak period has two movements. A real 0 is possible: two movements in the same minute.

**Long-haul.** There is no universal definition of a long-haul flight. This project adopts
**≥ 3,450 statute miles** (about 3,000 nautical miles; 3,000 nmi = 3,452.3 statute miles), using the
ICAO long-range distance boundary as the project threshold.
- **Distance priority:**
  1. BTS `Distance`, when it is a positive number.
  2. Otherwise, the great-circle distance between the two airports' OurAirports coordinates (already retrieved, passed in).
  3. Otherwise unknown. **Unknown distance is never classified as short-haul.**
- **Population:** every scheduled flight touching the airport (cancelled flights included).
- **Limitation:** both distances are airport-to-airport, not the path the aircraft actually flew.
- **Coverage:** in April 2026, ANC had 0 long-haul flights in On-Time; its longest domestic route,
  ANC–ATL, is 3,417 mi. International long-haul traffic (e.g. Asia cargo) is only visible through OpenSky.

## Passenger demand: `metrics/passengers.py` (BTS T-100 rows; passengers = enplaned at the airport)

| Row | Function | Definition |
|---|---|---|
| 26 | `annual_passenger_totals` | Sum of the monthly `total_passengers` per year, plus the number of known months and whether all 12 are known. |
| 27 | `yoy_monthly_growth` | `(P(Y,m) − P(Y−1,m)) / P(Y−1,m)`; `None` if either value is unknown or the prior year is 0. |
| **28** | `passenger_cagr` | **CAGR = (P_last / P_first) ^ (1 / (last_year − first_year)) − 1** |
| **29** | `seasonality_index` | **index(m) = average passengers in month m over the selected years / average passengers over all months of those years** (approved). |

**CAGR rules** (project decision):
- Use annual totals of **complete** calendar years only. **2020 and 2021 are always excluded.**
- Use the most recent **contiguous** run of at least two eligible years. Years are never bridged,
  so an excluded or incomplete year ends a run.
- By default the trend period starts in 2022 (`earliest_year=2022`).
- The result returns the first year, last year, number of years, number of periods, both passenger
  totals and the CAGR. With fewer than two such years, the CAGR is `None` and a reason is given.
- ANC 2022–2025: +1.89% per year.

**Seasonality rules:** the caller selects the years explicitly, and every selected year must have
all 12 months.

## Airport metadata: `metrics/airports.py` (OurAirports rows)

| Row | Function | Definition |
|---|---|---|
| 4 | `us_state`, `is_new_england` | State = the text after `US-` in `iso_region`. `is_new_england` (CT, ME, MA, NH, RI, VT) is a convenience check for one airport only. **Airport discovery** is general and lives in the data layer (see below). |
| 32 | `runway_count` | Open = rows with `closed = 0`; closed and unknown are counted separately. A raw count across all surfaces. Physical runways only, not runway or terminal capacity. |
| 32 | `runway_surface_category`, `operational_runway_count` | **Operational runway** = open, not a helipad, hard surface (asphalt/concrete family). Water, unpaved and unknown/ambiguous surfaces are excluded, and the result says when the count is uncertain. The KPI layer uses this count. Classification table: `docs/KPIS.md`. Still a proxy: runway length, configuration and capacity are not modeled. |
| 14 | `great_circle_distance_miles`, `airport_distance_miles` | Haversine formula, Earth radius 3,958.7613 statute miles. |

## Airport discovery: `OurAirportsClient.find_airports` (data layer)

`find_airports(country="US", states=["MA", "CT"])` lists the airports that **exist** in a country
or in one or more states/regions, using OurAirports `airports.csv`. Nothing is hard-coded: a future agent
turns geography in plain language (e.g. "New England") into structured country/state codes.

- **Returns:** `ident`, `iata_code`, `icao_code`, `gps_code`, `local_code`, `name`, `type`,
  `scheduled_service`, `municipality`, `iso_country`, `iso_region`, `latitude_deg`, `longitude_deg`.
  Values are strings exactly as in the CSV, and a missing code is `""`.
- **Excluded by default:** `heliport`, `balloonport` and closed facilities. The CSV uses `closed`, while the
  data dictionary says `closed_airport`, so both are excluded. These are clearly not airports for airline terminal investment.
- **Kept by default:** small airports and seaplane bases, because many have scheduled airline service (e.g. in Alaska).
- **Optional filters:**
  - `types=[...]`: exact types to include. This replaces the default exclusion.
  - `require_iata=True`: only airports with an IATA code.
  - `scheduled_service_only=True`: OurAirports' own flag for current scheduled service.
- **Errors:** an unknown region code raises an error rather than returning an empty list.
- **Not a definition of "commercial":** OurAirports says which airports exist. Whether an airport has
  relevant commercial activity is confirmed later from BTS data (T-100 / On-Time), not from its type.
- **Real data:** California returns 764 entries by default (2,430 raw), of which 78 are large or medium.
  Alaska has 82 large or medium airports, but 218 entries are flagged with scheduled service.

## Not implemented (later layers or a decision needed)

- **Delay-based congestion.** `delay_metrics` provides the delay facts; combining them into congestion
  belongs to a later layer.
- **Capacity utilization, demand vs capacity, unmet demand (rows 33–35).** Proxy definitions are not decided.
- **All scores, normalization and ranking across airports (rows 36–40).** These are implemented in the KPI layer;
  see `docs/KPIS.md`.
