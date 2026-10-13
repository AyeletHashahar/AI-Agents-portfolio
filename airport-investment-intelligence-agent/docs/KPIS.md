# KPIs and the Investment Opportunity Score (V1)

Status as of 2026-10-04. Code: `kpis/` (deterministic, no API calls, no LLM). Weights: `config.py` (project root; re-exported by `kpis/config.py`).
Underlying metric definitions: `docs/METRICS.md`.

```
data_sources (retrieve)  ->  metrics (derive)  ->  kpis.build_airport_kpis (raw KPIs per airport)
    ->  kpis.score_comparison_group (percentile normalization -> family scores -> score -> rank)
```

A future agent decides **which** airports to compare and **how** to explain the results.
Python does every calculation and applies every business rule.

## Public API

| Function | Input | Output |
|---|---|---|
| `build_airport_kpis(airport, flights=, t100_rows=, runways=, first_date=, last_date=, airports_by_code=None)` | Already-retrieved On-Time rows, T-100 rows, OurAirports runways for one airport, and the analysis period | Raw KPI signals and context per family, Capacity Pressure proxy inputs, coverage, and missing signals |
| `score_comparison_group([record, ...])` | Records from `build_airport_kpis` | Normalized signals, family scores, Investment Opportunity Score, rank, missing-data details, and the raw records |

## The three KPI families

Only **signals** are normalized and scored. **Context** values are returned to explain the result but are never scored.

### Demand Growth (35%)
- **Signal:** `passenger_cagr`, the passenger CAGR from 2022 to the latest complete contiguous year
  (`metrics.passengers.passenger_cagr`; 2020 and 2021 are never included).
- **Context:** the CAGR years and passenger totals, and the latest complete annual passenger volume.
- **Not included:** year-over-year growth and seasonality.

### Operational Pressure (40%)
The scheduled basis is primary because it shows the infrastructure demand airlines planned.
The actual basis is context only.

| Signal | Raw value | Direction | Weight in family |
|---|---|---|---|
| `scheduled_p95_per_operational_runway` | Scheduled 95th-percentile movements per hour / operational runways | higher = more pressure | **0.50** |
| `scheduled_peak_spacing_minutes` | Scheduled median spacing inside peak periods | **lower** = more pressure | **0.20** |
| `departure_delayed_15_rate` | Share of eligible departures delayed ≥ 15 min | higher | **0.15** |
| `arrival_delayed_15_rate` | Share of eligible completed arrivals delayed ≥ 15 min | higher | **0.075** |
| `departure_delayed_60_rate` | Share of eligible departures delayed ≥ 60 min (extreme disruption) | higher | **0.075** |

- **Context:** median departure and arrival delay, cancellation rate, actual 95th percentile, actual peak
  spacing, scheduled 95th percentile, operational runways (with the reason if missing), runway detail by
  surface, and the raw count of open runways on all surfaces.
- **Operational runways** (`metrics.airports.operational_runway_count`): OurAirports runways that are open
  (`closed = 0`), not helipads (`le_ident` "H1", "H2", …), and have a **hard** surface. The surface
  classification is explained below.
- **When the signal is `None`:** if the operational count is 0, or **uncertain** (a runway's open/closed status
  is unknown, or an open runway's surface is unknown), the per-runway signal is `None` with a reason. It is
  never divided by a doubtful count.
- **Effect:** FAI has 2 operational runways (4 open rows include a water lane and a gravel strip), JNU has 1
  (its 2 include a water lane), and ANC has 3.
- **Still a crude proxy:** it does not model runway length or strength suitability, runway configuration,
  simultaneous-use restrictions or runway capacity.
- **Runway length is an explicit V1 limitation; no length rule is applied.** Investigation (2026-10-04) of the
  1,737 operational runways at U.S. large, medium and scheduled-service airports:
  - **The distribution is smooth.** 1st percentile 2,510 ft, 10th 3,884, median 6,500, 90th 10,301, with no
    natural gap to anchor a cutoff.
  - **Short general-aviation runways inflate some large-airport counts.** 15 large airports have a sub-5,000 ft
    hard runway, e.g. BOS 2,557 ft (counted 6, about 5 used by airlines), SNA 2,886, OAK 3,376, MDW 3,859.
  - **But real scheduled service also uses short runways**, e.g. Block Island BID at 2,502 ft and Alaska villages.
    Any cutoff would remove those: a 4,000 ft cutoff changes 176 of 904 airports, and 30 would lose every runway.
  - **The right minimum depends on aircraft type, which no source we use provides.** OurAirports `length_ft` also
    includes overruns and displaced thresholds.

**Runway surface classification** (`metrics.airports.runway_surface_category`). OurAirports `surface` is free
text: 668 distinct values in the 2026-10 data. The text is split into tokens.
- **Ignored when combined with a surface:** the condition grades `-G`/`-F`/`-P`/`-E` and `GRVD`/`GROOVED`.
- **The category is the one shared by all remaining tokens:**

| Category | Tokens | Operational? |
|---|---|---|
| hard | ASP, ASPH, ASPHALT, CON, CONC, CONCRETE, PEM, BIT | yes |
| unpaved | TURF, GRS, GRASS, SOD, DIRT, EARTH, GRE, GVL, GRVL, GRAVEL, SAND, SNOW, ICE, MATS, PSP | no |
| water | WATER, WAT | no |
| unknown | blank, unlisted codes (PER, UNK, GRV, a bare G/X/…), mixed categories (e.g. ASPH-TURF) | no, and the count is flagged uncertain |

- **At U.S. airports that are large, medium or have scheduled service**, open runway rows split into 1,761 hard,
  214 unpaved, 73 water and 4 unknown (PER at Cedar Rapids, UNK, GRV).
- **Gravel-only airports:** 144 scheduled-service airports, 136 of them in Alaska, have only gravel or turf
  runways. They get no operational runway count, so this signal is missing (`None`) for them, not 0.
- **Delays and cancellations are not proof of infrastructure congestion.** They can come from weather,
  airline operations or the wider network.

### Airport Scale & Connectivity (25%)

| Signal | Raw value | Weight in family |
|---|---|---|
| `annual_passengers` | Latest complete calendar-year passengers (T-100, enplaned) | **0.40** |
| `scheduled_departures` | Scheduled departures in the period (On-Time) | **0.20** |
| `unique_destinations` | Distinct destinations of scheduled departures (On-Time) | **0.20** |
| `long_haul_flights` | Flights ≥ 3,450 statute miles (existing definition and distance-source rules) | **0.10** |
| `long_haul_share` | Long-haul flights / flights with a known distance | **0.10** |

All signals here are "higher = more". **Coverage limitation:** BTS On-Time contains only domestic scheduled
passenger flights of reporting carriers. Destinations and long-haul values are therefore **not** the airport's
complete or global network, and international long-haul service is missing. Every record carries
`connectivity_is_complete_network: False` and a source-scope note so an agent can explain this.
For Alaska in April 2026, every airport has 0 long-haul flights in On-Time. Both long-haul signals are
then **non-discriminative** and are left out of the score, as described below.

## Capacity Pressure: a proxy, not scored

We have **no** terminal capacity, gate capacity or terminal utilization data, so Capacity Pressure can only
be a **proxy / synthesis signal**. It is **not** a measurement of unmet demand.

Each record exposes `capacity_pressure_inputs`, marked `is_proxy: True` and `scored: False`. It contains:
- passenger CAGR;
- latest annual passengers;
- seasonality: the index over the CAGR years, plus peak and trough months;
- the Operational Pressure signals.

It is **not** part of the Investment Opportunity Score. Its inputs already feed the three base families,
so scoring it again would double-count them. Interpreting it belongs to a later layer.

## Normalization: comparison-group percentile ranking

Each signal is ranked against the airports in the current comparison group. There are no global thresholds,
and the same airport can score differently in a different group.

- **Formula:** `score = 100 × (below + (equal − 1) / 2) / (m − 1)`
  - `m` = airports with a value for that signal;
  - `below` = airports with a strictly lower value;
  - `equal` = airports with the same value, including the airport itself.
- **Results:** lowest = 0, highest = 100. **Tied airports share the mid-rank score.** "Lower is more" signals
  use `100 − percentile`.
- **Rank:** competition ranking on the final score, with ties sharing a rank (1, 2, 2, 4). The output lists
  airports by rank and then by code.

**Signal status in the group** (`signal_status`):
- `ranked`: the signal is percentile-ranked.
- `non_discriminative`: every airport with a value has the **same** value, e.g. long-haul = 0 for ANC, FAI and JNU.
  It holds no ranking information, so it is **not scored** (otherwise it would add the same 50 points to everyone).
  The raw value is kept and the signal is listed in `non_discriminative_signals`.
- `not_rankable`: fewer than 2 airports have a value.

## Missing data, non-discriminative signals and coverage

A signal can be unusable for two **different** reasons, and they are tracked separately:

| | Meaning | Reduces the airport's coverage? |
|---|---|---|
| `missing` | **This airport** has no raw value (no On-Time data, runway count uncertain, …) | **yes** |
| `non_discriminative` / `not_rankable` | The **group** gives no ranking information for the signal | no |

- **Family score:** `Σ effective_weight × percentile` over the usable signals (ranked and not missing).
  - `effective_weight = configured weight / sum of the configured weights of the usable signals`. This is
    proportional redistribution; the configured weights in `config.py` never change.
  - Each family reports `score`, `coverage` (the share of the family's configured weight the airport has data
    for), `effective_weights`, `missing_signals`, `non_discriminative_signals` and `not_rankable_signals`.
  - A family with no usable signal has no score, and the reason is given.
  - **Family minimum coverage: `MIN_FAMILY_COVERAGE = 0.50`.** A family is scored only if the airport has data
    for at least 50% of the family's configured signal weight.
    - Below that, the family score is `None` and the result reports `insufficient_family_coverage`. Available raw
      and normalized signals stay visible.
    - This stops a minor signal from being stretched into the whole family weight. For example, arrival delays
      (0.075) alone cannot carry Operational Pressure's 40%.
    - Non-discriminative signals don't reduce this coverage.
- **Investment Opportunity Score:** `Σ effective_family_weight × family score` over the families with a score.
  - `effective_family_weight = INVESTMENT_SCORE_WEIGHTS[f] / sum over scored families`.
  - **Investment coverage** = `Σ INVESTMENT_SCORE_WEIGHTS[f] × family coverage`.
- **Minimum coverage (proposed V1): `MIN_INVESTMENT_SCORE_COVERAGE = 0.60`.**
  - This is the smallest combined weight of any two families (Demand Growth 0.35 + Scale & Connectivity 0.25).
  - An airport can lose one whole KPI family and still be scored, but not more.
  - An airport with no BTS On-Time data (coverage 0.45) or no T-100 data (0.55) gets `insufficient_coverage`:
    no score and no rank. Its partial family scores are still shown.
- **`score_status`:** `complete` (coverage 1.0), `partial` (scored, coverage < 1.0), `insufficient_coverage` or
  `comparison_group_required`. Every airport also reports `coverage`, `effective_family_weights` and
  `missing_families` (with reasons).
- **A missing value is `None`, never 0.** It is left out of that signal's ranking (the others are ranked among
  themselves), so a missing source neither rewards nor penalizes an airport.

## Investment Opportunity Score

Configured: `score = 0.40 × Operational Pressure + 0.35 × Demand Growth + 0.25 × Scale & Connectivity`.
When a family has no score, its weight is redistributed as described above and reported.

## Why a single airport has no score

A percentile is only meaningful relative to other airports. With fewer than 2 airports,
`score_comparison_group` returns `status: "comparison_group_required"`. Raw KPIs are still returned, but no
normalized values, family scores or 0–100 score. **No relative score is made up.**

## Example (real data, April 2026, OpenSky not used)

| Rank | Airport | Score | Operational Pressure | Demand Growth | Scale & Connectivity | Operational runways | Coverage |
|---|---|---|---|---|---|---|---|
| 1 | ANC | 72.5 | 75 | 50 | 100 | 3 | 1.0 |
| 2 | FAI | 45.25 | 10 | 100 | 25 | 2 | 1.0 |
| 3 | JNU | 32.25 | 65 | 0 | 25 | 1 | 1.0 |

- **Non-discriminative here:** `long_haul_flights` and `long_haul_share`, so Scale & Connectivity uses
  annual passengers 0.50, scheduled departures 0.25 and unique destinations 0.25.
- **Earlier version** (all open runways counted, ties scored as 50, strict coverage): ANC 80.0, FAI 46.5, JNU 23.5.
