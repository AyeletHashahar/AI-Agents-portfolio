# Agent Tools (V1)

Status as of 2026-10-12. Code: `agent_tools/`. These are the **only four** functions the LLM Agent may call.
They are plain Python functions returning compact, JSON-serializable, structured data. They write no prose
recommendations and never ask the user questions.

| The LLM Agent | The tools (deterministic Python) |
|---|---|
| Understands the request and resolves geography (e.g. "New England" → `US` + `ME, NH, VT, MA, RI, CT`) | Retrieve data and resolve airport identifiers |
| Chooses and chains tools | Calculate metrics and KPIs, normalize, score and rank |
| Writes the human explanation | Report coverage, missing data and limitations |

## Signatures

```python
resolve_airports(references: list[str]) -> dict
discover_airports(country: str, states: list[str] | None = None) -> dict
compare_airports(airport_codes: list[str], year: int | None = None, month: int | None = None,
                 allow_large_group: bool = False) -> dict
analyze_airport(airport_code: str, year: int | None = None, month: int | None = None) -> dict
```

Each tool also accepts a keyword-only `context=AgentToolContext()`. The application creates **one context per
session or conversation** and passes it to every call; the LLM never sees or manages it. Every result has `tool`,
`status` (`ok` | `partial` | `error`), `errors` and `warnings`.

## Session cache (`AgentToolContext`)

An in-memory `SessionCache` of **raw retrieved data**, shared by all tool calls in a session:

| Data | Cache key |
|---|---|
| BTS On-Time form rows | `(state, year, month)` |
| BTS T-100 passenger rows | Airport code (including "no rows") |
| Latest T-100 month | One entry |
| OurAirports parsed CSVs | Held by the shared OurAirports client |

- **Not cached:** final tool answers and failures. A failed download raises before anything is stored, so the
  next call retries.
- **Separation:** different months and different states are separate entries.
- **Source metadata:** the On-Time client records `form` (downloaded), `form_cached` (reused) or `bulk` (fallback).
- **Lifetime:** the cache lives only as long as the context object. There is no disk cache and nothing is shared
  between processes or between contexts; `context.clear()` empties it.
- **Effect (real data, Alaska, July 2026):** compare ANC/FAI/JNU, then analyze FAI and JNU, downloads Alaska once.
  Each follow-up analysis takes 0.1 s instead of about 19 s, with identical results.

## Automatic analysis limit

`AUTOMATIC_ANALYSIS_LIMIT = 30` is **not** a computational maximum. Up to 30 airports are compared directly. For a
larger group, `compare_airports` returns the error `comparison_group_requires_confirmation`, with
`details = {candidate_count, automatic_analysis_limit}`, unless `allow_large_group=True`.

With the override, **every** airport is analyzed and scored in **one** comparison group: never truncated,
pre-selected, split, or merged from separately normalized groups. A `large_group_confirmed` warning records it.

The Agent policy that uses this will be built later:
- **≤ 30 candidates:** the Agent may compare them all automatically.
- **> 30 candidates:** the Agent asks the user whether to analyze all of them (and then calls with
  `allow_large_group=True`), or to narrow by passenger volume, passenger growth, airport type, geography or
  specific airports. The tools never narrow on their own.

## resolve_airports

Turns airport **names or cities** from the user ("Ted Stevens Anchorage International Airport", "Fairbanks",
"Portland, OR", "ANC") into airport codes for `analyze_airport` / `compare_airports`, so the Agent never depends on
codes it remembers. Entity resolution only: it uses the local OurAirports data (`OurAirportsClient.search_airports`)
and makes no BTS request; the analysis tools still take codes.

- **Levels** (first level with a match wins): exact code (only when written as a 3-4 character upper-case code),
  exact normalized name, place (municipality, or a part of a "/"/"," municipality list, or a whole-word phrase of
  the airport name). An optional `, <state>` narrows to one state. If the whole reference matches nothing,
  `<airport> in <city>` (or `<airport>, <city>`) is tried: the airport part with the same levels, kept only where
  the city (or state) agrees with the data ("John Wayne Airport in Santa Ana" → SNA); a contradicting city gives
  `ambiguous` (`qualifier_mismatch`), and "the airport in <city>" is the city itself.
- **Normalization:** case, accents, punctuation, apostrophes ("O'Hare" = "OHare"), and `Intl`/`St`/`Ft`/`Mt`.
  No fuzzy or edit-distance matching.
- **Narrowing (data only):** among matches with OurAirports scheduled service, the one airport of the largest
  facility type is chosen ("Anchorage" → ANC, not the smaller scheduled Merrill Field, which is reported in
  `other_scheduled_candidates`). Two or more of the same type → `ambiguous`. A place reference is never resolved
  to an airport without scheduled service.
- **Per reference:** `found` (`airport` with `code`, `name`, `municipality`, `state`, `airport_type`,
  `scheduled_service`), `ambiguous` (`reason`, up to 5 `candidates`) or `not_found`. Unresolved references are also
  listed in `errors`; status is `ok`, `partial` or `error`.
- **Code returned:** the IATA code if it identifies exactly this airport, else the OurAirports ident. The
  OurAirports → BTS code rule still runs inside the analysis tools.

## discover_airports

- **Sources:** airports come from OurAirports (`find_airports`). Commercial-activity evidence comes from **one
  batched BTS T-100 query**, with identifiers resolved by the shared OurAirports → BTS rule.
- **No size filter:** there is no passenger threshold, and small airports are kept.
- **`commercial_activity` values:**
  - `confirmed`: BTS T-100 reports passengers in the latest 12 months of T-100 data.
  - `reported_zero`: BTS has recent records, but all with 0 passengers (e.g. cargo-only).
  - `historical_only`: BTS has records, none in the last 12 months.
  - `unknown`: no usable BTS evidence (identifier not found or ambiguous, or values missing). **Not zero activity.**
- **Compact output:** a single table, `columns` + `airports` (one row per listed airport). New England is
  **10.6 KB** (previously 39 KB). The columns are:
  - `code`: pass this to the other tools;
  - `name`, `municipality`, `state`, `airport_type`, `commercial_activity`;
  - `recent_12m_passengers`, `latest_annual_year`, `latest_annual_passengers`;
  - `passenger_cagr` and `cagr_years`: the existing Demand Growth CAGR, computed from rows already fetched,
    with no extra request;
  - `latest_bts_month`, `identifier_status`.
  - There are no operational KPIs here; those come from `compare_airports` and `analyze_airport`.
- **Aggregates:** `candidate_count`, `activity_counts` (every status, including 0), `excluded_counts_by_type`,
  `automatic_analysis_limit` and `exceeds_automatic_analysis_limit`. Discovery never narrows or selects airports;
  the Agent and the user decide.
- **What is not listed:**
  - Airports with **no BTS record and no OurAirports scheduled service** (private or general-aviation fields)
    are counted by type under `excluded_counts_by_type`.
  - Heliports, balloonports and closed facilities are never discovered.

## compare_airports

- **Group size:** up to `AUTOMATIC_ANALYSIS_LIMIT` (30) airports directly. Larger groups need
  `allow_large_group=True` and are then analyzed in full as one group (see *Automatic analysis limit*).
- **Codes:** any code from `discover_airports`, IATA, ICAO or local codes are accepted
  (`OurAirportsClient.lookup_code`). Unknown, ambiguous and non-U.S. codes become structured errors; the remaining
  airports are still compared and the status is `partial`.
- **Pipeline:** runs the existing `analysis.analyze_comparison_group` (data → KPIs → percentile scoring). The tool
  adds **no** scoring logic.
- **Compact output** (about 14 KB for 30 airports; the internal object is about 200 KB):
  - `metadata`: requested airports, analyzed and scored counts, analysis period, coverage summary,
    non-discriminative signals, weights, identifier issues and short factual notes.
  - `top_candidates` (top 5): rank, score, coverage and, per KPI family, the family score, each signal's raw
    value and its percentile, key context, excluded signals and warnings.
  - `ranking`: `{code, name, rank, investment_score, coverage}` for **every** scored airport.
  - `not_scored`: every unscored airport, with its name, status, coverage, identifier status, missing families and data
    availability. Nothing is dropped silently.

## analyze_airport

- **Pipeline:** the same pipeline, run on one airport.
- **Output sections:** `airport` (OurAirports and BTS identifiers, resolution status), `analysis_period`,
  `demand`, `operational_pressure`, `scale_connectivity`, `capacity_pressure` and `data_quality`.
  - `demand`: CAGR, annual passenger history and seasonality.
  - `operational_pressure` and `scale_connectivity`: raw signals plus context.
  - `capacity_pressure`: proxy inputs, marked `is_proxy: true` and `scored: false`.
  - `data_quality`: coverage, missing signals, sources and limitations.
- **No relative score:** `relative_scoring.investment_score` and `rank` are always `null`, with the reason. The score
  is a percentile within a comparison group, so a single airport has none.

## Analysis period

- **Explicit period:** if `year` **and** `month` are given, they are validated against the latest On-Time month BTS
  has published. If the period is later than that (or before 1987), the result is a `requested_period_unavailable` error.
- **No period given:** the tool uses the **latest month BTS has published**. It reads this from the official TranStats
  download page ("Latest Available Data: July 2026") with one request per session, kept in memory. The same month is
  used for **every** airport in a call (`same_month_for_all_airports: true`).
- **Passenger trend years:** these follow the complete-calendar-year CAGR rule and are reported separately. If they
  differ between airports, the result includes a `passenger_trend_period_differs` warning.

## Errors and warnings

Each issue has the form `{"code", "message", "airport"?, "details"?}`. The codes are:

| Code | Meaning |
|---|---|
| `invalid_argument` | Bad input (empty list, unknown region code, …) |
| `airport_not_found` | No airport with this code in OurAirports |
| `airport_code_ambiguous` | The code matches several airports |
| `airport_not_supported` | Not a U.S. airport, so there is no BTS data |
| `comparison_group_requires_confirmation` | More than 30 airports without `allow_large_group=True`; details: `candidate_count`, `automatic_analysis_limit` |
| `large_group_confirmed` (warning) | A group above 30 was analyzed in full after confirmation |
| `invalid_period` | Only one of year and month was given, or they are invalid |
| `requested_period_unavailable` | The requested month is not published |
| `data_source_unavailable` | A BTS source could not be reached |
| `identifier_not_found_in_bts` | No candidate code exists in BTS: **missing data, not zero activity** |
| `identifier_ambiguous` | Several BTS codes could match, so none was used |
| `insufficient_data` | A family or source is unavailable for an airport |
| `identifier_alternate` | A non-IATA BTS code was used (e.g. DJT → PBI) |
| `comparison_group_required` | Fewer than 2 airports, so there is no relative score |
| `duplicate_airport_codes` | A duplicate code was analyzed once |
| `passenger_trend_period_differs` | Airports' CAGR years differ |
| `ontime_bulk_fallback_used` | The TranStats form failed and the BTS monthly file was used |

The existing safe fallback (`allow_bulk_fallback`: a monthly file streamed into a temp file that deletes itself) is
enabled in the tools and reported when used. If a source still fails, the tool returns `data_source_unavailable`.
