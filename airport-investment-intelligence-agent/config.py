"""Central configuration of the Airport Investment Intelligence Agent.

The tunable behavior of the system in one place: the LLM and its budgets, conversation
memory, analysis thresholds, comparison guardrails and the Investment Opportunity Score
weights. The modules that use a setting import it from here (and re-export it under its
old name), so this file has no project imports and can be imported by anything.

NOT here, on purpose:
- secrets: GEMINI_API_KEY and the OpenSky credentials are read from environment variables /
  the .env file (never from this file);
- implementation details kept next to their code: tool and error names, data-source URLs and
  field names, parsing tokens, output-size bounds, signal directions and data facts such as
  which countries BTS covers.

Changing a value here changes results: Gemini-visible values (guardrail size, top-K, memory
results) also change the tool descriptions, and any change makes earlier live-evaluation runs
non-resumable (see evaluation/storage.py).
"""

# --- LLM / Agent ------------------------------------------------------------------------------

# Gemini model used by every Agent turn (AgentSession, Chainlit chat, live evaluation).
MODEL = "gemini-3.5-flash-lite"

# Per-request timeout in milliseconds; also sent to the server as the request deadline
# (30 s gave 504 DEADLINE_EXCEEDED on long tool-using turns).
TIMEOUT_MS = 120_000

# Budgets per user turn. A tool round = one model response whose tool calls were executed.
MAX_TOOL_ROUNDS = 6
# Normal model calls: every allowed tool round + one final answer. When a budget is reached,
# ONE extra finalization call (tools disabled) answers from the evidence already collected.
MAX_LLM_CALLS = MAX_TOOL_ROUNDS + 1

# --- Conversation / memory --------------------------------------------------------------------

# Completed earlier turns sent to the model with each new message. Older turns are reachable
# only through the retrieve_conversation_context tool.
RECENT_TURNS = 2

# Earlier turns returned by one retrieve_conversation_context call, however broad the query.
MAX_MEMORY_RESULTS = 3

# --- Agent tools: comparison guardrails -------------------------------------------------------

# compare_airports analyses up to this many airports without asking; a larger group needs the
# user's explicit authorisation (allow_large_group) and is then analysed in full as ONE group
# (benchmark 2026-10-04: ~40-75 s for ~30 airports). Not a computational maximum.
AUTOMATIC_ANALYSIS_LIMIT = 30

# compare_airports returns full KPI-family detail for this many top-ranked airports (every
# airport still gets its score, rank and coverage).
TOP_K = 5

# discover_airports: "recent" commercial activity = BTS T-100 passengers in this many latest
# months of T-100 data.
RECENT_ACTIVITY_MONTHS = 12

# --- Analysis thresholds (metrics layer, see docs/METRICS.md) ---------------------------------

# A flight is "delayed" at >= 15 minutes (U.S. DOT on-time threshold; BTS DepDel15/ArrDel15).
DELAY_THRESHOLD_MINUTES = 15
# Extreme-disruption indicator.
SEVERE_DELAY_THRESHOLD_MINUTES = 60
# Peak-traffic percentile used to compare airports' busy periods.
PEAK_PERCENTILE = 95
# Long-haul flight: >= 3,450 statute miles (~3,000 nautical miles, the ICAO long-range
# boundary adopted as the project threshold).
LONG_HAUL_THRESHOLD_MILES = 3450

# Passenger-growth (CAGR) trend window: complete calendar years from this year on, never
# including the COVID years.
TREND_EARLIEST_YEAR = 2022
EXCLUDED_TREND_YEARS = frozenset({2020, 2021})

# --- Investment Opportunity Score (KPI layer, see docs/KPIS.md) -------------------------------
# Keys are the KPI family / signal names used by kpis/ (checked there at import).

# Final score = weighted KPI families (project decision, V1). Capacity Pressure is deliberately
# NOT a component: it is a proxy synthesised from these same signals and would double-count them.
INVESTMENT_SCORE_WEIGHTS = {
    "operational_pressure": 0.40,
    "demand_growth": 0.35,
    "scale_connectivity": 0.25,
}

# Signals scored inside each family, with their weight inside the family (each family sums to
# 1.0). Operational Pressure and Scale & Connectivity weights are the PROPOSED V1 scheme.
FAMILY_SIGNAL_WEIGHTS = {
    "demand_growth": {
        "passenger_cagr": 1.00,                     # only signal (latest volume is context)
    },
    "operational_pressure": {                       # PROPOSED V1 internal weights
        "scheduled_p95_per_operational_runway": 0.50,      # primary: scheduled peak demand vs runways
        "scheduled_peak_spacing_minutes": 0.20,     # secondary
        "departure_delayed_15_rate": 0.15,          # delay pressure (0.30 in total) ...
        "arrival_delayed_15_rate": 0.075,           # ... arrivals weigh less than departures
        "departure_delayed_60_rate": 0.075,         # ... extreme disruption
    },
    "scale_connectivity": {                         # PROPOSED V1 internal weights
        "annual_passengers": 0.40,                  # primary
        "scheduled_departures": 0.20,               # secondary
        "unique_destinations": 0.20,                # connectivity (0.40 in total) ...
        "long_haul_flights": 0.10,
        "long_haul_share": 0.10,
    },
}

# An airport gets an Investment Score only if it has data for at least this share of the total
# configured weight. 0.60 = the smallest combined weight of any two families (0.35 + 0.25): an
# airport may lose one whole family, not two. Weights of unusable parts are redistributed
# proportionally over usable parts; the configured weights above never change.
MIN_INVESTMENT_SCORE_COVERAGE = 0.60

# A KPI family gets a score only if the airport has data for at least this share of the
# family's signal weight (stops a minor signal, e.g. arrival delays 0.075, from standing in
# for the whole family).
MIN_FAMILY_COVERAGE = 0.50
