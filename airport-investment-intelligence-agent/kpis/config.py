"""KPI-layer configuration: family names, signal directions and the scoring settings.

The tunable scoring settings - Investment Score family weights, signal weights inside each
family and the coverage thresholds - live in the project's config.py and are re-exported
here unchanged. Investment-score weights are project decisions (V1). The internal signal
weights inside Operational Pressure and Scale & Connectivity are the PROPOSED V1 scheme
documented in docs/KPIS.md - review them before relying on the score.
"""

from config import (
    FAMILY_SIGNAL_WEIGHTS,
    INVESTMENT_SCORE_WEIGHTS,
    MIN_FAMILY_COVERAGE,
    MIN_INVESTMENT_SCORE_COVERAGE,
)

DEMAND_GROWTH = "demand_growth"
OPERATIONAL_PRESSURE = "operational_pressure"
SCALE_CONNECTIVITY = "scale_connectivity"

# INVESTMENT_SCORE_WEIGHTS (config.py): final Investment Opportunity Score = the three base
# KPI families only. Capacity Pressure is deliberately NOT a component: it is a proxy
# synthesised from these same signals and would double-count them.
# FAMILY_SIGNAL_WEIGHTS (config.py): signals scored inside each family, with their weight
# inside the family (each family's weights sum to 1.0). Signal names are keys of
# build_airport_kpis()[family]["signals"].

# Direction of each signal: "higher" = a higher raw value means MORE of the KPI (higher
# score); "lower" = a lower raw value means more (shorter spacing = more pressure).
SIGNAL_DIRECTIONS = {
    "passenger_cagr": "higher",
    "scheduled_p95_per_operational_runway": "higher",
    "scheduled_peak_spacing_minutes": "lower",
    "departure_delayed_15_rate": "higher",
    "arrival_delayed_15_rate": "higher",
    "departure_delayed_60_rate": "higher",
    "annual_passengers": "higher",
    "scheduled_departures": "higher",
    "unique_destinations": "higher",
    "long_haul_flights": "higher",
    "long_haul_share": "higher",
}

# Relative (percentile) scoring needs at least this many airports in the comparison group.
MIN_COMPARISON_GROUP_SIZE = 2

_FAMILIES = {DEMAND_GROWTH, OPERATIONAL_PRESSURE, SCALE_CONNECTIVITY}
if set(INVESTMENT_SCORE_WEIGHTS) != _FAMILIES or set(FAMILY_SIGNAL_WEIGHTS) != _FAMILIES:
    raise ValueError(f"config.py: score weights must be keyed by exactly the KPI families {sorted(_FAMILIES)}.")
_unknown = {s for weights in FAMILY_SIGNAL_WEIGHTS.values() for s in weights} - set(SIGNAL_DIRECTIONS)
if _unknown:
    raise ValueError(f"config.py: unknown KPI signals in FAMILY_SIGNAL_WEIGHTS: {sorted(_unknown)}.")

# Coverage rule (V1, explicit redistribution). Within a family, and across families, the
# configured weights of unusable parts (missing for the airport, or without ranking
# information in the group) are redistributed proportionally over the usable parts. The
# configured weights above never change; the effective weights used are always reported.
#
# Investment score coverage = sum over families of
#     INVESTMENT_SCORE_WEIGHTS[family] * (share of that family's signal weight the airport
#     has data for). Non-discriminative / not-rankable signals are a property of the GROUP,
#     not missing data, so they do not reduce an airport's coverage.
# MIN_INVESTMENT_SCORE_COVERAGE (config.py), PROPOSED V1 minimum: 0.60 = the smallest combined
# weight of any two families (Demand Growth 0.35 + Scale & Connectivity 0.25). An airport can
# lose one whole family and still be scored, but not two; an airport without any BTS On-Time
# data (coverage 0.45) or without any T-100 data (0.55) is not scored.
#
# MIN_FAMILY_COVERAGE (config.py): a KPI family gets a score only if the airport has data for
# at least this share of the family's configured signal weight. Prevents a minor signal (e.g.
# arrival delays, 0.075) from being redistributed into the entire family weight.
# Non-discriminative signals do not reduce coverage (they are a group property, not missing data).
