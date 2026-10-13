"""Comparison-group normalization and the Investment Opportunity Score (no API calls).

score_comparison_group() takes the raw KPI records of several airports (from
build_airport_kpis), ranks every signal as a percentile WITHIN that group, combines the
usable signals into the three KPI family scores and the weighted Investment Opportunity
Score, and ranks the airports. Raw values are preserved next to every score.

Scores are RELATIVE to the airports in the group: there are no global thresholds, and the
same airport can score differently in a different group.

Two different reasons a signal can be left out of a score are tracked separately:
- missing:            THIS airport has no raw value (e.g. no On-Time data, runway count
                      uncertain). Reduces the airport's coverage.
- non_discriminative: the GROUP has the measurement, but every airport with a value has
                      the same value, so it carries no ranking information (it would add the
                      same 50 points to everyone). Does not reduce coverage.
  (not_rankable:      fewer than MIN_COMPARISON_GROUP_SIZE airports have a value, so the
                      signal cannot be ranked; also a group property.)
In both cases the configured weight is redistributed proportionally over the usable
signals, and the effective weights are reported. The configured weights never change.
"""

import math

from kpis.config import (
    FAMILY_SIGNAL_WEIGHTS,
    INVESTMENT_SCORE_WEIGHTS,
    MIN_COMPARISON_GROUP_SIZE,
    MIN_FAMILY_COVERAGE,
    MIN_INVESTMENT_SCORE_COVERAGE,
    SIGNAL_DIRECTIONS,
)

PERCENTILE_METHOD = (
    "Percentile rank within the comparison group, 0-100, mid-rank for ties: "
    "score = 100 * (below + (equal - 1) / 2) / (m - 1), where m = airports with a value for "
    "the signal, below = airports with a strictly lower value, equal = airports with the same "
    "value (including itself). Lowest = 0, highest = 100, tied airports share the average "
    "score. For 'lower is more' signals the score is 100 - percentile. Airports without a "
    "value are left out of the ranking. A signal where all values are equal is "
    "'non_discriminative' and is not scored."
)
SINGLE_AIRPORT_MESSAGE = (
    "A relative Investment Opportunity Score needs a comparison group of at least "
    f"{MIN_COMPARISON_GROUP_SIZE} airports. Raw KPIs are returned; no 0-100 score is fabricated."
)
_TOLERANCE = 1e-9   # float tolerance for coverage comparisons (e.g. 0.35 + 0.25 vs 0.60)


def _check_values(values):
    for airport, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Signal value for {airport} must be a finite number, got {value!r}.")


def percentile_scores(values, direction="higher"):
    """Percentile-rank one signal across a group (see PERCENTILE_METHOD).

    values: {airport: raw value or None}. direction: "higher" or "lower" (lower raw value
    = higher score). Returns {airport: score 0-100 or None}. None for airports without a
    value, and for everyone if fewer than MIN_COMPARISON_GROUP_SIZE airports have one.
    Pure ranking maths: all-equal values give 50 each. score_comparison_group does NOT use
    such a signal (see signal_group_status).
    """
    if direction not in ("higher", "lower"):
        raise ValueError("direction must be 'higher' or 'lower'.")
    _check_values(values)
    known = {airport: v for airport, v in values.items() if v is not None}
    scores = dict.fromkeys(values)
    m = len(known)
    if m < MIN_COMPARISON_GROUP_SIZE:
        return scores
    for airport, value in known.items():
        below = sum(other < value for other in known.values())
        equal = sum(other == value for other in known.values())
        percentile = 100 * (below + (equal - 1) / 2) / (m - 1)
        scores[airport] = percentile if direction == "higher" else 100 - percentile
    return scores


def signal_group_status(values):
    """Group-level status of one signal: "ranked", "non_discriminative" (all available
    values identical) or "not_rankable" (fewer than MIN_COMPARISON_GROUP_SIZE values)."""
    _check_values(values)
    known = [v for v in values.values() if v is not None]
    if len(known) < MIN_COMPARISON_GROUP_SIZE:
        return "not_rankable"
    if all(v == known[0] for v in known):
        return "non_discriminative"
    return "ranked"


def _competition_ranks(scores):
    """{airport: score} -> {airport: rank}, highest score = 1, ties share a rank (1, 2, 2, 4)."""
    return {airport: 1 + sum(other > score for other in scores.values())
            for airport, score in scores.items()}


def _effective_weights(base_weights, usable):
    """Proportionally redistribute base weights over the usable keys (sums to 1.0)."""
    total = sum(base_weights[key] for key in usable)
    return {key: base_weights[key] / total for key in usable} if total else {}


def _family_result(base_weights, raw_signals, normalized, statuses):
    """Score one KPI family for one airport from its usable signals."""
    missing = [s for s in base_weights if raw_signals[s] is None]
    non_discriminative = [s for s in base_weights
                          if raw_signals[s] is not None and statuses[s] == "non_discriminative"]
    not_rankable = [s for s in base_weights
                    if raw_signals[s] is not None and statuses[s] == "not_rankable"]
    usable = [s for s in base_weights if normalized[s] is not None]
    coverage = sum(base_weights[s] for s in base_weights if raw_signals[s] is not None)
    insufficient = coverage < MIN_FAMILY_COVERAGE - _TOLERANCE
    if insufficient:
        # Too little of this family is measured: no score, and no redistribution of a
        # minor signal into the whole family weight. Normalized signals stay visible.
        usable = []
    effective = _effective_weights(base_weights, usable)
    reason = None
    if insufficient:
        reason = (f"insufficient_family_coverage: {coverage:.3f} < {MIN_FAMILY_COVERAGE:.2f} "
                  f"(missing: {missing})")
    elif not usable:
        reason = ("No usable signal: " + "; ".join(
            f"{label}: {names}" for label, names in (("missing", missing),
                                                     ("non_discriminative", non_discriminative),
                                                     ("not_rankable", not_rankable)) if names))
    return {
        "score": sum(w * normalized[s] for s, w in effective.items()) if usable else None,
        "coverage": coverage,
        "insufficient_family_coverage": insufficient,
        "effective_weights": effective,
        "missing_signals": missing,
        "non_discriminative_signals": non_discriminative,
        "not_rankable_signals": not_rankable,
        "reason_if_no_score": reason,
    }


def score_comparison_group(kpi_records):
    """Normalize KPIs within a comparison group and compute the Investment Opportunity Score.

    Input: list of build_airport_kpis() records, one per airport (unique airport codes).
    Steps:
      1. Each signal gets a group status (signal_group_status). "ranked" signals are
         percentile-ranked across the airports that have a value (percentile_scores).
      2. Family score (per airport) = sum(effective weight * percentile) over the usable
         signals (ranked and not missing); effective weight = configured weight / sum of
         the configured weights of the usable signals. family coverage = share of the
         family's configured weight the airport has raw data for. No family score if
         family coverage < MIN_FAMILY_COVERAGE ("insufficient_family_coverage").
      3. Investment Opportunity Score = sum(effective family weight * family score) over
         families with a score, effective family weight = INVESTMENT_SCORE_WEIGHTS[f] /
         sum over scored families. Coverage = sum(INVESTMENT_SCORE_WEIGHTS[f] * family
         coverage). No score if coverage < MIN_INVESTMENT_SCORE_COVERAGE or no family has a
         score. Capacity Pressure inputs are passed through untouched and never scored.
      4. Rank = competition rank among airports with a score (ties share a rank).
    With fewer than MIN_COMPARISON_GROUP_SIZE airports nothing is normalized:
    status "comparison_group_required" and raw KPIs only.

    Returns: {"status", "message", "group_size", "weights", "min_investment_score_coverage",
    "percentile_method", "signal_status", "non_discriminative_signals", "airports": [...]}.
    Per airport: "airport", "rank", "investment_score", "score_status" ("complete" =
    coverage 1.0 | "partial" = scored with coverage < 1.0 | "insufficient_coverage" |
    "comparison_group_required"), "coverage",
    "effective_family_weights", "missing_families", "family_scores", "families" (per-family
    score, coverage, effective weights, missing / non-discriminative / not-rankable signals),
    "normalized_signals" and "raw_kpis" (the input record).
    Airports are listed by rank, then code; unscored airports last.
    """
    if not kpi_records:
        raise ValueError("kpi_records must contain at least one airport.")
    codes = [record["airport"] for record in kpi_records]
    if len(set(codes)) != len(codes):
        raise ValueError(f"Duplicate airport in comparison group: {codes}")
    group_ready = len(kpi_records) >= MIN_COMPARISON_GROUP_SIZE

    statuses = {family: {} for family in FAMILY_SIGNAL_WEIGHTS}
    normalized = {code: {family: {} for family in FAMILY_SIGNAL_WEIGHTS} for code in codes}
    for family, signal_weights in FAMILY_SIGNAL_WEIGHTS.items():
        for signal in signal_weights:
            raw = {r["airport"]: r["kpis"][family]["signals"][signal] for r in kpi_records}
            status = signal_group_status(raw) if group_ready else "comparison_group_required"
            statuses[family][signal] = status
            ranked = (percentile_scores(raw, SIGNAL_DIRECTIONS[signal]) if status == "ranked"
                      else dict.fromkeys(raw))
            for code, score in ranked.items():
                normalized[code][family][signal] = score

    results = []
    for record in kpi_records:
        code = record["airport"]
        families = {
            family: _family_result(weights, record["kpis"][family]["signals"],
                                   normalized[code][family], statuses[family])
            for family, weights in FAMILY_SIGNAL_WEIGHTS.items()
        }
        scored_families = [f for f in INVESTMENT_SCORE_WEIGHTS if families[f]["score"] is not None]
        coverage = sum(w * families[f]["coverage"] for f, w in INVESTMENT_SCORE_WEIGHTS.items())
        effective, investment = {}, None
        if not group_ready:
            status = "comparison_group_required"
        elif not scored_families or coverage < MIN_INVESTMENT_SCORE_COVERAGE - _TOLERANCE:
            status = "insufficient_coverage"
        else:
            effective = _effective_weights(INVESTMENT_SCORE_WEIGHTS, scored_families)
            investment = sum(w * families[f]["score"] for f, w in effective.items())
            # "complete" describes the AIRPORT's data (coverage 1.0). Signals or families left
            # out because the group has no variation are not missing data; they are listed
            # under families / missing_families instead.
            status = "complete" if coverage > 1 - _TOLERANCE else "partial"

        results.append({
            "airport": code,
            "rank": None,
            "investment_score": investment,
            "score_status": status,
            "coverage": round(coverage, 10),
            "effective_family_weights": effective,
            "missing_families": {f: families[f]["reason_if_no_score"]
                                 for f in INVESTMENT_SCORE_WEIGHTS if families[f]["score"] is None}
                                if group_ready else {},
            "family_scores": {f: families[f]["score"] for f in INVESTMENT_SCORE_WEIGHTS},
            "families": families,
            "normalized_signals": normalized[code],
            "raw_kpis": record,
        })

    ranks = _competition_ranks({r["airport"]: r["investment_score"] for r in results
                                if r["investment_score"] is not None})
    for result in results:
        result["rank"] = ranks.get(result["airport"])
    results.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["airport"]))

    scored = sum(r["investment_score"] is not None for r in results)
    if not group_ready:
        status, message = "comparison_group_required", SINGLE_AIRPORT_MESSAGE
    else:
        status = "scored"
        message = (f"{scored} of {len(results)} airports have an Investment Opportunity Score "
                   f"(minimum coverage {MIN_INVESTMENT_SCORE_COVERAGE:.2f}).")
    return {
        "status": status,
        "message": message,
        "group_size": len(results),
        "weights": {"investment": INVESTMENT_SCORE_WEIGHTS, "signals": FAMILY_SIGNAL_WEIGHTS},
        "min_investment_score_coverage": MIN_INVESTMENT_SCORE_COVERAGE,
        "min_family_coverage": MIN_FAMILY_COVERAGE,
        "percentile_method": PERCENTILE_METHOD,
        "signal_status": statuses,
        "non_discriminative_signals": [f"{family}.{signal}" for family, signals in statuses.items()
                                       for signal, s in signals.items() if s == "non_discriminative"],
        "airports": results,
    }
