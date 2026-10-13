"""Passenger-demand metrics calculated from BTS T-100 rows
(BtsT100Client.get_monthly_airport_stats output, one airport).

Input rows are dicts such as
{"reporting_month": "2026-04-01T00:00:00.000", "total_passengers": 166273.0, ...}.

What total_passengers measures (validated, see Data Coverage Matrix row 25): passengers
DEPARTING (enplaned at) the airport in that month, all carriers in T-100. Every metric
below inherits that definition.

Missing data: the data.bts.gov API omits fields that are null, so a row may have no
total_passengers. That is treated as UNKNOWN, never as 0.

No function here makes API calls.
"""

from statistics import mean

from config import EXCLUDED_TREND_YEARS, TREND_EARLIEST_YEAR

# Demand-trend (CAGR) project rules, decided 2026-10-01 (see config.py, docs/METRICS.md):
# COVID years 2020-2021 never part of a trend period; post-COVID analysis starts in 2022.


def _monthly_passengers(rows):
    """{(year, month): passengers or None}. Raises ValueError on a duplicate month."""
    months = {}
    for row in rows:
        reporting_month = row.get("reporting_month")
        try:
            key = (int(reporting_month[0:4]), int(reporting_month[5:7]))
        except (TypeError, ValueError):
            raise ValueError(f"Row has no valid reporting_month: {reporting_month!r}")
        if key in months:
            raise ValueError(
                f"Duplicate month {key[0]}-{key[1]:02d}: pass rows for one airport only."
            )
        value = row.get("total_passengers")
        months[key] = float(value) if isinstance(value, (int, float)) else None
    return months


def annual_passenger_totals(rows):
    """Yearly passengers = sum of monthly total_passengers (Matrix row 26 note).

    Input: T-100 rows for one airport.
    Calculation: for each calendar year, sum total_passengers over months with a known
    value.
    Returns: {year: {"passengers": sum or None, "months": n, "complete": bool}}, sorted by
    year. "months" counts months with a known value; "complete" is True only when all 12
    are known. A partial year's sum is NOT comparable with full years - check "complete".
    "passengers" is None when no month of that year has a known value.
    """
    totals = {}
    for (year, _month), passengers in sorted(_monthly_passengers(rows).items()):
        entry = totals.setdefault(year, {"passengers": None, "months": 0, "complete": False})
        if passengers is not None:
            entry["passengers"] = (entry["passengers"] or 0.0) + passengers
            entry["months"] += 1
    for entry in totals.values():
        entry["complete"] = entry["months"] == 12
    return totals


def yoy_monthly_growth(rows):
    """Year-over-year passenger growth for each month (Matrix row 27).

    Input: T-100 rows for one airport.
    Calculation: growth(Y, m) = (P(Y, m) - P(Y-1, m)) / P(Y-1, m), where P is
    total_passengers for calendar month m. Expressed as a fraction (0.05 = +5%).
    Returns: list (oldest first) of {"month": "YYYY-MM", "passengers",
    "prior_year_passengers", "growth"} for every month present in the input.
    growth is None when either value is unknown, the prior-year month is absent, or the
    prior-year value is 0 (growth from zero is undefined).
    """
    months = _monthly_passengers(rows)
    result = []
    for (year, month), passengers in sorted(months.items()):
        prior = months.get((year - 1, month))
        growth = None
        if passengers is not None and prior:  # prior is neither None nor 0
            growth = (passengers - prior) / prior
        result.append({
            "month": f"{year}-{month:02d}",
            "passengers": passengers,
            "prior_year_passengers": prior,
            "growth": growth,
        })
    return result


def passenger_cagr(rows, earliest_year=TREND_EARLIEST_YEAR):
    """Long-term passenger demand trend as CAGR (Matrix row 28, project definition).

    Input: T-100 rows for one airport; earliest_year (default 2022): first calendar year
    allowed in the trend period (post-COVID analysis).
    Eligible years: complete calendar years (all 12 months known, see
    annual_passenger_totals), not in EXCLUDED_TREND_YEARS (2020, 2021), and
    >= earliest_year (None = no lower bound).
    Sequence: the MOST RECENT run of at least two consecutive eligible years. Years are
    never bridged: an excluded or incomplete year ends a run, so the sequence is always
    contiguous.
    With the default earliest_year=2022 and complete data this is 2022 .. latest complete
    year.
    Calculation: CAGR = (P_last / P_first) ** (1 / (last_year - first_year)) - 1, with P the
    annual passenger totals of the first and last years of the sequence. Fraction
    (0.02 = +2% per year).
    Returns: {"first_year", "last_year", "years" (years in the sequence, inclusive),
    "periods" (last_year - first_year, the CAGR exponent denominator), "first_passengers",
    "last_passengers", "cagr", "eligible_years", "reason"}. When fewer than two consecutive
    eligible years exist (or P_first is 0) all values are None, cagr is None and "reason"
    explains why - no trend is invented.
    """
    totals = annual_passenger_totals(rows)
    eligible = sorted(
        year for year, entry in totals.items()
        if entry["complete"]
        and year not in EXCLUDED_TREND_YEARS
        and (earliest_year is None or year >= earliest_year)
    )
    result = {
        "first_year": None, "last_year": None, "years": None, "periods": None,
        "first_passengers": None, "last_passengers": None, "cagr": None,
        "eligible_years": eligible, "reason": None,
    }

    # Split eligible years into runs of consecutive years; use the most recent run that
    # has at least two years (a single year cannot give a growth rate).
    runs = []
    for year in eligible:
        if runs and year == runs[-1][-1] + 1:
            runs[-1].append(year)
        else:
            runs.append([year])
    usable = [run for run in runs if len(run) >= 2]
    if not usable:
        result["reason"] = "Fewer than two consecutive complete eligible years."
        return result
    run = usable[-1]

    first, last = run[0], run[-1]
    p_first, p_last = totals[first]["passengers"], totals[last]["passengers"]
    result.update({
        "first_year": first, "last_year": last, "years": len(run), "periods": last - first,
        "first_passengers": p_first, "last_passengers": p_last,
    })
    if p_first == 0:
        result["reason"] = "First-year passengers are 0; CAGR is undefined."
        return result
    result["cagr"] = (p_last / p_first) ** (1 / (last - first)) - 1
    return result


def seasonality_index(rows, years):
    """Monthly seasonal index of passengers (Matrix row 29) - simple-average method.
    Approved project definition (2026-10-01).

    Input: T-100 rows for one airport, and `years`: the calendar years to use, chosen
    explicitly by the caller (e.g. to leave out the 2020-2021 COVID years). Every listed
    year must have all 12 months with a known value, otherwise ValueError (a partial year
    would bias the pattern).
    Calculation: index(m) = mean of P(y, m) over the given years
                            / mean of P(y, m) over all given years and months.
    1.0 = an average month; 1.25 = 25% above the average month. The 12 indexes average 1.0.
    Returns: {1: index, ..., 12: index}.
    """
    years = sorted(set(years))
    if not years:
        raise ValueError("years must contain at least one calendar year.")
    months = _monthly_passengers(rows)

    for year in years:
        missing = [m for m in range(1, 13) if months.get((year, m)) is None]
        if missing:
            raise ValueError(f"Year {year} is incomplete: no passenger value for months {missing}.")

    overall = mean(months[(year, m)] for year in years for m in range(1, 13))
    if overall == 0:
        raise ValueError("Average monthly passengers is 0 for the given years; index undefined.")
    return {
        m: mean(months[(year, m)] for year in years) / overall
        for m in range(1, 13)
    }
