"""Flight-activity metrics calculated from BTS On-Time rows (BtsOnTimeClient.get_flights output).

Input rows are dicts with the BtsOnTimeClient FIELDS, all values as raw strings, e.g.
{"FlightDate": "2026-04-01", "Origin": "ANC", "Dest": "MSP", "CRSDepTime": "2115",
 "DepDelay": "-1.00", "CRSArrTime": "0538", "ArrDelay": "-4.00", "Cancelled": "0.00",
 "Diverted": "0.00", ...}

Coverage reminder: On-Time contains only reporting carriers' domestic scheduled passenger
flights (~25% of ANC departures in Apr-2026). Every metric here describes that subset only.

Times
- All On-Time clock times are LOCAL time of the airport where the event happens, so
  departures and arrivals at one airport share the same local clock.
- time_basis="scheduled": CRSDepTime / CRSArrTime. Every scheduled flight has these,
  including cancelled ones, so this describes the PLANNED schedule.
- time_basis="actual": only movements that actually happened at the airport - departures
  with Cancelled == 0, arrivals with Cancelled == 0 and Diverted == 0 (same definition as
  "operated"/"completed" in flight_counts) - at scheduled time + DepDelay / ArrDelay
  minutes. Everything else (cancelled, diverted arrivals, unknown status, missing delay)
  is excluded and the number excluded is always returned. Adding the delay (instead of
  parsing DepTime / ArrTime) keeps the correct date when a delay crosses midnight, and
  avoids ArrTime of diverted flights, which is the landing time at another airport.
  Note: BTS keeps DepTime/DepDelay for some cancelled flights (left the gate, then
  cancelled); these are NOT counted as actual departures.
- Arrival date (accepted project limitation): BTS On-Time does not provide an explicit
  arrival date in our selected fields; FlightDate is the scheduled DEPARTURE date. For
  overnight flights we assume the arrival is on the following day when the arrival clock
  time (CRSArrTime) is earlier than the departure clock time (CRSDepTime). This can be
  wrong only for a flight that departs just after local midnight and arrives "earlier" on
  the clock because it crossed time zones westward - rare. No time-zone data or other
  source is added for this edge case.

Project definitions (decided 2026-10-01, see docs/METRICS.md)
- Peak traffic for airport comparison = 95th percentile of hourly movement counts
  (peak_hour_traffic). busiest_hour (the maximum) is kept as a descriptive building block.
- Peak-period spacing = median minutes between consecutive movements inside peak periods
  defined from that same hourly distribution (peak_period_spacing).
- Long-haul = route distance >= 3,450 statute miles (long_haul_summary).
- Delayed flight = delay >= 15 minutes (delay_metrics).

No function here makes API calls.
"""

import math
from datetime import datetime, time, timedelta
from statistics import median

from config import DELAY_THRESHOLD_MINUTES, LONG_HAUL_THRESHOLD_MILES, PEAK_PERCENTILE, SEVERE_DELAY_THRESHOLD_MINUTES
from metrics.airports import airport_distance_miles

TIME_BASES = ("scheduled", "actual")
MOVEMENT_TYPES = ("all", "departures", "arrivals")

# Project definitions (see config.py and docs/METRICS.md): delayed >= 15 min (U.S. DOT /
# BTS DepDel15), severe delay >= 60 min, peak percentile 95, long-haul >= 3,450 statute miles
# (~3,000 nmi; there is no universal long-haul definition, the ICAO long-range boundary is used).


# --- parsing helpers -------------------------------------------------------------------

def _flag(value):
    """BTS indicator '1.00' / '0.00' -> True / False; anything else -> None (unknown)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == 1:
        return True
    if number == 0:
        return False
    return None


def _minutes(value):
    """BTS minutes string ('-13.00') -> float, or None if missing/unparseable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _local_datetime(flight_date, hhmm):
    """'2026-04-01' + '2115' -> datetime(2026, 4, 1, 21, 15). '2400' = midnight at day end.
    Returns None if either value is missing or malformed."""
    try:
        day = datetime.strptime(flight_date, "%Y-%m-%d")
        hours, minutes = int(hhmm[:2]), int(hhmm[2:])
    except (TypeError, ValueError):
        return None
    valid_clock_time = 0 <= hours <= 23 and 0 <= minutes <= 59
    if len(hhmm) != 4 or not (valid_clock_time or hhmm == "2400"):
        return None
    return day + timedelta(hours=hours, minutes=minutes)


def _scheduled_departure(flight):
    return _local_datetime(flight.get("FlightDate"), flight.get("CRSDepTime"))


def _scheduled_arrival(flight):
    departure = _scheduled_departure(flight)
    arrival = _local_datetime(flight.get("FlightDate"), flight.get("CRSArrTime"))
    if departure is None or arrival is None:
        return None
    if arrival < departure:  # overnight flight: arrives the next calendar day
        arrival += timedelta(days=1)
    return arrival


def _check_choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}.")


def _airport_movements(flights, airport, time_basis, movement):
    """Return (list of (local datetime, 'departure'|'arrival'), number excluded)."""
    _check_choice("time_basis", time_basis, TIME_BASES)
    _check_choice("movement", movement, MOVEMENT_TYPES)
    airport = airport.strip().upper()

    times, excluded = [], 0
    for flight in flights:
        cancelled = _flag(flight.get("Cancelled"))
        events = []
        if flight.get("Origin") == airport and movement in ("all", "departures"):
            # Operated departure: Cancelled == 0 (a diverted flight still departed).
            operated = cancelled is False
            events.append(("departure", _scheduled_departure(flight), "DepDelay", operated))
        if flight.get("Dest") == airport and movement in ("all", "arrivals"):
            # Operated arrival: not cancelled and not diverted (it landed here).
            operated = cancelled is False and _flag(flight.get("Diverted")) is False
            events.append(("arrival", _scheduled_arrival(flight), "ArrDelay", operated))

        for kind, moment, delay_field, operated in events:
            if moment is not None and time_basis == "actual":
                # BTS keeps DepTime/DepDelay for some flights that left the gate and were
                # then cancelled, so "operated" is checked explicitly, not via a missing delay.
                delay = _minutes(flight.get(delay_field))
                moment = moment + timedelta(minutes=delay) if operated and delay is not None else None
            if moment is None:
                excluded += 1
            else:
                times.append((moment, kind))
    return times, excluded


# --- metrics ---------------------------------------------------------------------------

def flight_counts(flights, airport):
    """Departures, arrivals and total flight volume at `airport` (Matrix rows 6-8).

    Input: On-Time rows; `airport` is the BTS 3-letter code (e.g. 'ANC').
    Calculation, per direction:
      - scheduled = rows with Origin == airport (departures) / Dest == airport (arrivals)
      - cancelled = Cancelled == 1
      - departures operated = Cancelled == 0 (a diverted flight still departed)
      - arrivals: diverted = Cancelled == 0 and Diverted == 1 (did not land at airport);
                  completed = Cancelled == 0 and Diverted == 0
      - unknown = rows whose Cancelled/Diverted value is missing or unreadable
    total.scheduled = departures.scheduled + arrivals.scheduled
    total.operated  = departures.operated + arrivals.completed (movements that happened)
    Returns: {"departures": {...}, "arrivals": {...}, "total": {"scheduled", "operated"}}.
    Empty input gives all zeros (zero flights in the data is a real count).
    """
    airport = airport.strip().upper()
    dep = {"scheduled": 0, "cancelled": 0, "operated": 0, "unknown": 0}
    arr = {"scheduled": 0, "cancelled": 0, "diverted": 0, "completed": 0, "unknown": 0}

    for flight in flights:
        cancelled = _flag(flight.get("Cancelled"))
        if flight.get("Origin") == airport:
            dep["scheduled"] += 1
            if cancelled is None:
                dep["unknown"] += 1
            elif cancelled:
                dep["cancelled"] += 1
            else:
                dep["operated"] += 1
        if flight.get("Dest") == airport:
            arr["scheduled"] += 1
            diverted = _flag(flight.get("Diverted"))
            if cancelled is True:
                arr["cancelled"] += 1
            elif cancelled is None or diverted is None:
                arr["unknown"] += 1
            elif diverted:
                arr["diverted"] += 1
            else:
                arr["completed"] += 1

    return {
        "departures": dep,
        "arrivals": arr,
        "total": {
            "scheduled": dep["scheduled"] + arr["scheduled"],
            "operated": dep["operated"] + arr["completed"],
        },
    }


def cancellation_rate(flights, airport):
    """Share of scheduled flights at `airport` that were cancelled (Matrix row 24).

    Input: On-Time rows (cancelled flights remain as rows in BTS data).
    Calculation: count(Cancelled == 1) / count(rows with a known Cancelled value), over
    rows where Origin == airport or Dest == airport (each row counted once).
    Rows with a missing/unreadable Cancelled value are excluded from both counts and
    reported as "unknown".
    Returns: {"scheduled", "cancelled", "unknown", "rate"}; rate is None if no row has a
    known Cancelled value (rate is undefined, not 0).
    """
    airport = airport.strip().upper()
    scheduled = cancelled = unknown = 0
    for flight in flights:
        if airport not in (flight.get("Origin"), flight.get("Dest")):
            continue
        flag = _flag(flight.get("Cancelled"))
        if flag is None:
            unknown += 1
            continue
        scheduled += 1
        cancelled += flag
    rate = cancelled / scheduled if scheduled else None
    return {"scheduled": scheduled, "cancelled": cancelled, "unknown": unknown, "rate": rate}


def _delay_summary(delays, base):
    """Fill mean/median/delayed counts into `base` from a list of known delay minutes."""
    delayed = sum(delay >= DELAY_THRESHOLD_MINUTES for delay in delays)
    severe = sum(delay >= SEVERE_DELAY_THRESHOLD_MINUTES for delay in delays)
    base.update({
        "known_delay": len(delays),
        "unknown_delay": base["eligible"] - len(delays),
        "mean_delay_minutes": sum(delays) / len(delays) if delays else None,
        "median_delay_minutes": median(delays) if delays else None,
        "delayed_15_plus": delayed,
        "delayed_15_plus_rate": delayed / len(delays) if delays else None,
        "delayed_60_plus": severe,
        "delayed_60_plus_rate": severe / len(delays) if delays else None,
    })
    return base


def delay_metrics(flights, airport):
    """Departure and arrival delay summary at `airport` (Matrix rows 19 and 22).

    Input: On-Time rows; `airport` BTS code.
    Departures: rows with Origin == airport. Eligible = Cancelled == 0 (operated; a
      diverted flight still departed, so its departure delay counts). Delay = DepDelay.
    Arrivals: rows with Dest == airport. Eligible = Cancelled == 0 and Diverted == 0
      (completed arrivals that landed here). Delay = ArrDelay.
    Cancelled flights are never "delayed"; diverted arrivals are not completed arrivals.
    Rows with a missing/unreadable Cancelled or Diverted value are counted as
    "unknown_status" and are not eligible.
    Per direction:
      known_delay   = eligible flights with a readable delay value
      unknown_delay = eligible flights whose delay is missing/unreadable (NOT treated as 0)
      mean_delay_minutes / median_delay_minutes over known delays. Negative delays (early)
        are kept as they are, so the mean/median can be negative.
      delayed_15_plus = known delays >= 15 minutes (DELAY_THRESHOLD_MINUTES; exactly 15
        counts as delayed - the standard U.S. DOT definition, also used by BTS DepDel15)
      delayed_15_plus_rate = delayed_15_plus / known_delay
      delayed_60_plus / delayed_60_plus_rate: the same with delay >= 60 minutes
        (SEVERE_DELAY_THRESHOLD_MINUTES; extreme-disruption indicator)
    Returns: {"threshold_minutes", "severe_threshold_minutes", "departures": {...},
    "arrivals": {...}}. Each direction
    has "scheduled", "cancelled", "unknown_status", "eligible" plus the delay fields above;
    arrivals also have "diverted". Means, medians and rates are None when no known delay
    exists.
    """
    airport = airport.strip().upper()
    dep = {"scheduled": 0, "cancelled": 0, "unknown_status": 0, "eligible": 0}
    arr = {"scheduled": 0, "cancelled": 0, "diverted": 0, "unknown_status": 0, "eligible": 0}
    dep_delays, arr_delays = [], []

    for flight in flights:
        cancelled = _flag(flight.get("Cancelled"))
        if flight.get("Origin") == airport:
            dep["scheduled"] += 1
            if cancelled is None:
                dep["unknown_status"] += 1
            elif cancelled:
                dep["cancelled"] += 1
            else:
                dep["eligible"] += 1
                delay = _minutes(flight.get("DepDelay"))
                if delay is not None:
                    dep_delays.append(delay)
        if flight.get("Dest") == airport:
            arr["scheduled"] += 1
            diverted = _flag(flight.get("Diverted"))
            if cancelled is True:
                arr["cancelled"] += 1
            elif cancelled is None or diverted is None:
                arr["unknown_status"] += 1
            elif diverted:
                arr["diverted"] += 1
            else:
                arr["eligible"] += 1
                delay = _minutes(flight.get("ArrDelay"))
                if delay is not None:
                    arr_delays.append(delay)

    return {
        "threshold_minutes": DELAY_THRESHOLD_MINUTES,
        "severe_threshold_minutes": SEVERE_DELAY_THRESHOLD_MINUTES,
        "departures": _delay_summary(dep_delays, dep),
        "arrivals": _delay_summary(arr_delays, arr),
    }


def unique_destinations(flights, airport):
    """Distinct destination airports of scheduled departures from `airport`.

    Input: On-Time rows; `airport` BTS code.
    Population: rows with Origin == airport (scheduled departures, including cancelled
    ones - the same scheduled-network population as long_haul_summary).
    Calculation: the set of non-empty Dest codes.
    Returns: {"count", "destinations" (sorted codes), "departures"}.
    Coverage: On-Time is domestic reporting-carrier flights only, so this is NOT the
    airport's complete (e.g. international or cargo) network.
    """
    airport = airport.strip().upper()
    departures = [f for f in flights if f.get("Origin") == airport]
    destinations = sorted({f.get("Dest") for f in departures if f.get("Dest")})
    return {"count": len(destinations), "destinations": destinations, "departures": len(departures)}


def hourly_movement_counts(flights, airport, *, time_basis, movement="all"):
    """Number of movements at `airport` in each local clock hour (Matrix row 9).

    Input: On-Time rows; time_basis "scheduled" or "actual" (required, see module
    docstring); movement "all" (departures + arrivals), "departures" or "arrivals".
    Calculation: each departure (Origin == airport) and/or arrival (Dest == airport) is
    placed in the local clock hour of its time; movements are counted per hour.
    Returns: {"counts": {hour_start_datetime: count}, "excluded": n}. Only hours with at
    least one movement appear; the data does not define the observation period, so
    hours with no movements are not listed (they are not "unknown", just absent).
    """
    times, excluded = _airport_movements(flights, airport, time_basis, movement)
    counts = {}
    for moment, _kind in times:
        hour = moment.replace(minute=0, second=0, microsecond=0)
        counts[hour] = counts.get(hour, 0) + 1
    return {"counts": dict(sorted(counts.items())), "excluded": excluded}


def busiest_hour(hourly_counts):
    """The single local clock hour with the most movements (building block for row 10).

    Input: the "counts" dict from hourly_movement_counts.
    Calculation: maximum hourly count; ties go to the earliest hour.
    Returns: {"hour_start", "movements", "tied_hours"} or None for empty input.
    Note: this is the literal busiest hour. Whether "peak-hour traffic" should be this
    maximum or a percentile (e.g. a typical busy hour) is still undecided in the Matrix.
    """
    if not hourly_counts:
        return None
    peak = max(hourly_counts.values())
    peak_hours = sorted(hour for hour, count in hourly_counts.items() if count == peak)
    return {"hour_start": peak_hours[0], "movements": peak, "tied_hours": len(peak_hours)}


def movement_spacing_minutes(flights, airport, *, time_basis, movement="all"):
    """Minutes between consecutive movements at `airport` (building block for row 11).

    Input: as hourly_movement_counts.
    Calculation: movement times are sorted; the gap between each consecutive pair on the
    SAME local calendar date is recorded (overnight gaps are not spacing). Movements at
    the same minute give a gap of 0.
    Returns: {"gaps": [minutes, ...] in time order, "excluded": n}.
    Note: which summary of the gaps (minimum, median, during peak hours only, ...) the
    project should use is not defined in the Matrix, so none is chosen here.
    """
    times, excluded = _airport_movements(flights, airport, time_basis, movement)
    ordered = sorted(moment for moment, _kind in times)
    gaps = [
        (later - earlier).total_seconds() / 60
        for earlier, later in zip(ordered, ordered[1:])
        if earlier.date() == later.date()
    ]
    return {"gaps": gaps, "excluded": excluded}


# --- peak traffic (project definition: 95th percentile) --------------------------------

def percentile_linear(values, p):
    """p-th percentile (0-100) by linear interpolation between closest ranks.

    Method (Hyndman & Fan type 7; same as Excel PERCENTILE.INC and NumPy's default):
      sort the n values x[0..n-1]; rank = (n - 1) * p / 100;
      result = x[floor(rank)] + (rank - floor(rank)) * (x[floor(rank) + 1] - x[floor(rank)]).
    Returns None for empty input. Deterministic; ties need no special handling.
    """
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100.")
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * p / 100
    lower = math.floor(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def _hourly_distribution(flights, airport, time_basis, first_date, last_date, movement):
    """Hourly movement counts for EVERY local clock hour from first_date 00:00 to last_date
    23:00 (hours without movements count as 0), plus the sorted movement times inside
    that period. Returns (counts, moments, excluded, outside_period)."""
    if first_date > last_date:
        raise ValueError("first_date must not be after last_date.")
    times, excluded = _airport_movements(flights, airport, time_basis, movement)

    start = datetime.combine(first_date, time.min)
    end = datetime.combine(last_date + timedelta(days=1), time.min)  # exclusive
    hours = int((end - start) / timedelta(hours=1))
    counts = {start + timedelta(hours=i): 0 for i in range(hours)}

    moments, outside = [], 0
    for moment, _kind in times:
        if start <= moment < end:
            counts[moment.replace(minute=0, second=0, microsecond=0)] += 1
            moments.append(moment)
        else:
            outside += 1
    return counts, sorted(moments), excluded, outside


def peak_hour_traffic(flights, airport, *, time_basis, first_date, last_date, movement="all"):
    """Peak traffic for airport comparison: 95th percentile of hourly movement counts
    (Matrix row 10, project definition).

    Input: On-Time rows; time_basis "scheduled" or "actual" (definitions in the module
    docstring); first_date / last_date (datetime.date, inclusive): the analysis period in
    the airport's local dates, e.g. 2026-04-01 .. 2026-04-30 for one month of data.
    Calculation:
      1. Count movements in every local clock hour of the period. Hours with no movements
         are included as 0, so every hour of the period is part of the distribution.
      2. value = percentile_linear(all hourly counts, 95).
    Movements outside the period (e.g. an overnight arrival on the 1st of the next month)
    are not counted and are reported as "outside_period".
    Returns: {"time_basis", "percentile", "value", "hours", "movements", "excluded",
    "outside_period"}. value is None when no movement falls inside the period (peak traffic
    cannot be characterised from no data - not reported as 0).
    Limitations: hours are naive local clock hours (24 per day); on daylight-saving change
    days one clock hour is missing or doubled. The first hours of the period can miss
    overnight arrivals whose FlightDate is the day before the period.
    """
    counts, moments, excluded, outside = _hourly_distribution(
        flights, airport, time_basis, first_date, last_date, movement)
    value = percentile_linear(counts.values(), PEAK_PERCENTILE) if moments else None
    return {
        "time_basis": time_basis,
        "percentile": PEAK_PERCENTILE,
        "value": value,
        "hours": len(counts),
        "movements": len(moments),
        "excluded": excluded,
        "outside_period": outside,
    }


def peak_period_spacing(flights, airport, *, time_basis, first_date, last_date, movement="all"):
    """Median minutes between consecutive movements during peak periods (Matrix row 11,
    project definition).

    Input: as peak_hour_traffic (same period, same hourly distribution).
    Calculation:
      1. threshold = the 95th-percentile value from peak_hour_traffic (same inputs).
      2. Peak hours = clock hours with count >= threshold (hours exactly AT the threshold
         are peak) and at least 1 movement (an empty hour is never peak).
      3. Peak periods = runs of consecutive peak hours on the same local date (a run is
         split at local midnight, consistent with movement_spacing_minutes).
      4. Gaps = minutes between each pair of consecutive movements that both fall inside
         the same peak period. A peak period with fewer than two movements gives no gap.
      5. median_spacing_minutes = median of all gaps (statistics.median; for an even number
         of gaps, the mean of the two middle gaps).
    Returns: {"time_basis", "peak_threshold", "peak_hours", "peak_periods", "gaps",
    "median_spacing_minutes", "excluded", "outside_period"}. median_spacing_minutes is None
    when there is no movement in the period or no peak period has two movements - spacing
    cannot be measured, so it is not reported as 0. (A real 0 is possible: movements
    scheduled in the same minute.)
    """
    counts, moments, excluded, outside = _hourly_distribution(
        flights, airport, time_basis, first_date, last_date, movement)
    threshold = percentile_linear(counts.values(), PEAK_PERCENTILE) if moments else None

    period_of_hour = {}
    if threshold is not None:
        previous, period = None, -1
        for hour in sorted(h for h, c in counts.items() if c >= threshold and c >= 1):
            continues_run = (previous is not None and hour - previous == timedelta(hours=1)
                             and hour.date() == previous.date())
            if not continues_run:
                period += 1
            period_of_hour[hour] = period
            previous = hour

    def period_of(moment):
        return period_of_hour.get(moment.replace(minute=0, second=0, microsecond=0))

    gaps = [
        (later - earlier).total_seconds() / 60
        for earlier, later in zip(moments, moments[1:])
        if period_of(earlier) is not None and period_of(earlier) == period_of(later)
    ]
    return {
        "time_basis": time_basis,
        "peak_threshold": threshold,
        "peak_hours": len(period_of_hour),
        "peak_periods": len(set(period_of_hour.values())),
        "gaps": len(gaps),
        "median_spacing_minutes": median(gaps) if gaps else None,
        "excluded": excluded,
        "outside_period": outside,
    }


# --- long-haul (project definition: >= 3,450 statute miles) ----------------------------

def _valid_distance(value):
    """BTS Distance string -> positive finite float miles, or None."""
    try:
        miles = float(value)
    except (TypeError, ValueError):
        return None
    return miles if math.isfinite(miles) and miles > 0 else None


def long_haul_summary(flights, airport, airports_by_code=None):
    """Long-haul flight count and share at `airport` (Matrix rows 15-16, project definition).

    Input: On-Time rows; `airport` BTS code; optional `airports_by_code`: already-retrieved
    OurAirports airports.csv rows keyed by the BTS/IATA code (e.g. {"ANC": row, ...}),
    used only when a flight has no valid BTS Distance.
    Population: every row with Origin == airport or Dest == airport, counted once
    (scheduled flights, including cancelled ones - the route network that was scheduled).
    Distance, in priority order:
      1. BTS "Distance" (statute miles between airports) when it is a positive number;
      2. otherwise great-circle (haversine) distance between the two airports' OurAirports
         coordinates, when both airports are in airports_by_code with valid coordinates;
      3. otherwise unknown. Unknown-distance flights are NEVER classified as short-haul.
    Long-haul = distance >= LONG_HAUL_THRESHOLD_MILES (3,450 statute miles; exactly 3,450
    counts as long-haul).
    long_haul_share = long_haul / known_distance (unknown-distance flights are excluded
    from the denominator).
    Returns: {"threshold_miles", "flights", "known_distance", "unknown_distance",
    "long_haul", "not_long_haul", "long_haul_share", "distance_source": {"bts",
    "great_circle"}}. long_haul_share is None when no flight has a known distance.
    Limitation: both sources give airport-to-airport distance, NOT the path the aircraft
    actually flew (routings, holding, detours). There is no universal definition of
    long-haul; 3,450 miles is this project's explicit threshold.
    """
    airport = airport.strip().upper()
    airports_by_code = airports_by_code or {}
    result = {
        "threshold_miles": LONG_HAUL_THRESHOLD_MILES,
        "flights": 0, "known_distance": 0, "unknown_distance": 0,
        "long_haul": 0, "not_long_haul": 0, "long_haul_share": None,
        "distance_source": {"bts": 0, "great_circle": 0},
    }
    for flight in flights:
        if airport not in (flight.get("Origin"), flight.get("Dest")):
            continue
        result["flights"] += 1

        distance, source = _valid_distance(flight.get("Distance")), "bts"
        if distance is None:
            origin = airports_by_code.get(flight.get("Origin"))
            dest = airports_by_code.get(flight.get("Dest"))
            if origin and dest:
                try:
                    distance, source = airport_distance_miles(origin, dest), "great_circle"
                except ValueError:  # invalid coordinates -> unknown
                    distance = None

        if distance is None:
            result["unknown_distance"] += 1
            continue
        result["known_distance"] += 1
        result["distance_source"][source] += 1
        if distance >= LONG_HAUL_THRESHOLD_MILES:
            result["long_haul"] += 1
        else:
            result["not_long_haul"] += 1

    if result["known_distance"]:
        result["long_haul_share"] = result["long_haul"] / result["known_distance"]
    return result
