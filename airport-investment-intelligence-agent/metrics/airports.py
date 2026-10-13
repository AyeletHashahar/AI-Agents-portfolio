"""Airport metadata and infrastructure metrics calculated from OurAirports rows
(OurAirportsClient.get_airport / get_runways output). All values in those rows are strings.

No function here makes API calls.
"""

import re
from math import asin, cos, radians, sin, sqrt

# Geographic convenience only. Airport discovery is general and lives in the data layer:
# OurAirportsClient.find_airports(country="US", states=[...]). Interpreting a place name such
# as "New England" into states is the future agent's job; this constant is one such mapping.
NEW_ENGLAND_STATES = frozenset({"CT", "ME", "MA", "NH", "RI", "VT"})

# Mean Earth radius in statute miles (6,371.0088 km, IUGG mean radius). Statute miles
# match the unit of the BTS On-Time "Distance" field.
EARTH_RADIUS_MILES = 3958.7613


def us_state(airport):
    """Two-letter U.S. state code for an airports.csv row (Matrix row 4).

    Calculation: the text after 'US-' in iso_region, for iso_country == 'US'.
    Returns: e.g. 'AK', or None for non-U.S. airports or a missing/unexpected iso_region.
    Note: U.S. territories (PR, VI, GU, ...) have their own iso_country, so they return None.
    """
    region = airport.get("iso_region") or ""
    if airport.get("iso_country") != "US" or not region.startswith("US-"):
        return None
    state = region[3:]
    return state if len(state) == 2 and state.isalpha() else None


def is_new_england(airport):
    """True if the airport is in a New England state as defined in the Matrix (CT, ME, MA,
    NH, RI, VT); False for other U.S. states; None if the state cannot be determined.
    Convenience check for one airport row - not the airport-discovery mechanism (use
    OurAirportsClient.find_airports for that)."""
    state = us_state(airport)
    if state is None:
        return None
    return state in NEW_ENGLAND_STATES


def runway_count(runways):
    """Count runways by status (Matrix row 32: runway count = rows with closed = 0).

    Input: runways.csv rows for ONE airport (OurAirportsClient.get_runways).
    Calculation: closed == '0' -> open; closed == '1' -> closed; anything else -> unknown.
    Returns: {"open", "closed", "unknown"}. The project's runway count is "open"; if
    "unknown" > 0 the true count is uncertain.
    Note: this is physical runway count only - NOT runway capacity (movements per hour)
    and NOT terminal capacity. OurAirports rows also include helipads/waterways for some
    airports; this function counts every row the client returned.
    """
    counts = {"open": 0, "closed": 0, "unknown": 0}
    for runway in runways:
        status = (runway.get("closed") or "").strip()
        if status == "0":
            counts["open"] += 1
        elif status == "1":
            counts["closed"] += 1
        else:
            counts["unknown"] += 1
    return counts


# Runway surface vocabulary. OurAirports "surface" is free text (668 distinct values in the
# 2026-10 data); these token lists cover the values seen at U.S. commercial airports.
# Anything else is "unknown" - never guessed.
HARD_SURFACE_TOKENS = frozenset({"ASP", "ASPH", "ASPHALT", "CON", "CONC", "CONCRETE", "PEM", "BIT"})
UNPAVED_SURFACE_TOKENS = frozenset({
    "TURF", "GRS", "GRASS", "SOD", "DIRT", "EARTH", "GRE", "GVL", "GRVL", "GRAVEL",
    "SAND", "SNOW", "ICE", "MATS", "PSP",
})
WATER_SURFACE_TOKENS = frozenset({"WATER", "WAT"})
# Condition grades (Good / Fair / Poor / Excellent) and grooving: ignored when they follow a
# surface token ("ASPH-G", "CONC-GRVD"). On their own they are not a surface ("G" -> unknown).
SURFACE_QUALIFIER_TOKENS = frozenset({"G", "F", "P", "E", "GRVD", "GROOVED"})


def runway_surface_category(surface):
    """Classify an OurAirports runway surface string: "hard", "unpaved", "water" or "unknown".

    The text is split into tokens on non-letters (e.g. "ASPH-CONC-G" -> ASPH, CONC, G).
    Qualifier tokens are dropped when another token is present. The category is the one
    shared by ALL remaining tokens; a mix of categories (e.g. "ASPH-TURF", partly paved),
    an unlisted token (e.g. "PER", "UNK") or an empty value gives "unknown".
    """
    tokens = [t for t in re.split(r"[^A-Z]+", (surface or "").upper()) if t]
    surface_tokens = [t for t in tokens if t not in SURFACE_QUALIFIER_TOKENS]
    if not surface_tokens:
        return "unknown"
    categories = set()
    for token in surface_tokens:
        if token in HARD_SURFACE_TOKENS:
            categories.add("hard")
        elif token in UNPAVED_SURFACE_TOKENS:
            categories.add("unpaved")
        elif token in WATER_SURFACE_TOKENS:
            categories.add("water")
        else:
            return "unknown"
    return categories.pop() if len(categories) == 1 else "unknown"


def _is_helipad(runway):
    """OurAirports lists helipads in runways.csv with idents like "H1", "H2"."""
    return re.fullmatch(r"H\d*", (runway.get("le_ident") or "").strip().upper()) is not None


def operational_runway_count(runways):
    """Runways usable by conventional commercial passenger aircraft - a conservative proxy.

    Input: runways.csv rows for ONE airport.
    Operational runway = closed == "0", not a helipad (le_ident "H1", ...) and a "hard"
    surface (runway_surface_category). Water lanes, unpaved (gravel, turf, dirt, ...) and
    unknown/ambiguous surfaces are not counted.
    Returns: {"operational", "open_by_surface": {"hard", "unpaved", "water", "unknown"},
    "helipads", "closed", "unknown_status", "count_is_certain"}. count_is_certain is False
    when a runway's open/closed status is unknown or an open runway's surface is unknown -
    the true operational count could then be higher.
    Limitation: still a proxy. It does not model runway length or strength suitability,
    runway configuration, simultaneous-use restrictions or runway capacity.
    """
    result = {"operational": 0, "open_by_surface": {"hard": 0, "unpaved": 0, "water": 0, "unknown": 0},
              "helipads": 0, "closed": 0, "unknown_status": 0}
    for runway in runways:
        if _is_helipad(runway):
            result["helipads"] += 1
            continue
        status = (runway.get("closed") or "").strip()
        if status == "1":
            result["closed"] += 1
        elif status != "0":
            result["unknown_status"] += 1
        else:
            category = runway_surface_category(runway.get("surface"))
            result["open_by_surface"][category] += 1
            if category == "hard":
                result["operational"] += 1
    result["count_is_certain"] = (result["unknown_status"] == 0
                                  and result["open_by_surface"]["unknown"] == 0)
    return result


def great_circle_distance_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in statute miles between two points (Matrix row 14).

    Input: latitudes/longitudes in decimal degrees (numbers or numeric strings, as in
    OurAirports latitude_deg / longitude_deg).
    Calculation: haversine formula on a sphere with EARTH_RADIUS_MILES.
    Returns: float miles, or None if any coordinate is missing or not a number.
    Note: great-circle distance is the shortest path over the Earth's surface, not the
    distance actually flown. For flights in BTS On-Time, use its "Distance" field instead.
    """
    try:
        lat1, lon1, lat2, lon2 = (float(value) for value in (lat1, lon1, lat2, lon2))
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi, d_lambda = radians(lat2 - lat1), radians(lon2 - lon1)
    h = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * asin(min(1.0, sqrt(h)))


def airport_distance_miles(airport_a, airport_b):
    """Great-circle distance in miles between two airports.csv rows (None if coordinates missing)."""
    return great_circle_distance_miles(
        airport_a.get("latitude_deg"), airport_a.get("longitude_deg"),
        airport_b.get("latitude_deg"), airport_b.get("longitude_deg"),
    )
