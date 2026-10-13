"""OurAirports: airport discovery and metadata (codes, name, location, coordinates),
regions and runways.

Role: OurAirports tells us which airports EXIST in a country/state. Which of them have
relevant commercial activity is determined later from BTS data.

Source: https://ourairports.com/data/  (public domain CSV files)
Field definitions: https://ourairports.com/help/data-dictionary.html

No authentication. The CSV files are downloaded once and cached locally.
"""

import csv
import re
import unicodedata
from pathlib import Path

import requests

BASE_URL = "https://davidmegginson.github.io/ourairports-data"
CACHE_DIR = Path("data/raw/ourairports")

# Facility types excluded from airport discovery by default: clearly not airports for
# airline terminal investment. The data dictionary documents "closed_airport", but the
# actual airports.csv uses "closed", so both are listed.
# Small airports and seaplane bases are NOT excluded: in Alaska, for example, many of them
# have scheduled airline service. Whether an airport has relevant commercial activity is
# decided later from BTS data, not from the OurAirports type.
EXCLUDED_FACILITY_TYPES = frozenset({"heliport", "balloonport", "closed", "closed_airport"})

# Fields returned by find_airports (values as strings, exactly as in airports.csv;
# a missing code is an empty string).
DISCOVERY_FIELDS = [
    "ident", "iata_code", "icao_code", "gps_code", "local_code", "name", "type",
    "scheduled_service", "municipality", "iso_country", "iso_region",
    "latitude_deg", "longitude_deg",
    "keywords",   # may hold former/alternate codes; used by data_sources.identifiers
]

# Display order for find_airports results (largest facility types first).
_TYPE_ORDER = {"large_airport": 0, "medium_airport": 1, "small_airport": 2, "seaplane_base": 3}

# search_airports: words that describe airports in general. A place-phrase reference must
# contain at least one OTHER word ("International Airport" alone identifies nothing).
GENERIC_AIRPORT_WORDS = frozenset("""
airport airports international intl regional municipal county field airfield airpark jetport
national memorial the of
""".split())
_CODE_REFERENCE = re.compile(r"[A-Z0-9]{3,4}")
_IN_QUALIFIER = re.compile(r"^(.*\S)\s+in\s+(\S.*)$", re.IGNORECASE)    # greedy: splits at the LAST " in "
# Abbreviations written both ways in names and by users; expanded on BOTH sides.
_ABBREVIATIONS = {"intl": "international", "st": "saint", "ft": "fort", "mt": "mount"}


def normalize_reference(text):
    """Lower case, accents removed, apostrophes dropped ("O'Hare" -> "ohare"), every other
    non-alphanumeric character treated as a space, whitespace collapsed, and the few
    abbreviations in _ABBREVIATIONS expanded ("Intl" -> "international", "St." -> "saint")."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"['\u2019`]", "", text)
    return " ".join(_ABBREVIATIONS.get(word, word) for word in re.findall(r"[a-z0-9]+", text))


def _municipality_names(municipality):
    """Normalized forms a city reference may equal: the whole municipality and each part of
    a "/" or "," list ("Raleigh/Durham", "Honolulu, Oahu"). Hyphenated names are NOT split
    ("Winston-Salem" is not "Salem")."""
    parts = [municipality] + re.split(r"[/,]", municipality)
    return frozenset(name for name in map(normalize_reference, parts) if name)


def _candidate_order(row):
    """Deterministic candidate order: scheduled service first, larger facility, name, ident."""
    return (row["scheduled_service"] != "yes", _TYPE_ORDER.get(row["type"], 9), row["name"], row["ident"])


class OurAirportsClient:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        # Parsed CSV rows, kept in memory for the life of this client (no extra disk cache).
        self._rows = {}
        self._runways_by_ident = None
        self._code_owners = None
        self._search_source = self._search = None     # search_airports index

    def _load_csv(self, filename, refresh=False):
        """Download <filename> once (or again if refresh=True) and return its rows as dicts.
        Parsed rows are kept in memory, so each file is parsed once per client."""
        if filename in self._rows and not refresh:
            return self._rows[filename]
        path = self.cache_dir / filename
        if refresh or not path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            response = requests.get(f"{BASE_URL}/{filename}", timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)

        with open(path, encoding="utf-8", newline="") as f:
            self._rows[filename] = list(csv.DictReader(f))
        if filename == "runways.csv":
            self._runways_by_ident = None
        if filename == "airports.csv":
            self._code_owners = None
        return self._rows[filename]

    def code_owners(self):
        """{code: set of airport idents} for every iata_code and local_code in airports.csv.
        Used to check that an alternate identifier is not another airport's code."""
        if self._code_owners is None:
            owners = {}
            for row in self._load_csv("airports.csv"):
                for code in {row["iata_code"], row["local_code"]} - {""}:
                    owners.setdefault(code, set()).add(row["ident"])
            self._code_owners = owners
        return self._code_owners

    def get_airport(self, code):
        """Return the airports.csv row for an IATA (e.g. 'ANC') or ICAO (e.g. 'PANC') code.

        Returns None if no airport matches.
        """
        code = code.strip().upper()
        for row in self._load_csv("airports.csv"):
            if code in (row["iata_code"], row["icao_code"], row["ident"]):
                return row
        return None

    def find_airports(self, country, states=None, types=None, require_iata=False,
                      scheduled_service_only=False):
        """Discover airports from airports.csv by country and optional states/regions.

        country: ISO 3166-1 alpha-2 code, e.g. "US".
        states:  optional list of subdivision codes within that country, e.g. ["CA"] or
                 ["MA", "CT"] (full codes like "US-CA" are also accepted). None = whole
                 country. Codes unknown to OurAirports regions.csv raise ValueError, so a
                 typo is not mistaken for "no airports".
        types:   optional OurAirports types to include (e.g. ["large_airport",
                 "medium_airport"]). None = every type except EXCLUDED_FACILITY_TYPES
                 (heliports, balloonports, closed facilities).
        require_iata: keep only airports that have an IATA code.
        scheduled_service_only: keep only airports OurAirports marks scheduled_service ==
                 "yes" (its own flag for current scheduled airline service).

        Returns: list of dicts with DISCOVERY_FIELDS, sorted by region, facility type
        (large first) and name. This lists airports that EXIST; it does not say which have
        commercial activity - check that with BTS data (T-100 / On-Time).
        """
        country = (country or "").strip().upper()
        if len(country) != 2 or not country.isalpha():
            raise ValueError(f"country must be a 2-letter ISO code, got {country!r}.")

        regions = None
        if states is not None:
            if isinstance(states, str):
                states = [states]
            if not states:
                raise ValueError("states must not be empty; pass None for the whole country.")
            regions = set()
            for state in states:
                code = state.strip().upper()
                regions.add(code if code.startswith(f"{country}-") else f"{country}-{code}")
            known = {row["code"] for row in self._load_csv("regions.csv")}
            unknown = sorted(regions - known)
            if unknown:
                raise ValueError(f"Unknown region code(s) for {country}: {unknown}")

        wanted_types = set(types) if types is not None else None

        found = [
            {field: row[field] for field in DISCOVERY_FIELDS}
            for row in self._load_csv("airports.csv")
            if row["iso_country"] == country
            and (regions is None or row["iso_region"] in regions)
            and (row["type"] in wanted_types if wanted_types is not None
                 else row["type"] not in EXCLUDED_FACILITY_TYPES)
            and (not require_iata or row["iata_code"])
            and (not scheduled_service_only or row["scheduled_service"] == "yes")
        ]
        return sorted(found, key=lambda a: (a["iso_region"], _TYPE_ORDER.get(a["type"], 9), a["name"]))

    def lookup_code(self, code):
        """Find the ONE airport an airport code refers to (e.g. a code passed in by the Agent).

        Match order, stopping at the first level with any match:
          1. exact ident / iata_code / icao_code / gps_code;
          2. exact local_code;
          3. exact upper-case 3-character code in keywords (e.g. Palm Beach "PBI").
        Returns {"status": "found" | "ambiguous" | "not_found", "airport": row or None,
        "matched_on": field name or None, "matches": [idents]}. Several airports at the
        same level -> "ambiguous" (never guessed). No name matching.
        """
        code = (code or "").strip().upper()
        rows = self._load_csv("airports.csv")
        levels = [
            ("identifier", lambda r: code in (r["ident"], r["iata_code"], r["icao_code"], r["gps_code"])),
            ("local_code", lambda r: r["local_code"] == code),
            ("keywords", lambda r: code in re.split(r"[^A-Za-z0-9]+", r["keywords"])),
        ]
        for matched_on, test in levels:
            if not code:
                break
            matches = [r for r in rows if test(r)]
            if len(matches) == 1:
                return {"status": "found", "airport": matches[0], "matched_on": matched_on,
                        "matches": [matches[0]["ident"]]}
            if matches:
                return {"status": "ambiguous", "airport": None, "matched_on": matched_on,
                        "matches": sorted(r["ident"] for r in matches)}
        return {"status": "not_found", "airport": None, "matched_on": None, "matches": []}

    def search_airports(self, text, country="US"):
        """Resolve a natural-language airport reference (code, name or city) to ONE airport
        of `country`, deterministically and conservatively, from the local airports.csv.

        An optional trailing ", <state>" (code or name from regions.csv, e.g. "Portland, OR" or
        "Portland, Oregon") restricts every level to that region. Heliports, balloonports and
        closed facilities are never matched by name or place. Levels, stopping at the first
        level with any match:
          1. code  - only if the reference is WRITTEN as a code (3-4 upper-case letters or
                     digits, e.g. "ANC", "PANC"): lookup_code() within this country/region;
          2. name  - normalized airport name == normalized reference;
          3. place - the reference equals the normalized municipality or one part of a "/"
                     or "," municipality list ("Honolulu, Oahu"), OR occurs as a whole-word
                     phrase inside the normalized airport name ("Fairbanks", "Logan",
                     "Juneau International"). A leading "the" and a trailing "airport(s)" are
                     ignored, and the reference must contain a non-generic word (see
                     GENERIC_AIRPORT_WORDS).
        Narrowing uses ONLY the data, never city-specific rules:
          - code/name level, exactly one match -> found ("unique_match");
          - otherwise, among the matches with OurAirports scheduled_service == "yes", exactly
            one has the largest facility type (large > medium > small > seaplane base) ->
            found ("only_scheduled_airport" if it is the only scheduled match, otherwise
            "largest_scheduled_airport");
          - otherwise -> ambiguous, with "reason": "several_scheduled_airports",
            "no_scheduled_airport" (a place reference is never auto-resolved to an airport
            without scheduled service, even if it is the only match) or
            "several_airports_for_code".
        City qualifier: only if the WHOLE reference matches nothing, a reference of the form
        "<airport> in <place>" (split at the last " in ") or "<airport>, <place>" (place not a
        state) is searched again as <airport> with the same levels, keeping only matches whose
        municipality (or one part of a "/"/"," municipality list) equals <place>, or whose state
        is <place> ("John Wayne Airport in Santa Ana" -> SNA; "Logan Airport in Billings" ->
        BIL). The usual narrowing then applies to those matches. If <airport> matches but none
        of its matches is in <place>, the result is ambiguous ("qualifier_mismatch", with the
        <airport> matches as candidates): the qualifier contradicts the data, so nothing is
        chosen. If <airport> is only generic words ("the airport in Anchorage"), <place> is
        resolved as a place reference instead.
        No fuzzy/edit-distance matching. No network request beyond the usual one-time
        airports.csv download.

        Returns {"status": "found" | "ambiguous" | "not_found", "matched_on": "code" | "name" |
        "place" | None, "selection": see above | None, "reason": see above | None,
        "airport": row | None, "candidates": rows (found: every match, the selected one
        included; ambiguous: the competing scheduled airports, or every match if none is
        scheduled), "match_count": int, "region": e.g. "US-OR" | None, "qualifier": the place
        qualifier used | None}. Candidates are in a deterministic order: scheduled service first,
        larger facility type, name, ident.
        """
        country = (country or "").strip().upper()
        text = (text or "").strip()
        region = None
        if "," in text:
            head, tail = text.rsplit(",", 1)
            region = self._region_code(country, tail)
            if region:
                text = head.strip()
        found = self._search_levels(text, country, region)
        if found["status"] == "not_found":
            match = _IN_QUALIFIER.match(text)
            head, qualifier = match.groups() if match else (text.rsplit(",", 1) if "," in text else (text, None))
            if qualifier and qualifier.strip() and head.strip():
                if set(normalize_reference(head).split()) <= GENERIC_AIRPORT_WORDS | {"a", "an"}:
                    found = self._search_levels(qualifier.strip(), country, region)   # "the airport in X" = X
                else:
                    found = self._search_levels(head.strip(), country, region, qualifier.strip())
        return found

    def _search_levels(self, text, country, region, qualifier=None):
        """The code -> name -> place levels of search_airports (see there)."""
        def result(status, matched_on=None, selection=None, airport=None, candidates=(), match_count=0,
                   reason=None):
            return {"status": status, "matched_on": matched_on, "selection": selection, "reason": reason,
                    "airport": airport, "candidates": sorted(candidates, key=_candidate_order),
                    "match_count": match_count, "region": region, "qualifier": qualifier}

        def in_scope(row):
            return row["iso_country"] == country and (region is None or row["iso_region"] == region)

        place = normalize_reference(qualifier) if qualifier else None
        place_region = self._region_code(country, qualifier) if qualifier else None

        def in_place(row):
            return place in _municipality_names(row["municipality"]) or row["iso_region"] == place_region

        unqualified = []                   # matches rejected only because they are not in the place

        if _CODE_REFERENCE.fullmatch(text):
            found = self.lookup_code(text)
            if found["airport"]:
                rows = [found["airport"]]
            else:
                by_ident = {row["ident"]: row for row in self._load_csv("airports.csv")}
                rows = [by_ident[ident] for ident in found["matches"]]
            rows = [row for row in rows if in_scope(row)]
            if qualifier:
                unqualified += [row for row in rows if not in_place(row)]
                rows = [row for row in rows if in_place(row)]
            if len(rows) == 1:
                return result("found", "code", "unique_match", rows[0], rows, 1)
            if rows:
                return result("ambiguous", "code", None, None, rows, len(rows), "several_airports_for_code")

        reference = normalize_reference(text)
        if not reference:
            return result("not_found", reason="qualifier_mismatch" if unqualified else None)
        phrase = re.sub(r"^the ", "", reference)
        phrase = re.sub(r" airports?$", "", phrase) or phrase
        levels = [("name", lambda name, cities: name == reference)]
        if set(phrase.split()) - GENERIC_AIRPORT_WORDS:
            levels.append(("place", lambda name, cities: phrase in cities or f" {phrase} " in f" {name} "))
        index = [entry for entry in self._search_index() if in_scope(entry[0])]
        for matched_on, test in levels:
            matches = [row for row, name, cities in index if test(name, cities)]
            if qualifier:
                unqualified += [row for row in matches if not in_place(row)]
                matches = [row for row in matches if in_place(row)]
            if matches:
                return self._narrow(matches, matched_on, result)
        if unqualified:
            unique = list({row["ident"]: row for row in unqualified}.values())
            return result("ambiguous", None, None, None, unique, len(unique), "qualifier_mismatch")
        return result("not_found")

    @staticmethod
    def _narrow(matches, matched_on, result):
        """Data-only narrowing of name/place matches (see search_airports)."""
        if len(matches) == 1 and matched_on == "name":
            return result("found", matched_on, "unique_match", matches[0], matches, 1)
        scheduled = [row for row in matches if row["scheduled_service"] == "yes"]
        if not scheduled:
            return result("ambiguous", matched_on, None, None, matches, len(matches), "no_scheduled_airport")
        top_tier = min(_TYPE_ORDER.get(row["type"], 9) for row in scheduled)
        top = [row for row in scheduled if _TYPE_ORDER.get(row["type"], 9) == top_tier]
        if len(top) > 1:
            return result("ambiguous", matched_on, None, None, scheduled, len(matches), "several_scheduled_airports")
        if len(matches) == 1:
            selection = "unique_match"
        else:
            selection = "only_scheduled_airport" if len(scheduled) == 1 else "largest_scheduled_airport"
        return result("found", matched_on, selection, top[0], matches, len(matches))

    def _search_index(self):
        """[(row, normalized name, normalized municipality names)] for every searchable
        facility, built once per loaded airports.csv."""
        rows = self._load_csv("airports.csv")
        if self._search_source is not rows:
            self._search_source = rows
            self._search = [(row, normalize_reference(row["name"]), _municipality_names(row["municipality"]))
                            for row in rows if row["type"] not in EXCLUDED_FACILITY_TYPES]
        return self._search

    def _region_code(self, country, text):
        """'OR' / 'US-OR' / 'Oregon' -> 'US-OR' for this country, or None."""
        wanted = normalize_reference(text)
        if not wanted:
            return None
        for row in self._load_csv("regions.csv"):
            if row["iso_country"] == country and wanted in (normalize_reference(row["code"]),
                                                           normalize_reference(row["code"].split("-", 1)[-1]),
                                                           normalize_reference(row["name"])):
                return row["code"]
        return None

    def get_region_name(self, iso_region):
        """Return the regions.csv name for an iso_region code (e.g. 'US-AK' -> 'Alaska'), or None."""
        for row in self._load_csv("regions.csv"):
            if row["code"] == iso_region:
                return row["name"]
        return None

    def get_runways(self, airport_ident):
        """Return all runways.csv rows for an airport ident (normally the ICAO code, e.g. 'PANC')."""
        runways = self._load_csv("runways.csv")
        if self._runways_by_ident is None:
            index = {}
            for row in runways:
                index.setdefault(row["airport_ident"], []).append(row)
            self._runways_by_ident = index
        return list(self._runways_by_ident.get(airport_ident.strip().upper(), []))
