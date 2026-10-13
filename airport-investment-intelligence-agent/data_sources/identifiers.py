"""Resolve an OurAirports airport to the 3-character code BTS uses for it.

OurAirports and BTS usually agree (BTS code == OurAirports iata_code), but not always.
Example found 2026-10-04: Palm Beach is listed in OurAirports with iata_code, icao_code,
gps_code and local_code all "DJT" (renamed airport); BTS still reports it as "PBI". The old
code survives only in OurAirports "keywords" ("PBI, KPBI, MFW, South Florida"), a field the
data dictionary describes as including former names and alternate codes.

Conservative, deterministic rule (no airport-name matching):
  1. EXACT: if iata_code is a code present in BTS, use it. Alternates are not considered.
  2. ALTERNATE: otherwise consider, in order, local_code (FAA code) and 3-character
     codes in keywords. A candidate is eligible only if it is present in BTS AND is not the
     iata_code or local_code of any OTHER OurAirports airport (otherwise it could be a
     different airport - e.g. local "BVU" of Boulder City is the IATA code of Beluga, AK).
     Exactly one eligible candidate -> resolved; several -> "ambiguous" (not merged).
  3. Otherwise "not_found_in_bts": no candidate code exists in BTS. This is an identifier
     result, NOT evidence of zero activity.
The OurAirports identifiers and the BTS code actually used are both kept.
"""

import re

_CODE = re.compile(r"[A-Z0-9]{3}")


def bts_code_candidates(airport):
    """Candidate BTS codes for an airports.csv row, in priority order:
    [(code, source), ...] with source "iata_code", "local_code" or "keywords"."""
    candidates, seen = [], set()

    def add(code, source):
        code = (code or "").strip().upper()
        if _CODE.fullmatch(code) and code not in seen:
            seen.add(code)
            candidates.append((code, source))

    add(airport.get("iata_code"), "iata_code")
    add(airport.get("local_code"), "local_code")
    # Keyword codes must already be written as codes (upper case / digits): ordinary words
    # such as "Air" or "Bay" are not turned into candidate codes.
    for token in re.split(r"[^A-Za-z0-9]+", airport.get("keywords") or ""):
        if _CODE.fullmatch(token):
            add(token, "keywords")
    return candidates


def resolve_bts_code(airport, bts_codes, code_owners):
    """Resolve one airports.csv row to its BTS code (see module docstring).

    airport:     airports.csv row (needs ident, iata_code, local_code, keywords).
    bts_codes:   set of codes that exist in BTS (e.g. codes with T-100 rows).
    code_owners: {code: set of OurAirports idents using it as iata_code or local_code}
                 (OurAirportsClient.code_owners()).
    Returns {"ident", "ourairports_iata", "bts_code", "source", "status", "candidates",
    "detail"}; status is "exact", "alternate", "ambiguous" or "not_found_in_bts" and
    bts_code is None unless status is "exact" or "alternate".
    """
    candidates = bts_code_candidates(airport)
    ident = airport.get("ident")
    result = {
        "ident": ident,
        "ourairports_iata": airport.get("iata_code") or None,
        "bts_code": None,
        "source": None,
        "status": "not_found_in_bts",
        "candidates": [code for code, _ in candidates],
        "detail": None,
    }

    iata = next((code for code, source in candidates if source == "iata_code"), None)
    if iata and iata in bts_codes:
        result.update(bts_code=iata, source="iata_code", status="exact")
        return result

    eligible, rejected = [], []
    for code, source in candidates:
        if source == "iata_code" or code not in bts_codes:
            continue
        other_owners = code_owners.get(code, set()) - {ident}
        if other_owners:
            rejected.append(f"{code} (also used by {sorted(other_owners)})")
        else:
            eligible.append((code, source))

    if len(eligible) == 1:
        code, source = eligible[0]
        result.update(bts_code=code, source=source, status="alternate",
                      detail=f"OurAirports iata_code {iata!r} not in BTS; {source} {code!r} is.")
    elif len(eligible) > 1:
        result.update(status="ambiguous",
                      detail=f"Several alternate codes exist in BTS: {[c for c, _ in eligible]}")
    else:
        result["detail"] = ("No candidate code exists in BTS"
                            + (f"; rejected as other airports' codes: {rejected}" if rejected else "")
                            + ". Not evidence of zero activity.")
    return result
