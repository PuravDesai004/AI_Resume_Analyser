"""Offline city normalization and bounded location preference bonuses.

Coordinates/aliases: GeoNames through geonamescache (no network at runtime).
Distances are approximate straight-line distances between city coordinates,
not driving distances or commute estimates.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import re
import unicodedata

import geonamescache

# The complete hierarchy, in points. Cutoffs are inclusive.
PREFERRED_CITY_BONUS = 5
DISTANCE_BANDS = ((100, 4), (200, 3), (300, 2))
MAX_NEARBY_KM = DISTANCE_BANDS[-1][0]


@dataclass(frozen=True)
class City:
    identifier: int
    name: str
    country: str
    latitude: float
    longitude: float
    population: int
    region: str


@dataclass(frozen=True)
class LocationPreference:
    bonus: int = 0
    distance_km: float | None = None
    reason: str = "No location preference"


def normalize_name(value):
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


@lru_cache(maxsize=1)
def _gazetteer():
    database = geonamescache.GeonamesCache(min_city_population=15000)
    cities, names = {}, {}
    for record in database.get_cities().values():
        city = City(record["geonameid"], record["name"], record["countrycode"],
                    record["latitude"], record["longitude"], record["population"], record["admin1code"])
        cities[city.identifier] = city
        aliases = {record["name"], *record.get("alternatenames", [])}
        for alias in aliases:
            name = normalize_name(alias)
            if len(name) >= 3 and not name.isdigit():
                names.setdefault(name, set()).add(city.identifier)
    countries = {}
    for record in database.get_countries().values():
        for alias in (record["name"], record["iso"], record["iso3"]):
            countries[normalize_name(alias)] = record["iso"]
    countries.update({"uk": "GB", "united states of america": "US", "south korea": "KR"})
    states = {}
    for record in database.get_us_states().values():
        states[normalize_name(record["name"])] = record["code"]
        states[normalize_name(record["code"])] = record["code"]
    return cities, names, countries, states


_ARRANGEMENT = re.compile(r"\b(?:fully remote|remote|hybrid|on[ -]?site|in[ -]?person|office based|office|based in)\b", re.I)


@lru_cache(maxsize=512)
def resolve_city(location):
    """Resolve exact aliases with optional country/US state qualifiers.

    Bare ambiguous names resolve only if the largest city has at least ten times
    the population of the next candidate. Otherwise a qualifier is required.
    Unknown places and multiple city alternatives do not receive guessed bonuses.
    """
    if not location or not location.strip():
        return None
    cities, names, countries, states = _gazetteer()
    parts = [normalize_name(_ARRANGEMENT.sub("", part))
             for part in re.split(r"[,;()|/]+", location)]
    parts = [part for part in parts if part]
    if not parts or (parts[0] in countries and parts[0] not in names):
        return None  # A country/region is not a city coordinate.
    # Exact aliases only: no substring match (Mumbai != Navi Mumbai), no fuzzy
    # match that might silently put a misspelled city in the wrong country.
    candidates = [cities[i] for i in names.get(parts[0], ())]
    if parts[0] in countries:
        # City-states such as Singapore are valid; an unrelated city's alias
        # such as India -> Inđija (Serbia) is not a country coordinate.
        candidates = [c for c in candidates if c.country == countries[parts[0]]
                      and normalize_name(c.name) == parts[0]]
    if not candidates:
        return None
    qualifiers = parts[1:]
    country_hints = {countries[q] for q in qualifiers if q in countries and q not in states}
    if len(country_hints) > 1:
        return None
    if country_hints:
        candidates = [c for c in candidates if c.country in country_hints]
    # Do not select one arbitrarily from e.g. "Mumbai / Pune". A qualifier
    # such as Ontario in "London, Ontario, Canada" must not be confused with
    # the US city named Ontario.
    if any(q in names and q not in countries and q not in states
           and any(not country_hints or cities[i].country in country_hints for i in names[q])
           for q in qualifiers):
        return None
    state_hints = {states[q] for q in qualifiers if q in states}
    if state_hints:
        in_state = [c for c in candidates if c.country == "US" and c.region in state_hints]
        # CA can mean California or Canada. Use California only when a city
        # candidate fits it; otherwise allow the explicit Canada qualifier.
        if in_state:
            candidates = in_state
        elif not country_hints and any(q in countries for q in qualifiers):
            codes = {countries[q] for q in qualifiers if q in countries}
            candidates = [c for c in candidates if c.country in codes]
        else:
            return None
    if len(candidates) == 1:
        return candidates[0]
    if not candidates or qualifiers:
        return None
    candidates.sort(key=lambda c: (-c.population, c.identifier))
    if candidates[0].population >= 10 * max(candidates[1].population, 1):
        return candidates[0]
    return None


def distance_km(a, b):
    lat_a, lon_a, lat_b, lon_b = map(math.radians, [a.latitude, a.longitude, b.latitude, b.longitude])
    haversine = (math.sin((lat_b - lat_a) / 2) ** 2
                 + math.cos(lat_a) * math.cos(lat_b) * math.sin((lon_b - lon_a) / 2) ** 2)
    return 6371.0088 * 2 * math.asin(math.sqrt(max(0, min(1, haversine))))


def bonus_for_distance(distance):
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Distance must be finite and non-negative")
    for cutoff, bonus in DISTANCE_BANDS:
        if distance <= cutoff:
            return bonus
    return 0


def location_preference(preferred, job_location):
    if not preferred or not preferred.strip():
        return LocationPreference()
    origin = resolve_city(preferred)
    if origin is None:
        return LocationPreference(reason="Preferred city unknown or ambiguous; add city and country")
    destination = resolve_city(job_location)
    if destination is None:
        return LocationPreference(reason="Job city unspecified, unknown or ambiguous")
    if origin.identifier == destination.identifier:
        return LocationPreference(PREFERRED_CITY_BONUS, 0.0, f"Preferred city: {origin.name}")
    distance = distance_km(origin, destination)
    bonus = bonus_for_distance(distance)  # Apply the cutoff before rounding.
    reason = (f"Within {next(cutoff for cutoff, points in DISTANCE_BANDS if points == bonus)} km of {origin.name}"
              if bonus else f"Beyond {MAX_NEARBY_KM} km of {origin.name}")
    return LocationPreference(bonus, round(distance, 1), reason)
