"""
asteroid_fetcher.py
-------------------
Fetches near-Earth asteroid data from:
  1. NASA SSD/CNEOS close-approach API  – upcoming approaches list
  2. NASA JPL SBDB API                  – orbital elements per object
  3. NASA NeoWs                         – fallback / supplement

No API key required for basic usage (DEMO_KEY: 30 req/hr).
Set API_KEY below to your own key for higher limits.

SBDB API ref: https://ssd-api.jpl.nasa.gov/doc/sbdb.html
  Valid params: sstr | spk | des  (exactly one required)
  Optional booleans: full-prec, alt-des, alt-spk, no-orbit,
                     phys-par, ca-data, vi-data, sat, discovery, ...
  Do NOT send cov=0 or phys=0 — those are unrecognized and cause 400.
"""

import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import time
import urllib.parse

from physics import Body, AU, G, DAY, _keplerian_to_cartesian

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
NEOWS_BASE      = "https://api.nasa.gov/neo/rest/v1"
SSD_BASE        = "https://ssd-api.jpl.nasa.gov"
API_KEY         = "DEMO_KEY"   # swap for your key: https://api.nasa.gov
REQUEST_TIMEOUT = 20
MAX_ASTEROIDS   = 12

ASTEROID_COLORS = [
    (200, 150,  80), (180, 120,  60), (220, 180, 100),
    (160, 110,  50), (210, 160,  90), (190, 140,  70),
]

GM_SUN = G * 1.989e30


# ---------------------------------------------------------------------------
# SSD/CNEOS close-approach data API
# ---------------------------------------------------------------------------

def fetch_ssd_close_approaches(days_ahead: int = 60) -> list[dict]:
    """Return [{designation, dist_au}, ...] for upcoming Earth close approaches."""
    today  = datetime.now(timezone.utc).date()
    future = today + timedelta(days=days_ahead)
    url    = f"{SSD_BASE}/cad.api"
    params = {
        "date-min": today.isoformat(),
        "date-max": future.isoformat(),
        "dist-max": "0.10",
        "sort":     "dist",
        "limit":    str(MAX_ASTEROIDS * 4),
    }
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[SSD/CAD] Request failed: {e}")
        return []

    fields  = data.get("fields", [])
    records = data.get("data",   [])
    print(f"[SSD/CAD] {len(records)} close approaches found")

    idx = {f: i for i, f in enumerate(fields)}
    results = []
    for row in records:
        des  = row[idx["des"]]  if "des"  in idx else None
        dist = row[idx["dist"]] if "dist" in idx else None
        if des:
            results.append({"designation": str(des).strip(), "dist_au": dist})
    return results


# ---------------------------------------------------------------------------
# JPL Small-Body Database API
# ---------------------------------------------------------------------------

def fetch_sbdb_elements(designation: str) -> Optional[dict]:
    """
    Fetch osculating orbital elements from SBDB.

    Correct valid parameters per docs (https://ssd-api.jpl.nasa.gov/doc/sbdb.html):
      - sstr / spk / des  (exactly one, required)
      - full-prec         (boolean flag, optional)
    Do NOT send cov=0, phys=0, or any other unrecognised params → 400 error.

    Spaces must be %20-encoded; requests' params= uses +, so build URL manually.
    """
    enc = urllib.parse.quote(designation.strip(), safe="")
    # Only send sstr and full-prec — nothing else
    url = f"{SSD_BASE}/sbdb.api?sstr={enc}&full-prec=1"

    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 300:
            # Multiple objects matched — pick the first pdes and retry with des=
            matches = resp.json().get("list", [])
            if not matches:
                return None
            pdes = matches[0].get("pdes", "")
            if not pdes:
                return None
            print(f"[SBDB] '{designation}' matched multiple → retrying with des={pdes}")
            enc2 = urllib.parse.quote(pdes.strip(), safe="")
            resp = requests.get(
                f"{SSD_BASE}/sbdb.api?des={enc2}&full-prec=1",
                timeout=REQUEST_TIMEOUT,
            )

        if resp.status_code == 400:
            msg = ""
            try:
                msg = resp.json().get("message", "")
            except Exception:
                pass
            print(f"[SBDB] 400 for '{designation}': {msg[:100]}")
            return None

        resp.raise_for_status()
        data = resp.json()

    except Exception as e:
        print(f"[SBDB] Request error for '{designation}': {e}")
        return None

    orb = data.get("orbit")
    if not orb:
        print(f"[SBDB] No orbit data for '{designation}'")
        return None

    # elements array: each entry has "name" and "value"
    elem = {}
    for entry in orb.get("elements", []):
        name = entry.get("name")
        val  = entry.get("value")
        if name and val is not None:
            elem[name] = val

    # epoch is a plain Julian-day float string at top level of orbit dict
    try:
        epoch_jd = float(orb.get("epoch", 2451545.0))
    except (TypeError, ValueError):
        epoch_jd = 2451545.0

    obj      = data.get("object", {})
    fullname = (obj.get("fullname") or obj.get("des") or designation).strip()

    required = ("a", "e", "i", "om", "w", "ma")
    missing  = [k for k in required if k not in elem]
    if missing:
        print(f"[SBDB] Missing elements {missing} for '{designation}'")
        return None

    try:
        return {
            "name":     fullname,
            "a_au":     float(elem["a"]),
            "e":        float(elem["e"]),
            "i_deg":    float(elem["i"]),
            "Om_deg":   float(elem["om"]),
            "w_deg":    float(elem["w"]),
            "M0_deg":   float(elem["ma"]),
            "epoch_jd": epoch_jd,
        }
    except (KeyError, ValueError, TypeError) as ex:
        print(f"[SBDB] Parse error for '{designation}': {ex}")
        return None


# ---------------------------------------------------------------------------
# NeoWs fallback
# ---------------------------------------------------------------------------

def fetch_neows_feed(max_days: int = 7) -> list[dict]:
    """NeoWs /feed is capped at 7 days per request."""
    today = datetime.now(timezone.utc).date()
    start = today
    end   = start + timedelta(days=min(max_days, 7) - 1)
    url   = f"{NEOWS_BASE}/feed"
    params = {
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "api_key":    API_KEY,
    }
    neos = []
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        for objects in data.get("near_earth_objects", {}).values():
            neos.extend(objects)
        print(f"[NeoWs] {len(neos)} NEOs for {start} → {end}")
    except Exception as e:
        print(f"[NeoWs/feed] Failed: {e}")
    return neos


def fetch_neows_detail(neo_id: str) -> Optional[dict]:
    """Fetch full orbital_data for a single NEO by NeoWs numeric id."""
    url = f"{NEOWS_BASE}/neo/{neo_id}"
    try:
        resp = requests.get(url, params={"api_key": API_KEY}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        detail = resp.json()
    except Exception as e:
        print(f"[NeoWs/detail] Failed for id={neo_id}: {e}")
        return None

    od = detail.get("orbital_data", {})
    if not od:
        return None

    required_keys = (
        "semi_major_axis", "eccentricity", "inclination",
        "ascending_node_longitude", "perihelion_argument", "mean_anomaly",
    )
    if not all(k in od for k in required_keys):
        return None

    try:
        return {
            "name":     detail.get("name", neo_id),
            "a_au":     float(od["semi_major_axis"]),
            "e":        float(od["eccentricity"]),
            "i_deg":    float(od["inclination"]),
            "Om_deg":   float(od["ascending_node_longitude"]),
            "w_deg":    float(od["perihelion_argument"]),
            "M0_deg":   float(od["mean_anomaly"]),
            "epoch_jd": float(od.get("epoch_osculation", 2451545.0)),
        }
    except (KeyError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Epoch propagation
# ---------------------------------------------------------------------------

def _propagate_mean_anomaly(M0_deg: float, a_au: float, epoch_jd: float) -> float:
    """Advance mean anomaly from catalog epoch to J2000.0 (JD 2451545.0)."""
    J2000_JD  = 2451545.0
    dt_days   = J2000_JD - epoch_jd
    n_deg_day = 0.9856076686 / (a_au ** 1.5)
    return (M0_deg + n_deg_day * dt_days) % 360.0


# ---------------------------------------------------------------------------
# Element dict → Body
# ---------------------------------------------------------------------------

def elements_to_body(params: dict, color: tuple = (200, 150, 80)) -> Optional[Body]:
    try:
        M_j2000 = _propagate_mean_anomaly(
            params["M0_deg"], params["a_au"], params.get("epoch_jd", 2451545.0)
        )
        pos, vel = _keplerian_to_cartesian(
            a_au   = params["a_au"],
            e      = params["e"],
            i_deg  = params["i_deg"],
            Om_deg = params["Om_deg"],
            w_deg  = params["w_deg"],
            M0_deg = M_j2000,
            mu     = GM_SUN,
        )
        return Body(
            name        = params["name"],
            mass        = 1e12,
            pos         = pos,
            vel         = vel,
            radius      = 5e5,
            color       = color,
            is_asteroid = True,
            designation = params.get("designation"),
            trail_max   = 150,
        )
    except Exception as e:
        print(f"[elements_to_body] Failed for '{params.get('name','?')}': {e}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_asteroids(max_count: int = MAX_ASTEROIDS) -> List[Body]:
    bodies: List[Body] = []
    seen:   set        = set()

    # ── SSD/SBDB path ────────────────────────────────────────────────────────
    approaches = fetch_ssd_close_approaches(days_ahead=90)
    for approach in approaches:
        if len(bodies) >= max_count:
            break
        des = approach["designation"]
        if des in seen:
            continue
        seen.add(des)

        params = fetch_sbdb_elements(des)
        if not params:
            continue
        params["designation"] = des

        color = ASTEROID_COLORS[len(bodies) % len(ASTEROID_COLORS)]
        body  = elements_to_body(params, color)
        if body:
            bodies.append(body)
            print(f"[Asteroid] {body.name:<30s}  closest {approach.get('dist_au','?')} AU")
        time.sleep(0.2)

    # ── NeoWs fallback ───────────────────────────────────────────────────────
    if len(bodies) < max_count:
        neos = fetch_neows_feed(max_days=7)
        for neo in neos:
            if len(bodies) >= max_count:
                break
            name = neo.get("name", "")
            if name in seen:
                continue
            seen.add(name)

            neo_id = neo.get("id")
            if not neo_id:
                continue
            params = fetch_neows_detail(str(neo_id))
            if not params:
                continue

            color = ASTEROID_COLORS[len(bodies) % len(ASTEROID_COLORS)]
            body  = elements_to_body(params, color)
            if body:
                bodies.append(body)
                print(f"[Asteroid] {body.name:<30s}  (NeoWs fallback)")
            time.sleep(0.3)

    print(f"[fetch_asteroids] Loaded {len(bodies)} asteroid(s)")
    return bodies


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asteroids = fetch_asteroids(max_count=5)
    print()
    for ast in asteroids:
        r_au  = np.linalg.norm(ast.pos) / AU
        v_kms = np.linalg.norm(ast.vel) / 1000
        print(f"  {ast.name:<32s}  r={r_au:.3f} AU   |v|={v_kms:.2f} km/s")