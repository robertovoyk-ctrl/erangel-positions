"""Collect PUBG tournament data: matches + telemetry for Erangel."""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PUBG_API_KEY")
BASE = "https://api.pubg.com"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/vnd.api+json",
}
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
EXTRACTED = Path("data/extracted")
EXTRACTED.mkdir(parents=True, exist_ok=True)

TARGET_ERANGEL = 9999
RATE_CALLS = []
RATE_LIMIT = 10
RATE_WINDOW = 60


def rate_wait():
    now = time.time()
    RATE_CALLS[:] = [t for t in RATE_CALLS if now - t < RATE_WINDOW]
    if len(RATE_CALLS) >= RATE_LIMIT:
        sleep_for = RATE_WINDOW - (now - RATE_CALLS[0]) + 0.5
        if sleep_for > 0:
            print(f"  rate limit: sleeping {sleep_for:.1f}s", flush=True)
            time.sleep(sleep_for)
    RATE_CALLS.append(time.time())


def api_get(path, cache_file=None):
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text())
    rate_wait()
    url = f"{BASE}{path}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 429:
        retry = int(r.headers.get("Retry-After", 60))
        print(f"  429 rate limited, sleeping {retry}s", flush=True)
        time.sleep(retry)
        return api_get(path, cache_file)
    r.raise_for_status()
    data = r.json()
    if cache_file:
        cache_file.write_text(json.dumps(data))
    return data


def extract_telemetry(url, match_id):
    """Download telemetry and extract only the data we need."""
    out_file = EXTRACTED / f"{match_id}.json"
    if out_file.exists():
        return True

    print(f"    downloading telemetry...", flush=True)
    try:
        r = requests.get(url, timeout=180, stream=True)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"    telemetry error: {e}", flush=True)
        return False

    circles = []
    positions = []
    kills = []
    placements = {}

    for ev in events:
        t = ev.get("_T")

        if t == "LogGameStatePeriodic":
            gs = ev.get("gameState", {})
            circles.append({
                "elapsed": gs.get("elapsedTime", 0),
                "white_x": gs.get("poisonGasWarningPosition", {}).get("x", 0),
                "white_y": gs.get("poisonGasWarningPosition", {}).get("y", 0),
                "white_r": gs.get("poisonGasWarningRadius", 0),
                "blue_x": gs.get("safetyZonePosition", {}).get("x", 0),
                "blue_y": gs.get("safetyZonePosition", {}).get("y", 0),
                "blue_r": gs.get("safetyZoneRadius", 0),
                "alive": gs.get("numAliveTeams", 0),
                "alive_players": gs.get("numAlivePlayers", 0),
            })

        elif t == "LogPlayerPosition":
            ch = ev.get("character", {})
            loc = ch.get("location", {})
            positions.append({
                "name": ch.get("name", ""),
                "team": ch.get("teamId", 0),
                "x": loc.get("x", 0),
                "y": loc.get("y", 0),
                "z": loc.get("z", 0),
                "elapsed": ev.get("elapsedTime", 0),
            })

        elif t in ("LogPlayerKillV2", "LogPlayerKill"):
            killer = ev.get("killer", {}) or {}
            victim = ev.get("victim", {}) or {}
            vloc = victim.get("location", {}) or {}
            kills.append({
                "killer": killer.get("name", ""),
                "killer_team": killer.get("teamId", 0),
                "victim": victim.get("name", ""),
                "victim_team": victim.get("teamId", 0),
                "x": vloc.get("x", 0),
                "y": vloc.get("y", 0),
                "z": vloc.get("z", 0),
                "cause": ev.get("damageCauserName", ""),
                "distance": ev.get("distance", 0),
            })

        elif t == "LogMatchEnd":
            for result in ev.get("characters", []):
                ch = result.get("character", result)
                team = ch.get("teamId", 0)
                rank = ch.get("ranking", 0)
                if team and rank:
                    placements[team] = rank

    # Assign phases based on white zone (poisonGasWarning) radius changes
    circles.sort(key=lambda c: c["elapsed"])
    phase = 0
    last_br = None
    for c in circles:
        br = round(c["white_r"])
        if last_br is not None and br != last_br:
            phase += 1
        last_br = br
        c["phase"] = phase

    extracted = {
        "match_id": match_id,
        "circles": circles,
        "positions": positions,
        "kills": kills,
        "placements": placements,
    }

    out_file.write_text(json.dumps(extracted))
    size_kb = out_file.stat().st_size / 1024
    print(f"    extracted: {len(circles)} circles, {len(positions)} positions, "
          f"{len(kills)} kills, {len(placements)} teams ({size_kb:.0f} KB)", flush=True)
    return True


def gather_match_ids_from_cached_tournaments():
    """Collect all match IDs from already-cached tournament detail files."""
    all_mids = []
    seen = set()
    for f in sorted(RAW.glob("tournament_*.json")):
        data = json.loads(f.read_text())
        matches = data.get("data", {}).get("relationships", {}).get("matches", {}).get("data", [])
        for m in matches:
            mid = m["id"]
            if mid not in seen:
                seen.add(mid)
                all_mids.append(mid)
    return all_mids


def process_match_id(mid, erangel_ids_set):
    """Check one match: is it Erangel? If so, extract telemetry. Returns True if Erangel."""
    if mid in erangel_ids_set:
        return False

    match_cache = RAW / f"match_{mid}.json"
    try:
        mdata = api_get(f"/shards/tournament/matches/{mid}", match_cache)
    except requests.HTTPError as e:
        print(f"    match error: {e}", flush=True)
        return False

    map_name = mdata.get("data", {}).get("attributes", {}).get("mapName", "")
    if map_name != "Baltic_Main":
        return False

    tel_url = None
    for item in mdata.get("included", []):
        if item.get("type") == "asset" and "telemetry" in item.get("attributes", {}).get("name", "").lower():
            tel_url = item["attributes"]["URL"]
            break

    if not tel_url:
        return False

    return extract_telemetry(tel_url, mid)


def main():
    print("=== PUBG Erangel Data Collection (all tournaments) ===", flush=True)

    # Step 1: Load tournament list
    tourney_cache = RAW / "tournaments.json"
    data = api_get("/tournaments", tourney_cache)
    tournaments = data.get("data", [])
    print(f"Total tournaments: {len(tournaments)}", flush=True)

    # Sort newest first
    sorted_tournaments = []
    for t in tournaments:
        created = t.get("attributes", {}).get("createdAt", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                sorted_tournaments.append((dt, t))
            except ValueError:
                pass
    sorted_tournaments.sort(key=lambda x: x[0], reverse=True)

    # Step 2: Gather match IDs from CACHED tournament details (free, no API)
    cached_match_ids = gather_match_ids_from_cached_tournaments()
    cached_tourney_count = len(list(RAW.glob("tournament_*.json")))
    print(f"Cached tournament details: {cached_tourney_count}", flush=True)
    print(f"Match IDs from cache: {len(cached_match_ids)}", flush=True)

    existing = set(f.stem for f in EXTRACTED.glob("*.json"))
    erangel_count = len(existing)
    erangel_ids = set(existing)
    print(f"Already extracted Erangel: {erangel_count}", flush=True)

    if erangel_count >= TARGET_ERANGEL:
        print(f"Already at target. Done!", flush=True)
        return

    # Step 3: Process cached match IDs first (most are cache hits)
    print(f"\n=== Phase 1: Processing {len(cached_match_ids)} matches from cached tournaments ===", flush=True)
    checked = 0
    for mid in cached_match_ids:
        if mid in erangel_ids:
            continue
        checked += 1
        if process_match_id(mid, erangel_ids):
            erangel_count += 1
            erangel_ids.add(mid)
            print(f"  Erangel #{erangel_count} ({checked} checked)", flush=True)
        elif checked % 100 == 0:
            print(f"  checked {checked}, {erangel_count} Erangel so far...", flush=True)

        if erangel_count >= TARGET_ERANGEL:
            break

    print(f"\nAfter phase 1: {erangel_count} Erangel from {checked} matches checked", flush=True)

    if erangel_count >= TARGET_ERANGEL:
        save_and_exit(erangel_ids, checked, erangel_count)
        return

    # Step 4: Fetch MORE tournament details and process their matches
    print(f"\n=== Phase 2: Fetching remaining tournament details ===", flush=True)
    cached_tids = set(f.stem.replace("tournament_", "") for f in RAW.glob("tournament_*.json"))

    for i, (dt, t) in enumerate(sorted_tournaments):
        tid = t["id"]
        if tid in cached_tids:
            continue

        cache = RAW / f"tournament_{tid}.json"
        try:
            tdata = api_get(f"/tournaments/{tid}", cache)
        except requests.HTTPError as e:
            print(f"  [{i+1}] {tid}: error {e}", flush=True)
            continue

        matches = tdata.get("data", {}).get("relationships", {}).get("matches", {}).get("data", [])
        mids = [m["id"] for m in matches]
        new_count = 0

        for mid in mids:
            if mid in erangel_ids:
                continue
            checked += 1
            if process_match_id(mid, erangel_ids):
                erangel_count += 1
                erangel_ids.add(mid)
                new_count += 1
                print(f"  Erangel #{erangel_count} from {tid} ({checked} total checked)", flush=True)

            if erangel_count >= TARGET_ERANGEL:
                break

        if new_count == 0 and len(mids) > 0:
            pass
        if erangel_count >= TARGET_ERANGEL:
            break

        if (i + 1) % 20 == 0:
            print(f"  tournaments processed: {i+1}, Erangel: {erangel_count}", flush=True)

    save_and_exit(erangel_ids, checked, erangel_count)


def save_and_exit(erangel_ids, checked, erangel_count):
    (RAW / "erangel_match_ids.json").write_text(json.dumps(sorted(erangel_ids), indent=2))
    print(f"\n=== Done! ===", flush=True)
    print(f"Checked {checked} matches total", flush=True)
    print(f"Erangel matches with telemetry: {erangel_count}", flush=True)


if __name__ == "__main__":
    main()
