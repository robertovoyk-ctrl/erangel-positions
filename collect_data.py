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

TARGET_ERANGEL = 50
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
                "white_x": gs.get("safetyZonePosition", {}).get("x", 0),
                "white_y": gs.get("safetyZonePosition", {}).get("y", 0),
                "white_r": gs.get("safetyZoneRadius", 0),
                "blue_x": gs.get("poisonGasWarningPosition", {}).get("x", 0),
                "blue_y": gs.get("poisonGasWarningPosition", {}).get("y", 0),
                "blue_r": gs.get("poisonGasWarningRadius", 0),
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

    # Assign phases based on blue zone (poisonGasWarning) radius changes
    circles.sort(key=lambda c: c["elapsed"])
    phase = 0
    last_br = None
    for c in circles:
        br = round(c["blue_r"])
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


def main():
    print("=== PUBG Erangel Data Collection ===", flush=True)

    # Step 1: Load tournament list
    tourney_cache = RAW / "tournaments.json"
    data = api_get("/tournaments", tourney_cache)
    tournaments = data.get("data", [])
    print(f"Total tournaments: {len(tournaments)}", flush=True)

    # Filter last 12 months, sort newest first
    cutoff = datetime(2025, 10, 29, tzinfo=timezone.utc)
    recent = []
    for t in tournaments:
        created = t.get("attributes", {}).get("createdAt", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if dt >= cutoff:
                    recent.append((dt, t))
            except ValueError:
                pass
    recent.sort(key=lambda x: x[0], reverse=True)
    print(f"Recent tournaments (since {cutoff.date()}): {len(recent)}", flush=True)

    # Step 2: Fetch tournament details and collect match IDs
    print("\n=== Fetching tournament details ===", flush=True)
    all_matches = []
    for i, (dt, t) in enumerate(recent):
        tid = t["id"]
        cache = RAW / f"tournament_{tid}.json"
        cached = cache.exists()
        try:
            tdata = api_get(f"/tournaments/{tid}", cache)
        except requests.HTTPError as e:
            print(f"  [{i+1}/{len(recent)}] {tid}: error {e}", flush=True)
            continue

        matches = tdata.get("data", {}).get("relationships", {}).get("matches", {}).get("data", [])
        mids = [m["id"] for m in matches]
        if not cached:
            print(f"  [{i+1}/{len(recent)}] {tid}: {len(mids)} matches (fetched)", flush=True)
        all_matches.extend(mids)

    print(f"\nTotal match IDs: {len(all_matches)}", flush=True)

    # Check how many Erangel matches we already have extracted
    existing = list(EXTRACTED.glob("*.json"))
    print(f"Already extracted: {len(existing)} matches", flush=True)

    if len(existing) >= TARGET_ERANGEL:
        erangel_ids = [f.stem for f in existing]
        (RAW / "erangel_match_ids.json").write_text(json.dumps(erangel_ids, indent=2))
        print(f"\nAlready have {len(existing)} Erangel matches. Done!", flush=True)
        return

    # Step 3: Fetch match details, filter Erangel, download telemetry
    print(f"\n=== Fetching matches (target: {TARGET_ERANGEL} Erangel) ===", flush=True)
    erangel_count = len(existing)
    erangel_ids = [f.stem for f in existing]
    total_checked = 0

    for i, mid in enumerate(all_matches):
        if mid in erangel_ids:
            continue

        # Check if we already know this match's map from cache
        match_cache = RAW / f"match_{mid}.json"
        try:
            mdata = api_get(f"/shards/tournament/matches/{mid}", match_cache)
        except requests.HTTPError as e:
            print(f"  [{i+1}] error: {e}", flush=True)
            continue

        total_checked += 1
        map_name = mdata.get("data", {}).get("attributes", {}).get("mapName", "")

        if map_name != "Baltic_Main":
            if total_checked % 50 == 0:
                print(f"  checked {total_checked} matches, {erangel_count} Erangel so far...", flush=True)
            continue

        # It's Erangel — find telemetry URL
        tel_url = None
        for item in mdata.get("included", []):
            if item.get("type") == "asset" and "telemetry" in item.get("attributes", {}).get("name", "").lower():
                tel_url = item["attributes"]["URL"]
                break

        if not tel_url:
            print(f"  [{total_checked}] {mid[:16]}: Erangel but no telemetry URL", flush=True)
            continue

        print(f"  [{total_checked}] {mid[:16]}: Erangel! ({erangel_count+1}/{TARGET_ERANGEL})", flush=True)

        if extract_telemetry(tel_url, mid):
            erangel_count += 1
            erangel_ids.append(mid)

        if erangel_count >= TARGET_ERANGEL:
            print(f"\nReached target of {TARGET_ERANGEL} Erangel matches!", flush=True)
            break

    # Save final list
    (RAW / "erangel_match_ids.json").write_text(json.dumps(erangel_ids, indent=2))
    print(f"\n=== Done! ===", flush=True)
    print(f"Checked {total_checked} matches total", flush=True)
    print(f"Erangel matches with telemetry: {erangel_count}", flush=True)


if __name__ == "__main__":
    main()
