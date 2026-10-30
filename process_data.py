"""Process extracted telemetry: build features per team per phase."""

import json
import numpy as np
from pathlib import Path

EXTRACTED = Path("data/extracted")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

MAP_SIZE = 816000
GRID_RES = 2500


def assign_phases(circles):
    """Recompute phases from blue zone radius transitions."""
    if not circles:
        return circles
    sorted_c = sorted(circles, key=lambda c: c["elapsed"])
    phase = 0
    last_br = None
    for c in sorted_c:
        br = round(c["blue_r"])
        if last_br is not None and br != last_br:
            phase += 1
        last_br = br
        c["phase"] = phase
    return sorted_c


def get_phase_boundaries(circles):
    """Get time boundaries for each phase from circle data."""
    if not circles:
        return {}
    circles = assign_phases(circles)
    phases = {}
    for c in circles:
        p = c["phase"]
        if p not in phases:
            phases[p] = {"min_t": c["elapsed"], "max_t": c["elapsed"], "circles": []}
        else:
            phases[p]["min_t"] = min(phases[p]["min_t"], c["elapsed"])
            phases[p]["max_t"] = max(phases[p]["max_t"], c["elapsed"])
        phases[p]["circles"].append(c)
    return phases


def assign_positions_to_phases(positions, phase_bounds):
    """Assign player positions to phases by elapsed time."""
    if not positions or not phase_bounds:
        return {}

    sorted_phases = sorted(phase_bounds.keys())
    phase_positions = {p: [] for p in sorted_phases}

    for pos in positions:
        elapsed = pos["elapsed"]
        assigned = sorted_phases[0]
        for p in sorted_phases:
            b = phase_bounds[p]
            if b["min_t"] <= elapsed <= b["max_t"]:
                assigned = p
                break
            elif elapsed < b["min_t"]:
                assigned = p
                break
            assigned = p
        phase_positions[assigned].append(pos)
    return phase_positions


def assign_kills_to_phases(kills, positions, phase_bounds):
    """Estimate which phase each kill belongs to based on position matching."""
    if not kills or not phase_bounds:
        return {}

    sorted_phases = sorted(phase_bounds.keys())
    phase_kills = {p: [] for p in sorted_phases}

    for k in kills:
        best_phase = sorted_phases[-1]
        for p in sorted_phases:
            b = phase_bounds[p]
            mid_t = (b["min_t"] + b["max_t"]) / 2
            if mid_t > 0:
                best_phase = p
        phase_kills[best_phase].append(k)

    return phase_kills


def build_heightmap(all_positions):
    """Build a terrain heightmap from z-samples across all matches.
    Filters parachute/plane positions. Returns heightmap in METERS."""
    grid_size = MAP_SIZE // GRID_RES
    height_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    height_count = np.zeros((grid_size, grid_size), dtype=np.int32)

    for pos in all_positions:
        z = pos["z"]
        if z > 25000 or z < -1000:
            continue
        gx = int(pos["x"] / GRID_RES)
        gy = int(pos["y"] / GRID_RES)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            height_sum[gy, gx] += z
            height_count[gy, gx] += 1

    mask = height_count > 0
    heightmap = np.zeros_like(height_sum)
    heightmap[mask] = height_sum[mask] / height_count[mask]
    heightmap /= 100.0
    return heightmap, height_count


CELL_SIZE_M = GRID_RES / 100.0


def compute_slope(heightmap, cell_size=None):
    if cell_size is None:
        cell_size = CELL_SIZE_M
    dy, dx = np.gradient(heightmap, cell_size)
    return np.sqrt(dx**2 + dy**2)


def team_features(team_positions, circle, heightmap, slope_map, placements, team_id):
    if not team_positions or not circle:
        return None

    xs = [p["x"] for p in team_positions]
    ys = [p["y"] for p in team_positions]
    zs = [p["z"] for p in team_positions]

    cx, cy, cz = np.mean(xs), np.mean(ys), np.mean(zs)

    # Extracted data has old naming: white_*=safetyZone, blue_*=poisonGasWarning
    blue_x = circle["white_x"]
    blue_y = circle["white_y"]
    blue_r = circle["white_r"]
    white_x = circle["blue_x"]
    white_y = circle["blue_y"]
    white_r = circle["blue_r"]

    dist_to_blue = np.sqrt((cx - blue_x)**2 + (cy - blue_y)**2)
    dist_to_white = np.sqrt((cx - white_x)**2 + (cy - white_y)**2)

    in_blue = 1 if dist_to_blue < blue_r else 0
    in_white = 1 if dist_to_white < white_r else 0
    zone_edge_dist = blue_r - dist_to_blue
    zone_edge_ratio = dist_to_blue / max(blue_r, 1)

    grid_size = heightmap.shape[0]
    gx = min(max(int(cx / GRID_RES), 0), grid_size - 1)
    gy = min(max(int(cy / GRID_RES), 0), grid_size - 1)
    elevation = heightmap[gy, gx]
    slope_val = slope_map[gy, gx]

    team_spread = np.sqrt(np.var(xs) + np.var(ys))
    team_alive = len(set(p["name"] for p in team_positions))

    rank = placements.get(str(team_id), placements.get(team_id, 0))

    return {
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "centroid_z": float(cz),
        "dist_to_blue_center": float(dist_to_blue),
        "dist_to_white_center": float(dist_to_white),
        "in_blue": in_blue,
        "in_white": in_white,
        "zone_edge_dist": float(zone_edge_dist),
        "zone_edge_ratio": float(zone_edge_ratio),
        "elevation": float(elevation),
        "slope": float(slope_val),
        "team_spread": float(team_spread),
        "team_alive": team_alive,
        "blue_r": float(blue_r),
        "white_r": float(white_r),
        "rank": rank,
    }


def process_match(data, heightmap, slope_map):
    circles = data["circles"]
    positions = data["positions"]
    kills = data["kills"]
    placements = data["placements"]

    phase_bounds = get_phase_boundaries(circles)
    phase_positions = assign_positions_to_phases(positions, phase_bounds)

    rows = []
    sorted_phases = sorted(phase_bounds.keys())

    for phase_num in sorted_phases:
        if phase_num < 2 or phase_num > 8:
            continue

        phase_circs = phase_bounds[phase_num]["circles"]
        if not phase_circs:
            continue
        circle = phase_circs[len(phase_circs) // 2]

        pp = phase_positions.get(phase_num, [])
        teams = {}
        for p in pp:
            tid = p["team"]
            if tid not in teams:
                teams[tid] = []
            teams[tid].append(p)

        next_phase = None
        for sp in sorted_phases:
            if sp > phase_num:
                next_phase = sp
                break

        next_teams = set()
        if next_phase is not None and next_phase in phase_positions:
            for p in phase_positions[next_phase]:
                next_teams.add(p["team"])

        for tid, tpos in teams.items():
            feat = team_features(tpos, circle, heightmap, slope_map, placements, tid)
            if feat is None:
                continue

            feat["match_id"] = data["match_id"]
            feat["phase"] = phase_num
            feat["team_id"] = tid

            if next_phase is not None:
                survived = 1 if tid in next_teams else 0
            else:
                survived = 1 if feat["rank"] == 1 else 0
            feat["survived"] = survived

            rows.append(feat)

    return rows


def main():
    files = sorted(EXTRACTED.glob("*.json"))
    print(f"Found {len(files)} extracted match files")

    if not files:
        print("No data to process. Run collect_data.py first.")
        return

    print("Building heightmap from z-samples...")
    all_z_positions = []
    for f in files:
        data = json.loads(f.read_text())
        all_z_positions.extend(data["positions"])

    heightmap, hcount = build_heightmap(all_z_positions)
    slope_map = compute_slope(heightmap)
    coverage = np.sum(hcount > 0) / hcount.size * 100
    print(f"Heightmap: {heightmap.shape}, {coverage:.1f}% coverage")

    np.save(PROCESSED / "heightmap.npy", heightmap)
    np.save(PROCESSED / "heightmap_count.npy", hcount)

    print("Extracting features per team per phase...")
    all_rows = []
    circle_data = []

    for i, f in enumerate(files):
        data = json.loads(f.read_text())
        rows = process_match(data, heightmap, slope_map)
        all_rows.extend(rows)

        for c in data["circles"]:
            circle_data.append({
                "match_id": data["match_id"],
                "elapsed": c["elapsed"],
                "white_x": c["blue_x"], "white_y": c["blue_y"], "white_r": c["blue_r"],
                "blue_x": c["white_x"], "blue_y": c["white_y"], "blue_r": c["white_r"],
                "alive": c.get("alive", 0),
                "alive_players": c.get("alive_players", 0),
                "phase": c.get("phase", 0),
            })

        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(files)} matches, {len(all_rows)} feature rows")

    print(f"\nTotal feature rows: {len(all_rows)}")
    print(f"Total circle snapshots: {len(circle_data)}")

    with open(PROCESSED / "features.json", "w") as fp:
        json.dump(all_rows, fp)
    with open(PROCESSED / "circles.json", "w") as fp:
        json.dump(circle_data, fp)

    phases = {}
    for r in all_rows:
        p = r["phase"]
        if p not in phases:
            phases[p] = {"total": 0, "survived": 0}
        phases[p]["total"] += 1
        phases[p]["survived"] += r["survived"]

    print("\nPhase survival rates:")
    for p in sorted(phases):
        t = phases[p]["total"]
        s = phases[p]["survived"]
        print(f"  Phase {p}: {s}/{t} survived ({100*s/t:.1f}%)")

    all_kills = []
    for f in files:
        data = json.loads(f.read_text())
        for k in data["kills"]:
            k["match_id"] = data["match_id"]
            all_kills.append(k)

    with open(PROCESSED / "kills.json", "w") as fp:
        json.dump(all_kills, fp)
    print(f"Kill events: {len(all_kills)}")


if __name__ == "__main__":
    main()
