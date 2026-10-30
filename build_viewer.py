"""Build the interactive HTML viewer with embedded data."""

import base64
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

PROCESSED = Path("data/processed")
EXTRACTED = Path("data/extracted")
MODEL_DIR = Path("model")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
ASSETS = Path("assets")

MAP_SIZE = 816000
GRID_RES = 2500


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_viewer_data():
    with open(PROCESSED / "features.json") as f:
        features = json.load(f)
    with open(PROCESSED / "circles.json") as f:
        circles = json.load(f)
    with open(PROCESSED / "kills.json") as f:
        all_kills = json.load(f)
    with open(MODEL_DIR / "model_info.json") as f:
        model_info = json.load(f)
    with open(MODEL_DIR / "train_matches.json") as f:
        train_matches = set(json.load(f))

    kills = [k for k in all_kills if k.get("match_id") in train_matches]
    print(f"Kills: {len(kills)} from {len(train_matches)} train matches "
          f"(filtered {len(all_kills) - len(kills)} test-match kills)")

    # --- Aggregate circle data per phase ---
    phase_circles = {}
    for c in circles:
        p = c.get("phase", 0)
        if p < 1 or p > 9:
            continue
        if p not in phase_circles:
            phase_circles[p] = {"wx": [], "wy": [], "wr": [], "bx": [], "by": [], "br": []}
        phase_circles[p]["wx"].append(c["white_x"])
        phase_circles[p]["wy"].append(c["white_y"])
        phase_circles[p]["wr"].append(c["white_r"])
        phase_circles[p]["bx"].append(c["blue_x"])
        phase_circles[p]["by"].append(c["blue_y"])
        phase_circles[p]["br"].append(c["blue_r"])

    avg_circles = {}
    for p, data in phase_circles.items():
        avg_circles[p] = {
            "wx": float(np.mean(data["wx"])),
            "wy": float(np.mean(data["wy"])),
            "wr": float(np.mean(data["wr"])),
            "bx": float(np.mean(data["bx"])),
            "by": float(np.mean(data["by"])),
            "br": float(np.mean(data["br"])),
            "count": len(data["wx"]),
        }

    # --- Phase radii (median) ---
    phase_radii = {}
    for p, data in phase_circles.items():
        phase_radii[p] = {
            "white_r": float(np.median(data["wr"])),
            "blue_r": float(np.median(data["br"])),
        }

    # --- Circle shift vectors per phase ---
    circle_shifts = {}
    sorted_phases = sorted(avg_circles.keys())
    for i, p in enumerate(sorted_phases):
        if i == 0:
            continue
        prev = sorted_phases[i - 1]
        dx = avg_circles[p]["wx"] - avg_circles[prev]["wx"]
        dy = avg_circles[p]["wy"] - avg_circles[prev]["wy"]
        mag = max(np.sqrt(dx**2 + dy**2), 1)
        circle_shifts[p] = {"dx": round(dx / mag, 3), "dy": round(dy / mag, 3)}

    # --- Heatmaps (position, for optional global overlay) ---
    heatmaps = {}
    for phase in range(2, 9):
        phase_feat = [f for f in features if f["phase"] == phase and f["survived"] == 1]
        if not phase_feat:
            continue
        grid = np.zeros((64, 64))
        for f in phase_feat:
            gx = min(int(f["centroid_x"] / MAP_SIZE * 64), 63)
            gy = min(int(f["centroid_y"] / MAP_SIZE * 64), 63)
            grid[gy, gx] += 1
        grid = gaussian_filter(grid, sigma=2)
        if grid.max() > 0:
            grid = grid / grid.max()
        heatmaps[phase] = (grid * 255).astype(np.uint8).tolist()

    # --- Kill heatmap ---
    kgrid = np.zeros((64, 64))
    for k in kills:
        if k["x"] == 0 and k["y"] == 0:
            continue
        gx = min(int(k["x"] / MAP_SIZE * 64), 63)
        gy = min(int(k["y"] / MAP_SIZE * 64), 63)
        kgrid[gy, gx] += 1
    kgrid = gaussian_filter(kgrid, sigma=2)
    if kgrid.max() > 0:
        kgrid = kgrid / kgrid.max()
    kill_heatmap = (kgrid * 255).astype(np.uint8).tolist()

    # --- Per-phase stats ---
    phase_stats = {}
    for phase in range(1, 10):
        pf = [f for f in features if f["phase"] == phase]
        if not pf:
            continue
        survived = sum(1 for f in pf if f["survived"])
        phase_stats[phase] = {
            "teams": len(pf),
            "survived": survived,
            "rate": round(survived / len(pf) * 100, 1) if pf else 0,
        }

    # --- Circle scatter ---
    circle_scatter = {}
    for phase in range(1, 10):
        pc = phase_circles.get(phase, {})
        if not pc.get("wx"):
            continue
        pts = list(zip(
            [round(x / MAP_SIZE, 4) for x in pc["wx"]],
            [round(y / MAP_SIZE, 4) for y in pc["wy"]],
            [round(r / MAP_SIZE, 4) for r in pc["wr"]],
        ))
        if len(pts) > 200:
            idx = np.random.default_rng(42).choice(len(pts), 200, replace=False)
            pts = [pts[i] for i in idx]
        circle_scatter[phase] = pts

    # --- Heightmap (downsampled to 64x64) ---
    hm_path = PROCESSED / "heightmap.npy"
    if hm_path.exists():
        raw_hm = np.load(hm_path)
        from scipy.ndimage import zoom
        scale = 64 / raw_hm.shape[0]
        hm64 = zoom(raw_hm, scale, order=1)
        cell_m = MAP_SIZE / 100.0 / 64  # meters per cell
        dy, dx = np.gradient(hm64, cell_m)
        slope64 = np.sqrt(dx**2 + dy**2)
        heightmap_grid = np.round(hm64, 1).tolist()
        slope_grid = np.round(slope64, 4).tolist()
    else:
        heightmap_grid = [[0] * 64 for _ in range(64)]
        slope_grid = [[0] * 64 for _ in range(64)]

    # --- Kill positions (raw, normalized) from training matches only ---
    kills_by_phase = {}
    for f_path in sorted(EXTRACTED.glob("*.json")):
        data = json.loads(f_path.read_text())
        if data["match_id"] not in train_matches:
            continue
        match_circles = data["circles"]
        if not match_circles:
            continue
        sorted_c = sorted(match_circles, key=lambda c: c["elapsed"])
        phase = 0
        last_br = None
        phase_times = {}
        for c in sorted_c:
            br = round(c["blue_r"])
            if last_br is not None and br != last_br:
                phase += 1
            last_br = br
            if phase not in phase_times:
                phase_times[phase] = {"min": c["elapsed"], "max": c["elapsed"]}
            phase_times[phase]["max"] = c["elapsed"]

        for k in data["kills"]:
            if k["x"] == 0 and k["y"] == 0:
                continue
            # rough phase assignment: find the phase whose time range is closest
            best_p = 2
            for p_kills in sorted(phase_times.keys()):
                if p_kills < 2:
                    continue
                best_p = p_kills
            sp = str(best_p)
            if sp not in kills_by_phase:
                kills_by_phase[sp] = []
            kills_by_phase[sp].append([
                round(k["x"] / MAP_SIZE, 4),
                round(k["y"] / MAP_SIZE, 4),
            ])

    # Better approach: just store all kills flat with normalized coords
    all_kills_xy = []
    for k in kills:
        if k["x"] == 0 and k["y"] == 0:
            continue
        all_kills_xy.append([round(k["x"] / MAP_SIZE, 4), round(k["y"] / MAP_SIZE, 4)])

    # --- Raw kill density grid (unsmoothed, 64x64) for per-cell scoring ---
    kill_density = np.zeros((64, 64))
    for k in kills:
        if k["x"] == 0 and k["y"] == 0:
            continue
        gx = min(int(k["x"] / MAP_SIZE * 64), 63)
        gy = min(int(k["y"] / MAP_SIZE * 64), 63)
        kill_density[gy, gx] += 1
    # light smooth
    kill_density = gaussian_filter(kill_density, sigma=1.5)
    if kill_density.max() > 0:
        kill_density = kill_density / kill_density.max()
    kill_density_grid = np.round(kill_density, 3).tolist()

    # --- Team positions per phase ---
    team_pos = {}
    for f in features:
        p = str(f["phase"])
        if p not in team_pos:
            team_pos[p] = []
        team_pos[p].append([
            round(f["centroid_x"] / MAP_SIZE, 4),
            round(f["centroid_y"] / MAP_SIZE, 4),
            f["survived"],
        ])

    # --- Prediction grid ---
    pred_grid = {}
    pred_file = PROCESSED / "prediction_grid.json"
    if pred_file.exists():
        with open(pred_file) as f:
            pred_grid = json.load(f)

    return {
        "avg_circles": {str(k): v for k, v in avg_circles.items()},
        "phase_radii": {str(k): v for k, v in phase_radii.items()},
        "circle_shifts": {str(k): v for k, v in circle_shifts.items()},
        "heatmaps": {str(k): v for k, v in heatmaps.items()},
        "kill_heatmap": kill_heatmap,
        "kill_density": kill_density_grid,
        "kills_xy": all_kills_xy,
        "phase_stats": {str(k): v for k, v in phase_stats.items()},
        "circle_scatter": {str(k): v for k, v in circle_scatter.items()},
        "model_info": model_info,
        "pred_grid": pred_grid,
        "heightmap": heightmap_grid,
        "slope": slope_grid,
        "team_pos": team_pos,
    }


def main():
    print("Building viewer data...")
    viewer_data = build_viewer_data()

    print("Encoding map image...")
    map_b64 = encode_image(ASSETS / "erangel.png")

    print("Writing HTML viewer...")
    html = generate_html(viewer_data, map_b64)
    out_path = OUTPUT / "viewer.html"
    out_path.write_text(html)
    print(f"Viewer saved to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def generate_html(data, map_b64):
    data_json = json.dumps(data, separators=(",", ":"))
    # Use placeholder replacement to avoid f-string {{ }} escaping
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json).replace("__MAP_B64__", map_b64)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erangel Positioning Map</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
  background: #0d1117;
  color: #c9d1d9;
  overflow-x: hidden;
}
.header {
  background: linear-gradient(135deg, #161b22, #1a2332);
  border-bottom: 1px solid #30363d;
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.header h1 { font-size: 18px; font-weight: 600; color: #e6edf3; }
.badge {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
}
.badge-green { background: #238636; color: #fff; }
.badge-blue { background: #1f6feb; color: #fff; }
.badge-amber { background: #9e6a03; color: #fff; }
.badge-dim { background: #30363d; color: #8b949e; font-size: 11px; }
.container {
  display: flex;
  gap: 0;
  height: calc(100vh - 52px);
}
.sidebar {
  width: 320px;
  min-width: 320px;
  background: #161b22;
  border-right: 1px solid #30363d;
  overflow-y: auto;
  padding: 12px 16px;
}
.sidebar h3 {
  font-size: 11px;
  font-weight: 600;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 14px 0 6px;
}
.sidebar h3:first-child { margin-top: 0; }
.phase-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 6px 10px;
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 6px;
  color: #c9d1d9;
  cursor: pointer;
  margin: 3px 0;
  font-size: 12px;
  transition: all 0.15s;
}
.phase-btn:hover { background: #30363d; }
.phase-btn.active { background: #1f6feb22; border-color: #1f6feb; color: #58a6ff; }
.phase-btn .num {
  width: 22px; height: 22px; border-radius: 50%;
  background: #30363d;
  display: flex; align-items: center; justify-content: center;
  font-weight: 600; font-size: 11px;
}
.phase-btn.active .num { background: #1f6feb; color: #fff; }
.phase-btn .rate { font-size: 11px; color: #8b949e; margin-left: auto; }
.toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 0;
  font-size: 12px;
}
.toggle { position: relative; width: 34px; height: 18px; }
.toggle input { opacity: 0; width: 0; height: 0; }
.toggle .slider {
  position: absolute; inset: 0;
  background: #30363d; border-radius: 9px; cursor: pointer; transition: 0.2s;
}
.toggle .slider::before {
  content: ''; position: absolute;
  width: 14px; height: 14px; left: 2px; top: 2px;
  background: #c9d1d9; border-radius: 50%; transition: 0.2s;
}
.toggle input:checked + .slider { background: #1f6feb; }
.toggle input:checked + .slider::before { transform: translateX(16px); }
.model-row {
  display: flex; justify-content: space-between;
  font-size: 11px; padding: 2px 0; color: #8b949e;
}
.model-row .v { color: #c9d1d9; font-weight: 500; }
.model-row .warn { color: #d29922; }
.advisor-panel {
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
}
.advisor-panel .hint {
  color: #8b949e; font-size: 12px; text-align: center; padding: 16px 0;
}
.spot-card {
  display: flex; gap: 8px; padding: 8px 0;
  border-bottom: 1px solid #21262d;
}
.spot-card:last-child { border-bottom: none; }
.spot-num {
  width: 24px; height: 24px; border-radius: 50%;
  background: #238636; color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 12px; flex-shrink: 0;
}
.spot-num.low-conf { background: #30363d; border: 2px dashed #484f58; color: #8b949e; }
.spot-info { flex: 1; min-width: 0; }
.spot-reason { font-size: 12px; color: #e6edf3; }
.spot-meta { font-size: 11px; color: #8b949e; margin-top: 2px; }
.spot-meta .conf-low { color: #d29922; }
.map-area {
  flex: 1;
  display: flex; align-items: center; justify-content: center;
  background: #010409;
  position: relative; overflow: hidden;
}
#mapCanvas { cursor: crosshair; }
.tooltip {
  position: fixed;
  background: #1c2128;
  border: 1px solid #444c56;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 11px;
  pointer-events: none;
  z-index: 100;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  display: none;
  max-width: 260px;
}
.tooltip .tt-title { font-weight: 600; color: #e6edf3; margin-bottom: 3px; }
.tooltip .tt-row { display: flex; justify-content: space-between; gap: 12px; color: #8b949e; }
.tooltip .tt-val { color: #c9d1d9; }
.legend {
  position: absolute; bottom: 12px; right: 12px;
  background: #1c2128cc; border: 1px solid #30363d;
  border-radius: 8px; padding: 8px 10px; font-size: 10px;
}
.legend .row { display: flex; align-items: center; gap: 5px; margin: 2px 0; }
.legend .swatch { width: 12px; height: 12px; border-radius: 2px; }
</style>
</head>
<body>
<div class="header">
  <h1>Erangel Positioning Map</h1>
  <span class="badge badge-green" id="matchCount"></span>
  <span class="badge" id="modelBadge"></span>
  <span class="badge badge-dim" id="honesty"></span>
</div>
<div class="container">
  <div class="sidebar">
    <h3>Phase</h3>
    <div id="phaseButtons"></div>

    <h3>Advisor</h3>
    <div class="advisor-panel" id="advisorPanel">
      <div class="hint">Click the map to place a zone circle</div>
    </div>

    <h3>Layers</h3>
    <div class="toggle-row">
      <span>Position heatmap</span>
      <label class="toggle"><input type="checkbox" id="toggleHeat"><span class="slider"></span></label>
    </div>
    <div class="toggle-row">
      <span>Kill heatmap</span>
      <label class="toggle"><input type="checkbox" id="toggleKills"><span class="slider"></span></label>
    </div>
    <div class="toggle-row">
      <span>Circle scatter</span>
      <label class="toggle"><input type="checkbox" id="toggleCircles"><span class="slider"></span></label>
    </div>
    <div class="toggle-row">
      <span>Avg. zones</span>
      <label class="toggle"><input type="checkbox" id="toggleAvgZones" checked><span class="slider"></span></label>
    </div>

    <h3>Model (phase <span id="modelPhaseLabel">5</span>)</h3>
    <div id="modelCard" style="font-size:11px"></div>
  </div>
  <div class="map-area">
    <canvas id="mapCanvas"></canvas>
    <div class="tooltip" id="tooltip"></div>
    <div class="legend">
      <div class="row"><div class="swatch" style="background:#fff;border:1px solid #fff"></div> White zone</div>
      <div class="row"><div class="swatch" style="background:#3b82f6"></div> Blue zone</div>
      <div class="row"><div class="swatch" style="background:#22c55e"></div> Best spots</div>
      <div class="row"><div class="swatch" style="background:#ef4444"></div> Kill zones</div>
      <div class="row"><div class="swatch" style="background:#eab308;border-radius:50%"></div> Placed circle</div>
      <div class="row"><div class="swatch" style="background:rgba(255,120,0,0.6)"></div> Position heat</div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const MAP_B64 = "data:image/png;base64,__MAP_B64__";
const M = 816000;

let currentPhase = 5;
let canvasSize = 800;
let placedCircle = null;
let topSpots = [];
let entryRoute = [];

const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const mapImg = new Image();
mapImg.src = MAP_B64;

function resizeCanvas() {
  const area = document.querySelector('.map-area');
  canvasSize = Math.min(area.clientWidth, area.clientHeight) - 16;
  canvasSize = Math.max(400, canvasSize);
  canvas.width = canvasSize;
  canvas.height = canvasSize;
  render();
}

mapImg.onload = () => {
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
};

function px(v) { return v / M * canvasSize; }
function fromPx(v) { return v / canvasSize * M; }
function norm(v) { return v / M; }

// --- Terrain lookups ---
function getHeight(nx, ny) {
  const gx = Math.min(Math.floor(nx * 64), 63);
  const gy = Math.min(Math.floor(ny * 64), 63);
  if (gx < 0 || gy < 0) return 0;
  return DATA.heightmap[gy][gx];
}
function getSlope(nx, ny) {
  const gx = Math.min(Math.floor(nx * 64), 63);
  const gy = Math.min(Math.floor(ny * 64), 63);
  if (gx < 0 || gy < 0) return 0;
  return DATA.slope[gy][gx];
}
function getKillDensity(nx, ny) {
  const gx = Math.min(Math.floor(nx * 64), 63);
  const gy = Math.min(Math.floor(ny * 64), 63);
  if (gx < 0 || gy < 0) return 0;
  return DATA.kill_density[gy][gx];
}

// --- LR model scoring ---
function scoreLR(features) {
  const lr = DATA.model_info.lr_params;
  if (!lr) return 0.5;
  let z = lr.intercept;
  for (let i = 0; i < lr.coefs.length; i++) {
    const scaled = (features[i] - lr.scaler_mean[i]) / (lr.scaler_std[i] || 1);
    z += lr.coefs[i] * scaled;
  }
  return 1 / (1 + Math.exp(-z));
}

function buildFeatureVec(x, y, zoneX, zoneY, zoneR, phase) {
  const medians = DATA.model_info.phase_medians[String(phase)];
  if (!medians) return null;
  const nx = x / M, ny = y / M;
  const distBlue = Math.sqrt((x - zoneX) ** 2 + (y - zoneY) ** 2);
  const distWhite = medians.dist_to_white_center;
  const inBlue = distBlue < zoneR ? 1 : 0;
  const inWhite = medians.in_white;
  const edgeDist = zoneR - distBlue;
  const edgeRatio = distBlue / Math.max(zoneR, 1);
  const elev = getHeight(nx, ny);
  const slope = getSlope(nx, ny);
  const spread = medians.team_spread;
  const alive = medians.team_alive;
  const blueR = zoneR;
  const whiteR = medians.white_r;
  return [distBlue, distWhite, inBlue, inWhite, edgeDist, edgeRatio,
          elev, slope, spread, alive, blueR, whiteR, phase];
}

// --- Confidence: count nearby team positions ---
function getConfidence(nx, ny, phase) {
  const tp = DATA.team_pos[String(phase)];
  if (!tp) return 0;
  const thresh = 0.04; // ~330m radius
  let count = 0;
  for (const [tx, ty, surv] of tp) {
    if (surv && Math.abs(tx - nx) < thresh && Math.abs(ty - ny) < thresh) {
      const d = Math.sqrt((tx - nx) ** 2 + (ty - ny) ** 2);
      if (d < thresh) count++;
    }
  }
  return count;
}

// --- Kills near a point ---
function getKillsNear(nx, ny, radius) {
  const r2 = radius * radius;
  const result = [];
  for (const [kx, ky] of DATA.kills_xy) {
    const d2 = (kx - nx) ** 2 + (ky - ny) ** 2;
    if (d2 < r2) result.push([kx, ky]);
  }
  return result;
}

// --- Reason generation ---
function generateReason(spot, zoneX, zoneY, zoneR, phase) {
  const nx = spot.x / M, ny = spot.y / M;
  const parts = [];

  // Elevation
  const elev = getHeight(nx, ny);
  const avgElev = getHeight(norm(zoneX), norm(zoneY));
  const elevDiff = elev - avgElev;
  if (elevDiff > 5) parts.push("high ground +" + Math.round(elevDiff) + "m");

  // Zone edge
  const edgeDist = (zoneR - Math.sqrt((spot.x - zoneX) ** 2 + (spot.y - zoneY) ** 2)) / 100;
  if (edgeDist > 50) parts.push(Math.round(edgeDist) + "m inside edge");
  else if (edgeDist > 0) parts.push("edge position " + Math.round(edgeDist) + "m in");

  // Kill density
  const kd = getKillDensity(nx, ny);
  if (kd < 0.1) parts.push("low death zone");
  else if (kd > 0.5) parts.push("contested area");

  // Slope (cover)
  const slope = getSlope(nx, ny);
  if (slope > 0.01) parts.push("terrain cover");

  // Circle shift direction
  const shift = DATA.circle_shifts[String(phase)];
  if (shift) {
    const toSpotX = (spot.x - zoneX);
    const toSpotY = (spot.y - zoneY);
    const mag = Math.sqrt(toSpotX * toSpotX + toSpotY * toSpotY) || 1;
    const dot = (toSpotX / mag) * shift.dx + (toSpotY / mag) * shift.dy;
    if (dot > 0.5) parts.push("circle pulls this way");
  }

  return parts.length > 0 ? parts.join(", ") : "model-favored position";
}

// --- Analyze placed circle ---
function analyzeCircle() {
  if (!placedCircle) { topSpots = []; entryRoute = []; return; }

  const phase = currentPhase;
  const pr = DATA.phase_radii[String(phase)];
  const radius = pr ? pr.blue_r : 150000;
  const cx = placedCircle.x, cy = placedCircle.y;

  const gridRes = 28;
  const candidates = [];

  for (let i = 0; i < gridRes; i++) {
    for (let j = 0; j < gridRes; j++) {
      const x = cx - radius + (2 * radius * i / (gridRes - 1));
      const y = cy - radius + (2 * radius * j / (gridRes - 1));
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (dist > radius) continue;
      if (x < 0 || x > M || y < 0 || y > M) continue;

      const fv = buildFeatureVec(x, y, cx, cy, radius, phase);
      if (!fv) continue;

      const modelScore = scoreLR(fv);
      const nx = x / M, ny = y / M;
      const kd = getKillDensity(nx, ny);
      const elev = getHeight(nx, ny);
      const slope = getSlope(nx, ny);
      const combined = modelScore * 0.6 + (1 - kd) * 0.2 + Math.min(elev / 200, 1) * 0.1 + Math.min(slope / 0.02, 1) * 0.1;

      candidates.push({ x, y, score: combined, modelProb: modelScore, kd, elev, slope });
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  topSpots = [];
  const minSep = radius * 0.2;
  for (const c of candidates) {
    if (topSpots.length >= 5) break;
    const tooClose = topSpots.some(s =>
      Math.sqrt((s.x - c.x) ** 2 + (s.y - c.y) ** 2) < minSep
    );
    if (tooClose) continue;

    const nx = c.x / M, ny = c.y / M;
    c.confidence = getConfidence(nx, ny, phase);
    c.reason = generateReason(c, cx, cy, radius, phase);
    topSpots.push(c);
  }

  computeEntryRoute(cx, cy, radius, phase);
  updateAdvisorPanel();
}

// --- Entry route: simple A* from previous circle edge to best spot ---
function computeEntryRoute(cx, cy, radius, phase) {
  entryRoute = [];
  if (topSpots.length === 0) return;
  if (phase <= 2) return;

  const prevAC = DATA.avg_circles[String(phase - 1)];
  if (!prevAC) return;

  const target = topSpots[0];
  const dx = target.x - prevAC.bx;
  const dy = target.y - prevAC.by;
  const mag = Math.sqrt(dx * dx + dy * dy) || 1;
  const startX = prevAC.bx + (dx / mag) * prevAC.br;
  const startY = prevAC.by + (dy / mag) * prevAC.br;

  // Simple straight-line route with waypoints avoiding high-kill cells
  const steps = 20;
  const route = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    let rx = startX + (target.x - startX) * t;
    let ry = startY + (target.y - startY) * t;

    // Nudge away from kill hotspots
    const nx = rx / M, ny = ry / M;
    const kd = getKillDensity(nx, ny);
    if (kd > 0.3 && i > 0 && i < steps) {
      // Try perpendicular offsets
      const perpX = -(target.y - startY) / mag;
      const perpY = (target.x - startX) / mag;
      const offset = radius * 0.15;
      const kd1 = getKillDensity((rx + perpX * offset) / M, (ry + perpY * offset) / M);
      const kd2 = getKillDensity((rx - perpX * offset) / M, (ry - perpY * offset) / M);
      if (kd1 < kd) { rx += perpX * offset * 0.6; ry += perpY * offset * 0.6; }
      else if (kd2 < kd) { rx -= perpX * offset * 0.6; ry -= perpY * offset * 0.6; }

      // Also prefer higher ground
      const h0 = getHeight(rx / M, ry / M);
      const h1 = getHeight((rx + perpX * offset) / M, (ry + perpY * offset) / M);
      const h2 = getHeight((rx - perpX * offset) / M, (ry - perpY * offset) / M);
      if (h1 > h0 + 200 && kd1 <= kd) { rx += perpX * offset * 0.3; ry += perpY * offset * 0.3; }
      else if (h2 > h0 + 200 && kd2 <= kd) { rx -= perpX * offset * 0.3; ry -= perpY * offset * 0.3; }
    }
    route.push({ x: rx, y: ry });
  }
  entryRoute = route;
}

// --- Advisor panel HTML ---
function updateAdvisorPanel() {
  const panel = document.getElementById('advisorPanel');
  if (!placedCircle || topSpots.length === 0) {
    panel.innerHTML = '<div class="hint">Click the map to place a zone circle</div>';
    return;
  }

  const pr = DATA.phase_radii[String(currentPhase)];
  const radius = pr ? pr.blue_r : 150000;
  let html = '<div style="font-size:11px;color:#8b949e;margin-bottom:8px">' +
    'Phase ' + currentPhase + ' circle &middot; ' + Math.round(radius / 100) + 'm radius</div>';

  topSpots.forEach((s, i) => {
    const isLow = s.confidence < 3;
    html += '<div class="spot-card">' +
      '<div class="spot-num' + (isLow ? ' low-conf' : '') + '">' + (i + 1) + '</div>' +
      '<div class="spot-info">' +
        '<div class="spot-reason">' + s.reason + '</div>' +
        '<div class="spot-meta">' +
          (s.confidence > 0 ? s.confidence + ' teams backed this' : 'no team data') +
          ' &middot; ' + Math.round(s.modelProb * 100) + '% survival' +
          (isLow ? ' <span class="conf-low">low confidence</span>' : '') +
        '</div>' +
      '</div>' +
    '</div>';
  });

  panel.innerHTML = html;
}

// --- Drawing ---
function drawCircleStroke(cx, cy, r, color, lw, dash) {
  ctx.beginPath();
  ctx.arc(px(cx), px(cy), px(r), 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  if (dash) ctx.setLineDash(dash);
  else ctx.setLineDash([]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function renderHeatmap(grid, r, g, b, maxAlpha) {
  if (!grid) return;
  const rows = grid.length, cols = grid[0].length;
  const cw = canvasSize / cols, ch = canvasSize / rows;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const v = grid[y][x];
      if (v < 5) continue;
      const a = (v / 255) * maxAlpha;
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
      ctx.fillRect(x * cw, y * ch, cw + 0.5, ch + 0.5);
    }
  }
}

function render() {
  ctx.clearRect(0, 0, canvasSize, canvasSize);
  ctx.drawImage(mapImg, 0, 0, canvasSize, canvasSize);
  const p = String(currentPhase);

  // Position heatmap (off by default)
  if (document.getElementById('toggleHeat').checked && DATA.heatmaps[p]) {
    renderHeatmap(DATA.heatmaps[p], 255, 120, 0, 0.6);
  }

  // Kill heatmap
  if (document.getElementById('toggleKills').checked) {
    renderHeatmap(DATA.kill_heatmap, 255, 20, 60, 0.5);
  }

  // Circle scatter
  if (document.getElementById('toggleCircles').checked && DATA.circle_scatter[p]) {
    ctx.globalAlpha = 0.15;
    for (const [cx, cy, r] of DATA.circle_scatter[p]) {
      ctx.beginPath();
      ctx.arc(cx * canvasSize, cy * canvasSize, r * canvasSize, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // Average zones
  if (document.getElementById('toggleAvgZones').checked && DATA.avg_circles[p]) {
    const ac = DATA.avg_circles[p];
    drawCircleStroke(ac.wx, ac.wy, ac.wr, 'rgba(255,255,255,0.7)', 2, null);
    drawCircleStroke(ac.bx, ac.by, ac.br, 'rgba(59,130,246,0.6)', 1.5, [6, 4]);
  }

  // --- Placed circle and advisor overlay ---
  if (placedCircle) {
    const pr = DATA.phase_radii[p];
    const radius = pr ? pr.blue_r : 150000;
    const pcx = px(placedCircle.x), pcy = px(placedCircle.y), prr = px(radius);

    // Circle outline
    ctx.beginPath();
    ctx.arc(pcx, pcy, prr, 0, Math.PI * 2);
    ctx.strokeStyle = '#eab308';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Clip for overlays inside circle
    ctx.save();
    ctx.beginPath();
    ctx.arc(pcx, pcy, prr, 0, Math.PI * 2);
    ctx.clip();

    // Light scoring overlay inside circle
    const gridN = 40;
    const cellSz = prr * 2 / gridN;
    for (let gi = 0; gi < gridN; gi++) {
      for (let gj = 0; gj < gridN; gj++) {
        const sx = pcx - prr + gi * cellSz + cellSz / 2;
        const sy = pcy - prr + gj * cellSz + cellSz / 2;
        const dx = sx - pcx, dy = sy - pcy;
        if (dx * dx + dy * dy > prr * prr) continue;
        const mx = fromPx(sx), my = fromPx(sy);
        const fv = buildFeatureVec(mx, my, placedCircle.x, placedCircle.y, radius, currentPhase);
        if (!fv) continue;
        const prob = scoreLR(fv);
        const kd = getKillDensity(mx / M, my / M);
        if (prob > 0.7 && kd < 0.3) {
          const a = Math.min((prob - 0.7) / 0.3 * 0.25, 0.25);
          ctx.fillStyle = 'rgba(34,197,94,' + a + ')';
          ctx.fillRect(sx - cellSz / 2, sy - cellSz / 2, cellSz, cellSz);
        }
      }
    }

    // Red kill markers inside circle
    const nr = norm(radius) * 1.05;
    const ncx = norm(placedCircle.x), ncy = norm(placedCircle.y);
    const nearKills = getKillsNear(ncx, ncy, nr);
    ctx.fillStyle = 'rgba(239,68,68,0.5)';
    for (const [kx, ky] of nearKills) {
      const kpx = kx * canvasSize, kpy = ky * canvasSize;
      ctx.beginPath();
      ctx.arc(kpx, kpy, Math.max(3, canvasSize / 250), 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();

    // Entry route
    if (entryRoute.length > 1) {
      ctx.beginPath();
      ctx.moveTo(px(entryRoute[0].x), px(entryRoute[0].y));
      for (let i = 1; i < entryRoute.length; i++) {
        ctx.lineTo(px(entryRoute[i].x), px(entryRoute[i].y));
      }
      ctx.strokeStyle = 'rgba(34,197,94,0.7)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Arrow at start
      const a0 = entryRoute[0], a1 = entryRoute[1];
      const ax = px(a0.x), ay = px(a0.y);
      ctx.beginPath();
      ctx.arc(ax, ay, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#22c55e';
      ctx.fill();
    }

    // Top 5 green markers
    topSpots.forEach((s, i) => {
      const sx = px(s.x), sy = px(s.y);
      const isLow = s.confidence < 3;
      const markerR = Math.max(10, canvasSize / 60);

      // Outer ring
      ctx.beginPath();
      ctx.arc(sx, sy, markerR, 0, Math.PI * 2);
      if (isLow) {
        ctx.strokeStyle = '#6b7280';
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(107,114,128,0.3)';
      } else {
        ctx.fillStyle = 'rgba(34,197,94,0.85)';
      }
      ctx.fill();

      // Number
      ctx.fillStyle = '#fff';
      ctx.font = 'bold ' + Math.max(11, canvasSize / 70) + 'px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), sx, sy + 1);
    });
  }

  // Phase label
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(8, 8, 110, 28);
  ctx.fillStyle = '#e6edf3';
  ctx.font = 'bold 14px -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('Phase ' + currentPhase, 14, 27);
}

// --- Click handler ---
canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mapX = fromPx(e.clientX - rect.left);
  const mapY = fromPx(e.clientY - rect.top);
  if (mapX < 0 || mapX > M || mapY < 0 || mapY > M) return;

  placedCircle = { x: mapX, y: mapY };
  analyzeCircle();
  render();
});

// --- Build UI ---
function buildUI() {
  const container = document.getElementById('phaseButtons');
  for (let i = 2; i <= 8; i++) {
    const stats = DATA.phase_stats[String(i)];
    const btn = document.createElement('button');
    btn.className = 'phase-btn' + (i === currentPhase ? ' active' : '');
    btn.innerHTML =
      '<div class="num">' + i + '</div>' +
      '<div style="flex:1;text-align:left">Phase ' + i + '</div>' +
      '<div class="rate">' + (stats ? stats.rate + '%' : '-') + '</div>';
    btn.addEventListener('click', () => {
      currentPhase = i;
      document.querySelectorAll('.phase-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (placedCircle) analyzeCircle();
      updateModelCard();
      render();
    });
    container.appendChild(btn);
  }

  // Model honesty header
  const mi = DATA.model_info;
  const lift = mi.lift || (mi.accuracy - mi.baseline_accuracy);
  document.getElementById('matchCount').textContent = mi.n_matches + ' matches';

  if (lift < 0.05) {
    document.getElementById('modelBadge').className = 'badge badge-amber';
    document.getElementById('modelBadge').textContent =
      'Model +' + (lift * 100).toFixed(1) + 'pp over baseline';
    document.getElementById('honesty').textContent =
      (mi.accuracy * 100).toFixed(1) + '% vs always-survive ' + (mi.baseline_accuracy * 100).toFixed(1) + '%';
  } else {
    document.getElementById('modelBadge').className = 'badge badge-blue';
    document.getElementById('modelBadge').textContent =
      (mi.accuracy * 100).toFixed(1) + '% (+' + (lift * 100).toFixed(1) + 'pp over baseline)';
    document.getElementById('honesty').textContent =
      'AUC ' + (mi.auc || 0).toFixed(2) + ' | baseline ' + (mi.baseline_accuracy * 100).toFixed(1) + '%';
  }

  updateModelCard();

  // Toggle listeners
  ['toggleHeat', 'toggleKills', 'toggleCircles', 'toggleAvgZones'].forEach(id => {
    document.getElementById(id).addEventListener('change', render);
  });
}

function updateModelCard() {
  const p = String(currentPhase);
  document.getElementById('modelPhaseLabel').textContent = currentPhase;
  const pm = DATA.model_info.phase_metrics ? DATA.model_info.phase_metrics[p] : null;
  const mc = document.getElementById('modelCard');
  if (!pm) { mc.innerHTML = '<div class="model-row"><span>No data</span></div>'; return; }

  const liftBad = pm.lift < 0.05;
  mc.innerHTML =
    '<div class="model-row"><span>Model accuracy</span><span class="v">' + (pm.accuracy * 100).toFixed(1) + '%</span></div>' +
    '<div class="model-row"><span>Baseline (always-survive)</span><span class="v">' + (pm.baseline * 100).toFixed(1) + '%</span></div>' +
    '<div class="model-row"><span>Lift</span><span class="v' + (liftBad ? ' warn' : '') + '">+' +
      (pm.lift * 100).toFixed(1) + 'pp' + (liftBad ? ' (marginal)' : '') + '</span></div>' +
    '<div class="model-row"><span>ROC AUC</span><span class="v">' + pm.auc.toFixed(2) + '</span></div>' +
    '<div class="model-row"><span>Survival rate</span><span class="v">' + (pm.survival_rate * 100).toFixed(1) + '%</span></div>' +
    '<div class="model-row"><span>Samples</span><span class="v">' + pm.n_samples + '</span></div>';
}

// Tooltip
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = fromPx(e.clientX - rect.left);
  const y = fromPx(e.clientY - rect.top);
  const nx = x / M, ny = y / M;

  const tt = document.getElementById('tooltip');
  let info = '<div class="tt-title">' + Math.round(x / 100) + 'm, ' + Math.round(y / 100) + 'm</div>';

  const ac = DATA.avg_circles[String(currentPhase)];
  if (ac) {
    const dist = Math.sqrt((x - ac.wx) ** 2 + (y - ac.wy) ** 2);
    info += '<div class="tt-row"><span>To zone center</span><span class="tt-val">' + Math.round(dist / 100) + 'm</span></div>';
  }

  const elev = getHeight(nx, ny);
  if (elev > 0) info += '<div class="tt-row"><span>Elevation</span><span class="tt-val">' + Math.round(elev) + '</span></div>';

  const kd = getKillDensity(nx, ny);
  info += '<div class="tt-row"><span>Kill density</span><span class="tt-val">' + (kd * 100).toFixed(0) + '%</span></div>';

  tt.innerHTML = info;
  tt.style.display = 'block';
  tt.style.left = (e.clientX + 14) + 'px';
  tt.style.top = (e.clientY - 8) + 'px';
});

canvas.addEventListener('mouseleave', () => {
  document.getElementById('tooltip').style.display = 'none';
});

buildUI();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
