"""Build the interactive HTML viewer with embedded data."""

import base64
import json
import numpy as np
from pathlib import Path

PROCESSED = Path("data/processed")
EXTRACTED = Path("data/extracted")
MODEL_DIR = Path("model")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
ASSETS = Path("assets")


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_viewer_data():
    """Prepare compact data for the viewer."""
    with open(PROCESSED / "features.json") as f:
        features = json.load(f)
    with open(PROCESSED / "circles.json") as f:
        circles = json.load(f)
    with open(PROCESSED / "kills.json") as f:
        kills = json.load(f)
    with open(MODEL_DIR / "model_info.json") as f:
        model_info = json.load(f)

    # Aggregate circle data per phase
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

    # Heatmap data: surviving positions per phase (downsampled)
    heatmaps = {}
    for phase in range(4, 9):
        phase_feat = [f for f in features if f["phase"] == phase and f["survived"] == 1]
        if not phase_feat:
            continue
        grid = np.zeros((64, 64))
        for f in phase_feat:
            gx = min(int(f["centroid_x"] / 816000 * 64), 63)
            gy = min(int(f["centroid_y"] / 816000 * 64), 63)
            grid[gy, gx] += 1
        # Gaussian blur
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=2)
        if grid.max() > 0:
            grid = grid / grid.max()
        # Quantize to 0-255
        grid_int = (grid * 255).astype(np.uint8)
        heatmaps[phase] = grid_int.tolist()

    # Kill heatmap
    kgrid = np.zeros((64, 64))
    for k in kills:
        if k["x"] == 0 and k["y"] == 0:
            continue
        gx = min(int(k["x"] / 816000 * 64), 63)
        gy = min(int(k["y"] / 816000 * 64), 63)
        kgrid[gy, gx] += 1
    from scipy.ndimage import gaussian_filter
    kgrid = gaussian_filter(kgrid, sigma=2)
    if kgrid.max() > 0:
        kgrid = kgrid / kgrid.max()
    kill_heatmap = (kgrid * 255).astype(np.uint8).tolist()

    # Per-phase stats
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

    # Individual circle positions for scatter display
    circle_scatter = {}
    for phase in range(1, 10):
        pc = phase_circles.get(phase, {})
        if not pc.get("wx"):
            continue
        pts = list(zip(
            [round(x / 816000, 4) for x in pc["wx"]],
            [round(y / 816000, 4) for y in pc["wy"]],
            [round(r / 816000, 4) for r in pc["wr"]],
        ))
        # Sample if too many
        if len(pts) > 200:
            idx = np.random.default_rng(42).choice(len(pts), 200, replace=False)
            pts = [pts[i] for i in idx]
        circle_scatter[phase] = pts

    # Prediction grid
    pred_grid = {}
    pred_file = PROCESSED / "prediction_grid.json"
    if pred_file.exists():
        with open(pred_file) as f:
            pred_grid = json.load(f)

    return {
        "avg_circles": {str(k): v for k, v in avg_circles.items()},
        "heatmaps": {str(k): v for k, v in heatmaps.items()},
        "kill_heatmap": kill_heatmap,
        "phase_stats": {str(k): v for k, v in phase_stats.items()},
        "circle_scatter": {str(k): v for k, v in circle_scatter.items()},
        "model_info": model_info,
        "pred_grid": pred_grid,
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
    data_json = json.dumps(data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erangel Positioning Map</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
  background: #0d1117;
  color: #c9d1d9;
  overflow-x: hidden;
}}
.header {{
  background: linear-gradient(135deg, #161b22, #1a2332);
  border-bottom: 1px solid #30363d;
  padding: 16px 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}}
.header h1 {{
  font-size: 20px;
  font-weight: 600;
  color: #e6edf3;
}}
.header .badge {{
  background: #238636;
  color: #fff;
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}}
.container {{
  display: flex;
  gap: 0;
  height: calc(100vh - 57px);
}}
.sidebar {{
  width: 300px;
  min-width: 300px;
  background: #161b22;
  border-right: 1px solid #30363d;
  overflow-y: auto;
  padding: 16px;
}}
.sidebar h3 {{
  font-size: 13px;
  font-weight: 600;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 16px 0 8px;
}}
.sidebar h3:first-child {{ margin-top: 0; }}
.phase-btn {{
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 8px 12px;
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 6px;
  color: #c9d1d9;
  cursor: pointer;
  margin: 4px 0;
  font-size: 13px;
  transition: all 0.15s;
}}
.phase-btn:hover {{ background: #30363d; border-color: #484f58; }}
.phase-btn.active {{ background: #1f6feb22; border-color: #1f6feb; color: #58a6ff; }}
.phase-btn .num {{
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #30363d;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 12px;
}}
.phase-btn.active .num {{ background: #1f6feb; color: #fff; }}
.phase-btn .info {{ flex: 1; text-align: left; }}
.phase-btn .rate {{
  font-size: 12px;
  color: #8b949e;
}}
.toggle-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 0;
  font-size: 13px;
}}
.toggle {{
  position: relative;
  width: 36px;
  height: 20px;
}}
.toggle input {{ opacity: 0; width: 0; height: 0; }}
.toggle .slider {{
  position: absolute;
  inset: 0;
  background: #30363d;
  border-radius: 10px;
  cursor: pointer;
  transition: 0.2s;
}}
.toggle .slider::before {{
  content: '';
  position: absolute;
  width: 16px;
  height: 16px;
  left: 2px;
  top: 2px;
  background: #c9d1d9;
  border-radius: 50%;
  transition: 0.2s;
}}
.toggle input:checked + .slider {{ background: #1f6feb; }}
.toggle input:checked + .slider::before {{ transform: translateX(16px); }}
.stats-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 8px 0;
}}
.stat-card {{
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 10px;
  text-align: center;
}}
.stat-card .val {{
  font-size: 20px;
  font-weight: 700;
  color: #e6edf3;
}}
.stat-card .label {{
  font-size: 11px;
  color: #8b949e;
  margin-top: 2px;
}}
.model-card {{
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 12px;
  margin: 8px 0;
}}
.model-card .row {{
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  padding: 3px 0;
}}
.model-card .row .k {{ color: #8b949e; }}
.model-card .row .v {{ color: #e6edf3; font-weight: 500; }}
.map-area {{
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #010409;
  position: relative;
  overflow: hidden;
}}
#mapCanvas {{
  cursor: crosshair;
}}
.tooltip {{
  position: fixed;
  background: #1c2128;
  border: 1px solid #444c56;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 12px;
  pointer-events: none;
  z-index: 100;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  display: none;
  max-width: 240px;
}}
.tooltip .tt-title {{
  font-weight: 600;
  color: #e6edf3;
  margin-bottom: 4px;
}}
.tooltip .tt-row {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  color: #8b949e;
}}
.tooltip .tt-val {{ color: #c9d1d9; }}
.legend {{
  position: absolute;
  bottom: 16px;
  right: 16px;
  background: #1c2128cc;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 10px;
  font-size: 11px;
}}
.legend .row {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
.legend .swatch {{ width: 14px; height: 14px; border-radius: 3px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Erangel Positioning Map</h1>
  <span class="badge" id="matchCount"></span>
  <span class="badge" style="background:#1f6feb" id="modelBadge"></span>
</div>
<div class="container">
  <div class="sidebar">
    <h3>Phase</h3>
    <div id="phaseButtons"></div>

    <h3>Layers</h3>
    <div class="toggle-row">
      <span>Position heatmap</span>
      <label class="toggle"><input type="checkbox" id="toggleHeat" checked><span class="slider"></span></label>
    </div>
    <div class="toggle-row">
      <span>Kill heatmap</span>
      <label class="toggle"><input type="checkbox" id="toggleKills"><span class="slider"></span></label>
    </div>
    <div class="toggle-row">
      <span>Circle scatter</span>
      <label class="toggle"><input type="checkbox" id="toggleCircles" checked><span class="slider"></span></label>
    </div>
    <div class="toggle-row">
      <span>Avg. zones</span>
      <label class="toggle"><input type="checkbox" id="toggleAvgZones" checked><span class="slider"></span></label>
    </div>
    <div class="toggle-row">
      <span>Model prediction</span>
      <label class="toggle"><input type="checkbox" id="togglePred"><span class="slider"></span></label>
    </div>

    <h3>Stats</h3>
    <div class="stats-grid">
      <div class="stat-card"><div class="val" id="statTeams">-</div><div class="label">Teams</div></div>
      <div class="stat-card"><div class="val" id="statRate">-</div><div class="label">Survival %</div></div>
    </div>

    <h3>Model</h3>
    <div class="model-card" id="modelCard"></div>

    <h3>Feature Importance</h3>
    <div id="featureList" style="font-size:12px"></div>
  </div>
  <div class="map-area">
    <canvas id="mapCanvas"></canvas>
    <div class="tooltip" id="tooltip"></div>
    <div class="legend">
      <div class="row"><div class="swatch" style="background:#fff;border:1px solid #fff"></div> White zone (safe)</div>
      <div class="row"><div class="swatch" style="background:#3b82f6;border:1px solid #3b82f6"></div> Blue zone</div>
      <div class="row"><div class="swatch" style="background:rgba(255,80,0,0.7)"></div> Position heat</div>
      <div class="row"><div class="swatch" style="background:rgba(255,0,60,0.7)"></div> Kill heat</div>
      <div class="row"><div class="swatch" style="background:rgba(0,200,100,0.5)"></div> Model: high survival</div>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};
const MAP_B64 = "data:image/png;base64,{map_b64}";

let currentPhase = 5;
let canvasSize = 800;
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const mapImg = new Image();
mapImg.src = MAP_B64;

function resizeCanvas() {{
  const area = document.querySelector('.map-area');
  canvasSize = Math.min(area.clientWidth, area.clientHeight) - 20;
  canvasSize = Math.max(400, canvasSize);
  canvas.width = canvasSize;
  canvas.height = canvasSize;
  render();
}}

mapImg.onload = () => {{
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
}};

function toPixel(mapCoord) {{
  return mapCoord / 816000 * canvasSize;
}}

function drawCircle(cx, cy, r, color, lineWidth, dash) {{
  ctx.beginPath();
  ctx.arc(toPixel(cx), toPixel(cy), toPixel(r), 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  if (dash) ctx.setLineDash(dash);
  else ctx.setLineDash([]);
  ctx.stroke();
  ctx.setLineDash([]);
}}

function renderHeatmap(grid, r, g, b, maxAlpha) {{
  if (!grid) return;
  const rows = grid.length;
  const cols = grid[0].length;
  const cellW = canvasSize / cols;
  const cellH = canvasSize / rows;
  for (let y = 0; y < rows; y++) {{
    for (let x = 0; x < cols; x++) {{
      const v = grid[y][x];
      if (v < 5) continue;
      const a = (v / 255) * maxAlpha;
      ctx.fillStyle = `rgba(${{r}},${{g}},${{b}},${{a}})`;
      ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
    }}
  }}
}}

function renderPredGrid(phase) {{
  const pg = DATA.pred_grid[String(phase)];
  if (!pg) return;
  const probs = pg.probs;
  const rows = probs.length;
  const cols = probs[0].length;
  const cellW = canvasSize / cols;
  const cellH = canvasSize / rows;
  for (let y = 0; y < rows; y++) {{
    for (let x = 0; x < cols; x++) {{
      const p = probs[y][x];
      if (p < 0.3) continue;
      const a = Math.min((p - 0.3) / 0.7 * 0.5, 0.5);
      const r = Math.round((1 - p) * 255);
      const g = Math.round(p * 200);
      ctx.fillStyle = `rgba(${{r}},${{g}},60,${{a}})`;
      ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
    }}
  }}
}}

function render() {{
  ctx.clearRect(0, 0, canvasSize, canvasSize);
  ctx.drawImage(mapImg, 0, 0, canvasSize, canvasSize);

  const p = String(currentPhase);

  // Prediction grid
  if (document.getElementById('togglePred').checked) {{
    renderPredGrid(currentPhase);
  }}

  // Position heatmap
  if (document.getElementById('toggleHeat').checked && DATA.heatmaps[p]) {{
    renderHeatmap(DATA.heatmaps[p], 255, 120, 0, 0.6);
  }}

  // Kill heatmap
  if (document.getElementById('toggleKills').checked) {{
    renderHeatmap(DATA.kill_heatmap, 255, 20, 60, 0.5);
  }}

  // Circle scatter
  if (document.getElementById('toggleCircles').checked && DATA.circle_scatter[p]) {{
    const pts = DATA.circle_scatter[p];
    ctx.globalAlpha = 0.15;
    pts.forEach(([cx, cy, r]) => {{
      ctx.beginPath();
      ctx.arc(cx * canvasSize, cy * canvasSize, r * canvasSize, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }});
    ctx.globalAlpha = 1;
  }}

  // Average zones
  if (document.getElementById('toggleAvgZones').checked && DATA.avg_circles[p]) {{
    const ac = DATA.avg_circles[p];
    drawCircle(ac.wx, ac.wy, ac.wr, 'rgba(255,255,255,0.8)', 2.5, null);
    drawCircle(ac.bx, ac.by, ac.br, 'rgba(59,130,246,0.7)', 2, [6, 4]);
  }}

  // Phase number overlay
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(8, 8, 120, 32);
  ctx.fillStyle = '#e6edf3';
  ctx.font = 'bold 16px -apple-system, sans-serif';
  ctx.fillText('Phase ' + currentPhase, 16, 30);
}}

// Phase buttons
function buildUI() {{
  const container = document.getElementById('phaseButtons');
  for (let i = 1; i <= 9; i++) {{
    const stats = DATA.phase_stats[String(i)];
    const btn = document.createElement('button');
    btn.className = 'phase-btn' + (i === currentPhase ? ' active' : '');
    btn.innerHTML = `
      <div class="num">${{i}}</div>
      <div class="info">Phase ${{i}}</div>
      <div class="rate">${{stats ? stats.rate + '%' : '-'}}</div>
    `;
    btn.addEventListener('click', () => {{
      currentPhase = i;
      document.querySelectorAll('.phase-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      updateStats();
      render();
    }});
    container.appendChild(btn);
  }}

  // Model info
  const mi = DATA.model_info;
  document.getElementById('matchCount').textContent = mi.n_matches + ' matches';
  document.getElementById('modelBadge').textContent = mi.model_type + ' ' + (mi.accuracy * 100).toFixed(1) + '%';

  const mc = document.getElementById('modelCard');
  mc.innerHTML = `
    <div class="row"><span class="k">Model</span><span class="v">${{mi.model_type}}</span></div>
    <div class="row"><span class="k">Accuracy</span><span class="v">${{(mi.accuracy * 100).toFixed(1)}}%</span></div>
    <div class="row"><span class="k">CV Score</span><span class="v">${{(mi.cv_mean * 100).toFixed(1)}}% ± ${{(mi.cv_std * 100).toFixed(1)}}</span></div>
    <div class="row"><span class="k">Samples</span><span class="v">${{mi.n_samples.toLocaleString()}}</span></div>
    <div class="row"><span class="k">Survival rate</span><span class="v">${{(mi.class_balance * 100).toFixed(1)}}%</span></div>
  `;

  // Feature importance
  const fl = document.getElementById('featureList');
  if (mi.feature_importance) {{
    const entries = Object.entries(mi.feature_importance).sort((a, b) => b[1] - a[1]);
    entries.forEach(([name, val]) => {{
      const pct = (val * 100).toFixed(1);
      fl.innerHTML += `<div style="margin:3px 0;display:flex;align-items:center;gap:6px">
        <div style="width:${{val*200}}px;height:6px;background:#1f6feb;border-radius:3px;min-width:2px"></div>
        <span style="color:#8b949e;white-space:nowrap">${{name}} (${{pct}}%)</span>
      </div>`;
    }});
  }}

  // Toggle listeners
  ['toggleHeat','toggleKills','toggleCircles','toggleAvgZones','togglePred'].forEach(id => {{
    document.getElementById(id).addEventListener('change', render);
  }});

  updateStats();
}}

function updateStats() {{
  const stats = DATA.phase_stats[String(currentPhase)];
  document.getElementById('statTeams').textContent = stats ? stats.teams : '-';
  document.getElementById('statRate').textContent = stats ? stats.rate + '%' : '-';
}}

// Tooltip on hover
canvas.addEventListener('mousemove', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / canvasSize * 816000;
  const y = (e.clientY - rect.top) / canvasSize * 816000;

  const tt = document.getElementById('tooltip');
  const mapX = (x / 100).toFixed(0);
  const mapY = (y / 100).toFixed(0);

  const p = String(currentPhase);
  let info = `<div class="tt-title">Position ${{mapX}}m, ${{mapY}}m</div>`;

  const ac = DATA.avg_circles[p];
  if (ac) {{
    const dist = Math.sqrt((x - ac.wx) ** 2 + (y - ac.wy) ** 2);
    const inZone = dist < ac.wr;
    info += `<div class="tt-row"><span>To zone center</span><span class="tt-val">${{(dist/100).toFixed(0)}}m</span></div>`;
    info += `<div class="tt-row"><span>In white zone</span><span class="tt-val">${{inZone ? 'Yes' : 'No'}}</span></div>`;
  }}

  // Show predicted survival if available
  const pg = DATA.pred_grid[p];
  if (pg) {{
    const rows = pg.probs.length;
    const gx = Math.min(Math.floor(x / 816000 * rows), rows - 1);
    const gy = Math.min(Math.floor(y / 816000 * rows), rows - 1);
    const prob = pg.probs[gy][gx];
    info += `<div class="tt-row"><span>Survival prob</span><span class="tt-val">${{(prob*100).toFixed(1)}}%</span></div>`;
  }}

  tt.innerHTML = info;
  tt.style.display = 'block';
  tt.style.left = (e.clientX + 16) + 'px';
  tt.style.top = (e.clientY - 10) + 'px';
}});

canvas.addEventListener('mouseleave', () => {{
  document.getElementById('tooltip').style.display = 'none';
}});

buildUI();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
