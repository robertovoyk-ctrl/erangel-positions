# Erangel Positioning Map

Data-driven positioning guide for PUBG Erangel, built from pro tournament match telemetry.

## What it does

Analyzes competitive PUBG tournament matches on Erangel to identify optimal positioning strategies across zone phases. Includes:

- **Position heatmaps** — where surviving teams position in each phase (2–8)
- **Kill heatmaps** — where fights happen most
- **Circle scatter** — distribution of zone locations across matches
- **Survival prediction model** — GradientBoosting classifier predicting team survival based on positioning features
- **Interactive viewer** — HTML map with toggleable layers, phase navigation, and tooltip stats

## Pipeline

```
collect_data.py  →  process_data.py  →  train_model.py  →  generate_heatmaps.py  →  build_viewer.py
```

1. **collect_data.py** — Fetches tournament data from the PUBG API, filters Erangel matches, downloads telemetry from the CDN, and extracts circles, positions, kills, and placements into compact JSON files.

2. **process_data.py** — Builds a terrain heightmap from z-samples, assigns zone phases from blue zone transitions, and computes per-team features (distance to zone center, elevation, slope, team spread, etc.).

3. **train_model.py** — Trains GradientBoosting and Logistic Regression classifiers on phase survival, picks the best, and generates a prediction grid for the viewer.

4. **generate_heatmaps.py** — Renders static heatmap PNGs per phase with the Erangel map as background.

5. **build_viewer.py** — Builds a self-contained HTML viewer with embedded map and data.

## Key features used by the model

| Feature | Description |
|---------|-------------|
| `zone_edge_ratio` | Distance to white zone center / zone radius |
| `team_spread` | How spread out the team is |
| `zone_edge_dist` | Signed distance to zone edge (positive = inside) |
| `elevation` | Terrain height at team centroid |
| `slope` | Terrain steepness |
| `dist_to_white_center` | Absolute distance to safe zone center |
| `team_alive` | Number of living team members |

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` with your PUBG API key:

```
PUBG_API_KEY=your_key_here
```

## Run

```bash
python collect_data.py   # fetches data (rate-limited, takes ~20 min)
python run_pipeline.py   # runs processing → training → heatmaps → viewer
```

Or run each step individually.

## Output

- `output/viewer.html` — interactive map viewer (open in browser)
- `output/heatmap_phase_*.png` — static heatmaps per phase
- `output/heatmap_kills.png` — kill density heatmap
- `model/model_info.json` — model accuracy and feature importance

## Data

Uses the [PUBG API](https://developer.pubg.com/) tournament endpoints. Telemetry is downloaded from the PUBG CDN (no rate limit). All raw and processed data is cached in `data/` (gitignored).

Map image from [pubg/api-assets](https://github.com/pubg/api-assets).

## Requirements

- Python 3.8+
- PUBG API developer key
- ~500 MB disk space for telemetry cache
