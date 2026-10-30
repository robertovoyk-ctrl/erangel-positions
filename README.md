# Erangel Positioning Map

**[Open the live map](https://1artemenko1.github.io/erangel-positions/)** — pick a phase, click to drop a circle, see where to rotate.

Data-driven positioning guide for PUBG Erangel, built from 199 pro tournament matches.

## How to use

1. Open the live map link above.
2. Choose a zone phase (2–8) in the sidebar.
3. Click anywhere on the map to place a zone circle.
4. The advisor shows the top 5 positions inside that circle, ranked by predicted survival probability. A green entry route suggests how to rotate in from the previous phase.

Toggle layers in the sidebar: position heatmaps, kill zones, circle scatter, average zone outlines.

## Data source

All data comes from the [PUBG API](https://developer.pubg.com/) tournament endpoints. Telemetry for each match is downloaded from the PUBG CDN. Only Erangel (`Baltic_Main`) matches are used. 199 matches produced usable features across phases 2–8.

## Method

For each team in each phase, we extract 13 positioning features: distance to the blue zone edge, depth inside the safe zone, terrain elevation and slope, team spread, team alive count, and zone radii.

A GradientBoosting classifier predicts whether a team survives to the next phase. Training uses a match-based 80/20 split (no match appears in both train and test). The model is scored with logistic regression coefficients so the viewer can run predictions in-browser without a server.

**Top features by importance:**

| Feature | Importance |
|---------|-----------|
| `zone_edge_dist` (depth inside blue zone) | 0.43 |
| `team_spread` (how tight the team plays) | 0.25 |
| `team_alive` (living teammates) | 0.09 |
| `blue_r` (current zone radius) | 0.05 |
| `zone_edge_ratio` (relative zone position) | 0.05 |

## Honest limits

- **87.5% accuracy vs 81.1% always-survive baseline** — a +6.4pp lift. The model helps most in phases 5–8 where survival is no longer the overwhelming majority class. In phases 2–3, almost everyone survives and the model adds little.
- **ROC AUC 0.88** — the model separates survivors from eliminated teams reasonably well, but positioning is only one factor. Aim, loot, and team coordination matter and are not captured here.
- **199 matches** is a moderate dataset. Circle positions and meta shift between patches, so results reflect the tournament meta at data collection time.
- Kill density and circle scatter are computed from training matches only to avoid leaking test data into the viewer.

## Pipeline

```
collect_data.py → process_data.py → train_model.py → generate_heatmaps.py → build_viewer.py
```

## Run locally

```bash
pip install -r requirements.txt
```

Create `.env` with your PUBG API key:

```
PUBG_API_KEY=your_key_here
```

```bash
python collect_data.py   # fetch matches (~20 min, rate-limited)
python run_pipeline.py   # process → train → heatmaps → viewer
open output/viewer.html
```

## License

Map image from [pubg/api-assets](https://github.com/pubg/api-assets).
