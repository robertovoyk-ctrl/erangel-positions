"""Generate static heatmap images per phase."""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROCESSED = Path("data/processed")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

MAP_SIZE = 816000
IMG_SIZE = 1024


def coord_to_px(x, y):
    px = int(x / MAP_SIZE * IMG_SIZE)
    py = int(y / MAP_SIZE * IMG_SIZE)
    return max(0, min(IMG_SIZE - 1, px)), max(0, min(IMG_SIZE - 1, py))


def make_heatmap(points_x, points_y, size=IMG_SIZE, sigma=8):
    """Create a gaussian-smoothed heatmap from points."""
    from scipy.ndimage import gaussian_filter
    heatmap = np.zeros((size, size), dtype=np.float64)
    for x, y in zip(points_x, points_y):
        px, py = coord_to_px(x, y)
        heatmap[py, px] += 1
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def heatmap_to_rgba(heatmap, colormap="hot"):
    """Convert heatmap values to RGBA with transparency."""
    h, w = heatmap.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            v = heatmap[y, x]
            if v < 0.01:
                continue
            alpha = min(int(v * 200), 200)
            if v < 0.33:
                r, g, b = int(v * 3 * 255), 0, int(v * 3 * 128)
            elif v < 0.66:
                t = (v - 0.33) * 3
                r, g, b = 255, int(t * 200), 0
            else:
                t = (v - 0.66) * 3
                r, g, b = 255, 200 + int(t * 55), int(t * 100)
            rgba[y, x] = [r, g, b, alpha]
    return rgba


def draw_circle(draw, cx, cy, r, color, width=2):
    """Draw a circle on the image."""
    px, py = coord_to_px(cx, cy)
    pr = int(r / MAP_SIZE * IMG_SIZE)
    draw.ellipse([px - pr, py - pr, px + pr, py + pr], outline=color, width=width)


def main():
    features_file = PROCESSED / "features.json"
    circles_file = PROCESSED / "circles.json"
    kills_file = PROCESSED / "kills.json"

    if not features_file.exists():
        print("No processed data. Run process_data.py first.")
        return

    with open(features_file) as f:
        features = json.load(f)
    with open(circles_file) as f:
        circles = json.load(f)
    with open(kills_file) as f:
        kills = json.load(f)

    # Try to load Erangel map as background
    map_img = None
    map_path = Path("assets/erangel.png")
    if map_path.exists():
        map_img = Image.open(map_path).resize((IMG_SIZE, IMG_SIZE)).convert("RGBA")

    for phase in range(2, 9):
        print(f"\nPhase {phase}:")

        # Surviving team positions
        phase_feat = [f for f in features if f["phase"] == phase and f["survived"] == 1]
        if not phase_feat:
            print(f"  no data for phase {phase}")
            continue

        xs = [f["centroid_x"] for f in phase_feat]
        ys = [f["centroid_y"] for f in phase_feat]
        print(f"  {len(phase_feat)} surviving team positions")

        # Position heatmap
        heatmap = make_heatmap(xs, ys, IMG_SIZE, sigma=12)
        heat_rgba = heatmap_to_rgba(heatmap)
        heat_img = Image.fromarray(heat_rgba, "RGBA")

        if map_img:
            base = map_img.copy()
        else:
            base = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (20, 25, 30, 255))

        base = Image.alpha_composite(base, heat_img)

        # Draw average circles for this phase
        phase_circles = [c for c in circles if c.get("phase", 0) == phase]
        if phase_circles:
            avg_wx = np.mean([c["white_x"] for c in phase_circles])
            avg_wy = np.mean([c["white_y"] for c in phase_circles])
            avg_wr = np.mean([c["white_r"] for c in phase_circles])
            avg_bx = np.mean([c["blue_x"] for c in phase_circles])
            avg_by = np.mean([c["blue_y"] for c in phase_circles])
            avg_br = np.mean([c["blue_r"] for c in phase_circles])

            draw = ImageDraw.Draw(base)
            draw_circle(draw, avg_wx, avg_wy, avg_wr, (255, 255, 255, 180), 3)
            draw_circle(draw, avg_bx, avg_by, avg_br, (60, 120, 255, 150), 2)

        # Add title
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, IMG_SIZE, 40], fill=(0, 0, 0, 180))
        draw.text((10, 8), f"Phase {phase} - Surviving Team Positions ({len(phase_feat)} teams)",
                  fill=(255, 255, 255))

        base.save(OUTPUT / f"heatmap_phase_{phase}.png")
        print(f"  saved heatmap_phase_{phase}.png")

    # Kill heatmap (all phases)
    print("\nKill heatmap (all phases):")
    kx = [k["x"] for k in kills if k["x"] != 0]
    ky = [k["y"] for k in kills if k["y"] != 0]
    print(f"  {len(kx)} kill events")

    if kx:
        heatmap = make_heatmap(kx, ky, IMG_SIZE, sigma=10)
        heat_rgba = heatmap_to_rgba(heatmap)
        heat_img = Image.fromarray(heat_rgba, "RGBA")

        if map_img:
            base = map_img.copy()
        else:
            base = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (20, 25, 30, 255))

        base = Image.alpha_composite(base, heat_img)
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, IMG_SIZE, 40], fill=(0, 0, 0, 180))
        draw.text((10, 8), f"Kill Heatmap - All Phases ({len(kx)} kills)", fill=(255, 255, 255))
        base.save(OUTPUT / f"heatmap_kills.png")
        print("  saved heatmap_kills.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
