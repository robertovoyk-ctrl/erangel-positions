"""Run the full pipeline: process data, train model, generate heatmaps, build viewer."""

import subprocess
import sys
from pathlib import Path


def run(script):
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script], cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print(f"\nERROR: {script} failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    extracted = Path("data/extracted")
    if not list(extracted.glob("*.json")):
        print("No extracted data found. Run collect_data.py first.")
        sys.exit(1)

    n = len(list(extracted.glob("*.json")))
    print(f"Found {n} extracted match files")

    run("process_data.py")
    run("train_model.py")
    run("generate_heatmaps.py")
    run("build_viewer.py")

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    for f in sorted(Path("output").glob("*")):
        print(f"  {f} ({f.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
