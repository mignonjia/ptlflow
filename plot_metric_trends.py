"""Plot 9 metric trends in a 3x3 grid from step_average_summaries.csv.

Usage:
    python plot_metric_trends.py \
        --csv /path/to/step_average_summaries.csv \
        --output /path/to/metric_trends_3x3.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "mf_epe_mean",
    "mf_angle_err_mean",
    "mf_cosine_mean",
    "mf_mag_ratio_mean",
    "pixel_epe_mean_mean",
    "px_angle_rmse_mean",
    "fl_all_mean",
    "foe_dist_mean",
    "flow_kl_2d_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3x3 metric trends from step_average_summaries.csv",
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to step_average_summaries.csv")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    return parser.parse_args()


def _to_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    rows = sorted(rows, key=lambda r: int(float(r["train_step"])))
    steps = [int(float(r["train_step"])) for r in rows]

    # Convert 0,1000,2000,... into 0,1,2,... on x-axis if possible.
    if steps and all(step % 1000 == 0 for step in steps):
        x_vals = [step // 1000 for step in steps]
        x_label = "Training Step Index (x1000)"
    else:
        x_vals = list(range(len(steps)))
        x_label = "Training Step Index"

    x_ticks = list(range(min(x_vals), max(x_vals) + 1))

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    flat_axes = axes.flatten()

    for ax, metric in zip(flat_axes, METRICS):
        y_vals = []
        for r in rows:
            v = _to_float(r.get(metric))
            y_vals.append(np.nan if v is None else v)

        ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, markersize=4)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_ticks)
        ax.set_xlabel(x_label)
        ax.tick_params(axis="x", labelbottom=True)

        # Mark best point in red.
        if metric == "mf_cosine_mean":
            best_idx = int(np.nanargmax(y_vals))
        elif metric == "mf_mag_ratio_mean":
            best_idx = int(np.nanargmin(np.abs(np.asarray(y_vals) - 1.0)))
        else:
            best_idx = int(np.nanargmin(y_vals))
        ax.scatter([x_vals[best_idx]], [y_vals[best_idx]], color="red", s=30, zorder=3)

    fig.suptitle("Metric Trends Across Training Steps", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
