"""Evaluate action-conditioned video generation by comparing optical flow divergence.

Compares optical flow between ground truth and generated videos to measure how
well a world model follows action conditioning. Produces per-frame metrics,
time-series plots, flow visualizations, and summary scores.

Supports two modes:
    1. GT mode (default): Compare flows from GT video vs generated video.
    2. Synthetic mode (--synthetic): Compare synthetic flow (from actions) vs
       generated video flow. No GT video needed.

Metrics:
    - EPE (End-Point Error): L2 distance between flow vectors, per-pixel and global mean.
    - MF Angular Error: Angle between mean GT and Gen flow vectors (degrees). 0=perfect, 180=opposite.
    - MF Cosine Similarity: Cosine of mean flow vectors. 1=same direction, -1=opposite.
    - MF Magnitude Ratio: Gen speed / GT speed. 1=same speed, <1=too slow, >1=too fast.
    - Per-pixel Angular Error: Average angular error across all pixels (degrees).
    - Fl-all: Fraction of pixels where flow is wrong (EPE > 3px AND > 5% of GT magnitude).

Usage:
    # Single pair (GT mode)
    python eval_flow_divergence.py \
        --gt_video /mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/to_shao/1_wasd_only/05.mp4 \
        --gen_video /mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/to_shao/1_wasd_only/05_wangame.mp4 \
        --output_dir outputs/flow_eval/1_wasd_only_05

    # Batch mode: entire scenario directory
    python eval_flow_divergence.py \
        --scenario_dir /mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/to_shao/camera \
        --output_dir outputs/flow_eval/camera

    # Synthetic mode (no GT video needed)
    python eval_flow_divergence.py \
        --synthetic \
        --gen_video /mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/to_shao/camera/01_wangame.mp4 \
        --action_file /mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/to_shao/camera4hold_alpha1/01_action.npy \
        --calibration calibration.json \
        --output_dir outputs/flow_eval_synth/camera_02
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import ptlflow
from ptlflow.utils.flow_utils import flow_to_rgb
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.utils import tensor_dict_to_numpy
from scipy.integrate import trapezoid

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str) -> List[np.ndarray]:
    """Read all frames from an MP4 file as a list of HWC uint8 BGR arrays."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Optical flow computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_flow_sequence(
    model: torch.nn.Module,
    frames: List[np.ndarray],
    io_adapter: IOAdapter,
) -> List[np.ndarray]:
    """Compute optical flow for each consecutive frame pair.

    Returns a list of (T-1) flow arrays, each with shape HxWx2 (u, v).
    """
    flows = []
    for i in tqdm(range(len(frames) - 1), desc="Computing flow", leave=False):
        inputs = io_adapter.prepare_inputs([frames[i], frames[i + 1]])
        preds = model(inputs)
        preds["images"] = inputs["images"]
        preds = io_adapter.unscale(preds)
        preds_npy = tensor_dict_to_numpy(preds)
        flows.append(preds_npy["flows"])  # HxWx2
    return flows


# ---------------------------------------------------------------------------
# Flow warping
# ---------------------------------------------------------------------------

def flow_warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp an image using an optical flow field.

    Given flow F at pixel (x, y), the source pixel is at (x + F_u, y + F_v).
    Uses bilinear interpolation.

    Parameters
    ----------
    img : np.ndarray, shape HxWxC, uint8
    flow : np.ndarray, shape HxWx2 (u=horizontal, v=vertical)

    Returns
    -------
    np.ndarray, shape HxWxC, uint8 — the warped image.
    """
    h, w = flow.shape[:2]
    # Build remap coordinates: for each pixel (x,y), sample from (x+u, y+v)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
    warped = cv.remap(img, map_x, map_y, cv.INTER_LINEAR,
                      borderMode=cv.BORDER_REPLICATE)
    return warped


def compute_warp_error(frame_t: np.ndarray, frame_t1: np.ndarray,
                       flow: np.ndarray) -> float:
    """Compute photometric warp error: warp frame_t using flow, compare to frame_t1.

    Returns the mean absolute error (L1) in [0, 255] scale, averaged over pixels and channels.
    """
    warped = flow_warp(frame_t, flow)
    # Convert to float for accurate difference
    diff = np.abs(warped.astype(np.float32) - frame_t1.astype(np.float32))
    return float(diff.mean())


# ---------------------------------------------------------------------------
# Focus of Expansion (FOE) estimation
# ---------------------------------------------------------------------------

def estimate_foe(flow: np.ndarray, step: int = 8,
                 min_mag: float = 0.5) -> Tuple[float, float, float]:
    """Estimate the Focus of Expansion from an optical flow field.

    Each flow vector at pixel (x, y) with direction (u, v) defines a line
    through (x, y). The FOE is the point where all these lines converge.
    We solve this as a least-squares problem:

        For each pixel: v * fx - u * fy = v * x - u * y

    This gives A @ [fx, fy]^T = b, solved via least squares.

    Parameters
    ----------
    flow : np.ndarray, shape HxWx2
    step : int — subsample spacing (use every step-th pixel for speed)
    min_mag : float — skip pixels with flow magnitude below this

    Returns
    -------
    (fx, fy, residual) — FOE position in pixel coords and mean residual.
        Residual indicates fit quality (low = good radial pattern).
    """
    H, W = flow.shape[:2]
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.ravel()
    xx = xx.ravel()
    uu = flow[yy, xx, 0]  # horizontal flow
    vv = flow[yy, xx, 1]  # vertical flow

    mag = np.sqrt(uu ** 2 + vv ** 2)
    valid = mag > min_mag
    if valid.sum() < 10:
        # Not enough flow to estimate FOE
        return W / 2.0, H / 2.0, float("inf")

    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    uu = uu[valid].astype(np.float64)
    vv = vv[valid].astype(np.float64)

    # v * fx - u * fy = v * x - u * y
    A = np.column_stack([vv, -uu])
    b = vv * xx - uu * yy

    # Solve with least squares
    result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    fx, fy = result

    # Compute mean residual (how well the flow fits a radial pattern)
    fit_err = np.abs(A @ result - b)
    mean_residual = float(fit_err.mean())

    return float(fx), float(fy), mean_residual


# ---------------------------------------------------------------------------
# Flow distribution comparison
# ---------------------------------------------------------------------------

def compute_flow_kl_2d(flow_a: np.ndarray, flow_b: np.ndarray,
                       n_angle_bins: int = 36, n_mag_bins: int = 20,
                       min_mag: float = 0.5) -> float:
    """Compute KL divergence between 2D (angle, magnitude) histograms of two flow fields.

    Builds a joint histogram of flow angle (circular, 0-360°) and flow magnitude
    (log-spaced) for each flow field, then computes KL(P_a || P_b) with Laplace smoothing.

    Parameters
    ----------
    flow_a, flow_b : np.ndarray, shape HxWx2
    n_angle_bins : number of angular bins (10° each by default)
    n_mag_bins : number of magnitude bins (log-spaced)
    min_mag : minimum magnitude to include (skip near-zero flow)

    Returns
    -------
    float — KL divergence in nats. 0 = identical distributions.
    """
    def _build_hist(flow):
        u, v = flow[:, :, 0].ravel(), flow[:, :, 1].ravel()
        mag = np.sqrt(u ** 2 + v ** 2)
        angle = np.degrees(np.arctan2(v, u)) % 360  # [0, 360)

        valid = mag >= min_mag
        if valid.sum() < 10:
            return None
        mag = mag[valid]
        angle = angle[valid]

        # Log-spaced magnitude bins from min_mag to max observed
        mag_max = max(mag.max(), min_mag + 1.0)
        mag_edges = np.logspace(np.log10(min_mag), np.log10(mag_max), n_mag_bins + 1)
        angle_edges = np.linspace(0, 360, n_angle_bins + 1)

        hist, _, _ = np.histogram2d(angle, mag, bins=[angle_edges, mag_edges])
        return hist

    hist_a = _build_hist(flow_a)
    hist_b = _build_hist(flow_b)

    if hist_a is None or hist_b is None:
        return 0.0

    # Laplace smoothing and normalize to probability distributions
    eps = 1.0  # Laplace smoothing count
    p = (hist_a + eps) / (hist_a + eps).sum()
    q = (hist_b + eps) / (hist_b + eps).sum()

    # KL(P || Q) = sum(P * log(P / Q))
    kl = float((p * np.log(p / q)).sum())
    return kl


# ---------------------------------------------------------------------------
# Per-frame metric computation
# ---------------------------------------------------------------------------

def compute_frame_metrics(
    flow_gt: np.ndarray,
    flow_gen: np.ndarray,
    grid_size: int = 8,
    min_mag: float = 0.5,
    max_mag_pct: float = 80.0,
) -> Dict[str, float]:
    """Compute all metrics comparing two flow fields for a single frame pair.

    Parameters
    ----------
    flow_gt, flow_gen : np.ndarray, shape HxWx2
    grid_size : int
    min_mag : float
        Minimum flow magnitude to include (filters near-zero vectors).
    max_mag_pct : float
        Percentile above which to exclude (filters extreme outliers).
        E.g. 80.0 means drop the top 20% by magnitude.

    Returns
    -------
    dict with metric names → scalar values
    """
    metrics = {}

    # --- Per-pixel magnitude maps ---
    gt_mag_map = np.linalg.norm(flow_gt, axis=2)   # HxW
    gen_mag_map = np.linalg.norm(flow_gen, axis=2)  # HxW

    # --- Build magnitude filter mask ---
    # Use the max magnitude at each pixel across both fields
    max_mag_map = np.maximum(gt_mag_map, gen_mag_map)
    mag_hi = np.percentile(max_mag_map, max_mag_pct)
    mag_mask = (max_mag_map >= min_mag) & (max_mag_map <= mag_hi)
    n_valid = mag_mask.sum()
    metrics["mag_filter_kept"] = float(n_valid / max(mag_mask.size, 1))

    # --- Mean flow vector metrics (computed on filtered pixels) ---
    if n_valid > 0:
        gt_filtered = flow_gt[mag_mask]   # (N, 2)
        gen_filtered = flow_gen[mag_mask]  # (N, 2)
        mean_gt = gt_filtered.mean(axis=0)
        mean_gen = gen_filtered.mean(axis=0)
    else:
        mean_gt = flow_gt.reshape(-1, 2).mean(axis=0)
        mean_gen = flow_gen.reshape(-1, 2).mean(axis=0)

    # Mean flow EPE (global motion vector error)
    metrics["mf_epe"] = float(np.linalg.norm(mean_gt - mean_gen))

    # Mean flow angular error & cosine similarity
    # Tiny mean-flow vectors are directionally unstable. Treat one-small/one-large
    # as orthogonal mismatch rather than letting a near-zero vector produce a
    # spuriously good angle by chance.
    mf_min_mag = 0.1
    mag_gt = np.linalg.norm(mean_gt)
    mag_gen = np.linalg.norm(mean_gen)
    if mag_gt < mf_min_mag and mag_gen < mf_min_mag:
        metrics["mf_angle_err"] = 0.0
        metrics["mf_cosine"] = 1.0
    elif mag_gt < mf_min_mag or mag_gen < mf_min_mag:
        metrics["mf_angle_err"] = 90.0
        metrics["mf_cosine"] = 0.0
    elif mag_gt > 1e-6 and mag_gen > 1e-6:
        cos_sim = np.dot(mean_gt, mean_gen) / (mag_gt * mag_gen)
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        metrics["mf_angle_err"] = float(np.degrees(np.arccos(cos_sim)))
        metrics["mf_cosine"] = cos_sim
    else:
        metrics["mf_angle_err"] = 0.0
        metrics["mf_cosine"] = 1.0  # both near-zero = agree on "no motion"

    # Mean flow magnitude ratio: gen_speed / gt_speed (1.0 = same speed)
    metrics["mf_mag_ratio"] = float(mag_gen / mag_gt) if mag_gt > 1e-6 else 1.0

    # --- Per-pixel EPE (filtered) ---
    epe_map = np.linalg.norm(flow_gt - flow_gen, axis=2)  # HxW
    if n_valid > 0:
        metrics["pixel_epe_mean"] = float(epe_map[mag_mask].mean())
        metrics["pixel_epe_max"] = float(epe_map[mag_mask].max())
    else:
        metrics["pixel_epe_mean"] = float(epe_map.mean())
        metrics["pixel_epe_max"] = float(epe_map.max())

    # --- Per-pixel angular error (filtered) ---
    valid = mag_mask & (gt_mag_map > 0.5) & (gen_mag_map > 0.5)
    if valid.sum() > 0:
        dot = (flow_gt[:, :, 0] * flow_gen[:, :, 0] +
               flow_gt[:, :, 1] * flow_gen[:, :, 1])
        cos_map = np.clip(dot / (gt_mag_map * gen_mag_map + 1e-8), -1.0, 1.0)
        angle_map = np.degrees(np.arccos(cos_map))
        metrics["px_angle_rmse"] = float(np.sqrt((angle_map[valid] ** 2).mean()))
    else:
        metrics["px_angle_rmse"] = 0.0

    # --- Grid-based EPE (filtered) ---
    H, W = epe_map.shape
    gh = H // grid_size
    gw = W // grid_size
    grid_epe_values = []
    for gi in range(grid_size):
        for gj in range(grid_size):
            cell_mask = mag_mask[gi * gh : (gi + 1) * gh, gj * gw : (gj + 1) * gw]
            cell_epe = epe_map[gi * gh : (gi + 1) * gh, gj * gw : (gj + 1) * gw]
            if cell_mask.sum() > 0:
                grid_epe_values.append(float(cell_epe[cell_mask].mean()))
            else:
                grid_epe_values.append(float(cell_epe.mean()))
    metrics["grid_epe_mean"] = float(np.mean(grid_epe_values))
    metrics["grid_epe_max"] = float(np.max(grid_epe_values))

    # --- Fl-all (filtered) ---
    if n_valid > 0:
        outlier = (epe_map > 3.0) & (epe_map > 0.05 * gt_mag_map) & mag_mask
        metrics["fl_all"] = float(outlier.sum() / n_valid)
    else:
        outlier = (epe_map > 3.0) & (epe_map > 0.05 * gt_mag_map)
        metrics["fl_all"] = float(outlier.mean())

    # --- Focus of Expansion (FOE) ---
    foe_gt_x, foe_gt_y, foe_gt_res = estimate_foe(flow_gt)
    foe_gen_x, foe_gen_y, foe_gen_res = estimate_foe(flow_gen)
    metrics["foe_gt_x"] = foe_gt_x
    metrics["foe_gt_y"] = foe_gt_y
    metrics["foe_gen_x"] = foe_gen_x
    metrics["foe_gen_y"] = foe_gen_y
    # FOE distance: how far apart the estimated heading directions are (pixels)
    metrics["foe_dist"] = float(np.sqrt((foe_gt_x - foe_gen_x) ** 2 +
                                         (foe_gt_y - foe_gen_y) ** 2))
    # FOE fit residuals: how well the flow fits a radial pattern (lower = more translational)
    metrics["foe_gt_residual"] = foe_gt_res
    metrics["foe_gen_residual"] = foe_gen_res

    # --- 2D flow distribution KL divergence ---
    # Joint (angle, magnitude) histogram comparison
    metrics["flow_kl_2d"] = compute_flow_kl_2d(flow_gt, flow_gen)

    return metrics


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------

def compute_temporal_metrics(
    frame_metrics_list: List[Dict[str, float]],
) -> Dict:
    """Aggregate per-frame metrics into temporal summary statistics."""
    n_frames = len(frame_metrics_list)
    if n_frames == 0:
        return {}

    metric_names = list(frame_metrics_list[0].keys())
    series = {name: np.array([m[name] for m in frame_metrics_list]) for name in metric_names}

    summary = {"n_frames": n_frames}

    for name, vals in series.items():
        summary[f"{name}_mean"] = float(vals.mean())
        summary[f"{name}_std"] = float(vals.std())
        summary[f"{name}_max"] = float(vals.max())
        summary[f"{name}_auc"] = float(trapezoid(vals) / max(n_frames - 1, 1))

    # Divergence onset: first frame where 5-frame moving average of pixel_epe_mean
    # exceeds 2x the median of the first 5 frames
    epe_series = series["pixel_epe_mean"]
    window = min(5, n_frames)
    if n_frames >= window:
        baseline = float(np.median(epe_series[:window]))
        threshold = max(baseline * 2.0, 1.0)
        kernel = np.ones(window) / window
        smoothed = np.convolve(epe_series, kernel, mode="valid")
        divergence_frame = None
        for i, val in enumerate(smoothed):
            if val > threshold:
                divergence_frame = i
                break
        summary["divergence_onset_frame"] = divergence_frame
        summary["divergence_threshold"] = threshold
    else:
        summary["divergence_onset_frame"] = None
        summary["divergence_threshold"] = None

    return summary


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_visualizations(
    frames_gt: List[np.ndarray],
    frames_gen: List[np.ndarray],
    flows_gt: List[np.ndarray],
    flows_gen: List[np.ndarray],
    frame_metrics_list: List[Dict[str, float]],
    summary: Dict,
    output_dir: Path,
    fps: float = 10.0,
) -> None:
    """Generate all output visualizations including comparison video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_flows = len(flows_gt)

    # --- Static time-series plot ---
    metric_names = list(frame_metrics_list[0].keys())
    groups = {
        "EPE (px)": ["mf_epe", "pixel_epe_mean"],
        "Direction": ["mf_cosine", "mf_angle_err"],
        "Speed": ["mf_mag_ratio"],
        "Local Err": ["px_angle_rmse", "fl_all"],
        "FOE (px)": ["foe_dist"],
        "KL Div": ["flow_kl_2d"],
    }

    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 12), sharex=True)
    frame_indices = np.arange(n_flows)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    div_frame = summary.get("divergence_onset_frame")

    for ax, (group_name, names) in zip(axes, groups.items()):
        all_handles = []
        all_labels = []
        if len(names) == 2:
            ax_right = ax.twinx()
            axes_list = [ax, ax_right]
        else:
            axes_list = [ax] * len(names)
        for ci, (name, target_ax) in enumerate(zip(names, axes_list)):
            if name in metric_names:
                vals = [m[name] for m in frame_metrics_list]
                line, = target_ax.plot(frame_indices, vals, label=name, linewidth=1.5,
                                       color=colors[ci % len(colors)])
                all_handles.append(line)
                all_labels.append(name)
        ax.set_ylabel(group_name)
        ax.legend(all_handles, all_labels, loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        if div_frame is not None:
            ax.axvline(x=div_frame, color="red", linestyle="--", alpha=0.7)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Flow Divergence Metrics Over Time", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "timeseries.png", dpi=150)
    plt.close(fig)

    # --- Comparison video ---
    print("  Rendering comparison video...")
    _generate_comparison_video(
        frames_gt, frames_gen, flows_gt, flows_gen,
        frame_metrics_list, summary, output_dir, fps,
    )


def _render_timeseries_base(
    frame_metrics_list: List[Dict[str, float]],
    summary: Dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Pre-render the time series plots as a BGR image (no cursor yet)."""
    n_flows = len(frame_metrics_list)
    frame_indices = np.arange(n_flows)
    metric_names = list(frame_metrics_list[0].keys())

    groups = [
        ("EPE (px)", ["mf_epe", "pixel_epe_mean"]),
        ("Direction", ["mf_cosine", "mf_angle_err"]),
        ("Speed", ["mf_mag_ratio"]),
        ("Local Err", ["px_angle_rmse", "fl_all"]),
        ("FOE (px)", ["foe_dist"]),
        ("KL Div", ["flow_kl_2d"]),
    ]

    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, axes = plt.subplots(len(groups), 1, figsize=(fig_w, fig_h), sharex=True)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    div_frame = summary.get("divergence_onset_frame")
    for ax, (group_name, names) in zip(axes, groups):
        all_handles = []
        all_labels = []
        # Use twin axes if metrics have very different scales
        if len(names) == 2:
            ax_right = ax.twinx()
            ax_right.tick_params(labelsize=6)
            axes_list = [ax, ax_right]
        else:
            axes_list = [ax] * len(names)
        for ci, (name, target_ax) in enumerate(zip(names, axes_list)):
            if name in metric_names:
                vals = [m[name] for m in frame_metrics_list]
                line, = target_ax.plot(frame_indices, vals, label=name, linewidth=1.2,
                                       color=colors[ci % len(colors)])
                all_handles.append(line)
                all_labels.append(name)
        ax.set_ylabel(group_name, fontsize=7)
        ax.legend(all_handles, all_labels, loc="upper left", fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        if div_frame is not None:
            ax.axvline(x=div_frame, color="red", linestyle="--", alpha=0.5, linewidth=1)

    axes[-1].set_xlabel("Frame", fontsize=7)
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    buf = buf[:, :, :3].copy()
    plt.close(fig)

    img = cv.resize(buf, (width, height), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img


def _draw_flow_arrows(img: np.ndarray, flow: np.ndarray, step: int = 16,
                      scale: float = 3.0, min_length: float = 0.5) -> np.ndarray:
    """Draw flow vectors as arrows with black outline on an image (in-place).

    Parameters
    ----------
    img : BGR uint8 image to draw on (modified in-place and returned)
    flow : HxWx2 optical flow (must match img dimensions)
    step : spacing between arrows in pixels
    scale : multiplier on arrow length for visibility
    min_length : minimum flow magnitude to draw (skip tiny vectors)
    """
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            mag = (fx ** 2 + fy ** 2) ** 0.5
            if mag < min_length:
                continue
            end_x = int(x + fx * scale)
            end_y = int(y + fy * scale)
            # Black outline
            cv.arrowedLine(img, (x, y), (end_x, end_y), (0, 0, 0),
                           3, cv.LINE_AA, tipLength=0.3)
            # White fill
            cv.arrowedLine(img, (x, y), (end_x, end_y), (255, 255, 255),
                           1, cv.LINE_AA, tipLength=0.3)
    return img


def _draw_foe_crosshair(img: np.ndarray, fx: float, fy: float,
                        color: Tuple[int, int, int], label: str = "",
                        size: int = 15, thickness: int = 2) -> None:
    """Draw a crosshair at the FOE location on an image (in-place)."""
    h, w = img.shape[:2]
    ix, iy = int(round(fx)), int(round(fy))
    # Clamp to image bounds for drawing, but still draw if close to edge
    if -size <= ix <= w + size and -size <= iy <= h + size:
        cv.line(img, (ix - size, iy), (ix + size, iy), color, thickness, cv.LINE_AA)
        cv.line(img, (ix, iy - size), (ix, iy + size), color, thickness, cv.LINE_AA)
        cv.circle(img, (ix, iy), size // 2, color, 1, cv.LINE_AA)
        if label:
            cv.putText(img, label, (ix + size + 2, iy - 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv.LINE_AA)


def _make_epe_heatmap_bgr(flow_gt: np.ndarray, flow_gen: np.ndarray, max_epe: float) -> np.ndarray:
    """Create an EPE heatmap as a BGR uint8 image, same size as flow."""
    epe_map = np.linalg.norm(flow_gt - flow_gen, axis=2)
    normalized = np.clip(epe_map / max(max_epe, 1e-3) * 255, 0, 255).astype(np.uint8)
    heatmap = cv.applyColorMap(normalized, cv.COLORMAP_HOT)
    return heatmap


def _make_warp_diff_bgr(frame_t: np.ndarray, frame_t1: np.ndarray,
                         flow: np.ndarray) -> np.ndarray:
    """Create a warp error visualization as a BGR heatmap."""
    warped = flow_warp(frame_t, flow)
    diff = np.abs(warped.astype(np.float32) - frame_t1.astype(np.float32))
    # Average across channels, scale to 0-255
    diff_gray = diff.mean(axis=2)
    max_val = max(diff_gray.max(), 1.0)
    normalized = np.clip(diff_gray / max_val * 255, 0, 255).astype(np.uint8)
    heatmap = cv.applyColorMap(normalized, cv.COLORMAP_INFERNO)
    return heatmap


def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a text label bar on top of an image."""
    h, w = img.shape[:2]
    bar_h = 22
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    cv.putText(bar, text, (4, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), 1, cv.LINE_AA)
    return np.vstack([bar, img])


def _put_metrics_text(img: np.ndarray, metrics: Dict[str, float], frame_idx: int) -> np.ndarray:
    """Draw metric values onto a panel."""
    lines = [
        f"Frame {frame_idx}",
        "",
        f"MF-EPE:      {metrics['mf_epe']:.2f} px",
        f"MF-Angle:    {metrics['mf_angle_err']:.1f} deg",
        f"MF-Cosine:   {metrics['mf_cosine']:.3f}",
        f"MF-MagRatio: {metrics['mf_mag_ratio']:.2f}",
        "",
        f"Pixel EPE:   {metrics['pixel_epe_mean']:.2f} px",
        f"Px Ang RMSE: {metrics['px_angle_rmse']:.1f} deg",
        f"Fl-all:      {metrics['fl_all'] * 100:.1f} %",
        "",
        f"FOE Dist:    {metrics['foe_dist']:.1f} px",
        f"Flow KL-2D:  {metrics['flow_kl_2d']:.4f}",
    ]
    y = 20
    for line in lines:
        if line == "":
            y += 8
            continue
        cv.putText(img, line, (8, y), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                   (255, 255, 255), 1, cv.LINE_AA)
        y += 20
    return img


def _generate_comparison_video(
    frames_gt: List[np.ndarray],
    frames_gen: List[np.ndarray],
    flows_gt: List[np.ndarray],
    flows_gen: List[np.ndarray],
    frame_metrics_list: List[Dict[str, float]],
    summary: Dict,
    output_dir: Path,
    fps: float = 10.0,
) -> None:
    """Render an MP4 comparison video with all panels.

    Layout (3 columns x 2 rows + time series strip):
    ┌───────────┬───────────┬───────────┐
    │ GT Frame  │ GT Flow   │ EPE Hmap  │
    ├───────────┼───────────┼───────────┤
    │ Gen Frame │ Gen Flow  │ Metrics   │
    ├───────────┴───────────┴───────────┤
    │      Time Series (with cursor)    │
    └───────────────────────────────────┘
    """
    n_flows = len(flows_gt)
    H, W = flows_gt[0].shape[:2]

    pw, ph = W, H  # panel width, height

    # Shared flow color scale
    flow_max_radius = max(
        np.sqrt(f[:, :, 0] ** 2 + f[:, :, 1] ** 2).max()
        for f in flows_gt + flows_gen
    )

    # Shared EPE color scale (95th percentile for stability)
    all_epe_maxes = []
    for i in range(n_flows):
        epe = np.linalg.norm(flows_gt[i] - flows_gen[i], axis=2)
        all_epe_maxes.append(np.percentile(epe, 95))
    epe_color_max = max(np.percentile(all_epe_maxes, 95), 1.0)

    # Pre-render time series base image
    n_cols = 3
    ts_width = pw * n_cols
    ts_height = ph
    ts_base = _render_timeseries_base(frame_metrics_list, summary, ts_width, ts_height)

    # Cursor x mapping
    left_margin_frac = 0.08
    right_margin_frac = 0.02
    plot_x_start = int(ts_width * left_margin_frac)
    plot_x_end = int(ts_width * (1.0 - right_margin_frac))
    plot_x_range = plot_x_end - plot_x_start

    label_h = 22
    total_w = pw * n_cols
    row_h = ph + label_h
    total_h = row_h * 2 + ts_height

    video_path = output_dir / "comparison.mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(video_path), fourcc, fps, (total_w, total_h))

    for i in tqdm(range(n_flows), desc="Rendering video", leave=False):
        gt_frame = cv.resize(frames_gt[i], (pw, ph))
        gen_frame = cv.resize(frames_gen[i], (pw, ph))

        gt_flow_rgb = flow_to_rgb(flows_gt[i], flow_max_radius=flow_max_radius)
        gt_flow_bgr = cv.resize(cv.cvtColor(gt_flow_rgb, cv.COLOR_RGB2BGR), (pw, ph))
        _draw_flow_arrows(gt_flow_bgr, cv.resize(flows_gt[i], (pw, ph)))
        # Draw FOE crosshair on GT flow (cyan)
        m = frame_metrics_list[i]
        _draw_foe_crosshair(gt_flow_bgr, m["foe_gt_x"], m["foe_gt_y"],
                            color=(255, 255, 0), label="FOE")

        gen_flow_rgb = flow_to_rgb(flows_gen[i], flow_max_radius=flow_max_radius)
        gen_flow_bgr = cv.resize(cv.cvtColor(gen_flow_rgb, cv.COLOR_RGB2BGR), (pw, ph))
        _draw_flow_arrows(gen_flow_bgr, cv.resize(flows_gen[i], (pw, ph)))
        # Draw FOE crosshair on Gen flow (green)
        _draw_foe_crosshair(gen_flow_bgr, m["foe_gen_x"], m["foe_gen_y"],
                            color=(0, 255, 0), label="FOE")

        epe_heatmap = cv.resize(
            _make_epe_heatmap_bgr(flows_gt[i], flows_gen[i], epe_color_max), (pw, ph))

        # Metrics panel
        metrics_panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        _put_metrics_text(metrics_panel, frame_metrics_list[i], i)

        # Add labels
        row1 = np.hstack([
            _put_label(gt_frame, "GT Frame"),
            _put_label(gt_flow_bgr, "GT Flow"),
            _put_label(epe_heatmap, "EPE Heatmap"),
        ])
        row2 = np.hstack([
            _put_label(gen_frame, "Gen Frame"),
            _put_label(gen_flow_bgr, "Gen Flow"),
            _put_label(metrics_panel, "Metrics"),
        ])

        # Time series with cursor
        ts_frame = ts_base.copy()
        cursor_x = plot_x_start + int(i / max(n_flows - 1, 1) * plot_x_range)
        cv.line(ts_frame, (cursor_x, 0), (cursor_x, ts_height), (0, 255, 0), 2)

        composite = np.vstack([row1, row2, ts_frame])

        if composite.shape[1] != total_w or composite.shape[0] != total_h:
            composite = cv.resize(composite, (total_w, total_h))

        writer.write(composite)

    writer.release()
    print(f"  Wrote comparison video: {video_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_pair(
    gt_video: str,
    gen_video: str,
    output_dir: str,
    model_name: str = "dpflow",
    ckpt: str = "things",
    grid_size: int = 8,
    no_viz: bool = False,
    model: Optional[torch.nn.Module] = None,
    io_adapter: Optional[IOAdapter] = None,
) -> Dict:
    """Run full evaluation pipeline on a single GT/generated video pair."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Extract frames and get FPS ---
    print(f"Extracting frames from GT: {gt_video}")
    frames_gt = extract_frames(gt_video)
    cap_tmp = cv.VideoCapture(gt_video)
    src_fps = cap_tmp.get(cv.CAP_PROP_FPS)
    cap_tmp.release()
    if src_fps <= 0:
        src_fps = 10.0
    print(f"Extracting frames from Gen: {gen_video}")
    frames_gen = extract_frames(gen_video)

    n_gt = len(frames_gt)
    n_gen = len(frames_gen)
    n_frames = min(n_gt, n_gen)
    if n_gt != n_gen:
        print(f"  Warning: frame count mismatch (GT={n_gt}, Gen={n_gen}), using first {n_frames}")
        frames_gt = frames_gt[:n_frames]
        frames_gen = frames_gen[:n_frames]

    print(f"  {n_frames} frames, resolution {frames_gt[0].shape[1]}x{frames_gt[0].shape[0]}")

    if n_frames < 2:
        print("  Error: need at least 2 frames for flow computation")
        return {}

    # --- Load model if not provided ---
    if model is None:
        print(f"Loading flow model: {model_name} ({ckpt})")
        model = ptlflow.get_model(model_name, ckpt)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

    if io_adapter is None:
        io_adapter = IOAdapter(
            output_stride=model.output_stride,
            input_size=frames_gt[0].shape[:2],
            cuda=torch.cuda.is_available(),
        )

    # --- Compute flow ---
    print("Computing GT flows...")
    flows_gt = compute_flow_sequence(model, frames_gt, io_adapter)
    print("Computing Generated flows...")
    flows_gen = compute_flow_sequence(model, frames_gen, io_adapter)

    # --- Compute per-frame metrics ---
    print("Computing metrics...")
    frame_metrics_list = []
    for i in tqdm(range(len(flows_gt)), desc="Metrics", leave=False):
        m = compute_frame_metrics(
            flows_gt[i], flows_gen[i],
            grid_size=grid_size,
        )
        m["frame_idx"] = i
        frame_metrics_list.append(m)

    # --- Temporal analysis ---
    summary = compute_temporal_metrics(frame_metrics_list)
    summary["gt_video"] = gt_video
    summary["gen_video"] = gen_video

    # --- Write CSV ---
    csv_path = output_path / "metrics.csv"
    fieldnames = list(frame_metrics_list[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_metrics_list)
    print(f"  Wrote {csv_path}")

    # --- Write summary JSON ---
    json_path = output_path / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote {json_path}")

    # --- Visualizations ---
    if not no_viz:
        print("Generating visualizations...")
        generate_visualizations(
            frames_gt, frames_gen, flows_gt, flows_gen,
            frame_metrics_list, summary, output_path, fps=src_fps,
        )
        print(f"  Visualizations saved to {output_path}")

    # --- Print summary ---
    print("\n--- Summary ---")
    for key in ["mf_epe_mean", "mf_angle_err_mean", "mf_cosine_mean",
                 "mf_mag_ratio_mean", "pixel_epe_mean_mean",
                 "px_angle_rmse_mean", "fl_all_mean",
                 "foe_dist_mean", "flow_kl_2d_mean"]:
        if key in summary:
            val = summary[key]
            if "fl_all" in key:
                print(f"  {key}: {val * 100:.1f}%")
            elif "angle" in key:
                print(f"  {key}: {val:.1f} deg")
            else:
                print(f"  {key}: {val:.3f}")
    div = summary.get("divergence_onset_frame")
    if div is not None:
        print(f"  divergence_onset_frame: {div}")
    else:
        print("  divergence_onset_frame: none detected")
    print()

    return summary


@torch.no_grad()
def evaluate_pair_synthetic(
    gen_video: str,
    action_path: str,
    calibration_path: str,
    output_dir: str,
    model_name: str = "dpflow",
    ckpt: str = "things",
    grid_size: int = 8,
    no_viz: bool = False,
    use_depth: bool = True,
    model: Optional[torch.nn.Module] = None,
    io_adapter: Optional[IOAdapter] = None,
    flow_generator: Optional[object] = None,
    gt_video: Optional[str] = None,
) -> Dict:
    """Run evaluation using synthetic flow from actions vs generated video flow.

    The reference flow is computed analytically from the action data using the
    Longuet-Higgins physics model. GT video is optional (shown for reference only).

    Parameters
    ----------
    gen_video : str
        Path to generated video.
    action_path : str
        Path to action .npy file.
    calibration_path : str
        Path to calibration.json.
    output_dir : str
        Output directory for results.
    use_depth : bool
        If True, estimate depth from generated frames for translation flow.
    flow_generator : optional SyntheticFlowGenerator
        Pre-initialized generator (for batch mode to avoid reloading).
    gt_video : optional str
        Path to GT video (shown for visual reference only, not used in metrics).
    """
    from synthetic_flow import SyntheticFlowGenerator, load_calibration

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Extract frames ---
    print(f"Extracting frames from Gen: {gen_video}")
    frames_gen = extract_frames(gen_video)
    cap_tmp = cv.VideoCapture(gen_video)
    src_fps = cap_tmp.get(cv.CAP_PROP_FPS)
    cap_tmp.release()
    if src_fps <= 0:
        src_fps = 10.0

    n_frames = len(frames_gen)
    print(f"  {n_frames} frames, resolution {frames_gen[0].shape[1]}x{frames_gen[0].shape[0]}")

    if n_frames < 2:
        print("  Error: need at least 2 frames for flow computation")
        return {}

    # --- Load GT frames for reference (optional) ---
    # Auto-discover GT video if not provided: {id}_wangame.mp4 -> {id}.mp4
    frames_gt = None
    if gt_video is None:
        gen_p = Path(gen_video)
        gt_auto = gen_p.parent / gen_p.name.replace("_wangame.mp4", ".mp4")
        if gt_auto != gen_p and gt_auto.exists():
            gt_video = str(gt_auto)
    if gt_video is not None:
        gt_path = Path(gt_video)
        if gt_path.exists():
            print(f"Loading GT frames for reference: {gt_video}")
            frames_gt = extract_frames(str(gt_path))
            frames_gt = frames_gt[:n_frames]

    # --- Load actions ---
    print(f"Loading actions from: {action_path}")
    actions = np.load(action_path, allow_pickle=True).item()
    n_actions = len(actions["keyboard"])
    n_use = min(n_frames, n_actions)
    if n_frames != n_actions:
        print(f"  Warning: frame/action count mismatch ({n_frames} vs {n_actions}), using {n_use}")
        frames_gen = frames_gen[:n_use]
        if frames_gt is not None:
            frames_gt = frames_gt[:n_use]
        actions = {
            key: value[:n_use] if hasattr(value, "__getitem__") else value
            for key, value in actions.items()
        }
        n_frames = n_use

    # --- Load flow model if not provided ---
    if model is None:
        print(f"Loading flow model: {model_name} ({ckpt})")
        model = ptlflow.get_model(model_name, ckpt)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

    if io_adapter is None:
        io_adapter = IOAdapter(
            output_stride=model.output_stride,
            input_size=frames_gen[0].shape[:2],
            cuda=torch.cuda.is_available(),
        )

    # --- Initialize synthetic flow generator ---
    if flow_generator is None:
        calibration = load_calibration(calibration_path)
        flow_generator = SyntheticFlowGenerator(
            calibration=calibration,
            frame_shape=frames_gen[0].shape[:2],
            use_depth=use_depth,
        )
        if use_depth:
            print("Loading depth model...")
            flow_generator.load_depth_model()

    # --- Compute generated video flows ---
    print("Computing Generated flows...")
    flows_gen = compute_flow_sequence(model, frames_gen, io_adapter)

    # --- Generate synthetic flows (with depth maps) ---
    print("Generating synthetic flows...")
    flows_synth, depth_maps = flow_generator.generate_flow_sequence(
        actions=actions,
        frames=frames_gen if use_depth else None,
        return_depths=True,
    )

    # Match lengths (flows = n_frames - 1, synthetic may differ)
    n_flows = min(len(flows_gen), len(flows_synth))
    flows_gen = flows_gen[:n_flows]
    flows_synth = flows_synth[:n_flows]
    depth_maps = depth_maps[:n_flows]

    # --- Compute per-frame metrics ---
    print("Computing metrics...")
    frame_metrics_list = []
    for i in tqdm(range(n_flows), desc="Metrics", leave=False):
        m = compute_frame_metrics(
            flows_synth[i], flows_gen[i],
            grid_size=grid_size,
        )
        m["frame_idx"] = i
        frame_metrics_list.append(m)

    # --- Temporal analysis ---
    summary = compute_temporal_metrics(frame_metrics_list)
    summary["gen_video"] = gen_video
    summary["action_file"] = action_path
    summary["mode"] = "synthetic"

    # --- Write CSV ---
    csv_path = output_path / "metrics.csv"
    fieldnames = list(frame_metrics_list[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_metrics_list)
    print(f"  Wrote {csv_path}")

    # --- Write summary JSON ---
    json_path = output_path / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote {json_path}")

    # --- Visualizations ---
    if not no_viz:
        print("Generating visualizations...")
        generate_visualizations_synthetic(
            frames_gen, flows_synth, flows_gen,
            frame_metrics_list, summary, output_path,
            fps=src_fps, actions=actions, depth_maps=depth_maps,
            frames_gt=frames_gt,
        )
        print(f"  Visualizations saved to {output_path}")

    # --- Print summary ---
    print("\n--- Summary (Synthetic Mode) ---")
    for key in ["mf_epe_mean", "mf_angle_err_mean", "mf_cosine_mean",
                 "mf_mag_ratio_mean", "pixel_epe_mean_mean",
                 "px_angle_rmse_mean", "fl_all_mean",
                 "foe_dist_mean", "flow_kl_2d_mean"]:
        if key in summary:
            val = summary[key]
            if "fl_all" in key:
                print(f"  {key}: {val * 100:.1f}%")
            elif "angle" in key:
                print(f"  {key}: {val:.1f} deg")
            else:
                print(f"  {key}: {val:.3f}")
    print()

    return summary


# ---------------------------------------------------------------------------
# Action overlay helpers (for synthetic mode visualization)
# ---------------------------------------------------------------------------

_KEY_NAMES = ["W", "S", "A", "D", "left", "right"]
_KEY_ICONS = {"W": "W", "A": "A", "S": "S", "D": "D", "left": "L", "right": "R"}


def _overlay_actions(frame: np.ndarray, keyboard_vec: np.ndarray,
                     mouse_vec: np.ndarray) -> None:
    """Draw keyboard and mouse indicators on a frame (in-place)."""
    key_size = (30, 30)
    top_margin = 15
    left_margin = 15
    gap = 3
    key_positions = {
        "W": (left_margin + key_size[0] + gap, top_margin),
        "A": (left_margin, top_margin + key_size[1] + gap),
        "S": (left_margin + key_size[0] + gap, top_margin + key_size[1] + gap),
        "D": (left_margin + (key_size[0] + gap) * 2, top_margin + key_size[1] + gap),
        "left": (left_margin + (key_size[0] + gap) * 3 + 10, top_margin + key_size[1] + gap),
        "right": (left_margin + (key_size[0] + gap) * 4 + 15, top_margin + key_size[1] + gap),
    }

    for key_name, (x, y) in key_positions.items():
        idx = _KEY_NAMES.index(key_name)
        is_pressed = bool(keyboard_vec[idx] > 0.5)
        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv.rectangle(overlay, (x, y), (x + key_size[0], y + key_size[1]), color, -1)
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        icon = _KEY_ICONS[key_name]
        text_size = cv.getTextSize(icon, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (key_size[0] - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv.putText(frame, icon, (text_x, text_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Mouse crosshair
    if mouse_vec is not None and len(mouse_vec) >= 2:
        h, w = frame.shape[:2]
        r = 25
        cx = w - 15 - r
        cy = top_margin + r
        dx = int(mouse_vec[1] * r * 8)
        dy = int(-mouse_vec[0] * r * 8)
        max_arrow = r - 5
        dx = max(-max_arrow, min(max_arrow, dx))
        dy = max(-max_arrow, min(max_arrow, dy))
        cv.circle(frame, (cx, cy), r, (50, 50, 50), -1)
        cv.circle(frame, (cx, cy), r, (200, 200, 200), 1)
        cv.line(frame, (cx - r + 5, cy), (cx + r - 5, cy), (100, 100, 100), 1)
        cv.line(frame, (cx, cy - r + 5), (cx, cy + r - 5), (100, 100, 100), 1)
        if abs(dx) > 1 or abs(dy) > 1:
            cv.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy),
                           (0, 255, 0), 2, tipLength=0.3)


def _depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Convert a depth map (HxW float) to a BGR colormap image."""
    if depth is None:
        return None
    # Normalize to 0-255 (clip outliers at 5th/95th percentile)
    d = depth.copy()
    lo, hi = np.percentile(d, [5, 95])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    d = np.clip((d - lo) / (hi - lo), 0, 1)
    d_uint8 = (d * 255).astype(np.uint8)
    # Use INFERNO colormap (close = warm, far = cool)
    return cv.applyColorMap(d_uint8, cv.COLORMAP_INFERNO)


def generate_visualizations_synthetic(
    frames_gen: List[np.ndarray],
    flows_synth: List[np.ndarray],
    flows_gen: List[np.ndarray],
    frame_metrics_list: List[Dict[str, float]],
    summary: Dict,
    output_dir: Path,
    fps: float = 10.0,
    actions: Optional[dict] = None,
    depth_maps: Optional[List] = None,
    frames_gt: Optional[List[np.ndarray]] = None,
) -> None:
    """Generate visualizations for synthetic mode.

    Layout (3 columns x 3 rows + time series):
    +---------------+--------------+-------------+
    | GT Frame      | Gen Frame    | Depth Map   |
    +  (actions)    |  (actions)   |             |
    +---------------+--------------+-------------+
    | Synth Flow    | Gen Flow     | EPE Hmap    |
    |               |              |             |
    +---------------+--------------+-------------+
    |  Metrics      |   Time Series (cursor)     |
    +---------------+----------------------------+
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_flows = min(len(flows_synth), len(flows_gen))

    # --- Static time-series plot ---
    metric_names = list(frame_metrics_list[0].keys())
    groups = {
        "EPE (px)": ["mf_epe", "pixel_epe_mean"],
        "Direction": ["mf_cosine", "mf_angle_err"],
        "Speed": ["mf_mag_ratio"],
        "Local Err": ["px_angle_rmse", "fl_all"],
        "FOE (px)": ["foe_dist"],
        "KL Div": ["flow_kl_2d"],
    }

    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 12), sharex=True)
    frame_indices = np.arange(n_flows)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    div_frame = summary.get("divergence_onset_frame")

    for ax, (group_name, names) in zip(axes, groups.items()):
        all_handles = []
        all_labels = []
        if len(names) == 2:
            ax_right = ax.twinx()
            axes_list = [ax, ax_right]
        else:
            axes_list = [ax] * len(names)
        for ci, (name, target_ax) in enumerate(zip(names, axes_list)):
            if name in metric_names:
                vals = [m[name] for m in frame_metrics_list]
                line, = target_ax.plot(frame_indices, vals, label=name, linewidth=1.5,
                                       color=colors[ci % len(colors)])
                all_handles.append(line)
                all_labels.append(name)
        ax.set_ylabel(group_name)
        ax.legend(all_handles, all_labels, loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        if div_frame is not None:
            ax.axvline(x=div_frame, color="red", linestyle="--", alpha=0.7)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Flow Divergence Metrics Over Time (Synthetic Mode)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "timeseries.png", dpi=150)
    plt.close(fig)

    # --- Comparison video ---
    print("  Rendering comparison video...")
    _generate_comparison_video_synthetic(
        frames_gen, flows_synth, flows_gen,
        frame_metrics_list, summary, output_dir, fps,
        actions=actions, depth_maps=depth_maps, frames_gt=frames_gt,
    )


def _generate_comparison_video_synthetic(
    frames_gen: List[np.ndarray],
    flows_synth: List[np.ndarray],
    flows_gen: List[np.ndarray],
    frame_metrics_list: List[Dict[str, float]],
    summary: Dict,
    output_dir: Path,
    fps: float = 10.0,
    actions: Optional[dict] = None,
    depth_maps: Optional[List] = None,
    frames_gt: Optional[List[np.ndarray]] = None,
) -> None:
    """Render comparison video for synthetic mode.

    Layout (3 columns x 3 rows):
    +---------------+--------------+-------------+
    | GT Frame      | Gen Frame    | Depth Map   |
    |  (actions)    |  (actions)   |             |
    +---------------+--------------+-------------+
    | Synth Flow    | Gen Flow     | EPE Heatmap |
    |               |              |             |
    +---------------+--------------+-------------+
    |  Metrics      |  Time Series (with cursor) |
    +---------------+----------------------------+
    """
    n_flows = min(len(flows_synth), len(flows_gen))
    H, W = flows_gen[0].shape[:2]
    pw, ph = W, H

    # Shared flow color scale
    flow_max_radius = max(
        np.sqrt(f[:, :, 0] ** 2 + f[:, :, 1] ** 2).max()
        for f in flows_synth + flows_gen
    )

    # Shared EPE color scale
    all_epe_maxes = []
    for i in range(n_flows):
        epe = np.linalg.norm(flows_synth[i] - flows_gen[i], axis=2)
        all_epe_maxes.append(np.percentile(epe, 95))
    epe_color_max = max(np.percentile(all_epe_maxes, 95), 1.0)

    # Pre-render time series (spans 2 columns)
    n_cols = 3
    ts_width = pw * 2  # time series spans right 2 columns
    ts_height = ph
    ts_base = _render_timeseries_base(frame_metrics_list, summary, ts_width, ts_height)

    left_margin_frac = 0.08
    right_margin_frac = 0.02
    plot_x_start = int(ts_width * left_margin_frac)
    plot_x_end = int(ts_width * (1.0 - right_margin_frac))
    plot_x_range = plot_x_end - plot_x_start

    label_h = 22
    total_w = pw * n_cols
    row_h = ph + label_h
    total_h = row_h * 3  # 3 rows (row3 = metrics + timeseries)

    video_path = output_dir / "comparison.mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(video_path), fourcc, fps, (total_w, total_h))

    for i in tqdm(range(n_flows), desc="Rendering video", leave=False):
        # --- Row 1: GT Frame | Gen Frame | Depth Map ---

        # GT frame (or placeholder)
        if frames_gt is not None and i < len(frames_gt):
            gt_frame = cv.resize(frames_gt[i].copy(), (pw, ph))
        else:
            gt_frame = np.zeros((ph, pw, 3), dtype=np.uint8)
            cv.putText(gt_frame, "GT not available", (pw // 4, ph // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv.LINE_AA)

        gen_frame = cv.resize(frames_gen[i].copy(), (pw, ph))

        # Overlay actions on both frames
        if actions is not None:
            kb = actions["keyboard"]
            ms = actions["mouse"]
            action_idx = min(i, len(kb) - 1)
            _overlay_actions(gt_frame, kb[action_idx], ms[action_idx])
            _overlay_actions(gen_frame, kb[action_idx], ms[action_idx])

        # Depth map
        if depth_maps is not None and i < len(depth_maps) and depth_maps[i] is not None:
            depth_bgr = _depth_to_colormap(depth_maps[i])
            depth_panel = cv.resize(depth_bgr, (pw, ph))
        else:
            depth_panel = np.zeros((ph, pw, 3), dtype=np.uint8)
            cv.putText(depth_panel, "No depth (constant Z=1)", (pw // 6, ph // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv.LINE_AA)

        row1 = np.hstack([
            _put_label(gt_frame, "GT Frame (reference)"),
            _put_label(gen_frame, "Gen Frame"),
            _put_label(depth_panel, "Depth Map (gen)"),
        ])

        # --- Row 2: Synth Flow | Gen Flow | EPE Heatmap ---
        synth_flow_rgb = flow_to_rgb(flows_synth[i], flow_max_radius=flow_max_radius)
        synth_flow_bgr = cv.resize(cv.cvtColor(synth_flow_rgb, cv.COLOR_RGB2BGR), (pw, ph))
        _draw_flow_arrows(synth_flow_bgr, cv.resize(flows_synth[i], (pw, ph)))
        m = frame_metrics_list[i]
        _draw_foe_crosshair(synth_flow_bgr, m["foe_gt_x"], m["foe_gt_y"],
                            color=(255, 255, 0), label="FOE")

        gen_flow_rgb = flow_to_rgb(flows_gen[i], flow_max_radius=flow_max_radius)
        gen_flow_bgr = cv.resize(cv.cvtColor(gen_flow_rgb, cv.COLOR_RGB2BGR), (pw, ph))
        _draw_flow_arrows(gen_flow_bgr, cv.resize(flows_gen[i], (pw, ph)))
        _draw_foe_crosshair(gen_flow_bgr, m["foe_gen_x"], m["foe_gen_y"],
                            color=(0, 255, 0), label="FOE")

        epe_heatmap = cv.resize(
            _make_epe_heatmap_bgr(flows_synth[i], flows_gen[i], epe_color_max), (pw, ph))

        row2 = np.hstack([
            _put_label(synth_flow_bgr, "Synthetic Flow"),
            _put_label(gen_flow_bgr, "Gen Flow"),
            _put_label(epe_heatmap, "EPE Heatmap"),
        ])

        # --- Row 3: Metrics | Time Series ---
        metrics_panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        _put_metrics_text(metrics_panel, frame_metrics_list[i], i)
        metrics_with_label = _put_label(metrics_panel, "Metrics")

        ts_frame = ts_base.copy()
        cursor_x = plot_x_start + int(i / max(n_flows - 1, 1) * plot_x_range)
        cv.line(ts_frame, (cursor_x, 0), (cursor_x, ts_height), (0, 255, 0), 2)
        # Add label bar to timeseries to match height
        ts_with_label = _put_label(ts_frame, "Time Series")

        row3 = np.hstack([metrics_with_label, ts_with_label])

        composite = np.vstack([row1, row2, row3])
        if composite.shape[1] != total_w or composite.shape[0] != total_h:
            composite = cv.resize(composite, (total_w, total_h))

        writer.write(composite)

    writer.release()
    print(f"  Wrote comparison video: {video_path}")


def evaluate_scenario(
    scenario_dir: str,
    output_dir: str,
    model_name: str = "dpflow",
    ckpt: str = "things",
    grid_size: int = 8,
    no_viz: bool = False,
) -> List[Dict]:
    """Run evaluation on all video pairs in a scenario directory."""
    scenario_path = Path(scenario_dir)
    output_path = Path(output_dir)

    gt_videos = sorted(scenario_path.glob("*.mp4"))
    gt_videos = [v for v in gt_videos if "_wangame" not in v.stem]

    if not gt_videos:
        print(f"No GT videos found in {scenario_dir}")
        return []

    print(f"Loading flow model: {model_name} ({ckpt})")
    model = ptlflow.get_model(model_name, ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    first_frame = extract_frames(str(gt_videos[0]))[0]
    io_adapter = IOAdapter(
        output_stride=model.output_stride,
        input_size=first_frame.shape[:2],
        cuda=torch.cuda.is_available(),
    )

    summaries = []
    for gt_video in gt_videos:
        gen_video = gt_video.parent / f"{gt_video.stem}_wangame.mp4"
        if not gen_video.exists():
            print(f"  Skipping {gt_video.name}: no matching _wangame video")
            continue

        pair_name = gt_video.stem
        pair_output = output_path / pair_name
        print(f"\n{'='*60}")
        print(f"Evaluating pair: {pair_name}")
        print(f"{'='*60}")

        s = evaluate_pair(
            gt_video=str(gt_video),
            gen_video=str(gen_video),
            output_dir=str(pair_output),
            model_name=model_name,
            ckpt=ckpt,
            grid_size=grid_size,
            no_viz=no_viz,
            model=model,
            io_adapter=io_adapter,
        )
        summaries.append(s)

    if summaries:
        agg_path = output_path / "all_summaries.json"
        with open(agg_path, "w") as f:
            json.dump(summaries, f, indent=2, default=str)
        print(f"\nAggregated summaries written to {agg_path}")

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate action-conditioned video generation via optical flow divergence."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gt_video", type=str, help="Path to ground truth video")
    group.add_argument("--scenario_dir", type=str, help="Path to scenario directory for batch mode")
    group.add_argument("--synthetic", action="store_true",
                       help="Synthetic flow mode (no GT video needed)")

    parser.add_argument("--gen_video", type=str, help="Path to generated video")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="dpflow", help="Optical flow model name (default: dpflow)")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="hf://mignonjia/dpflow-ckpt/dpflow-things-2012b5d6.ckpt",
        help="Model checkpoint path",
    )
    parser.add_argument("--grid_size", type=int, default=8, help="Grid size for spatial EPE (default: 8)")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization generation")

    # Synthetic mode options
    parser.add_argument("--action_file", type=str,
                        help="Path to action .npy file (for --synthetic)")
    parser.add_argument("--calibration", type=str,
                        help="Path to calibration.json (for --synthetic)")
    parser.add_argument("--no_depth", action="store_true",
                        help="Skip depth estimation, use constant depth (for --synthetic)")
    parser.add_argument(
        "--validation_json",
        type=str,
        help="Path to validation JSON. Used for synthetic auto-sweep to load action_file by index.",
    )
    parser.add_argument(
        "--validation_idx",
        type=int,
        default=0,
        help="Validation sample index for synthetic auto-sweep (default: 0).",
    )
    parser.add_argument(
        "--validation_indices",
        type=str,
        default="",
        help="Validation indices for synthetic auto-sweep. Supports ranges and lists, "
             "e.g. 2-7 or 2,3,4,5,6,7. If set, overrides --validation_idx.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        help="Directory containing validation_step_* videos for synthetic auto-sweep.",
    )
    parser.add_argument(
        "--inference_steps_tag",
        type=int,
        default=40,
        help="Filename tag in validation videos: validation_step_*_inference_steps_{tag}_video_*.mp4",
    )
    parser.add_argument(
        "--train_steps",
        type=str,
        default="",
        help="Comma-separated training steps to evaluate (e.g. 0,100,200). "
             "If empty, auto-discover from files and use --step_stride filter.",
    )
    parser.add_argument(
        "--step_stride",
        type=int,
        default=1000,
        help="When auto-discovering training steps, only keep steps divisible by this value (default: 1000).",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="pixel_epe_mean_mean",
        help="Metric key used to pick best step from step-average table "
             "(default: pixel_epe_mean_mean).",
    )
    parser.add_argument(
        "--merge_with_existing_output",
        action="store_true",
        help="When set, rebuild all summaries/csv from all existing "
             "idx_*/step_*/summary.json under --output_dir (not only this run).",
    )
    return parser.parse_args()


def _resolve_action_path_from_validation_json(
    validation_json: str,
    validation_idx: int,
) -> str:
    """Load action path from validation json at a specific index."""
    validation_json_path = Path(validation_json)
    if not validation_json_path.exists():
        raise FileNotFoundError(f"validation json not found: {validation_json}")

    with open(validation_json_path, "r") as f:
        payload = json.load(f)

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Invalid validation json format: missing list field 'data' in {validation_json}")
    if validation_idx < 0 or validation_idx >= len(data):
        raise IndexError(
            f"validation_idx={validation_idx} out of range [0, {len(data) - 1}] for {validation_json}"
        )

    sample = data[validation_idx]
    action_path = sample.get("action_path")
    if not action_path:
        raise ValueError(
            f"No action_path found at data[{validation_idx}] in {validation_json}"
        )

    action_path = Path(action_path)
    if not action_path.is_absolute():
        action_path = (validation_json_path.parent / action_path).resolve()
    return str(action_path)


def _discover_validation_step_videos(
    video_root: str,
    validation_idx: int,
    inference_steps_tag: int,
) -> Dict[int, str]:
    """Return mapping: train_step -> video_path for matching validation videos."""
    root = Path(video_root)
    if not root.exists():
        raise FileNotFoundError(f"video_root not found: {video_root}")

    pattern = re.compile(
        rf"^validation_step_(\d+)_inference_steps_{inference_steps_tag}_video_{validation_idx}\.mp4$"
    )
    step_to_video: Dict[int, str] = {}
    for path in root.glob("validation_step_*_inference_steps_*_video_*.mp4"):
        m = pattern.match(path.name)
        if not m:
            continue
        step = int(m.group(1))
        step_to_video[step] = str(path)
    return step_to_video


def _parse_train_steps(train_steps: str) -> List[int]:
    """Parse comma-separated steps string into sorted unique integer list."""
    if not train_steps.strip():
        return []
    parsed = set()
    for token in train_steps.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.add(int(token))
    return sorted(parsed)


def _parse_validation_indices(
    validation_indices: str,
    fallback_idx: int,
) -> List[int]:
    """Parse validation indices from a range/list string.

    Examples
    --------
    "2-7" -> [2, 3, 4, 5, 6, 7]
    "2,4,6" -> [2, 4, 6]
    """
    spec = validation_indices.strip()
    if not spec:
        return [fallback_idx]

    parsed = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(
                    f"Invalid validation index range '{token}': end < start"
                )
            for idx in range(start, end + 1):
                parsed.add(idx)
        else:
            parsed.add(int(token))

    if not parsed:
        raise ValueError("No validation indices parsed from --validation_indices")
    return sorted(parsed)


def _load_existing_summaries_from_output(output_root: Path) -> List[Dict]:
    """Load per-(step, idx) summaries from output_root/idx_*/step_*/summary.json."""
    summaries: List[Dict] = []
    for idx_dir in sorted(output_root.glob("idx_*")):
        if not idx_dir.is_dir():
            continue
        try:
            idx = int(idx_dir.name.split("_", 1)[1])
        except Exception:
            continue

        for step_dir in sorted(idx_dir.glob("step_*")):
            if not step_dir.is_dir():
                continue
            try:
                step = int(step_dir.name.split("_", 1)[1])
            except Exception:
                continue
            summary_path = step_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
            except Exception:
                continue
            summary["train_step"] = step
            summary["validation_idx"] = idx
            summaries.append(summary)
    return summaries


def _plot_step_metric_trends(
    step_avg_rows: List[Dict],
    scalar_keys: List[str],
    output_root: Path,
) -> Optional[Path]:
    """Plot 9 metrics in a 3x3 grid over training steps.

    If train steps are multiples of 1000, x-axis is displayed as 0, 1, ..., 10
    (i.e., step / 1000), matching checkpoints 0, 1000, ..., 10000.
    """
    if not step_avg_rows:
        return None

    rows = sorted(step_avg_rows, key=lambda r: int(r["train_step"]))
    steps = [int(r["train_step"]) for r in rows]

    if steps and all(step % 1000 == 0 for step in steps):
        x_vals = [step // 1000 for step in steps]
        x_label = "Training Step Index (x1000)"
        x_ticks = list(range(min(x_vals), max(x_vals) + 1))
    else:
        x_vals = list(range(len(steps)))
        x_label = "Training Step Index"
        x_ticks = x_vals

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    flat_axes = axes.flatten()

    for ax, metric in zip(flat_axes, scalar_keys):
        y_vals = []
        for r in rows:
            v = r.get(metric)
            y_vals.append(np.nan if v is None else float(v))

        ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, markersize=4)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(x_label)
        ax.tick_params(axis="x", labelbottom=True)

        # Highlight best point per metric.
        if metric == "mf_cosine_mean":
            best_idx = int(np.nanargmax(y_vals))
        elif metric == "mf_mag_ratio_mean":
            arr = np.asarray(y_vals, dtype=np.float64)
            best_idx = int(np.nanargmin(np.abs(arr - 1.0)))
        else:
            best_idx = int(np.nanargmin(y_vals))
        ax.scatter([x_vals[best_idx]], [y_vals[best_idx]], color="red", s=30, zorder=3)

    for ax in flat_axes:
        ax.set_xticks(x_ticks)

    fig.suptitle("Metric Trends Across Training Steps", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = output_root / "metric_trends_3x3.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()

    if args.synthetic:
        if not args.calibration:
            print("Error: --calibration is required with --synthetic")
            sys.exit(1)

        # Mode A: original single-video synthetic evaluation
        if args.gen_video:
            if not args.action_file:
                print("Error: --action_file is required with --synthetic when --gen_video is provided")
                sys.exit(1)
            evaluate_pair_synthetic(
                gen_video=args.gen_video,
                action_path=args.action_file,
                calibration_path=args.calibration,
                output_dir=args.output_dir,
                model_name=args.model,
                ckpt=args.ckpt,
                grid_size=args.grid_size,
                no_viz=args.no_viz,
                use_depth=not args.no_depth,
            )
            return

        # Mode B: synthetic auto-sweep over validation_step_* videos
        if not args.validation_json:
            print("Error: --validation_json is required for synthetic auto-sweep")
            sys.exit(1)
        if not args.video_root:
            print("Error: --video_root is required for synthetic auto-sweep")
            sys.exit(1)

        validation_indices = _parse_validation_indices(
            args.validation_indices,
            args.validation_idx,
        )
        print(f"Using validation indices: {validation_indices}")

        idx_to_action_path: Dict[int, str] = {}
        idx_to_step_to_video: Dict[int, Dict[int, str]] = {}
        for idx in validation_indices:
            action_path = _resolve_action_path_from_validation_json(
                args.validation_json,
                idx,
            )
            idx_to_action_path[idx] = action_path
            print(f"Resolved action file from validation_json[{idx}]: {action_path}")

            step_to_video = _discover_validation_step_videos(
                video_root=args.video_root,
                validation_idx=idx,
                inference_steps_tag=args.inference_steps_tag,
            )
            if not step_to_video:
                print(
                    "Error: no matching videos found with pattern "
                    f"validation_step_*_inference_steps_{args.inference_steps_tag}_video_{idx}.mp4 "
                    f"under {args.video_root}"
                )
                sys.exit(1)
            idx_to_step_to_video[idx] = step_to_video

        requested_steps = _parse_train_steps(args.train_steps)
        if requested_steps:
            eval_step_set = set(requested_steps)
            for idx in validation_indices:
                idx_steps = set(idx_to_step_to_video[idx].keys())
                missing = sorted(eval_step_set - idx_steps)
                if missing:
                    print(f"Warning: idx {idx} missing requested steps: {missing}")
                eval_step_set &= idx_steps
            eval_steps = sorted(eval_step_set)
        else:
            eval_step_set = set(idx_to_step_to_video[validation_indices[0]].keys())
            for idx in validation_indices[1:]:
                eval_step_set &= set(idx_to_step_to_video[idx].keys())
            eval_steps = sorted(eval_step_set)
            stride = max(int(args.step_stride), 1)
            eval_steps = [s for s in eval_steps if s % stride == 0]

        if not eval_steps:
            print("Error: no steps selected for evaluation after filtering")
            sys.exit(1)

        print(
            f"Evaluating {len(eval_steps)} shared step(s) across {len(validation_indices)} indices: "
            f"{eval_steps}"
        )
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        summaries: List[Dict] = []
        for step in eval_steps:
            for idx in validation_indices:
                gen_video = idx_to_step_to_video[idx][step]
                action_path = idx_to_action_path[idx]
                pair_output = output_root / f"idx_{idx}" / f"step_{step}"
                print(f"\n{'='*60}")
                print(f"Synthetic evaluation at training step {step}, idx {idx}")
                print(f"Gen video: {gen_video}")
                print(f"Output: {pair_output}")
                print(f"{'='*60}")
                summary = evaluate_pair_synthetic(
                    gen_video=gen_video,
                    action_path=action_path,
                    calibration_path=args.calibration,
                    output_dir=str(pair_output),
                    model_name=args.model,
                    ckpt=args.ckpt,
                    grid_size=args.grid_size,
                    no_viz=args.no_viz,
                    use_depth=not args.no_depth,
                )
                if summary:
                    summary["train_step"] = step
                    summary["validation_idx"] = idx
                    summaries.append(summary)

        if summaries:
            if args.merge_with_existing_output:
                merged = _load_existing_summaries_from_output(output_root)
                if merged:
                    summaries = merged
                    print(
                        f"Loaded {len(summaries)} summaries from existing output tree under {output_root}"
                    )
                else:
                    print(
                        f"Warning: --merge_with_existing_output is set but no existing summaries found under {output_root}"
                    )

            # De-duplicate by (train_step, validation_idx), prefer later entries.
            deduped: Dict[Tuple[int, int], Dict] = {}
            for row in summaries:
                key = (int(row["train_step"]), int(row["validation_idx"]))
                deduped[key] = row
            summaries = list(deduped.values())
            summaries = sorted(
                summaries,
                key=lambda x: (x["train_step"], x["validation_idx"]),
            )
            json_path = output_root / "all_summaries.json"
            with open(json_path, "w") as f:
                json.dump(summaries, f, indent=2, default=str)
            print(f"\nWrote aggregated summaries: {json_path}")

            scalar_keys = [
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
            csv_path = output_root / "all_summaries.csv"
            with open(csv_path, "w", newline="") as f:
                fieldnames = ["train_step", "validation_idx"] + scalar_keys
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in summaries:
                    writer.writerow({k: row.get(k) for k in fieldnames})
            print(f"Wrote aggregated table: {csv_path}")

            # Aggregate averages across validation indices for each train step.
            step_avg_rows = []
            std_keys = [f"{k}_std" for k in scalar_keys]
            aggregate_steps = sorted({int(r["train_step"]) for r in summaries})
            for step in aggregate_steps:
                rows = [r for r in summaries if r["train_step"] == step]
                if not rows:
                    continue
                avg_row: Dict[str, float | int] = {
                    "train_step": step,
                    "num_indices": len(rows),
                }
                for k in scalar_keys:
                    vals = [r[k] for r in rows if r.get(k) is not None]
                    if vals:
                        avg_row[k] = float(np.mean(vals))
                        avg_row[f"{k}_std"] = float(np.std(vals))
                    else:
                        avg_row[k] = None
                        avg_row[f"{k}_std"] = None
                step_avg_rows.append(avg_row)

            step_avg_json = output_root / "step_average_summaries.json"
            with open(step_avg_json, "w") as f:
                json.dump(step_avg_rows, f, indent=2, default=str)
            print(f"Wrote step-average summaries: {step_avg_json}")

            step_avg_csv = output_root / "step_average_summaries.csv"
            with open(step_avg_csv, "w", newline="") as f:
                fieldnames = ["train_step", "num_indices"] + scalar_keys + std_keys
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in step_avg_rows:
                    writer.writerow({k: row.get(k) for k in fieldnames})
            print(f"Wrote step-average table: {step_avg_csv}")

            trend_fig = _plot_step_metric_trends(
                step_avg_rows=step_avg_rows,
                scalar_keys=scalar_keys,
                output_root=output_root,
            )
            if trend_fig is not None:
                print(f"Wrote metric trend figure: {trend_fig}")

            # Additional trend table with step-to-step deltas (based on step averages).
            trend_path = output_root / "score_change_trend.csv"
            delta_keys = [f"{k}_delta" for k in scalar_keys]
            with open(trend_path, "w", newline="") as f:
                fieldnames = ["train_step", "num_indices"] + scalar_keys + delta_keys
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                prev_row = None
                for row in step_avg_rows:
                    out_row = {k: row.get(k) for k in ["train_step", "num_indices"] + scalar_keys}
                    for k in scalar_keys:
                        cur = row.get(k)
                        prev = None if prev_row is None else prev_row.get(k)
                        out_row[f"{k}_delta"] = (
                            None if prev is None or cur is None else (cur - prev)
                        )
                    writer.writerow(out_row)
                    prev_row = row
            print(f"Wrote score-change trend: {trend_path}")

            # Pick best steps from step-average table for all scalar metrics.
            metric_rules: Dict[str, str] = {}
            for metric in scalar_keys:
                if metric == "mf_cosine_mean":
                    metric_rules[metric] = "max"
                elif metric == "mf_mag_ratio_mean":
                    metric_rules[metric] = "closest_to_1"
                else:
                    metric_rules[metric] = "min"

            best_by_metric: Dict[str, Dict[str, float | int | str]] = {}
            for metric, rule in metric_rules.items():
                valid_rows = [r for r in step_avg_rows if r.get(metric) is not None]
                if not valid_rows:
                    continue
                if rule == "max":
                    best_row = max(valid_rows, key=lambda x: x[metric])
                elif rule == "closest_to_1":
                    best_row = min(valid_rows, key=lambda x: abs(x[metric] - 1.0))
                else:
                    best_row = min(valid_rows, key=lambda x: x[metric])
                best_by_metric[metric] = {
                    "train_step": int(best_row["train_step"]),
                    "metric_value": float(best_row[metric]),
                    "selection_rule": rule,
                    "num_indices": int(best_row["num_indices"]),
                }

            best_all_info = {
                "validation_indices": sorted({int(r["validation_idx"]) for r in summaries}),
                "num_steps": len(step_avg_rows),
                "metrics": best_by_metric,
            }
            best_all_path = output_root / "best_steps_by_metric.json"
            with open(best_all_path, "w") as f:
                json.dump(best_all_info, f, indent=2, default=str)
            print(f"Wrote best-steps-by-metric summary: {best_all_path}")

            # Backward-compatible single metric summary.
            best_metric = args.best_metric
            if best_metric in best_by_metric:
                best_info = {
                    "best_metric": best_metric,
                    "selection_rule": best_by_metric[best_metric]["selection_rule"],
                    "best_train_step": best_by_metric[best_metric]["train_step"],
                    "best_metric_value": best_by_metric[best_metric]["metric_value"],
                    "num_indices": best_by_metric[best_metric]["num_indices"],
                    "validation_indices": best_all_info["validation_indices"],
                }
                best_path = output_root / "best_step.json"
                with open(best_path, "w") as f:
                    json.dump(best_info, f, indent=2, default=str)
                print(f"Wrote best-step summary: {best_path}")
            else:
                print(f"Warning: metric '{best_metric}' not found in step-average rows; skip best-step selection.")
    elif args.gt_video:
        if not args.gen_video:
            print("Error: --gen_video is required when using --gt_video")
            sys.exit(1)
        evaluate_pair(
            gt_video=args.gt_video,
            gen_video=args.gen_video,
            output_dir=args.output_dir,
            model_name=args.model,
            ckpt=args.ckpt,
            grid_size=args.grid_size,
            no_viz=args.no_viz,
        )
    else:
        evaluate_scenario(
            scenario_dir=args.scenario_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            ckpt=args.ckpt,
            grid_size=args.grid_size,
            no_viz=args.no_viz,
        )


if __name__ == "__main__":
    main()
