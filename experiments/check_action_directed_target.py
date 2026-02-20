#!/usr/bin/env python3
"""
Episode-level direction-to-target similarity using MOTION (velocity) from end-effector positions.

Each demo subfolder = one episode, containing many *.pkl (one per timestep).

Target for the episode:
  tgt = ee position from LAST pkl in the folder

For each timestep pair (t, t+1):
  pos_t  = ee_pos(t)
  pos_t1 = ee_pos(t+1)
  v      = pos_t1 - pos_t
  v_hat  = unit(v)
  g_hat  = unit(tgt - pos_t)

Similarity:
  dot_t   = v_hat · g_hat           in [-1, 1]
  angle_t = arccos(dot_t) (deg)     in [0, 180]

Per episode summaries:
  - median_angle_deg
  - mean_angle_deg
  - %dot>0
  - %dot>cos(angle_deg_thr)

Also records skip breakdown:
  - skipped_zero_vel (v_norm <= speed_eps)
  - skipped_goal_zero (g_norm ~ 0)
  - skipped_dist_eps (if dist_eps>0)
  - failed (exceptions)

Outputs:
  - episode_metrics.csv
  - boxplots: angles and %zero-velocity pairs
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# ---- pickle compat (numpy._core -> numpy.core) ----
class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pkl(path: Path) -> Any:
    with path.open("rb") as f:
        return _NumpyCompatUnpickler(f).load()


def get_array(obj: Any, key: str) -> np.ndarray:
    if isinstance(obj, dict):
        if key not in obj:
            raise KeyError(f"Missing key '{key}' (keys={list(obj)[:20]}...)")
        return np.asarray(obj[key])
    if hasattr(obj, key):
        return np.asarray(getattr(obj, key))
    raise KeyError(f"Missing key/attr '{key}' in {type(obj)}")


def _as_xyz(v: np.ndarray) -> np.ndarray:
    """
    Convert a stored pose-ish vector to xyz.
    - Accepts 1D (>=3) or (1,D) and takes first 3.
    NOTE: If your pose is quaternion-first (qw,qx,qy,qz,x,y,z), change slicing to v[-3:].
    """
    v = np.asarray(v)
    if v.ndim == 2 and v.shape[0] == 1:
        v = v[0]
    v = np.squeeze(v)
    if v.ndim != 1 or v.shape[0] < 3:
        raise ValueError(f"Expected vector with >=3 elems, got {v.shape}")
    return v[:3].astype(float)


def _unit(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v, dtype=float), 0.0
    return (v / n).astype(float), n


@dataclass
class EpisodeMetrics:
    name: str
    n_files: int
    n_pairs_total: int

    used: int
    skip_zero_vel: int
    skip_goal_zero: int
    skip_dist_eps: int
    failed: int

    pct_zero_vel_of_all_pairs: float
    pct_zero_vel_of_skips: float

    median_angle_deg: float
    mean_angle_deg: float
    pct_dot_gt_0: float
    pct_dot_gt_thr: float


def _save_boxplot(values: np.ndarray, outpath: Path, ylabel: str, title: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.boxplot(values, showmeans=True)
    plt.xticks([1], ["Episodes"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def episode_metrics_from_folder(
    folder: Path,
    *,
    pos_key: str,
    pattern: str,
    speed_eps: float,
    dist_eps: float,
    angle_deg_thr: float,
) -> EpisodeMetrics:
    paths = sorted(folder.glob(pattern))
    n_files = len(paths)
    n_pairs_total = max(0, n_files - 1)

    # default empty metrics
    def _empty() -> EpisodeMetrics:
        return EpisodeMetrics(
            name=folder.name,
            n_files=n_files,
            n_pairs_total=n_pairs_total,
            used=0,
            skip_zero_vel=0,
            skip_goal_zero=0,
            skip_dist_eps=0,
            failed=0,
            pct_zero_vel_of_all_pairs=float("nan"),
            pct_zero_vel_of_skips=float("nan"),
            median_angle_deg=float("nan"),
            mean_angle_deg=float("nan"),
            pct_dot_gt_0=float("nan"),
            pct_dot_gt_thr=float("nan"),
        )

    if n_files < 2:
        return _empty()

    # target from last pkl
    try:
        obj_last = load_pkl(paths[-1])
        tgt = _as_xyz(get_array(obj_last, pos_key))
    except Exception as e:
        m = _empty()
        m.failed = n_pairs_total
        print(f"[WARN] {folder.name}: failed to read target from last pkl: {paths[-1].name} ({e})")
        return m

    cos_thr = float(np.cos(np.deg2rad(angle_deg_thr)))

    dots: list[float] = []
    angles: list[float] = []

    used = 0
    failed = 0
    skip_zero_vel = 0
    skip_goal_zero = 0
    skip_dist_eps = 0

    for p_t, p_t1 in tqdm(list(zip(paths[:-1], paths[1:])), desc=f"{folder.name}", leave=False):
        try:
            obj_t = load_pkl(p_t)
            obj_t1 = load_pkl(p_t1)

            pos_t = _as_xyz(get_array(obj_t, pos_key))
            pos_t1 = _as_xyz(get_array(obj_t1, pos_key))

            v = pos_t1 - pos_t
            v_hat, v_norm = _unit(v)

            g = tgt - pos_t
            g_hat, g_norm = _unit(g)

            if v_norm <= speed_eps:
                skip_zero_vel += 1
                continue
            if g_norm <= 1e-12:
                skip_goal_zero += 1
                continue
            if dist_eps > 0.0 and g_norm <= dist_eps:
                skip_dist_eps += 1
                continue

            dot = float(np.clip(np.dot(v_hat, g_hat), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(dot)))

            dots.append(dot)
            angles.append(angle_deg)
            used += 1

        except Exception:
            failed += 1
            continue

    n_skipped_total = skip_zero_vel + skip_goal_zero + skip_dist_eps

    if n_pairs_total > 0:
        pct_zero_vel_of_all_pairs = 100.0 * (skip_zero_vel / n_pairs_total)
    else:
        pct_zero_vel_of_all_pairs = float("nan")

    if n_skipped_total > 0:
        pct_zero_vel_of_skips = 100.0 * (skip_zero_vel / n_skipped_total)
    else:
        pct_zero_vel_of_skips = float("nan")

    if used == 0:
        median_angle = mean_angle = pct0 = pctthr = float("nan")
    else:
        dots_np = np.asarray(dots, float)
        ang_np = np.asarray(angles, float)
        median_angle = float(np.median(ang_np))
        mean_angle = float(np.mean(ang_np))
        pct0 = float(np.mean(dots_np > 0.0) * 100.0)
        pctthr = float(np.mean(dots_np > cos_thr) * 100.0)

    print(
        f"{folder.name:30s} pairs={n_pairs_total:5d} used={used:5d} "
        f"skip_v0={skip_zero_vel:5d} ({pct_zero_vel_of_all_pairs:6.2f}%) "
        f"skip_g0={skip_goal_zero:5d} skip_dist={skip_dist_eps:5d} fail={failed:5d} "
        f"med_ang={median_angle:7.2f} mean_ang={mean_angle:7.2f} %dot>0={pct0:6.2f} %dot>thr={pctthr:6.2f}"
    )

    return EpisodeMetrics(
        name=folder.name,
        n_files=n_files,
        n_pairs_total=n_pairs_total,
        used=used,
        skip_zero_vel=skip_zero_vel,
        skip_goal_zero=skip_goal_zero,
        skip_dist_eps=skip_dist_eps,
        failed=failed,
        pct_zero_vel_of_all_pairs=pct_zero_vel_of_all_pairs,
        pct_zero_vel_of_skips=pct_zero_vel_of_skips,
        median_angle_deg=median_angle,
        mean_angle_deg=mean_angle,
        pct_dot_gt_0=pct0,
        pct_dot_gt_thr=pctthr,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/home/qcrvgs/test_100hz_withupdate/gello"),
    )
    ap.add_argument("--outdir", type=Path, default=Path("data_analyses/dec19_tact_angle_summary"))
    ap.add_argument("--pattern", default="*.pkl")
    ap.add_argument("--pos_key", default="ee_pos_quat")

    ap.add_argument("--speed_eps", type=float, default=1e-8, help="ignore pairs with ||velocity|| <= this")
    ap.add_argument("--dist_eps", type=float, default=0.0, help="if >0, ignore pairs with ||target-pos|| <= this")
    ap.add_argument("--angle_deg_thr", type=float, default=30.0, help="threshold angle in degrees for %dot>cos(thr)")

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    demo_dirs = sorted([d for d in args.root.iterdir() if d.is_dir() and any(d.glob(args.pattern))])
    if not demo_dirs:
        raise SystemExit(f"No demo folders with '{args.pattern}' under {args.root}")

    metrics: list[EpisodeMetrics] = []
    for d in tqdm(demo_dirs, desc="Episodes"):
        metrics.append(
            episode_metrics_from_folder(
                d,
                pos_key=args.pos_key,
                pattern=args.pattern,
                speed_eps=args.speed_eps,
                dist_eps=args.dist_eps,
                angle_deg_thr=args.angle_deg_thr,
            )
        )

    # --- save CSV with per-episode % zero-velocity skip ---
    csv_path = args.outdir / "episode_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode",
            "n_files",
            "n_pairs_total",
            "used",
            "skip_zero_vel",
            "skip_goal_zero",
            "skip_dist_eps",
            "failed",
            "pct_zero_vel_of_all_pairs",
            "pct_zero_vel_of_skips",
            "median_angle_deg",
            "mean_angle_deg",
            "pct_dot_gt_0",
            f"pct_dot_gt_cos_{int(args.angle_deg_thr)}deg",
        ])
        for m in metrics:
            w.writerow([
                m.name,
                m.n_files,
                m.n_pairs_total,
                m.used,
                m.skip_zero_vel,
                m.skip_goal_zero,
                m.skip_dist_eps,
                m.failed,
                m.pct_zero_vel_of_all_pairs,
                m.pct_zero_vel_of_skips,
                m.median_angle_deg,
                m.mean_angle_deg,
                m.pct_dot_gt_0,
                m.pct_dot_gt_thr,
            ])

    # --- plots ---
    med = np.array([m.median_angle_deg for m in metrics], float)
    mean = np.array([m.mean_angle_deg for m in metrics], float)
    pct0 = np.array([m.pct_dot_gt_0 for m in metrics], float)
    pctthr = np.array([m.pct_dot_gt_thr for m in metrics], float)
    pct_v0 = np.array([m.pct_zero_vel_of_all_pairs for m in metrics], float)

    if not np.any(np.isfinite(med)) and not np.any(np.isfinite(pct_v0)):
        raise SystemExit("No valid episodes (check pos_key / filters).")

    if np.any(np.isfinite(med)):
        _save_boxplot(
            med[np.isfinite(med)],
            args.outdir / "boxplot_episode_median_angle_deg.png",
            ylabel="Median angle (deg) between motion and goal (lower is better)",
            title="Episode-wise median direction error",
        )
        _save_boxplot(
            mean[np.isfinite(mean)],
            args.outdir / "boxplot_episode_mean_angle_deg.png",
            ylabel="Mean angle (deg) between motion and goal (lower is better)",
            title="Episode-wise mean direction error",
        )
        _save_boxplot(
            pct0[np.isfinite(pct0)],
            args.outdir / "boxplot_episode_percent_dot_gt_0.png",
            ylabel="% of pairs with dot > 0 (not moving away)",
            title="Episode-wise fraction of steps not moving away from target",
        )
        _save_boxplot(
            pctthr[np.isfinite(pctthr)],
            args.outdir / f"boxplot_episode_percent_dot_gt_cos_{int(args.angle_deg_thr)}deg.png",
            ylabel=f"% of pairs with angle <= {args.angle_deg_thr:g}° (dot > cos(thr))",
            title=f"Episode-wise fraction of steps strongly toward target (<= {args.angle_deg_thr:g}°)",
        )

    # your requested diagnostic: %pairs skipped due to zero velocity
    if np.any(np.isfinite(pct_v0)):
        _save_boxplot(
            pct_v0[np.isfinite(pct_v0)],
            args.outdir / "boxplot_episode_percent_zero_velocity_pairs.png",
            ylabel=f"% of consecutive pairs with ||velocity|| <= {args.speed_eps:g}",
            title="Episode-wise fraction of zero-velocity pairs",
        )

    print(f"\nSaved plots to: {args.outdir}")
    print(f"Saved per-episode CSV to: {csv_path}")


if __name__ == "__main__":
    main()
