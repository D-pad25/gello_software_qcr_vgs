#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def make_odd_window(n: int, preferred: int) -> int:
    w = min(preferred, n if n % 2 == 1 else n - 1)
    if w < 5:
        w = 5 if n >= 5 else n
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def smooth_and_differentiate(values: np.ndarray, t: np.ndarray, window: int = 21, polyorder: int = 3):
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)

    n = len(t)
    if n < 5:
        vel = finite_diff(values, t)
        acc = finite_diff(vel, t)
        return values.copy(), vel, acc

    dt = np.diff(t)
    mean_dt = float(np.mean(dt))

    # if timestamps are weird, fall back
    if mean_dt <= 0 or not np.isfinite(mean_dt):
        vel = finite_diff(values, t)
        acc = finite_diff(vel, t)
        return values.copy(), vel, acc

    w = make_odd_window(n, window)
    if w <= polyorder:
        w = polyorder + 2
        if w % 2 == 0:
            w += 1
        if w > n:
            w = n if n % 2 == 1 else n - 1

    pos_smooth = savgol_filter(values, window_length=w, polyorder=polyorder, axis=0, mode="interp")
    vel_smooth = savgol_filter(values, window_length=w, polyorder=polyorder, deriv=1, delta=mean_dt, axis=0, mode="interp")
    acc_smooth = savgol_filter(values, window_length=w, polyorder=polyorder, deriv=2, delta=mean_dt, axis=0, mode="interp")

    return pos_smooth, vel_smooth, acc_smooth



def load_pkl(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a dict")
    return data


def get_timestamp(data: dict[str, Any], fallback_idx: int | None = None) -> float | None:
    if fallback_idx is not None:
        return float(fallback_idx)
    return None


def sorted_pkl_files(demo_dir: Path) -> list[Path]:
    return sorted([p for p in demo_dir.rglob("*.pkl") if p.is_file()])


def finite_diff(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    if values.shape[0] < 2:
        return np.zeros_like(values)
    if values.shape[0] != t.shape[0]:
        raise ValueError("values and time must have same length")
    out = np.zeros_like(values)
    for d in range(values.shape[1]):
        out[:, d] = np.gradient(values[:, d], t, edge_order=1)
    return out


def choose_joint_positions(data: dict[str, Any], prefer_control_joint: bool = True) -> np.ndarray:
    if prefer_control_joint and "control_joint" in data:
        q = np.asarray(data["control_joint"], dtype=float).reshape(-1)
        if q.size > 0:
            return q
    if "joint_positions" in data:
        q = np.asarray(data["joint_positions"], dtype=float).reshape(-1)
        if q.size > 0:
            return q
    raise KeyError("No control_joint or joint_positions found")


def choose_logged_joint_velocity(data: dict[str, Any], target_dim: int) -> np.ndarray | None:
    if "joint_velocities" not in data:
        return None
    dq = np.asarray(data["joint_velocities"], dtype=float).reshape(-1)
    if dq.size == target_dim:
        return dq
    if dq.size > target_dim:
        return dq[:target_dim]
    return None


def choose_ee_position(data: dict[str, Any]) -> np.ndarray:
    if "ee_pos_quat" not in data:
        raise KeyError("No ee_pos_quat found")
    ee = np.asarray(data["ee_pos_quat"], dtype=float).reshape(-1)
    if ee.size < 3:
        raise ValueError("ee_pos_quat has fewer than 3 values")
    return ee[:3]


def load_demo(demo_dir: Path, prefer_control_joint: bool = True, hz: float = 60.0) -> dict[str, Any]:
    files = sorted_pkl_files(demo_dir)
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {demo_dir}")

    times = []
    joint_pos_list = []
    obs_joint_pos_list = []
    logged_joint_vel_list = []
    ee_pos_list = []

    for i, p in enumerate(files):
        data = load_pkl(p)
        times.append(get_timestamp(data, fallback_idx=i))
        q = choose_joint_positions(data, prefer_control_joint=prefer_control_joint)
        joint_pos_list.append(q)
        # always load joint_positions (robot observation) for comparison
        if "joint_positions" in data:
            obs_q = np.asarray(data["joint_positions"], dtype=float).reshape(-1)
            obs_joint_pos_list.append(obs_q if obs_q.size > 0 else q)
        else:
            obs_joint_pos_list.append(q)
        ee_pos_list.append(choose_ee_position(data))
        logged_joint_vel_list.append(choose_logged_joint_velocity(data, target_dim=q.size))

    joint_dim = len(joint_pos_list[0])
    if any(len(q) != joint_dim for q in joint_pos_list):
        raise ValueError("Inconsistent joint dimension across files")

    t = np.arange(len(files), dtype=float) / hz

    joint_pos = np.vstack(joint_pos_list)
    obs_joint_pos = np.vstack(obs_joint_pos_list)
    ee_pos = np.vstack(ee_pos_list)

    joint_pos_smooth = joint_pos.copy()
    joint_vel_from_pos = finite_diff(joint_pos, t)
    joint_acc = finite_diff(joint_vel_from_pos, t)

    obs_joint_vel = finite_diff(obs_joint_pos, t)
    obs_joint_acc = finite_diff(obs_joint_vel, t)

    ee_pos_smooth = ee_pos.copy()
    ee_vel = finite_diff(ee_pos, t)
    ee_acc = finite_diff(ee_vel, t)

    has_logged_joint_vel = all(v is not None for v in logged_joint_vel_list)
    logged_joint_vel = None
    logged_vel_matches_pos = False
    if has_logged_joint_vel:
        logged_joint_vel = np.vstack(logged_joint_vel_list)
        if logged_joint_vel.shape == joint_pos.shape:
            logged_vel_matches_pos = np.allclose(logged_joint_vel, joint_pos)

    return {
        "files": files,
        "t": t,
        "joint_pos": joint_pos_smooth,
        "joint_vel": joint_vel_from_pos,
        "joint_acc": joint_acc,
        "obs_joint_vel": obs_joint_vel,
        "obs_joint_acc": obs_joint_acc,
        "ee_pos": ee_pos_smooth,
        "ee_vel": ee_vel,
        "ee_acc": ee_acc,
        "logged_joint_vel": logged_joint_vel,
        "logged_vel_matches_pos": logged_vel_matches_pos,
    }


def plot_matrix(t: np.ndarray, y: np.ndarray, title: str, ylabel: str, labels: list[str], out_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(y.shape[1]):
        ax.plot(t, y[:, i], label=labels[i], linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(4, len(labels)), fontsize=9)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_vel_comparison(t: np.ndarray, ctrl_vel: np.ndarray, obs_vel: np.ndarray, labels: list[str], out_path: Path | None = None):
    n_joints = ctrl_vel.shape[1]
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5 * n_joints), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for i, ax in enumerate(axes):
        ax.plot(t, ctrl_vel[:, i], label="control_joint vel", linewidth=1.2)
        ax.plot(t, obs_vel[:, i], label="joint_positions vel", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Joint velocity: control_joint vs joint_positions (obs)", y=1.01)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def save_all_plots(results: dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = results["t"]
    joint_pos = results["joint_pos"]
    joint_vel = results["joint_vel"]
    obs_joint_vel = results["obs_joint_vel"]
    obs_joint_acc = results["obs_joint_acc"]
    ee_pos = results["ee_pos"]
    ee_vel = results["ee_vel"]
    ee_acc = results["ee_acc"]

    joint_labels = [f"j{i+1}" for i in range(joint_pos.shape[1])]
    obs_joint_labels = [f"j{i+1}" for i in range(obs_joint_vel.shape[1])]
    ee_labels = ["x", "y", "z"]

    figs = []
    figs.append(plot_matrix(t, joint_pos, "Joint position vs time", "joint position [rad]", joint_labels, out_dir / "joint_position.png"))
    figs.append(plot_matrix(t, obs_joint_vel, "Joint velocity vs time (from joint_positions)", "joint velocity [rad/s]", obs_joint_labels, out_dir / "joint_velocity.png"))
    figs.append(plot_vel_comparison(t, joint_vel, obs_joint_vel[:, :joint_vel.shape[1]], joint_labels, out_dir / "joint_velocity_comparison.png"))
    figs.append(plot_matrix(t, obs_joint_acc, "Joint acceleration vs time (from joint_positions)", "joint acceleration [rad/s^2]", obs_joint_labels, out_dir / "joint_acceleration.png"))
    figs.append(plot_matrix(t, ee_pos, "End-effector position vs time", "position [m]", ee_labels, out_dir / "ee_position.png"))
    figs.append(plot_matrix(t, ee_vel, "End-effector velocity vs time", "velocity [m/s]", ee_labels, out_dir / "ee_velocity.png"))
    figs.append(plot_matrix(t, ee_acc, "End-effector acceleration vs time", "acceleration [m/s^2]", ee_labels, out_dir / "ee_acceleration.png"))

    if results["logged_joint_vel"] is not None:
        figs.append(
            plot_matrix(
                t,
                results["logged_joint_vel"],
                "Logged joint_velocities field vs time",
                "logged joint velocity",
                joint_labels,
                out_dir / "joint_velocity_logged.png",
            )
        )

    return figs


def main():
    parser = argparse.ArgumentParser(description="Plot joint and EE kinematics from one demo folder of .pkl files.")
    parser.add_argument("demo_dir", type=Path, nargs="?", default="/home/qcrvgs/structured_slowspeed/gello/0326_180622/")
    parser.add_argument("--out-dir", type=Path, default="kenimatics_plots", help="Directory to save png plots. Default: <demo_dir>/plots")
    parser.add_argument(
        "--use-joint-positions",
        action="store_true",
        help="Use joint_positions instead of control_joint. By default, control_joint is used when available.",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Recording frequency in Hz (default: 60)")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    args = parser.parse_args()

    prefer_control_joint = not args.use_joint_positions
    results = load_demo(args.demo_dir, prefer_control_joint=prefer_control_joint, hz=args.hz)

    out_dir = args.out_dir if args.out_dir is not None else args.demo_dir / "plots"
    save_all_plots(results, out_dir)

    dt = np.diff(results["t"])
    mean_dt = float(np.mean(dt)) if len(dt) > 0 else float("nan")
    mean_hz = (1.0 / mean_dt) if len(dt) > 0 and mean_dt > 0 else float("nan")

    print(f"Loaded {len(results['files'])} frames from {args.demo_dir}")
    print(f"Saved plots to: {out_dir}")
    print(f"Joint dimension: {results['joint_pos'].shape[1]}")
    print(f"Mean dt: {mean_dt:.6f} s")
    print(f"Mean rate: {mean_hz:.3f} Hz")
    if results["logged_joint_vel"] is not None:
        print("logged joint_velocities field found")
        if results["logged_vel_matches_pos"]:
            print("WARNING: logged joint_velocities is numerically identical to joint positions in this demo")
            print("         so the plotted joint velocity uses finite differences of position instead")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
