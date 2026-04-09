#!/usr/bin/env python3
from pathlib import Path
import pickle

import numpy as np
import imageio
import cv2


class TactileVideo:
    def __init__(self, root, key="tactile_data", grid=(4, 8)):
        self.root = Path(root)
        self.key = key
        self.rows, self.cols = grid
        self.files = sorted(self.root.rglob("*.pkl"))

        # store (x, y, z)
        self.traj = []
        self.xlim = None
        self.ylim = None
        self.zlim = None

    def set_ee_limits(self, xlim, ylim, zlim):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

    # ---- tactile heatmap helpers ----
    def _pick_channel(self, tact_nx3: np.ndarray, channel: str) -> np.ndarray:
        if channel == "x":
            return tact_nx3[:, 0]
        if channel == "y":
            return tact_nx3[:, 1]
        return tact_nx3[:, 2]

    def _to_heatmap(self, grid_4x4: np.ndarray, vmin: float, vmax: float, cell_px: int, cmap: int) -> np.ndarray:
        denom = max(1e-6, (vmax - vmin))
        norm = (grid_4x4 - vmin) / denom
        norm = np.clip(norm, 0.0, 1.0)
        img_u8 = (norm * 255.0).astype(np.uint8)
        img_u8 = cv2.resize(img_u8, (4 * cell_px, 4 * cell_px), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(img_u8, cmap)
        return heat

    # ---- ee_pos_quat limits (fallback per-folder) ----
    def _compute_ee_limits(self):
        xs, ys, zs = [], [], []
        for p in self.files:
            try:
                with open(p, "rb") as f:
                    data = pickle.load(f)
            except Exception:
                continue
            ee = np.asarray(data.get("ee_pos_quat", []), dtype=np.float32)
            if ee.ndim >= 1 and ee.size >= 3:
                x, y, z = float(ee[0]), float(ee[1]), float(ee[2])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    xs.append(x); ys.append(y); zs.append(z)

        if not xs:
            print(f"[{self.root.name}][EE] no ee_pos_quat → auto limits")
            self.xlim = self.ylim = self.zlim = None
            return

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)

        dx = (xmax - xmin) or 1e-3
        dy = (ymax - ymin) or 1e-3
        dz = (zmax - zmin) or 1e-3

        self.xlim = (xmin - 0.05 * dx, xmax + 0.05 * dx)
        self.ylim = (ymin - 0.05 * dy, ymax + 0.05 * dy)
        self.zlim = (zmin - 0.05 * dz, zmax + 0.05 * dz)
        print(f"[{self.root.name}][EE] xlim={self.xlim}, ylim={self.ylim}, zlim={self.zlim}")

    # ---- trajectory panel ----
    def _traj_panel(self, panel_w: int, panel_h: int, plane: str = "xy") -> np.ndarray:
        """
        Draw EE trajectory into a BGR image.
        plane="xy" uses (x,y); plane="xz" uses (x,z).
        Uses fixed limits if self.xlim/... are set.
        """
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

        if not self.traj:
            cv2.putText(panel, "No ee_pos_quat", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            return panel

        if plane == "xy":
            idx_a, idx_b = 0, 1
            lim_a, lim_b = self.xlim, self.ylim
            label = "EE traj (x,y)"
            a_name, b_name = "x", "y"
        elif plane == "xz":
            idx_a, idx_b = 0, 2
            lim_a, lim_b = self.xlim, self.zlim
            label = "EE traj (x,z)"
            a_name, b_name = "x", "z"
        else:
            raise ValueError("plane must be 'xy' or 'xz'")

        # Extract finite points from (x,y,z)
        pts_world = [(p[idx_a], p[idx_b]) for p in self.traj
                     if np.isfinite(p[idx_a]) and np.isfinite(p[idx_b])]
        if len(pts_world) == 0:
            cv2.putText(panel, f"No {label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            return panel

        # Limits
        if lim_a is None or lim_b is None:
            as_, bs_ = zip(*pts_world)
            amin, amax = min(as_), max(as_)
            bmin, bmax = min(bs_), max(bs_)
        else:
            amin, amax = lim_a
            bmin, bmax = lim_b

        da = (amax - amin) if (amax - amin) != 0 else 1e-6
        db = (bmax - bmin) if (bmax - bmin) != 0 else 1e-6

        margin = 25
        x0, y0 = margin, margin
        x1, y1 = panel_w - margin, panel_h - margin

        cv2.rectangle(panel, (x0, y0), (x1, y1), (160, 160, 160), 1)

        def to_px(a, b):
            u = int(x0 + (a - amin) / da * (x1 - x0))
            v = int(y1 - (b - bmin) / db * (y1 - y0))
            return u, v

        pts = np.array([to_px(a, b) for (a, b) in pts_world], dtype=np.int32)

        if len(pts) >= 2:
            cv2.polylines(panel, [pts], isClosed=False, color=(255, 255, 255), thickness=2)
        cv2.circle(panel, tuple(pts[-1]), 4, (0, 255, 0), -1)

        cv2.putText(panel, label, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(panel, f"{a_name}[{amin:.3f},{amax:.3f}]", (10, panel_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{b_name}[{bmin:.3f},{bmax:.3f}]", (10, panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        return panel

    # ---- video ----
    def make_video(self, out_path, vmin, vmax, fps=10):
        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            try:
                out_path.unlink()
                print(f"[VIDEO] removed existing {out_path}")
            except PermissionError:
                print(f"[VIDEO][ERR] cannot delete {out_path} (open in another app?)")
                return

        if not self.files:
            print(f"[VIDEO] no .pkl files under {self.root}")
            return

        # fallback per-folder limits only if not preset by batch
        if self.xlim is None or self.ylim is None or self.zlim is None:
            self._compute_ee_limits()

        self.traj = []

        print(f"[VIDEO] {self.root.name} → {out_path} ({len(self.files)} frames)")
        with imageio.get_writer(str(out_path), fps=fps, macro_block_size=1) as w:
            for i, p in enumerate(self.files, 1):
                try:
                    frame = self._frame_from_file(p, vmin, vmax)
                except Exception as e:
                    print(f"[skip] {p}: {e}")
                    continue
                w.append_data(frame)
                if i == 1 or i % 100 == 0:
                    print(f"  {i}/{len(self.files)}")
        print(f"[VIDEO] done: {out_path}")

    # ---- single frame: heatmap + XY traj + XZ traj ----
    def _frame_from_file(self, pkl_path, vmin, vmax,
                         channel="z",
                         cell_px=60,
                         cmap=cv2.COLORMAP_TURBO,
                         show_g2=True,
                         swap_g1_g2=False,
                         autoscale=False):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        tact = np.asarray(data[self.key], dtype=np.float32)

        ee = np.asarray(data.get("ee_pos_quat", []), dtype=np.float32)
        if ee.ndim >= 1 and ee.size >= 3:
            x, y, z = float(ee[0]), float(ee[1]), float(ee[2])
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                self.traj.append((x, y, z))

        expected = (self.rows * self.cols, 3)  # (32,3) for 4x8
        if tact.shape != expected:
            raise ValueError(f"{pkl_path.name}: {tact.shape}, expected {expected}")

        # split into two 4x4 pads
        gA = tact[:16]
        gB = tact[16:]

        # Client renders: left = g2, right = g1
        if swap_g1_g2:
            g1, g2 = gA, gB
        else:
            g2, g1 = gA, gB

        ch1 = self._pick_channel(g1, channel)
        ch2 = self._pick_channel(g2, channel)

        if autoscale:
            vals = ch1 if not show_g2 else np.concatenate([ch1, ch2])
            vmin = float(np.percentile(vals, 5))
            vmax = float(np.percentile(vals, 95))
            if abs(vmax - vmin) < 1e-6:
                vmax = vmin + 1.0

        heat1 = self._to_heatmap(ch1.reshape(4, 4), vmin, vmax, cell_px, cmap)
        heat1 = cv2.flip(heat1, 0)

        heat2 = None
        if show_g2:
            heat2 = self._to_heatmap(ch2.reshape(4, 4), vmin, vmax, cell_px, cmap)
            heat2 = cv2.flip(heat2, 0)

        # Build canvas
        width_cells = 8 if show_g2 else 4
        canvas = np.zeros((4 * cell_px, width_cells * cell_px, 3), dtype=np.uint8)

        if show_g2 and heat2 is not None:
            canvas[:, :4 * cell_px, :] = heat2
            canvas[:, 4 * cell_px:8 * cell_px, :] = heat1
        else:
            canvas[:, :4 * cell_px, :] = heat1

        # Two panels: XY and XZ
        panel_w = 4 * cell_px
        panel_h = 4 * cell_px
        traj_xy = self._traj_panel(panel_w=panel_w, panel_h=panel_h, plane="xy")
        traj_xz = self._traj_panel(panel_w=panel_w, panel_h=panel_h, plane="xz")

        out_bgr = np.hstack([canvas, traj_xy, traj_xz])

        # OpenCV is BGR; imageio expects RGB
        frame_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb


def compute_global_ee_limits(in_root: Path, pad_frac: float = 0.05):
    """
    Scan ALL .pkl under in_root recursively and compute global x/y/z limits.
    """
    xs, ys, zs = [], [], []
    for p in sorted(in_root.rglob("*.pkl")):
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        ee = np.asarray(data.get("ee_pos_quat", []), dtype=np.float32)
        if ee.ndim >= 1 and ee.size >= 3:
            x, y, z = float(ee[0]), float(ee[1]), float(ee[2])
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                xs.append(x); ys.append(y); zs.append(z)

    if not xs:
        return None, None, None

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    dx = (xmax - xmin) or 1e-3
    dy = (ymax - ymin) or 1e-3
    dz = (zmax - zmin) or 1e-3

    xlim = (xmin - pad_frac * dx, xmax + pad_frac * dx)
    ylim = (ymin - pad_frac * dy, ymax + pad_frac * dy)
    zlim = (zmin - pad_frac * dz, zmax + pad_frac * dz)
    return xlim, ylim, zlim


def process_all_folders(
    base_data_dir,
    rel_in,
    rel_out,
    key="tactile_data",
    grid=(4, 8),
    fps=10,
    fixed_vmin=0.0,
    fixed_vmax=150.0,
):
    base = Path(base_data_dir)
    in_root = Path(rel_in)
    out_root = (base / rel_out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[BATCH] input:  {in_root}")
    print(f"[BATCH] output: {out_root}")

    # Global fixed xyz limits for comparability across ALL videos
    xlim, ylim, zlim = compute_global_ee_limits(in_root)
    print(f"[BATCH][EE GLOBAL] xlim={xlim}, ylim={ylim}, zlim={zlim}")

    # Iterate immediate subfolders (same behavior as your original code)
    for sub in sorted(in_root.iterdir()):
        if not sub.is_dir():
            continue
        tv = TactileVideo(sub, key=key, grid=grid)
        if not tv.files:
            print(f"[BATCH] skip {sub.name}: no .pkl files")
            continue

        if xlim is not None:
            tv.set_ee_limits(xlim, ylim, zlim)

        out_video = out_root / f"{sub.name}.mp4"
        tv.make_video(out_video, vmin=fixed_vmin, vmax=fixed_vmax, fps=fps)


if __name__ == "__main__":
    base_dir = Path.cwd()

    process_all_folders(
        base_data_dir=base_dir,
        rel_in="/home/qcrvgs/structured_slowspeed/gello",
        rel_out="data/video_1/",
        key="tactile_data",
        grid=(4, 8),
        fps=10,
        fixed_vmin=0.0,
        fixed_vmax=150.0,
    )