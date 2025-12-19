import os
import glob
import pickle
import time

import numpy as np
import cv2

# ========= CONFIG =========
DATA_DIR = "/home/qcrvgs/gello/nidhi/gello/1218_191615/"
pattern = os.path.join(DATA_DIR, "*.pkl")
paths = sorted(glob.glob(pattern))

if not paths:
    print("No .pkl files found.")
    raise SystemExit(1)

# ---- tactile display settings (same as your client) ----
CHANNEL = "z"                 # "x" | "y" | "z"
SHOW_G2 = True
AUTOSCALE = False
VMIN = 0.0
VMAX = 150.0
CELL_PX = 60
CMAP = cv2.COLORMAP_TURBO
SHOW_COLORBAR = True


def _pick_channel(tact_16x3: np.ndarray, channel: str) -> np.ndarray:
    if channel == "x":
        return tact_16x3[:, 0]
    if channel == "y":
        return tact_16x3[:, 1]
    return tact_16x3[:, 2]


def _to_heatmap(grid_4x4: np.ndarray, vmin: float, vmax: float, cell_px: int, cmap: int) -> np.ndarray:
    denom = max(1e-6, (vmax - vmin))
    norm = (grid_4x4 - vmin) / denom
    norm = np.clip(norm, 0.0, 1.0)
    img_u8 = (norm * 255.0).astype(np.uint8)
    img_u8 = cv2.resize(img_u8, (4 * cell_px, 4 * cell_px), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(img_u8, cmap)


def _make_colorbar(height: int, width: int, cmap: int) -> np.ndarray:
    grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    return cv2.applyColorMap(grad, cmap)


def _get_tactile_from_state(state: dict):
    """
    Returns (g1_16x3, g2_16x3) as float32 arrays, or (None, None).
    Supports:
      - state["tactile_data"] with shape (32,3)
      - state["tactile"] as dict containing "g1"/"g2"
      - state["tactile"] as tuple where first item is dict containing "g1"/"g2"
    """
    # Case 1: tactile_data (32,3)
    td = state.get("tactile_data", None)
    if td is not None:
        td = np.asarray(td, dtype=np.float32)
        if td.shape == (32, 3):
            return td[:16], td[16:]

    # Case 2: tactile dict/tuple from snapshot
    t = state.get("tactile", None)
    if isinstance(t, dict):
        g1 = np.asarray(t.get("g1", []), dtype=np.float32)
        g2 = np.asarray(t.get("g2", []), dtype=np.float32)
        if g1.shape == (16, 3) and g2.shape == (16, 3):
            return g1, g2

    if isinstance(t, (tuple, list)) and len(t) > 0 and isinstance(t[0], dict):
        snap = t[0]
        g1 = np.asarray(snap.get("g1", []), dtype=np.float32)
        g2 = np.asarray(snap.get("g2", []), dtype=np.float32)
        if g1.shape == (16, 3) and g2.shape == (16, 3):
            return g1, g2

    return None, None


def _render_tactile_panel(g1: np.ndarray, g2: np.ndarray):
    """
    Returns a BGR image panel with g1 (and optionally g2) heatmaps.
    """
    w_cells = 8 if SHOW_G2 else 4
    canvas = np.zeros((4 * CELL_PX, w_cells * CELL_PX, 3), dtype=np.uint8)

    if g1 is None or (SHOW_G2 and g2 is None):
        cv2.putText(canvas, "no tactile", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    ch1 = _pick_channel(g1, CHANNEL)
    g1_grid = ch1.reshape(4, 4)

    if SHOW_G2:
        ch2 = _pick_channel(g2, CHANNEL)
        vals = np.concatenate([ch1, ch2])
    else:
        ch2 = None
        vals = ch1

    if AUTOSCALE:
        vmin = float(np.percentile(vals, 5))
        vmax = float(np.percentile(vals, 95))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
    else:
        vmin, vmax = float(VMIN), float(VMAX)

    heat1 = _to_heatmap(g1_grid, vmin, vmax, CELL_PX, CMAP)
    canvas[:, :4 * CELL_PX, :] = heat1

    if SHOW_G2 and ch2 is not None:
        g2_grid = ch2.reshape(4, 4)
        heat2 = _to_heatmap(g2_grid, vmin, vmax, CELL_PX, CMAP)
        canvas[:, 4 * CELL_PX:8 * CELL_PX, :] = heat2

    txt = f"{CHANNEL}-channel  min {vals.min():.1f}  max {vals.max():.1f}  vmin {vmin:.1f}  vmax {vmax:.1f}"
    cv2.putText(canvas, txt, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if SHOW_COLORBAR:
        bar_h = max(20, CELL_PX // 3)
        bar = _make_colorbar(bar_h, canvas.shape[1], CMAP)
        canvas = np.vstack([canvas, bar])

    return canvas


def describe(x, name="state", indent=0, max_items=12):
    pad = "  " * indent

    # numpy arrays
    if isinstance(x, np.ndarray):
        print(f"{pad}{name}: np.ndarray shape={x.shape} dtype={x.dtype}")
        return

    # dicts
    if isinstance(x, dict):
        print(f"{pad}{name}: dict ({len(x)} keys)")
        for i, (k, v) in enumerate(x.items()):
            if i >= max_items:
                print(f"{pad}  ... (+{len(x)-max_items} more keys)")
                break
            describe(v, name=f"{k!r}", indent=indent + 1, max_items=max_items)
        return

    # lists/tuples
    if isinstance(x, (list, tuple)):
        tname = type(x).__name__
        print(f"{pad}{name}: {tname} (len={len(x)})")
        for i, v in enumerate(x[:max_items]):
            describe(v, name=f"[{i}]", indent=indent + 1, max_items=max_items)
        if len(x) > max_items:
            print(f"{pad}  ... (+{len(x)-max_items} more items)")
        return

    # scalars / other objects
    try:
        tname = type(x).__name__
        s = repr(x)
        if len(s) > 120:
            s = s[:120] + "..."
        print(f"{pad}{name}: {tname} = {s}")
    except Exception:
        print(f"{pad}{name}: {type(x).__name__}")


def _prep_rgb(img: np.ndarray, target_h: int):
    """
    Ensure uint8 BGR and resize to target_h (keep aspect).
    """
    if img is None:
        out = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        cv2.putText(out, "missing", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Many pipelines store RGB; OpenCV expects BGR for display. If yours looks “wrong”, flip it.
    # If your images already look correct, set this to False.
    RGB_TO_BGR = True
    if RGB_TO_BGR and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def main():
    win = "PKL viewer (wrist | base | tactile)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i = 0
    while 0 <= i < len(paths):
        path = paths[i]
        with open(path, "rb") as f:
            state = pickle.load(f)

        wrist = state.get("wrist_rgb", None)
        base = state.get("base_rgb", None)
        g1, g2 = _get_tactile_from_state(state)

        tact_panel = _render_tactile_panel(g1, g2)
        target_h = tact_panel.shape[0]

        wrist_panel = _prep_rgb(wrist, target_h)
        base_panel = _prep_rgb(base, target_h)

        # Compose a single canvas
        canvas = np.hstack([wrist_panel, base_panel, tact_panel])

        # overlay filename + index
        cv2.putText(canvas, f"{i+1}/{len(paths)}  {os.path.basename(path)}", (10, target_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win, canvas)

        # Controls:
        #   d / right arrow: next
        #   a / left arrow : prev
        #   space          : play (fast)
        #   q / esc        : quit
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q")):
            break
        elif key in (ord("d"), 83):  # 83 is right arrow on many systems
            i += 1
        elif key in (ord("a"), 81):  # 81 is left arrow on many systems
            i -= 1
        elif key == ord(" "):
            # quick playback until any key
            while True:
                i += 1
                if i >= len(paths):
                    i = len(paths) - 1
                    break
                with open(paths[i], "rb") as f:
                    state = pickle.load(f)


                print("\n=== FILE:", os.path.basename(path), "===")
                describe(state, "state")                    
                wrist = state.get("wrist_rgb", None)
                base = state.get("base_rgb", None)
                g1, g2 = _get_tactile_from_state(state)
                tact_panel = _render_tactile_panel(g1, g2)
                target_h = tact_panel.shape[0]
                wrist_panel = _prep_rgb(wrist, target_h)
                base_panel = _prep_rgb(base, target_h)
                canvas = np.hstack([wrist_panel, base_panel, tact_panel])
                cv2.putText(canvas, f"{i+1}/{len(paths)}  {os.path.basename(paths[i])}", (10, target_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(win, canvas)
                k2 = cv2.waitKey(30) & 0xFF
                if k2 != 255:  # any key interrupts
                    break
        else:
            # unrecognized -> stay
            pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
