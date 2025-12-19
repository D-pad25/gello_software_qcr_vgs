# experiments/tactile_client.py
from dataclasses import dataclass
import time
import tyro
import zmq
import pickle
import numpy as np
import cv2


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    zmq_port: int = 7001
    hz: int = 60

    # what to display
    channel: str = "z"          # "x", "y", or "z"
    show_g2: bool = True

    # normalization
    autoscale: bool = False     # if True, uses percentiles per-frame (or per-window)
    vmin: float = 0.0           # used when autoscale=False
    vmax: float = 150.0

    # rendering
    cell_px: int = 60           # size of each taxel cell in pixels
    cmap: int = cv2.COLORMAP_TURBO
    show_colorbar: bool = True


def _pick_channel(tact_16x3: np.ndarray, channel: str) -> np.ndarray:
    if channel == "x":
        return tact_16x3[:, 0]
    if channel == "y":
        return tact_16x3[:, 1]
    return tact_16x3[:, 2]  # default "z"


def _to_heatmap(grid_4x4: np.ndarray, vmin: float, vmax: float, cell_px: int, cmap: int) -> np.ndarray:
    # normalize to 0..255
    denom = max(1e-6, (vmax - vmin))
    norm = (grid_4x4 - vmin) / denom
    norm = np.clip(norm, 0.0, 1.0)
    img_u8 = (norm * 255.0).astype(np.uint8)

    # upscale so it's visible
    img_u8 = cv2.resize(img_u8, (4 * cell_px, 4 * cell_px), interpolation=cv2.INTER_NEAREST)

    # apply colormap (OpenCV expects single-channel 8-bit)
    heat = cv2.applyColorMap(img_u8, cmap)
    return heat


def _make_colorbar(height: int, width: int, cmap: int) -> np.ndarray:
    # horizontal gradient 0..255
    grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    bar = cv2.applyColorMap(grad, cmap)
    return bar


def main(args: Args):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.hostname}:{args.zmq_port}")
    print(f"[tactile_client] Connected to tcp://{args.hostname}:{args.zmq_port}")

    dt = 1.0 / float(args.hz)
    win = "tactile_heatmap"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        t0 = time.time()

        sock.send(b"GET")
        payload = sock.recv()
        
        data = pickle.loads(payload)
        now = time.time()
        recv_time = data.get("recv_time", None)
        if recv_time is not None:
            age_ms = (now - float(recv_time)) * 1000.0
        else:
            age_ms = None


        canvas = np.zeros((4 * args.cell_px, (8 if args.show_g2 else 4) * args.cell_px, 3), dtype=np.uint8)
        if age_ms is not None:
            cv2.putText(canvas, f"age {age_ms:.0f} ms", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if not data or not data.get("is_calibrated", False):
            cv2.putText(canvas, "waiting for calibration...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
        else:
            g1 = np.array(data["g1"], dtype=np.float32)  # (16,3)
            g2 = np.array(data["g2"], dtype=np.float32)  # (16,3)



            if args.autoscale:
                # robust scaling; adjust as needed
                vals = ch1 if not args.show_g2 else np.concatenate([ch1, _pick_channel(g2, args.channel)])
                vmin = float(np.percentile(vals, 5))
                vmax = float(np.percentile(vals, 95))
                if abs(vmax - vmin) < 1e-6:
                    vmax = vmin + 1.0
            else:
                vmin, vmax = float(args.vmin), float(args.vmax)
            ch1 = _pick_channel(g1, args.channel)  # (16,)
            g1_grid = ch1.reshape(4, 4)
            heat1 = _to_heatmap(g1_grid, vmin, vmax, args.cell_px, args.cmap)
            

            ch2 = _pick_channel(g2, args.channel)
            g2_grid = ch2.reshape(4, 4)
            heat2 = _to_heatmap(g2_grid, vmin, vmax, args.cell_px, args.cmap)
            
            heat1 = cv2.flip(heat1, 0)
            heat2 = cv2.flip(heat2, 0)
            canvas[:, :4 * args.cell_px, :] = heat2
            canvas[:, 4 * args.cell_px:8 * args.cell_px, :] = heat1

            # overlay stats
            all_vals = ch1 if not args.show_g2 else np.concatenate([ch1, _pick_channel(g2, args.channel)])
            txt = f"{args.channel}-channel  min {all_vals.min():.1f}  max {all_vals.max():.1f}  vmin {vmin:.1f}  vmax {vmax:.1f}"
            cv2.putText(canvas, txt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # optional colorbar strip under the heatmaps
            if args.show_colorbar:
                bar_h = max(20, args.cell_px // 3)
                bar = _make_colorbar(bar_h, canvas.shape[1], args.cmap)
                out = np.vstack([canvas, bar])

                # add vmin/vmax labels
                # cv2.putText(out, f"{vmin:.1f}", (10, out.shape[0] - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(out, f"{vmax:.1f}", (out.shape[1] - 60, out.shape[0] - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(win, out)
            else:
                cv2.imshow(win, canvas)

        # keyboard handling (ESC / q to quit)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(Args))
