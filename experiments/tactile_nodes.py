# experiments/tactile_node.py
from __future__ import annotations

from dataclasses import dataclass
import time
import threading
from typing import Optional, Dict, Any, Tuple

import numpy as np
import tyro
import zmq
import pickle

import websocket
import json
from scipy.signal import butter, filtfilt

# Optional local display
import cv2


# -----------------------------
# Sensor processing core
# -----------------------------
class SensorProcessor:
    """
    Runs inside a single OS process.
    WebSocket callbacks update 'latest' state protected by a lock.
    """

    def __init__(
        self,
        ip: str,
        port: int,
        *,
        sampling_rate: float = 100.0,
        filter_cutoff: float = 40.0,
        filter_order: int = 4,
        calibration_samples: int = 20,
        num_sensors: int = 16,
        history_length: int = 25,
        deadband_abs: float = 1.0,
        swap_groups: bool = False,   # swap ws["1"] and ws["2"]
        flip_g2: bool = True,        # reverse sensor order for group2
        get_time: bool = True,
        verbose_errors: bool = False,
    ):
        self.server_ip = ip
        self.server_port = port

        # Config
        self.sampling_rate = float(sampling_rate)
        self.filter_cutoff = float(filter_cutoff)
        self.filter_order = int(filter_order)
        self.calibration_samples = int(calibration_samples)
        self.num_sensors = int(num_sensors)
        self.history_length = int(history_length)
        self.deadband_abs = float(deadband_abs)
        self.swap_groups = bool(swap_groups)
        self.flip_g2 = bool(flip_g2)
        self.get_time = bool(get_time)
        self.verbose_errors = bool(verbose_errors)

        # State
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

        self.is_calibrated = False
        self.calibration_count = 0
        self.baseline_g1 = np.zeros((self.num_sensors, 3), dtype=np.float32)
        self.baseline_g2 = np.zeros((self.num_sensors, 3), dtype=np.float32)

        self.last_g1 = np.zeros((self.num_sensors, 3), dtype=np.float32)
        self.last_g2 = np.zeros((self.num_sensors, 3), dtype=np.float32)

        # For debugging / timing
        self.last_times = {
            "mess_creation_time": None,
            "mess_send_time": None,
        }

        # Precompute filter coeffs
        self._b, self._a = self._design_filter()

    def _design_filter(self):
        nyq = 0.5 * self.sampling_rate
        cutoff = min(max(self.filter_cutoff / nyq, 1e-6), 0.999999)
        b, a = butter(self.filter_order, cutoff, btype="low", analog=False)
        return b, a

    def _butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        # data: (16,3)
        return filtfilt(self._b, self._a, data, axis=0).astype(np.float32)

    def _extract_sensor_data(self, data: Dict[str, Any]) -> Tuple[Optional[list], Optional[list]]:
        try:
            if self.swap_groups:
                raw1 = data["2"]["data"]
                raw2 = data["1"]["data"]
            else:
                raw1 = data["1"]["data"]
                raw2 = data["2"]["data"]

            g1 = [int(v, 16) for v in raw1.split(",")]
            g2 = [int(v, 16) for v in raw2.split(",")]
            return g1, g2
        except Exception:
            if self.verbose_errors:
                print("[sensor] Invalid or missing sensor data")
            return None, None

    def _extract_time_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mess_creation_time": data.get("time", None),
            "mess_send_time": data.get("sendtime", None),
        }

    def _reshape_raw(self, g_raw: list) -> np.ndarray:
        # raw list length should be 16*3 = 48
        g = np.array([g_raw[i * 3:(i + 1) * 3] for i in range(self.num_sensors)], dtype=np.float32)
        return g

    def _calibrate_step(self, g1_raw: list, g2_raw: list):
        g1 = self._reshape_raw(g1_raw)
        g2 = self._reshape_raw(g2_raw)
        if self.flip_g2:
            g2 = g2[::-1, :]

        self.baseline_g1 += g1
        self.baseline_g2 += g2
        self.calibration_count += 1

        if self.calibration_count >= self.calibration_samples:
            self.baseline_g1 /= float(self.calibration_samples)
            self.baseline_g2 /= float(self.calibration_samples)
            self.is_calibrated = True
            print(f"[sensor] Calibration completed ({self.calibration_samples} samples)")

    def _process_step(self, g1_raw: list, g2_raw: list) -> Tuple[np.ndarray, np.ndarray]:
        g1 = self._reshape_raw(g1_raw)
        g2 = self._reshape_raw(g2_raw)
        if self.flip_g2:
            g2 = g2[::-1, :]

        adj1 = g1 - self.baseline_g1
        adj2 = g2 - self.baseline_g2

        f1 = self._butterworth_filter(adj1)
        f2 = self._butterworth_filter(adj2)

        if self.deadband_abs > 0:
            f1[np.abs(f1) < self.deadband_abs] = 0
            f2[np.abs(f2) < self.deadband_abs] = 0

        return f1, f2

    # --- websocket callbacks ---
    def _on_message(self, ws, message: str):
        recv_time = time.time()

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if data.get("message") == "Welcome":
            print("[sensor] Connected to WebSocket server")
            return

        g1_raw, g2_raw = self._extract_sensor_data(data)
        if g1_raw is None or g2_raw is None:
            return

        tdata = self._extract_time_data(data) if self.get_time else {}

        with self._lock:


            self.last_times= tdata

            if not self.is_calibrated:
                self._calibrate_step(g1_raw, g2_raw)
            else:
                f1, f2 = self._process_step(g1_raw, g2_raw)
                self.last_g1 = f1
                self.last_g2 = f2

    def _on_error(self, ws, err):
        # Keep it concise; too many prints can cause lag
        if self.verbose_errors:
            print(f"[sensor] WebSocket error: {err}")

    def _on_close(self, ws, code, reason):
        print(f"[sensor] WebSocket closed: {code}, {reason}")

    def _on_open(self, ws):
        print(f"[sensor] Connected to ws://{self.server_ip}:{self.server_port}")

    def run_forever(self):
        websocket.setdefaulttimeout(1)
        ws = websocket.WebSocketApp(
            f"ws://{self.server_ip}:{self.server_port}",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        # websocket-client run_forever blocks; it will reconnect depending on params
        while not self._stop_evt.is_set():
            try:
                ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as e:
                if self.verbose_errors:
                    print(f"[sensor] run_forever exception: {e}")
            time.sleep(0.2)

    def stop(self):
        self._stop_evt.set()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # copy to plain python types for pickling
            return {
                "is_calibrated": bool(self.is_calibrated),
                "g1": self.last_g1.tolist(),
                "g2": self.last_g2.tolist(),
                "tact_time": self.last_times,
            }


# -----------------------------
# ZMQ "camera-style" tactile server
# -----------------------------
class ZMQServerTactile:
    def __init__(self, sensor: SensorProcessor, *, host: str, port: int):
        self.sensor = sensor
        self.host = host
        self.port = int(port)

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(f"tcp://{self.host}:{self.port}")
        print(f"[tactile_node] ZMQ tactile server on tcp://{self.host}:{self.port}")

        self._stop_evt = threading.Event()

    def serve(self):
        # Poll so CTRL+C is responsive
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)

        while not self._stop_evt.is_set():
            events = dict(poller.poll(timeout=100))
            if self._sock not in events:
                continue

            msg = self._sock.recv()
            if msg != b"GET":
                self._sock.send(pickle.dumps({"error": "send GET"}))
                continue

            snap = self.sensor.snapshot()
            self._sock.send(pickle.dumps(snap))

    def stop(self):
        self._stop_evt.set()
        try:
            self._sock.close(0)
        except Exception:
            pass


# -----------------------------
# Optional OpenCV viewer (no queue)
# -----------------------------
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


def run_local_viewer(
    sensor: SensorProcessor,
    *,
    hz: float = 60.0,
    channel: str = "z",
    show_g2: bool = True,
    autoscale: bool = False,
    vmin: float = 0.0,
    vmax: float = 150.0,
    cell_px: int = 60,
    cmap: int = cv2.COLORMAP_TURBO,
):
    dt = 1.0 / float(hz)
    win = "tactile_local_viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        t0 = time.time()
        snap = sensor.snapshot()

        canvas = np.zeros((4 * cell_px, (8 if show_g2 else 4) * cell_px, 3), dtype=np.uint8)

        if not snap.get("is_calibrated", False):
            cv2.putText(canvas, "waiting for calibration...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
        else:
            g1 = np.array(snap["g1"], dtype=np.float32)  # (16,3)
            g2 = np.array(snap["g2"], dtype=np.float32)  # (16,3)

            ch1 = _pick_channel(g1, channel)
            g1_grid = ch1.reshape(4, 4)

            if show_g2:
                ch2 = _pick_channel(g2, channel)
                vals = np.concatenate([ch1, ch2])
            else:
                ch2 = None
                vals = ch1

            if autoscale:
                _vmin = float(np.percentile(vals, 5))
                _vmax = float(np.percentile(vals, 95))
                if abs(_vmax - _vmin) < 1e-6:
                    _vmax = _vmin + 1.0
            else:
                _vmin, _vmax = float(vmin), float(vmax)

            heat1 = _to_heatmap(g1_grid, _vmin, _vmax, cell_px, cmap)
            canvas[:, :4 * cell_px, :] = heat1

            if show_g2 and ch2 is not None:
                g2_grid = ch2.reshape(4, 4)
                heat2 = _to_heatmap(g2_grid, _vmin, _vmax, cell_px, cmap)
                canvas[:, 4 * cell_px:8 * cell_px, :] = heat2

            txt = f"{channel}-channel  min {vals.min():.1f}  max {vals.max():.1f}  vmin {_vmin:.1f} vmax {_vmax:.1f}"
            cv2.putText(canvas, txt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    cv2.destroyAllWindows()


# -----------------------------
# CLI + main
# -----------------------------
@dataclass
class Args:
    # XELA websocket server
    sensor_ip: str = "0.0.0.0"
    sensor_port: int = 5000

    # ZMQ tactile server (clients connect here)
    hostname: str = "127.0.0.1"
    zmq_port: int = 7001

    # calibration / processing
    calibration_samples: int = 20
    sampling_rate: float = 100.0
    filter_cutoff: float = 40.0
    filter_order: int = 4
    history_length: int = 25
    deadband_abs: float = 1.0
    swap_groups: bool = False
    flip_g2: bool = True

    # viewer
    use_viewer: bool = False
    viewer_hz: float = 100.0
    viewer_channel: str = "z"     # x/y/z
    viewer_show_g2: bool = True
    viewer_autoscale: bool = False
    viewer_vmin: float = 0.0
    viewer_vmax: float = 150.0
    viewer_cell_px: int = 60

    verbose_errors: bool = False


def main(args: Args):
    sensor = SensorProcessor(
        ip=args.sensor_ip,
        port=args.sensor_port,
        sampling_rate=args.sampling_rate,
        filter_cutoff=args.filter_cutoff,
        filter_order=args.filter_order,
        calibration_samples=args.calibration_samples,
        history_length=args.history_length,
        deadband_abs=args.deadband_abs,
        swap_groups=args.swap_groups,
        flip_g2=args.flip_g2,
        verbose_errors=args.verbose_errors,
    )

    # Start websocket ingest in background thread
    sensor_thread = threading.Thread(target=sensor.run_forever, daemon=True)
    sensor_thread.start()

    # Optional local viewer thread (no queue)
    viewer_thread = None
    if args.use_viewer:
        viewer_thread = threading.Thread(
            target=run_local_viewer,
            kwargs=dict(
                sensor=sensor,
                hz=args.viewer_hz,
                channel=args.viewer_channel,
                show_g2=args.viewer_show_g2,
                autoscale=args.viewer_autoscale,
                vmin=args.viewer_vmin,
                vmax=args.viewer_vmax,
                cell_px=args.viewer_cell_px,
            ),
            daemon=True,
        )
        viewer_thread.start()

    # ZMQ server in main thread (like camera server)
    server = ZMQServerTactile(sensor, host=args.hostname, port=args.zmq_port)

    try:
        server.serve()
    except KeyboardInterrupt:
        print("\n[tactile_node] Shutting down...")
    finally:
        server.stop()
        sensor.stop()


if __name__ == "__main__":
    main(tyro.cli(Args))