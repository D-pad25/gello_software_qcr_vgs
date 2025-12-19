# scripts/sensor.py
import json
import time
from typing import Optional, Tuple

import numpy as np
import websocket
from scipy.signal import butter, filtfilt
import multiprocessing as mp
import time

class SensorProcessor:
    """
    Connects to XELA websocket server, calibrates baselines, produces filtered tactile frames.
    Publishes:
      - tactile_dict: latest f1/f2 + timestamps
      - plot_queue: latest heatmap data (z channel by default)
    """

    def __init__(
        self,
        ip: str,
        port: int,
        mode: str = "raw_data",
        tactile_dict=None,
        plot_queue: Optional[mp.Queue] = None,
        *,
        sampling_rate_hz: float = 100.0,
        filter_cutoff_hz: float = 40.0,
        filter_order: int = 4,
        history_length: int = 25,
        num_sensors: int = 16,
        calibration_samples: int = 20,
        deadband_abs: float = 1.0,
        swap_groups: bool = False,   # if server indexing is reversed
        flip_g2: bool = True,        # keep if physical mounting needs it
    ):
        self.server_ip = ip
        self.server_port = port
        self.mode = mode

        self.tactile_dict = tactile_dict
        self.plot_queue = plot_queue

        self.sampling_rate = float(sampling_rate_hz)
        self.filter_cutoff = float(filter_cutoff_hz)
        self.filter_order = int(filter_order)
        self.history_length = int(history_length)
        self.num_sensors = int(num_sensors)

        self.calibration_samples = int(calibration_samples)
        self.calibration_count = 0
        self.is_calibrated = False

        self.deadband_abs = float(deadband_abs)
        self.swap_groups = bool(swap_groups)
        self.flip_g2 = bool(flip_g2)

        # History buffers are TIME-major: (H, 16, 3)
        self.sensor_data_group1 = np.zeros((self.history_length, self.num_sensors, 3), dtype=np.float32)
        self.sensor_data_group2 = np.zeros((self.history_length, self.num_sensors, 3), dtype=np.float32)

        self.baseline_group1 = np.zeros((self.num_sensors, 3), dtype=np.float32)
        self.baseline_group2 = np.zeros((self.num_sensors, 3), dtype=np.float32)

        # timestamps
        self.get_time = True
        self.mess_creation_time = None
        self.mess_send_time = None
        self.g1_time = None
        self.g2_time = None

        # precompute filter
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff = self.filter_cutoff / nyquist
        self._b, self._a = butter(self.filter_order, normalized_cutoff, btype="low", analog=False)

        # keep last frames
        self.last_f1 = np.zeros((self.num_sensors, 3), dtype=np.float32)
        self.last_f2 = np.zeros((self.num_sensors, 3), dtype=np.float32)

    # ---------- queue helper ----------
    def _put_latest(self, item):
        """Keep only the newest item in the queue (prevents queue-full freeze)."""
        if self.plot_queue is None:
            return
        try:
            self.plot_queue.get_nowait()  # drop stale
        except Exception:
            pass
        try:
            self.plot_queue.put_nowait(item)
        except Exception:
            pass

    # ---------- filtering ----------
    def butterworth_filter(self, data_time_major: np.ndarray) -> np.ndarray:
        # data_time_major shape: (H, 16, 3), filter over time axis=0
        return filtfilt(self._b, self._a, data_time_major, axis=0)

    # ---------- websocket callbacks ----------
    def _on_message(self, ws, message):
        recv_time = time.time()   # <-- define it here, every message

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print("Invalid JSON")
            return

        if data.get("message") == "Welcome":
            print("Connected to WebSocket server")
            return

        group1_raw, group2_raw = self._extract_sensor_data(data)
        if self.get_time:
            self.mess_creation_time, self.mess_send_time, self.g1_time, self.g2_time = self._extract_sensor_time_data(data)

        if group1_raw is None or group2_raw is None:
            return

        if not self.is_calibrated:
            self._calibrate_baselines(group1_raw, group2_raw)
        else:
            self._update_sensor_data(
                group1_raw, group2_raw,
                self.mess_creation_time, self.mess_send_time, self.g1_time, self.g2_time,
                recv_time,   # <-- pass it
            )


    def _on_error(self, ws, err):
        print(f"[sensor] WebSocket error: {err}")

    def _on_close(self, ws, code, reason):
        print(f"[sensor] WebSocket closed: {code}, {reason}")

    def _on_open(self, ws):
        print(f"[sensor] Connected to ws://{self.server_ip}:{self.server_port}")

    # ---------- core parsing ----------
    def _extract_sensor_data(self, data) -> Tuple[Optional[list], Optional[list]]:
        try:
            # XELA server often uses keys "1" and "2"
            a = [int(v, 16) for v in data["1"]["data"].split(",")]
            b = [int(v, 16) for v in data["2"]["data"].split(",")]

            if self.swap_groups:
                a, b = b, a

            return a, b
        except (ValueError, KeyError):
            print("[sensor] Invalid or missing sensor data")
            return None, None

    def _extract_sensor_time_data(self, data):
        try:
            mess_creation_time = data["time"]
            mess_send_time = data["sendtime"]
            g1_time = data["1"]["time"]
            g2_time = data["2"]["time"]
            if self.swap_groups:
                g1_time, g2_time = g2_time, g1_time
            return mess_creation_time, mess_send_time, g1_time, g2_time
        except (ValueError, KeyError):
            return None, None, None, None

    def _reshape_group(self, raw_list: list) -> np.ndarray:
        g = np.array([raw_list[i * 3:(i + 1) * 3] for i in range(self.num_sensors)], dtype=np.float32)
        return g

    # ---------- calibration ----------
    def _calibrate_baselines(self, g1_raw, g2_raw):
        g1 = self._reshape_group(g1_raw)
        g2 = self._reshape_group(g2_raw)

        if self.flip_g2:
            g2 = g2[::-1, :]

        # accumulate mean baseline
        self.baseline_group1 += g1
        self.baseline_group2 += g2
        self.calibration_count += 1

        # publish something so plot is alive during calibration
        self._put_latest((g1[:, 2], g2[:, 2], False))  # False => not calibrated yet

        if self.calibration_count >= self.calibration_samples:
            self.baseline_group1 /= float(self.calibration_samples)
            self.baseline_group2 /= float(self.calibration_samples)
            self.is_calibrated = True
            print("[sensor] Calibration completed")

    # ---------- streaming update ----------
    def _update_sensor_data(self, g1_raw, g2_raw, mess_creation_time, mess_send_time, g1_time, g2_time, recv_time):
        g1 = self._reshape_group(g1_raw)
        g2 = self._reshape_group(g2_raw)
        if self.flip_g2:
            g2 = g2[::-1, :]

        adj1 = g1 - self.baseline_group1
        adj2 = g2 - self.baseline_group2

        # time-major history update with adjusted samples
        self.sensor_data_group1 = np.roll(self.sensor_data_group1, -1, axis=0)
        self.sensor_data_group2 = np.roll(self.sensor_data_group2, -1, axis=0)
        self.sensor_data_group1[-1] = adj1
        self.sensor_data_group2[-1] = adj2

        # filter over TIME axis
        f1_hist = self.butterworth_filter(self.sensor_data_group1)
        f2_hist = self.butterworth_filter(self.sensor_data_group2)

        f1 = f1_hist[-1]
        f2 = f2_hist[-1]

        # simple deadband (absolute)
        f1[np.abs(f1) < self.deadband_abs] = 0
        f2[np.abs(f2) < self.deadband_abs] = 0

        self.last_f1 = f1
        self.last_f2 = f2

        # shared dict for consumers
        if self.tactile_dict is not None:
            self.tactile_dict["recv_time"] = float(recv_time) 
            self.tactile_dict["g1"] = f1.tolist()
            self.tactile_dict["g2"] = f2.tolist()
            self.tactile_dict["mess_creation_time"] = self.mess_creation_time
            self.tactile_dict["mess_send_time"] = self.mess_send_time
            self.tactile_dict["g1_time"] = self.g1_time
            self.tactile_dict["g2_time"] = self.g2_time
            self.tactile_dict["is_calibrated"] = True

        # plot queue: z channel heatmap
        if self.plot_queue is not None:
            pkt = (f1[:, 2].astype(np.float32), f2[:, 2].astype(np.float32))

            # drop any queued old frame(s)
            try:
                while True:
                    self.plot_queue.get_nowait()
            except Exception:
                pass

            try:
                self.plot_queue.put_nowait(pkt)
            except Exception:
                pass


    # ---------- public ----------
    def run_forever(self):
        websocket.setdefaulttimeout(1)
        ws = websocket.WebSocketApp(
            f"ws://{self.server_ip}:{self.server_port}",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        ws.run_forever()
