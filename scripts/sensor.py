# scripts/sensor.py
import websocket
import json
import numpy as np
from scipy.signal import butter, filtfilt
import multiprocessing as mp
from typing import Optional


class SensorProcessor:
    def __init__(self, ip, port, mode="raw_data",
                 tactile_dict=None,
                 plot_queue: Optional[mp.Queue] = None):
        self.server_ip = ip
        self.server_port = port
        self.mode = mode

        # Shared outputs
        self.tactile_dict = tactile_dict        # Manager().dict() from parent
        self.plot_queue = plot_queue            # Queue for plotting

        # Sensor Data Configuration
        self.sampling_rate = 100
        self.filter_cutoff = 40
        self.filter_order = 4
        self.history_length = 10
        self.num_sensors = 16

        self.sensor_data_group1 = np.zeros((self.history_length, self.num_sensors, 3))
        self.sensor_data_group2 = np.zeros((self.history_length, self.num_sensors, 3))
        self.baseline_group1 = np.zeros((self.num_sensors, 3))
        self.baseline_group2 = np.zeros((self.num_sensors, 3))
        self.is_calibrated = False
        self.calibration_samples = 10
        self.calibration_count = 0

    # ---------- filtering ----------

    def butterworth_filter(self, data):
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff = self.filter_cutoff / nyquist
        b, a = butter(self.filter_order, normalized_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data, axis=0)

    # ---------- websocket callbacks ----------

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print("Invalid JSON")
            return

        if data.get("message") == "Welcome":
            print("Connected to WebSocket server")
            return

        group1_raw, group2_raw = self._extract_sensor_data(data)
        if group1_raw is None or group2_raw is None:
            return

        if not self.is_calibrated:
            self._calibrate_baselines(group1_raw, group2_raw)
        else:
            self._update_sensor_data(group1_raw, group2_raw)

    def _on_error(self, ws, err):
        print(f"WebSocket error: {err}")

    def _on_close(self, ws, code, reason):
        print(f"WebSocket closed: {code}, {reason}")

    def _on_open(self, ws):
        print(f"Connected to {self.server_ip}:{self.server_port}")

    # ---------- core logic ----------

    def _extract_sensor_data(self, data):
        try:
            g1 = [int(v, 16) for v in data["1"]["data"].split(",")]
            g2 = [int(v, 16) for v in data["2"]["data"].split(",")]
            return g1, g2
        except (ValueError, KeyError):
            print("Invalid or missing sensor data")
            return None, None

    def _calibrate_baselines(self, g1_raw, g2_raw):
        g1 = np.array([g1_raw[i * 3:(i + 1) * 3] for i in range(self.num_sensors)])
        g2 = np.array([g2_raw[i * 3:(i + 1) * 3] for i in range(self.num_sensors)])
        g2 = g2[::-1, :]

        self.baseline_group1 += g1
        self.baseline_group2 += g2
        self.calibration_count += 1

        if self.calibration_count >= self.calibration_samples:
            self.baseline_group1 /= self.calibration_samples
            self.baseline_group2 /= self.calibration_samples
            self.is_calibrated = True
            print("Calibration completed")

    def _update_sensor_data(self, g1_raw, g2_raw):
        g1 = np.array([g1_raw[i * 3:(i + 1) * 3] for i in range(self.num_sensors)])
        g2 = np.array([g2_raw[i * 3:(i + 1) * 3] for i in range(self.num_sensors)])
        g2 = g2[::-1, :]

        adj1 = g1 - self.baseline_group1
        adj2 = g2 - self.baseline_group2

        f1 = self.butterworth_filter(adj1)
        f2 = self.butterworth_filter(adj2)
        f1[np.abs(f1) < 1] = 0
        f2[np.abs(f2) < 1] = 0

        self.sensor_data_group1 = np.roll(self.sensor_data_group1, -1, axis=0)
        self.sensor_data_group2 = np.roll(self.sensor_data_group2, -1, axis=0)
        self.sensor_data_group1[-1] = f1
        self.sensor_data_group2[-1] = f2

        # ---- write to shared objects ----
        if self.tactile_dict is not None:
            # Store latest frame as plain Python lists
            self.tactile_dict["g1"] = f1.tolist()
            self.tactile_dict["g2"] = f2.tolist()

        if self.plot_queue is not None:
            # Example: send the z-component for a heatmap
            g1_z = f1[:, 2]
            g2_z = f2[:, 2]
            try:
                self.plot_queue.put_nowait((g1_z, g2_z))
            except:
                pass  # queue full → drop frame

    # ---------- public run method ----------

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
