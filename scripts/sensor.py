import websocket
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt
from matplotlib.colors import LinearSegmentedColormap  # NEW

class SensorProcessor:
    def __init__(self, ip, port, mode="raw_data", enable_plot=True):
        self.server_ip = ip
        self.server_port = port
        self.mode = mode
        self.plot_enabled = enable_plot

        # Sensor Data Configuration
        # Sensor Data Configuration
        self.sampling_rate = 100  # Hz (adjust this to match your actual sampling rate)
        self.filter_cutoff = 10  # Hz (cutoff frequency for the low-pass filter)
        self.filter_order = 4  # Butterworth filter order
        self.history_length = 10
        self.num_sensors = 16
        self.sensor_data_group1 = np.zeros((self.history_length, self.num_sensors, 3))  # Updated shape
        self.sensor_data_group2 = np.zeros((self.history_length, self.num_sensors, 3))  # Updated shape
        self.baseline_group1 = np.zeros((self.num_sensors, 3))  # Updated shape
        self.baseline_group2 = np.zeros((self.num_sensors, 3))  # Updated shape
        self.is_calibrated = False
        self.calibration_samples = 10
        self.calibration_count = 0

        # Plot Setup
        self.figure = None
        self.ax_group1 = None
        self.ax_group2 = None
        self.lines_group1 = []
        self.lines_group2 = []
        self.im_group1 = None      # NEW: heatmap image
        self.im_group2 = None    
        self.lock = threading.Lock()

        if self.plot_enabled:
            self._setup_plot_heat()   # CHANGED



    def _setup_plot_heat(self):
        """Initialize Matplotlib heatmaps for visualizing sensor data."""
        self.figure, (self.ax_group2, self.ax_group1) = plt.subplots(1, 2, figsize=(10, 6))
import websocket
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt
from matplotlib.colors import LinearSegmentedColormap  # NEW

class SensorProcessor:
    def __init__(self, ip, port, mode="raw_data", enable_plot=True):
        self.server_ip = ip
        self.server_port = port
        self.mode = mode
        self.plot_enabled = enable_plot

        # Sensor Data Configuration
        # Sensor Data Configuration
        self.sampling_rate = 100  # Hz (adjust this to match your actual sampling rate)
        self.filter_cutoff = 40  # Hz (cutoff frequency for the low-pass filter)
        self.filter_order = 4  # Butterworth filter order
        self.history_length = 10
        self.num_sensors = 16
        self.sensor_data_group1 = np.zeros((self.history_length, self.num_sensors, 3))  # Updated shape
        self.sensor_data_group2 = np.zeros((self.history_length, self.num_sensors, 3))  # Updated shape
        self.baseline_group1 = np.zeros((self.num_sensors, 3))  # Updated shape
        self.baseline_group2 = np.zeros((self.num_sensors, 3))  # Updated shape
        self.is_calibrated = False
        self.calibration_samples = 10
        self.calibration_count = 0

        # Plot Setup
        self.figure = None
        self.ax_group1 = None
        self.ax_group2 = None
        self.lines_group1 = []
        self.lines_group2 = []
        self.im_group1 = None      # NEW: heatmap image
        self.im_group2 = None    
        self.lock = threading.Lock()

        if self.plot_enabled:
            self._setup_plot_heat()   # CHANGED



    def _setup_plot_heat(self):
        """Initialize Matplotlib heatmaps for visualizing sensor data."""
        self.figure, (self.ax_group2, self.ax_group1) = plt.subplots(1, 2, figsize=(10, 6))

        # Custom colormap: yellow -> green (like your other script)
        cmap = LinearSegmentedColormap.from_list(
            "yellow_green",
            [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],  # yellow -> green
            N=256,
        )

        zero_grid = np.zeros((4, 4))

        self.im_group1 = self.ax_group1.imshow(
            zero_grid, vmin=0, vmax=100, cmap=cmap, origin="lower"
        )
        self.im_group2 = self.ax_group2.imshow(
            zero_grid, vmin=0, vmax=100, cmap=cmap, origin="lower"
        )

        self.ax_group1.set_title("Group 1 Sensors (4x4)")
        self.ax_group2.set_title("Group 2 Sensors (4x4)")

        # Ticks so you can see the 4x4 layout
        ticks = np.arange(4)
        for ax in (self.ax_group1, self.ax_group2):
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

        # Colorbars
        self.figure.colorbar(self.im_group1, ax=self.ax_group1, label="Value")
        self.figure.colorbar(self.im_group2, ax=self.ax_group2, label="Value")

    def _update_plot_heat(self, frame):
        """Update heatmaps with the latest sensor values."""
        with self.lock:
            # latest timestep, z-component (index 2) → shape (16,)
            g1 = self.sensor_data_group1[-1, :, 2].copy()
            g2 = self.sensor_data_group2[-1, :, 2].copy()

        # reshape to 4x4
        g1_grid = g1.reshape(4, 4)
        g2_grid = g2.reshape(4, 4)

        self.im_group1.set_data(g1_grid)
        self.im_group2.set_data(g2_grid)

        return [self.im_group1, self.im_group2]
    
    def butterworth_filter(self, data):
        """
        Apply a Butterworth low-pass filter to the sensor data.

        Args:
            data (numpy array): Sensor data to filter.
        
        Returns:
            numpy array: Filtered data.
        """
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff = self.filter_cutoff / nyquist

        # Design the Butterworth filter
        b, a = butter(self.filter_order, normalized_cutoff, btype='low', analog=False)

        # Apply the filter along the last axis (sensor values)
        filtered_data = filtfilt(b, a, data, axis=0)

        return filtered_data


    def _setup_plot(self):
        """Initialize the Matplotlib plot for visualizing sensor data."""
        self.figure, (self.ax_group1, self.ax_group2) = plt.subplots(2, 1, figsize=(10, 8))

        self.lines_group1 = [
            self.ax_group1.plot([], [], label=f"Sensor 1-{i + 1}")[0] for i in range(self.num_sensors)
        ]
        self.lines_group2 = [
            self.ax_group2.plot([], [], label=f"Sensor 2-{i + 1}")[0] for i in range(self.num_sensors)
        ]

        for ax, title in zip([self.ax_group1, self.ax_group2], ["Group 1 Sensors", "Group 2 Sensors"]):
            ax.set_xlim(0, self.history_length)
            ax.set_ylim(-10, 1500)
            ax.set_xlabel("Time")
            ax.set_ylabel("Sensor Values")
            ax.set_title(title)
            ax.legend(loc="upper right")

    def _update_plot(self, frame):
        """Update the plot with new data."""
        with self.lock:
            # Update lines for group 1
            for i, line in enumerate(self.lines_group1):
                # Ensure only one dimension is used (e.g., x-dimension: [history_length, i, 0])
                line.set_data(
                    np.arange(self.history_length),  # x-axis: time
                    self.sensor_data_group1[:, i, 2]  # y-axis: x-dimension of sensor i
                )
            
            # Update lines for group 2
            for i, line in enumerate(self.lines_group2):
                line.set_data(
                    np.arange(self.history_length),
                    self.sensor_data_group2[:, i, 2]  # y-axis: x-dimension of sensor i
                )

        # Combine all lines for rendering
        return self.lines_group1 + self.lines_group2


    def start_plot(self):
        """Start the live plot."""
        if self.plot_enabled:
            self.ani = FuncAnimation(self.figure, self._update_plot_heat, blit=True, interval=10)
            plt.show()

    def process_message(self, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print("Invalid JSON message received")
            return

        if "message" in data and data["message"] == "Welcome":
            print("Connected to WebSocket server")
            return

        group1_raw, group2_raw = self._extract_sensor_data(data)
        if group1_raw is None or group2_raw is None:
            return

        with self.lock:
            if not self.is_calibrated:
                self._calibrate_baselines(group1_raw, group2_raw)
            else:
                self._update_sensor_data(group1_raw, group2_raw)

    def _extract_sensor_data(self, data):
        """Extract raw sensor data from incoming WebSocket message."""
        try:
            group1_raw = [int(value, 16) for value in data["1"]["data"].split(",")]
            group2_raw = [int(value, 16) for value in data["2"]["data"].split(",")]
            return group1_raw, group2_raw
        except (ValueError, KeyError):
            print("Invalid or missing sensor data")
            return None, None

    def _calibrate_baselines(self, group1_raw, group2_raw):
        """Calibrate the sensor baselines using the initial readings."""
        # Loop through each sensor index to extract [x, y, z]
        group1_values = np.array([group1_raw[i * 3: (i * 3) + 3] for i in range(self.num_sensors)])
        group2_values = np.array([group2_raw[i * 3: (i * 3) + 3] for i in range(self.num_sensors)])
        group2_values = group2_values[::-1, :]
        self.baseline_group1 += group1_values
        self.baseline_group2 += group2_values
        self.calibration_count += 1

        if self.calibration_count >= self.calibration_samples:
            self.baseline_group1 /= self.calibration_samples
            self.baseline_group2 /= self.calibration_samples
            self.is_calibrated = True
            print("Calibration completed")

    def _update_sensor_data(self, group1_raw, group2_raw):
        """Update sensor data arrays with calibrated and filtered values."""
        group1_values = np.array([group1_raw[i * 3: (i * 3) + 3] for i in range(self.num_sensors)])
        group2_values = np.array([group2_raw[i * 3: (i * 3) + 3] for i in range(self.num_sensors)])
        group2_values = group2_values[::-1, :]  # Reverse group2
        adjusted_group1 = group1_values - self.baseline_group1
        adjusted_group2 = group2_values - self.baseline_group2

        # Apply the Butterworth filter
        filtered_group1 = self.butterworth_filter(adjusted_group1)
        filtered_group2 = self.butterworth_filter(adjusted_group2)
        filtered_group1[np.abs(filtered_group1) < 1] = 0
        filtered_group2[np.abs(filtered_group2) < 1] = 0
        self.sensor_data_group1 = np.roll(self.sensor_data_group1, -1, axis=0)
        self.sensor_data_group2 = np.roll(self.sensor_data_group2, -1, axis=0)
        self.sensor_data_group1[-1] = filtered_group1
        self.sensor_data_group2[-1] = filtered_group2


    def start_websocket(self):
        """Start the WebSocket connection."""
        try:
            websocket.setdefaulttimeout(1)
            ws = websocket.WebSocketApp(
                f"ws://{self.server_ip}:{self.server_port}",
                on_open=lambda ws: print(f"Connected to {self.server_ip}:{self.server_port}"),
                on_message=lambda ws, msg: self.process_message(msg),
                on_error=lambda ws, err: print(f"WebSocket error: {err}"),
                on_close=lambda ws, code, reason: print(f"WebSocket closed: {code}, {reason}"),
            )
            threading.Thread(target=ws.run_forever, daemon=True).start()
        except Exception as e:
            print(f"Failed to start WebSocket: {e}")




if __name__ == "__main__":
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 5000

    processor = SensorProcessor(ip=SERVER_IP, port=SERVER_PORT, mode="raw_data", enable_plot=True)
