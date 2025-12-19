import threading
import zmq
import pickle
import time
import numpy as np


import threading
import zmq
import pickle
import time
import numpy as np


class ZMQClientTactile:
    """
    Background polling tactile client for a REP tactile server.
    Keeps latest snapshot; auto-recovers by recreating socket on any failure.
    """
    def __init__(self, host: str, port: int, hz: float = 100.0, timeout_ms: int = 200):
        self.host = host
        self.port = int(port)
        self.hz = float(hz)
        self.timeout_ms = int(timeout_ms)

        self._ctx = zmq.Context.instance()
        self._sock = None
        self._poller = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self._latest = {}
        self._ok = False
        self._last_ok_time = None

        self._connect()
        print(f"[tactile_client] Connected to tcp://{self.host}:{self.port}")

    def _connect(self):
        # (re)create socket + poller
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass

        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(f"tcp://{self.host}:{self.port}")
        self._sock.setsockopt(zmq.LINGER, 0)

        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            if self._sock is not None:
                self._sock.close(0)
        except Exception:
            pass

    def _request_once(self):
        """
        One GET round-trip with polling.
        Returns (snap: dict, ok: bool). Raises on any protocol/socket issue.
        """
        self._sock.send(b"GET")

        events = dict(self._poller.poll(self.timeout_ms))
        if self._sock not in events:
            raise TimeoutError("tactile recv timeout")

        payload = self._sock.recv()
        snap = pickle.loads(payload)
        if not isinstance(snap, dict):
            snap = {}

        ok = bool(snap) and bool(snap.get("is_calibrated", False))
        return snap, ok

    def _run(self):
        dt = 1.0 / max(1e-6, self.hz)
        while not self._stop.is_set():
            t0 = time.time()
            snap = {}
            ok = False

            try:
                snap, ok = self._request_once()
            except Exception:
                # Any error: reset the REQ socket to restore clean send/recv state.
                try:
                    self._connect()
                except Exception:
                    pass
                snap = {}
                ok = False

            with self._lock:
                self._latest = snap
                self._ok = ok
                if ok:
                    self._last_ok_time = time.time()

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def get_latest(self):
        with self._lock:
            return dict(self._latest), bool(self._ok), self._last_ok_time


    def get_obs(self):
        obs={}

        snap, ok, last_ok_time = self.get_latest()
        obs["tactile_ok"] = ok
        obs["tactile_last_ok_time"] = last_ok_time
        t = snap.get("tact_time", None)
        obs["tact_time"] = t 
        if ok:
            # If your server still sends g1/g2 lists:
            g1 = np.array(snap.get("g1", []), dtype=np.float32)  # (16,3)
            g2 = np.array(snap.get("g2", []), dtype=np.float32)  # (16,3)

            if g1.shape == (16, 3) and g2.shape == (16, 3):
                g1 = np.flipud(g1.reshape(4, 4, 3)).reshape(16, 3)
                g2 = np.flipud(g2.reshape(4, 4, 3)).reshape(16, 3)
                obs["tactile_data"] = np.concatenate([g2, g1], axis=0)  # (32,3)

            


 
        else:
            obs["tactile_data"] = None
    

        return obs