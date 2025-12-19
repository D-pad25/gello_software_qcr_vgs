import os
import glob
import pickle
import numpy as np
from datetime import datetime

# ====== CONFIGURE THIS ======
FOLDER = "/home/qcrvgs/gello/nidhi/gello/1218_191615/"
# ============================


def parse_filename_timestamp(path: str) -> float:
    """
    Parse filename like '2025-12-04T13-49-53.791178.pkl'
    into a UNIX timestamp (float seconds).
    """
    name = os.path.basename(path)
    name = name.replace(".pkl", "")
    dt = datetime.strptime(name, "%Y-%m-%dT%H-%M-%S.%f")
    return dt.timestamp()


def compute_freq_from_times(times, label: str):
    """
    Given a list of timestamps (seconds), compute instantaneous dt and freq stats.
    """
    arr = np.asarray([t for t in times if t is not None], dtype=float)
    if arr.size < 2:
        print(f"\nNo valid dt data for {label} (only {arr.size} samples).")
        return

    # Sort just in case
    arr = np.sort(arr)

    dt = np.diff(arr)
    # Keep only positive intervals
    dt = dt[dt > 0]
    if dt.size == 0:
        print(f"\nNo positive dt for {label}.")
        return

    freq = 1.0 / dt
    print(f"\n=== {label} ===")
    print(f"Num intervals: {dt.size}")
    print(f"Mean freq: {freq.mean():.3f} Hz")
    print(f"Std  freq: {freq.std():.3f} Hz")
    print(f"Min  freq: {freq.min():.3f} Hz")
    print(f"Max  freq: {freq.max():.3f} Hz")
    print(f"(Mean dt: {dt.mean()*1000:.3f} ms)")


def main():
    pkl_files = sorted(glob.glob(os.path.join(FOLDER, "*.pkl")))
    if not pkl_files:
        print("No .pkl files found in", FOLDER)
        return

    # PC times from filenames
    pc_times = []

    # Sensor times from tact_time[0..3]
    msg_creation_times = []
    send_times = []
    # g1_times = []
    # g2_times = []

    for path in pkl_files:
        # --- PC time from filename ---
        pc_ts = parse_filename_timestamp(path)
        pc_times.append(pc_ts)

        # --- load pkl ---
        with open(path, "rb") as f:
            data = pickle.load(f)

        # we expect 'tact_time' inside
        tact_time = None
        if isinstance(data, dict):
            tact_time = data.get("tact_time", None)

        if tact_time is None:
            # you can print here if you want to debug
            # print(f"{path}: no 'tact_time' key")
            msg_creation_times.append(None)
            send_times.append(None)
            # g1_times.append(None)
            # g2_times.append(None)
            continue

        tact_time = np.asarray(tact_time).ravel()
        if tact_time.size < 4:
            print(f"{path}: 'tact_time' has size {tact_time.size}, expected >= 4")
            msg_creation_times.append(None)
            send_times.append(None)
            continue

        # assuming order: [msg_creation, send_time, g1_time, g2_time]
        msg_creation_times.append(float(tact_time[0]))
        send_times.append(float(tact_time[1]))

    # --- Compute frequencies ---

    # 1) PC logging frequency (how fast pkls were written)
    compute_freq_from_times(pc_times, "PC logging frequency (from filenames)")

    # 2) Sensor time frequencies
    compute_freq_from_times(msg_creation_times, "Sensor msg_creation_time frequency")
    compute_freq_from_times(send_times,        "Sensor send_time frequency")



if __name__ == "__main__":
    main()
