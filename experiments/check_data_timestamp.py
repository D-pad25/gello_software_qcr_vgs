#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime
import statistics
import math

import matplotlib.pyplot as plt


def parse_timestamp_from_stem(stem: str) -> datetime:
    """
    Parse a datetime from filename stem.

    Supports:
      - 2025-11-21T15:31:16.360637
      - 2025-11-21T15-31-16.360637
    """
    # First try direct ISO8601
    try:
        return datetime.fromisoformat(stem)
    except ValueError:
        pass

    # Handle "T15-31-16.360637" style
    if "T" in stem:
        date_part, time_part = stem.split("T", 1)
        # Replace '-' in the time part with ':' (before the fractional seconds)
        if "-" in time_part:
            if "." in time_part:
                time_main, micro = time_part.split(".", 1)
                time_main = time_main.replace("-", ":")
                fixed = f"{date_part}T{time_main}.{micro}"
            else:
                time_main = time_part.replace("-", ":")
                fixed = f"{date_part}T{time_main}"
            return datetime.fromisoformat(fixed)

    # If we get here, it’s some unexpected format
    raise ValueError(f"Could not parse timestamp from stem: {stem}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse timestamp spacing between .pkl files."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/qcrvgs/gello_nidhi/gello/1128_204828",
        help="Directory containing .pkl files (default: current directory).",
    )
    args = parser.parse_args()

    pkldir = Path(args.dir)
    files = sorted(pkldir.glob("*.pkl"))
    if not files:
        print(f"No .pkl files found in {pkldir}")
        return

    # Parse timestamps
    timestamps = []
    for f in files:
        stem = f.stem  # e.g. '2025-11-21T15-31-16.360637'
        try:
            ts = parse_timestamp_from_stem(stem)
            timestamps.append(ts)
        except ValueError as e:
            print(f"Skipping {f.name}: {e}")

    if len(timestamps) < 2:
        print("Need at least 2 valid timestamps to compute deltas.")
        return

    # Sort timestamps just in case
    timestamps.sort()

    # Compute inter-sample deltas in milliseconds
    deltas_ms = []
    for t_prev, t_cur in zip(timestamps[:-1], timestamps[1:]):
        dt = (t_cur - t_prev).total_seconds() * 1000.0  # ms
        deltas_ms.append(dt)

    # Basic stats
    mean_dt = statistics.mean(deltas_ms)
    std_dt = statistics.pstdev(deltas_ms) if len(deltas_ms) > 1 else float("nan")

    print(f"Number of samples: {len(timestamps)}")
    print(f"Number of intervals: {len(deltas_ms)}")
    print(f"Mean Δt: {mean_dt:.3f} ms")
    print(f"Mean hz: {1000/(mean_dt):.3f}")
    print(f"Std  Δt: {std_dt:.3f} ms")

    # Plot Δt over index
    plt.figure()
    plt.plot(deltas_ms, marker=".", linestyle="-")
    plt.xlabel("Interval index")
    plt.ylabel("Δt between files (ms)")
    plt.title("Timestamp intervals between consecutive .pkl files")

    # Add a horizontal line at the mean
    plt.axhline(mean_dt, linestyle="--")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
