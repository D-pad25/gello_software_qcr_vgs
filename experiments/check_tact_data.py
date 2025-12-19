import os
import glob
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========= CONFIG =========
DATA_DIR = "/run/user/1001/gvfs/sftp:host=aqua.qut.edu.au,user=n11457830/mnt/hpccs01/home/n11457830/gello/dec19_tact/gello/1219_152156"   # <-- change this

pattern = os.path.join(DATA_DIR, "*.pkl")
paths = sorted(glob.glob(pattern))

if not paths:
    print("No .pkl files found.")
    exit()

nonzero_files = []

for path in paths:
    with open(path, "rb") as f:
        state = pickle.load(f)

    tactile = state["tactile_data"]        # shape (N, 3)
    z = tactile[:, 2]                      # Z channel

    has_nonzero = np.any(z != 0)

    if has_nonzero:
        nonzero_files.append(
            (path, float(z.min()), float(z.max()), float(z.mean()))
        )

total = len(paths)
nz = len(nonzero_files)

print(f"\n{nz} / {total} files have non-zero Z values.\n")

# Print first 20 examples
for i, (path, zmin, zmax, zmean) in enumerate(nonzero_files[:20], start=1):
    print(f"{i:3d}. {os.path.basename(path)}")
    print(f"     min={zmin:.4f}, max={zmax:.4f}, mean={zmean:.4f}")