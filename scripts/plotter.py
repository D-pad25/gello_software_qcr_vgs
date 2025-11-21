# scripts/plotter.py (or at bottom of sensor.py)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def run_plot_process(plot_queue):
    """Runs in its own process; reads from queue and plots heatmap or timeseries."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    g1_grid = np.zeros((4, 4))
    g2_grid = np.zeros((4, 4))

    im1 = ax1.imshow(g1_grid, vmin=0, vmax=100, origin="lower")
    im2 = ax2.imshow(g2_grid, vmin=0, vmax=100, origin="lower")

    def update(_frame):
        # drain queue; keep the last packet
        latest = None
        while True:
            try:
                latest = plot_queue.get_nowait()
            except Exception:
                break
        if latest is None:
            return [im1, im2]

        g1_flat, g2_flat = latest
        g1 = np.array(g1_flat).reshape(4, 4)
        g2 = np.array(g2_flat).reshape(4, 4)
        im1.set_data(g1)
        im2.set_data(g2)
        return [im1, im2]

    ani = FuncAnimation(fig, update, interval=50, blit=True)
    plt.show()
