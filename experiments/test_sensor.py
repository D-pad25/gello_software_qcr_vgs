# Put this at the bottom of scripts/sensor.py

from dataclasses import dataclass

import tyro
from scripts.sensor import SensorProcessor

# If your plotter is in scripts/plotter.py, prefer this import:
from scripts.plotter import run_plot_process
import multiprocessing as mp

# If running as a plain script and relative imports break, you may need:
# from plotter import run_plot_process


@dataclass
class SensorArgs:
    sensor_ip: str = "127.0.0.1"   # client destination (NOT 0.0.0.0)
    sensor_port: int = 5000
    calibration_samples: int = 300
    use_plot: bool = True


def run_sensor_process(ip: str, port: int, plot_queue, tactile_dict, calibration_samples: int):
    sensor = SensorProcessor(
        ip=ip,
        port=port,
        mode="raw_data",
        tactile_dict=tactile_dict,
        plot_queue=plot_queue,
    )
    sensor.calibration_samples = calibration_samples
    sensor.run_forever()


def main(args: SensorArgs):
    plot_queue = mp.Queue(maxsize=1) if args.use_plot else None

    manager = mp.Manager()
    tactile_dict = manager.dict()

    sensor_proc = mp.Process(
        target=run_sensor_process,
        args=(args.sensor_ip, args.sensor_port, plot_queue, tactile_dict, args.calibration_samples),
        daemon=False,
    )
    sensor_proc.start()

    plot_proc = None
    if args.use_plot:
        plot_proc = mp.Process(
            target=run_plot_process,
            args=(plot_queue,),
            daemon=False,
        )
        plot_proc.start()

    try:
        sensor_proc.join()
        if plot_proc is not None:
            plot_proc.join()
    except KeyboardInterrupt:
        pass
    finally:
        # terminate children cleanly
        for p in (plot_proc, sensor_proc):
            if p is not None and p.is_alive():
                p.terminate()
                p.join(timeout=2)
        try:
            manager.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(tyro.cli(SensorArgs))
