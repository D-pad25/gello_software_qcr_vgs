import datetime
import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action_joint: np.ndarray,
    action_ee: np.ndarray,
    delta_action_ee: np.ndarray,
) -> None:
    obs["control_joint"] = action_joint  # add action to obs
    obs["control_ee"] = action_ee        # add action to obs
    obs["delta_control_ee"] = delta_action_ee  # add action to obs

    # make folder if it doesn't exist
    folder.mkdir(exist_ok=True, parents=True)

    # ---- FIX: cross-platform safe timestamp ----
    safe_timestamp = timestamp.isoformat().replace(":", "-")

    recorded_file = folder / f"{safe_timestamp}.pkl"

    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)
