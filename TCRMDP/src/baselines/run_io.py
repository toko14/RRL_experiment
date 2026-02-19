import os
import uuid
from typing import Tuple

import random
import numpy as np
import torch


def setup_output_dir(base_dir: str) -> Tuple[str, str]:
    """
    Create a unique run directory and standard subdirectories.

    Returns
    -------
    output_dir : str
        Path to the created run directory.
    run_id : str
        UUID used for the run.
    """
    run_id = str(uuid.uuid4())
    output_dir = os.path.join(base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    for sub in ["policies", "critics", "metrics"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    return output_dir, run_id


def append_csv(csv_path: str, header: list[str], row: list) -> None:
    """Append one row to CSV, writing header if the file is new."""
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        import csv

        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def init_seeds(seed: int, device_str: str) -> tuple[torch.device, np.random.RandomState]:
    """Set seeds and return device and numpy RandomState."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    # Seed common RNG sources used across this repo:
    # - torch: network init / exploration noise
    # - numpy global RNG: used by td3 (np.random.uniform)
    # - python random: used by td3 replay buffer sampling (random.sample)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    rand_state = np.random.RandomState(seed)
    return device, rand_state

