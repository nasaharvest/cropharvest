from pathlib import Path
import h5py

import torch
import numpy as np
import random

from typing import Dict


DATAFOLDER_PATH = Path(__file__).parent.parent / "data"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_normalizing_dict(path_to_dict: Path) -> Dict[str, np.ndarray]:

    hf = h5py.File(path_to_dict, "r")
    return {"mean": hf.get("mean"), "std": hf.get("std")}
