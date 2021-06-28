import h5py
import numpy as np
import random
from pathlib import Path
from typing import Dict

try:
    import torch

    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

DATAFOLDER_PATH = Path(__file__).parent.parent / "data"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if TORCH_INSTALLED:
        torch.manual_seed(seed)


def load_normalizing_dict(path_to_dict: Path) -> Dict[str, np.ndarray]:

    hf = h5py.File(path_to_dict, "r")
    return {"mean": hf.get("mean"), "std": hf.get("std")}
