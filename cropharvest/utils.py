import h5py
from pathlib import Path
from tqdm import tqdm
from urllib.request import urlopen, Request
import numpy as np
import random
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


def download_from_url(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urlopen(Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def load_normalizing_dict(path_to_dict: Path) -> Dict[str, np.ndarray]:

    hf = h5py.File(path_to_dict, "r")
    return {"mean": hf.get("mean"), "std": hf.get("std")}
