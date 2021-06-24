from pathlib import Path
from tqdm import tqdm
from urllib.request import urlopen, Request

import torch
import numpy as np
import random


DATAFOLDER_PATH = Path(__file__).parent.parent / "data"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def download_from_url(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urlopen(Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)
