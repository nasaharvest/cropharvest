import h5py
from pathlib import Path
from tqdm import tqdm
from urllib.request import urlopen, Request
import numpy as np
import random
import geopandas
import collections
import functools

from cropharvest.config import LABELS_FILENAME

from typing import Dict, List

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


def deterministic_shuffle(x: List, seed: int) -> List:

    output_list: List = []
    x = x.copy()

    if seed % 2 == 0:
        seed = -seed
    while len(x) > 0:
        if abs(seed) >= len(x):
            is_negative = seed < 0
            seed = min(seed % len(x), len(x) - 1)
            if is_negative:
                # seed will now definitely be positive. This
                # ensures it will retain its original sign
                seed *= -1
        output_list.append(x.pop(seed))
        seed *= -1
    return output_list


class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


@memoized
def read_geopandas(file_path) -> geopandas.GeoDataFrame:
    return geopandas.read_file(str(file_path))
