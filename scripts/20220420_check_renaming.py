"""
After 20220418_renaming.py was run,
this script was used to check that
the renaming happened correctly

From 3 runs, this had a failure rate of 8, 10 and 6
(out of 1000), which is <1%. In all cases, I looked at
the failures and the corresponding latitudes and longitudes
were just outside of the margins, which I am happy to attribute
to rounding errors.
"""
from pathlib import Path
import xarray as xr
import re
import numpy as np
from random import shuffle
from tqdm import tqdm

from cropharvest.countries import BBox
from cropharvest.utils import DATAFOLDER_PATH


def _bbox_from_filepath(p: Path) -> BBox:
    """
    https://github.com/nasaharvest/crop-mask/blob/master/src/ETL/boundingbox.py#L24
    """
    decimals_in_p = re.findall(r"=-?\d*\.?\d*", p.stem)
    coords = [float(d[1:]) for d in decimals_in_p[0:4]]
    return BBox(min_lat=coords[0], min_lon=coords[1], max_lat=coords[2], max_lon=coords[3])


def isin(x: np.ndarray, val: float) -> bool:
    return (val >= x.min()) & (val <= x.max())


def check_file(path: Path) -> None:

    # extract expected info from the path
    bbox = _bbox_from_filepath(path)
    tif_file = xr.open_rasterio(path)

    # x is lon, y is lat
    lat, lon = bbox.get_centre(in_radians=False)
    x, y = tif_file.x.values, tif_file.y.values

    assert isin(x, lon) & isin(y, lat), f"{path} failed with {x}, {y} and {lat}, {lon}"


def main(renamed_path: Path, num_to_check: int = 1000):

    all_files = list(renamed_path.glob("*.tif"))
    shuffle(all_files)

    failed = 0
    for path_to_check in tqdm(all_files[:num_to_check]):
        try:
            check_file(path_to_check)
        except AssertionError as e:
            print(e)
            failed += 1

    print(f"{failed} files failed check out of {num_to_check}")


if __name__ == "__main__":
    main(Path(DATAFOLDER_PATH / "renamed_eo_data"))
