import geopandas
import pandas as pd
from datetime import datetime

from typing import List

from cropharvest.utils import DATASET_PATH
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY

from .utils import process_crop_non_crop


def load_sudan() -> geopandas.GeoDataFrame:

    output_dfs: List[geopandas.GeoDataFrame] = []

    filenames = ["sudan_crop", "sudan_non_crop"]

    for filename in filenames:
        filepath = DATASET_PATH / "sudan" / filename
        output_dfs.append(process_crop_non_crop(filepath))

    df = pd.concat(output_dfs)
    df["collection_date"] = datetime(2020, 10, 22)
    df = df.reset_index(drop=True)
    df["index"] = df.index
    df["export_end_date"] = datetime(2021, EXPORT_END_MONTH, EXPORT_END_DAY)

    return df
