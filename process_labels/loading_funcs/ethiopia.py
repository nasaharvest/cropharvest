import geopandas
import pandas as pd
from datetime import datetime

from typing import List

from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY
from .utils import process_crop_non_crop
from ..columns import RequiredColumns
from ..utils import DATASET_PATH


def load_ethiopia() -> geopandas.GeoDataFrame:

    output_dfs: List[geopandas.GeoDataFrame] = []

    filenames = [
        "ethiopia_crop.shp",
        "ethiopia_noncrop.shp",
        "ethiopia_crop_gt",
        "ethiopia_non_crop_gt",
    ]

    for filename in filenames:
        filepath = DATASET_PATH / "ethiopia" / filename
        output_dfs.append(process_crop_non_crop(filepath))

    df = pd.concat(output_dfs)
    df[RequiredColumns.COLLECTION_DATE] = datetime(2020, 10, 22)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2021, EXPORT_END_MONTH, EXPORT_END_DAY)
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df
