import geopandas
import pandas as pd
import numpy as np
from datetime import datetime
from shapely import wkt
from cropharvest.columns import RequiredColumns
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY

from ..utils import DATASET_PATH

from typing import List


def load_zambia_ceo():

    ceo_files = (DATASET_PATH / "zambia" / "ceo_labels").glob("*.csv")

    gdfs: List[geopandas.GeoDataFrame] = []
    for filepath in ceo_files:

        single_df = pd.read_csv(filepath)
        single_df[RequiredColumns.GEOMETRY] = single_df["sample_geom"].apply(wkt.loads)
        single_df["is_crop_mean"] = single_df.apply(lambda x: x["Crop/non-crop"] == "Crop", axis=1)
        gdfs.append(geopandas.GeoDataFrame(single_df, crs="epsg:4326"))

    df = pd.concat(gdfs)
    df = df.groupby("plotid").agg(
        {
            RequiredColumns.LON: "first",
            RequiredColumns.LAT: "first",
            RequiredColumns.GEOMETRY: "first",
            "is_crop_mean": "mean",
        }
    )

    df[RequiredColumns.COLLECTION_DATE] = datetime(2019, 1, 2)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2019, EXPORT_END_MONTH, EXPORT_END_DAY)
    df[RequiredColumns.IS_CROP] = np.where(df["is_crop_mean"] > 0.5, 1, 0)
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df
