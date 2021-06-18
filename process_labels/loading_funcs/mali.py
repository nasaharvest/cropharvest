import geopandas
import pandas as pd

from datetime import datetime

from .utils import process_crop_non_crop
from cropharvest.utils import DATASET_PATH
from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH


LABEL_TO_CLASSIFICATION = {
    "maize": "cereals",
    "sorghum": "cereals",
    "millet": "cereals",
    "rice": "cereals",
}


def load_mali_crop_noncrop():
    df = process_crop_non_crop(DATASET_PATH / "mali/mali_noncrop_2019")
    # not sure about this
    df["collection_date"] = datetime(2019, 1, 1)
    df["export_end_date"] = datetime(2020, EXPORT_END_MONTH, EXPORT_END_DAY)
    df = df.reset_index(drop=True)
    df["index"] = df.index
    return df


def load_mali():
    df = geopandas.read_file(DATASET_PATH / "mali/segou_bounds_07212020")

    df["lon"] = df.geometry.centroid.x.values
    df["lat"] = df.geometry.centroid.y.values
    df["collection_date"] = datetime(2020, 7, 21)
    df["is_crop"] = 1

    final_dfs = []
    for year in [2018, 2019]:
        year_df = df.copy()
        year_df["label"] = year_df[f"{year}_main_"]
        year_df["classification_label"] = year_df.apply(
            lambda x: LABEL_TO_CLASSIFICATION[x.label], axis=1
        )
        year_df["export_end_date"] = datetime(year + 1, EXPORT_END_MONTH, EXPORT_END_DAY)
        year_df = year_df.drop(
            columns=["2018_main_", "2018_other", "2019_main_", "2019_other", "2019_mai_1"]
        )
        final_dfs.append(year_df)

    df = pd.concat(final_dfs)
    df = df.reset_index(drop=True)
    df["index"] = df.index

    return df
