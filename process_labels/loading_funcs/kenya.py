import geopandas
import pandas as pd
import numpy as np
from datetime import datetime

from .utils import process_crop_non_crop, export_date_from_row, LATLON_CRS
from cropharvest.utils import DATASET_PATH
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY

from typing import List


LABEL_TO_CLASSIFICATION = {
    "Cassava": "root_tuber",
    "Maize": "cereals",
    "Sorghum": "cereals",
    "Bean": "leguminous",
    "Groundnut": "oilseeds",
    "Fallowland": "non_crop",
    "Millet": "cereals",
    "Tomato": "vegetables_melons",
    "Sugarcane": "sugar",
    "Sweetpotato": "root_tuber",
    "Banana": "fruits_nuts",
    "Soybean": "oilseeds",
    "Cabbage": "vegetables_melons",
}


def load_kenya():

    subfolders = [f"ref_african_crops_kenya_01_labels_0{i}" for i in [0, 1, 2]]

    dfs: List[geopandas.GeoDataFrame] = []
    for subfolder in subfolders:
        df = geopandas.read_file(
            DATASET_PATH
            / "kenya"
            / "ref_african_crops_kenya_01_labels"
            / subfolder
            / "labels.geojson"
        )
        df = df.rename(
            columns={
                "Latitude": "lat",
                "Longitude": "lon",
                "Planting Date": "planting_date",
                "Estimated Harvest Date": "harvest_date",
                "Crop1": "label",
                "Survey Date": "collection_date",
            }
        )
        df["planting_date"] = pd.to_datetime(df["planting_date"]).dt.to_pydatetime()
        df["harvest_date"] = pd.to_datetime(df["harvest_date"]).dt.to_pydatetime()
        df["collection_date"] = pd.to_datetime(df["collection_date"]).dt.to_pydatetime()
        df["export_end_date"] = df.apply(export_date_from_row, axis=1)
        df["classification_label"] = df.apply(lambda x: LABEL_TO_CLASSIFICATION[x.label], axis=1)
        df["is_crop"] = np.where((df["classification_label"] == "non_crop"), 0, 1)
        df = df.to_crs(LATLON_CRS)
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df["index"] = df.index
    return df


def load_kenya_non_crop():

    dfs: List[geopandas.GeoDataFrame] = []

    base = DATASET_PATH / "kenya" / "kenya_non_crop"

    folders_with_crs = [
        ("noncrop_labels_v2", "EPSG:32636"),
        ("noncrop_labels_set2", "EPSG:32636"),
        ("2019_gepro_noncrop", "EPSG:4326"),
        ("noncrop_water_kenya_gt", "EPSG:4326"),
        ("noncrop_kenya_gt", "EPSG:4326"),
        ("kenya_non_crop_test_polygons", "EPSG:4326"),
    ]
    for folder, crs in folders_with_crs:
        dfs.append(process_crop_non_crop(base / folder, org_crs=crs))

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df["index"] = df.index
    df["collection_date"] = datetime(2020, 4, 16)
    df["export_end_date"] = datetime(2020, EXPORT_END_MONTH, EXPORT_END_DAY)

    return df
