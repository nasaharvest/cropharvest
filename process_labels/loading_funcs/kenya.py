import geopandas
import pandas as pd
import numpy as np
from datetime import datetime

from .utils import process_crop_non_crop, export_date_from_row, LATLON_CRS
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY
from cropharvest.columns import RequiredColumns, NullableColumns
from ..utils import DATASET_PATH

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
                "Latitude": RequiredColumns.LAT,
                "Longitude": RequiredColumns.LON,
                "Planting Date": NullableColumns.PLANTING_DATE,
                "Estimated Harvest Date": NullableColumns.HARVEST_DATE,
                "Crop1": NullableColumns.LABEL,
                "Survey Date": RequiredColumns.COLLECTION_DATE,
            }
        )
        df[NullableColumns.PLANTING_DATE] = pd.to_datetime(
            df[NullableColumns.PLANTING_DATE]
        ).dt.to_pydatetime()
        df[NullableColumns.HARVEST_DATE] = pd.to_datetime(
            df[NullableColumns.HARVEST_DATE]
        ).dt.to_pydatetime()
        df[RequiredColumns.COLLECTION_DATE] = pd.to_datetime(
            df[RequiredColumns.COLLECTION_DATE]
        ).dt.to_pydatetime()
        df[RequiredColumns.EXPORT_END_DATE] = df.apply(export_date_from_row, axis=1)
        df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
            lambda x: LABEL_TO_CLASSIFICATION[x.label], axis=1
        )
        df[RequiredColumns.IS_CROP] = np.where(
            (df[NullableColumns.CLASSIFICATION_LABEL] == "non_crop"), 0, 1
        )
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
        output_df = process_crop_non_crop(base / folder, org_crs=crs)
        if (len(output_df.geometry.type.unique()) == 1) & (
            output_df.geometry.type.unique()[0] == "MultiPoint"
        ):
            # there are multipoints, but they all contain single points,
            # so this is safe to do
            output_df = output_df.explode()
        dfs.append()

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    df[RequiredColumns.COLLECTION_DATE] = datetime(2020, 4, 16)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2020, EXPORT_END_MONTH, EXPORT_END_DAY)

    return df
