import geopandas
import pandas as pd
from shapely.geometry import Point

from datetime import datetime

from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH
from cropharvest.columns import RequiredColumns, NullableColumns
from ..utils import DATASET_PATH


LABEL_TO_CLASSIFICATION = {
    "maize": "cereals",
    "sorghum": "cereals",
    "millet": "cereals",
    "rice": "cereals",
    "sesame": "oilseeds",
    "groundnuts": "oilseeds",
    "beans": "leguminous",
    "cotton": "other",
}


def load_malawi():
    df = pd.read_csv(
        DATASET_PATH / "malawi/helmets_crop_type_mapping_2022_04_06_16_20_56_356161.csv"
    )

    # currently don't include intercropped crops
    df = df[df["multiple_crops"] == "no"]

    df[RequiredColumns.LON] = df[
        "field_specification_assessment/_geopoint_widget_placementmap_longitude"
    ]
    df[RequiredColumns.LAT] = df[
        "field_specification_assessment/_geopoint_widget_placementmap_latitude"
    ]
    df[RequiredColumns.COLLECTION_DATE] = pd.to_datetime(df["today"])
    df[RequiredColumns.IS_CROP] = 1

    df[NullableColumns.LABEL] = df["current_season_crop/current_season_current_crop"]
    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2022, EXPORT_END_MONTH, EXPORT_END_DAY)
    df[RequiredColumns.GEOMETRY] = df.apply(
        lambda x: Point(x[RequiredColumns.LON], x[RequiredColumns.LAT]), axis=1
    )

    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index

    return geopandas.GeoDataFrame(df, geometry=RequiredColumns.GEOMETRY)
