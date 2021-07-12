import geopandas
from shapely.geometry import Point

from datetime import datetime

from .utils import process_crop_non_crop
from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH
from cropharvest.columns import RequiredColumns, NullableColumns
from ..utils import DATASET_PATH


LABEL_TO_CLASSIFICATION = {
    "maize": "cereals",
    "rice": "cereals",
    "forage": "other",
    "groundnuts": "oilseeds",
    "beans": "leguminous",
    "potato": "root_tuber",
    "cassava": "root_tuber",
    "sesame": "oilseeds",
    "cotton": "other",
}


def load_malawi_fao():
    df = geopandas.read_file(DATASET_PATH / "malawi/malawi_fao.geojson")

    # remove intercropped labels
    df = df[df.multiple_crops == "no"]

    df = df.rename(
        columns={
            "latitude_center": RequiredColumns.LAT,
            "longitude_center": RequiredColumns.LON,
            "current_season_current_crop": NullableColumns.LABEL,
        }
    )

    df[RequiredColumns.COLLECTION_DATE] = datetime(2021, 6, 1)
    df[RequiredColumns.IS_CROP] = 1
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2021, EXPORT_END_MONTH, EXPORT_END_DAY)
    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )

    df[RequiredColumns.GEOMETRY] = df.apply(
        lambda x: Point(x[RequiredColumns.LON], x[RequiredColumns.LAT]), axis=1
    )

    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index

    return df
