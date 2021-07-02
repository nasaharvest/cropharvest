import pandas as pd
import geopandas
from datetime import datetime
from shapely.geometry import Point

from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH
from cropharvest.columns import RequiredColumns, NullableColumns
from ..utils import DATASET_PATH


LABEL_TO_CLASSIFICATION = {"Maize": "cereals"}


def load_zimbabwe():

    df = pd.read_excel(
        DATASET_PATH / "zimbabwe/zimbabwe_fewsnet_2021_crop_tour_obs_crop_type2.xlsx",
        engine="openpyxl",
    )

    df[RequiredColumns.EXPORT_END_DATE] = datetime(2021, EXPORT_END_MONTH, EXPORT_END_DAY)
    df[RequiredColumns.COLLECTION_DATE] = datetime(2021, 3, 31)
    df[RequiredColumns.IS_CROP] = 1  # all maize labels
    df = df.rename(
        columns={
            "field_lat": RequiredColumns.LAT,
            "field_lon": RequiredColumns.LON,
            "crop_type": NullableColumns.LABEL,
        }
    )
    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index

    geometries = df.apply(lambda x: Point(x[RequiredColumns.LON], x[RequiredColumns.LAT]), axis=1)

    return geopandas.GeoDataFrame(df, geometry=geometries)
