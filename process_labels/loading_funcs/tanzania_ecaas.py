from pathlib import Path
import json
import geopandas
import pandas as pd
from datetime import datetime
from shapely.geometry import Point
from cropharvest.columns import RequiredColumns, NullableColumns
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY

# from ..utils import DATASET_PATH # ImportError: attempted relative import with no known parent package

from typing import List

LABEL_TO_CLASSIFICATION = {
    "rice": "cereals",
    "maize": "cereals",
}


def load_tanzania_ecaas():

    # Field_files = (DATASET_PATH / "tanzania/tanzania_rice_ecaas").glob("*.csv")
    file_path = Path("process_labels/raw_data/tanzania/tanzania_rice_ecaas")
    file_path = file_path / "Field_Mapper_Ver2_2022_04_12_15_21_48_825439.csv"

    gdf = geopandas.GeoDataFrame(crs="EPSG:32736")

    # for filepath in Field_files:
    df = pd.read_csv(file_path)
    # replace NaN with Rice
    df["consent_given/field_planted/primary_crop"].fillna("rice", inplace=True)

    # lat and long
    gdf[RequiredColumns.LAT] = df["consent_given/_field_center_latitude"]
    gdf[RequiredColumns.LON] = df["consent_given/_field_center_longitude"]
    gdf[RequiredColumns.GEOMETRY] = gdf.apply(
        lambda row: Point(row[RequiredColumns.LON], row[RequiredColumns.LAT]), axis=1
    )

    # collection date
    gdf[RequiredColumns.COLLECTION_DATE] = df["end"].apply(convert_date)

    # export date
    gdf[RequiredColumns.EXPORT_END_DATE] = datetime(2022, EXPORT_END_MONTH, EXPORT_END_DAY)

    # label and classification label
    gdf[NullableColumns.LABEL] = df["consent_given/field_planted/primary_crop"]
    gdf[NullableColumns.CLASSIFICATION_LABEL] = gdf.apply(
        lambda row: LABEL_TO_CLASSIFICATION[row[NullableColumns.LABEL]], axis=1
    )

    # manual inputs
    gdf[RequiredColumns.IS_CROP] = 1
    # fill the NANs in the harvest and planting date columns with one of their values
    df["consent_given/field_planted/planting_date"].fillna(
        "2022-01-20T00:00:00.000+03:00", inplace=True
    )

    df["consent_given/field_planted/harvesting_date"].fillna(
        "2022-05-01T00:00:00.000+03:00", inplace=True
    )

    gdf[NullableColumns.HARVEST_DATE] = df["consent_given/field_planted/harvesting_date"].apply(
        convert_date
    )
    gdf[NullableColumns.PLANTING_DATE] = df["consent_given/field_planted/planting_date"].apply(
        convert_date
    )

    df.reset_index(drop=True, inplace=True)
    gdf[RequiredColumns.INDEX] = df.index

    gdf = gdf.to_crs(epsg=4326)

    return print(gdf.head(10))


def convert_date(date_str):
    date_str = date_str.split("T")[0]
    date_str = date_str.split("-")
    year = date_str[0]
    month = date_str[1]
    day = date_str[2]
    return datetime(int(year), int(month), int(day))
