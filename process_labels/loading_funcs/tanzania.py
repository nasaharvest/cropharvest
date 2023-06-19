from pathlib import Path
import json
import geopandas
import pandas as pd
import numpy as np
from datetime import datetime
from shapely.geometry import Polygon, Point
from shapely import wkt
from cropharvest.columns import RequiredColumns, NullableColumns
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY

from .utils import _process_copernicusgeoglam, export_date_from_row
from ..utils import DATASET_PATH

from typing import Tuple, List


LABEL_TO_CLASSIFICATION = {
    "Dry Bean": "leguminous",
    "Sunflower": "oilseeds",
    "Bush Bean": "leguminous",
    "Safflower": "oilseeds",
    "White Sorghum": "cereals",
    "Yellow Maize": "cereals",
    "rice": "cereals",
    "maize": "cereals",
}


def convert_date(date_str):
    date_str = date_str.split("T")[0]
    date_str = date_str.split("-")
    year = date_str[0]
    month = date_str[1]
    day = date_str[2]
    return datetime(int(year), int(month), int(day))


def _load_single_stac(path_to_stac: Path) -> List[Tuple[Polygon, str, datetime, datetime]]:
    with (path_to_stac / "labels.geojson").open("r") as f:
        label_json = json.load(f)

        features = label_json["features"]
        fields: List[Tuple[Polygon, str, datetime, datetime]] = []
        for feature in features:
            fields.append(
                (
                    Polygon(feature["geometry"]["coordinates"][0]),
                    feature["properties"]["Crop"],
                    datetime.strptime(feature["properties"]["Planting Date"], "%Y-%m-%d"),
                    datetime.strptime(feature["properties"]["Estimated Harvest Date"], "%Y-%m-%d"),
                )
            )

    return fields


def load_tanzania():
    data_folder = DATASET_PATH / "tanzania"
    # first, get all files
    stac_folders = list(
        (data_folder / "ref_african_crops_tanzania_01_labels").glob(
            "ref_african_crops_tanzania_01_labels*"
        )
    )

    labels: List[str] = []
    polygons: List[Polygon] = []

    all_fields: List[Tuple[Polygon, str, datetime, datetime]] = []
    for stac_folder in stac_folders:
        fields = _load_single_stac(stac_folder)
        all_fields.extend(fields)

    polygons, labels, planting_date, harvest_date = map(list, zip(*all_fields))
    # in the absence of a collection date, we set the collection date to be equal to
    # the planting date
    df = geopandas.GeoDataFrame(
        data={
            NullableColumns.LABEL: labels,
            NullableColumns.PLANTING_DATE: planting_date,
            NullableColumns.HARVEST_DATE: harvest_date,
            RequiredColumns.COLLECTION_DATE: planting_date,
        },
        geometry=polygons,
        crs="EPSG:32736",
    )
    df = df.to_crs("EPSG:4326")

    # isolate the latitude and longitude
    df[RequiredColumns.LON] = df.geometry.centroid.x
    df[RequiredColumns.LAT] = df.geometry.centroid.y

    df[RequiredColumns.EXPORT_END_DATE] = df.apply(export_date_from_row, axis=1)
    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )
    df[RequiredColumns.IS_CROP] = 1
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df


def load_tanzania_ceo():
    ceo_files = (DATASET_PATH / "tanzania" / "ceo_labels").glob("*.csv")

    gdfs: List[geopandas.GeoDataFrame] = []
    for filepath in ceo_files:
        single_df = pd.read_csv(filepath)
        single_df[RequiredColumns.GEOMETRY] = single_df["sample_geom"].apply(wkt.loads)
        single_df["is_crop_mean"] = single_df.apply(
            lambda x: x["Crop/non-Crop"] == "Cropland", axis=1
        )
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
    # np.where(
    # (df['crops'] == 'no'), 0, 1)
    df[RequiredColumns.COLLECTION_DATE] = datetime(2019, 1, 2)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2019, EXPORT_END_MONTH, EXPORT_END_DAY)
    df[RequiredColumns.IS_CROP] = np.where(df["is_crop_mean"] > 0.5, 1, 0)
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df


def load_tanzania_ecaas():
    ecaas_files = (DATASET_PATH / "tanzania" / "tanzania_rice_ecaas").glob("*.csv")

    gdfs: List[geopandas.GeoDataFrame] = []
    for file_path in ecaas_files:
        gdf = geopandas.GeoDataFrame(crs="EPSG:4326")
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
        gdf[NullableColumns.HARVEST_DATE] = df[
            "consent_given/field_planted/harvesting_date"
        ].apply(convert_date)
        gdf[NullableColumns.PLANTING_DATE] = df["consent_given/field_planted/planting_date"].apply(
            convert_date
        )

        gdfs.append(gdf)

    df = pd.concat(gdfs)

    df = df.groupby([RequiredColumns.LON, RequiredColumns.LAT]).agg(
        {
            RequiredColumns.LAT: "first",
            RequiredColumns.LON: "first",
            RequiredColumns.GEOMETRY: "first",
            RequiredColumns.COLLECTION_DATE: "first",
            RequiredColumns.EXPORT_END_DATE: "first",
            NullableColumns.LABEL: "first",
            NullableColumns.CLASSIFICATION_LABEL: "first",
            RequiredColumns.IS_CROP: "first",
            NullableColumns.HARVEST_DATE: "first",
            NullableColumns.PLANTING_DATE: "first",
        }
    )

    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index

    return df


def load_tanzania_copernicusgeoglam():
    df = geopandas.read_file(
        DATASET_PATH
        / ("tanzania/copernicusgeoglam/cop4geoglam_tanzania_aoi_field_data_points.shp")
    )
    return _process_copernicusgeoglam(df, export_end_year=2023)
