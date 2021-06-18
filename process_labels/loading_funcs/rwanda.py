from pathlib import Path
import json
import geopandas
import pandas as pd
import numpy as np
from datetime import datetime
from shapely import wkt
from shapely.geometry import Polygon

from cropharvest.utils import DATASET_PATH, distance_between_latlons
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY

from typing import Tuple, List, Set


LABEL_TO_CLASSIFICATION = {
    "maize": "cereals",
    "structure": "non_crop",
    "banana": "fruits_nuts",
    "eggplants": "vegetables_melons",
    "climbing_beans": "leguminous",
    "cassava": "root_tuber",
    "forest": "non_crop",
    "fallow": "non_crop",
    "bush_beans": "leguminous",
    "mangoes": "fruits_nuts",
    "tomato_trees": "vegetables_melons",
    "sweet_potatoe": "root_tuber",
    "bare_earth": "non_crop",
    "peas": "leguminous",
    "irish_potatoe": "root_tuber",
    "carrots": "vegetables_melons",
    "avocados": "fruits_nuts",
    "natural_vegetation": "non_crop",
    "wheat": "cereals",
    "fodder": "other",
    "ordinary_beans": "leguminous",
    "passion_fruit": "fruits_nuts",
}


def _load_single_stac(path_to_stac: Path) -> Tuple[Polygon, str]:
    with (path_to_stac / "labels.json").open("r") as f:
        label_json = json.load(f)
        label = label_json["label"]
    with (path_to_stac / "stac.json").open("r") as f:
        stac_json = json.load(f)
        polygon = Polygon(stac_json["geometry"]["coordinates"][0])

    return polygon, label


def load_rwanda():

    data_folder = DATASET_PATH / "rwanda"
    # first, get all files
    stac_folders = list(
        (data_folder / "rti_rwanda_crop_type_labels").glob("rti_rwanda_crop_type_labels*")
    )

    labels: List[str] = []
    polygons: List[Polygon] = []

    for stac_folder in stac_folders:
        polygon, label = _load_single_stac(stac_folder)
        polygons.append(polygon)
        labels.append(label)

    df = geopandas.GeoDataFrame(data={"label": labels}, geometry=polygons)
    # isolate the latitude and longitude
    df["lon"] = df.geometry.centroid.x
    df["lat"] = df.geometry.centroid.y

    # remove unknown labels, and "other_known"
    df = df[df.label != "unknown"]
    df = df[df.label != "other_known"]

    unique_labels = df.label.unique()

    # minimum distance between points of the same labels
    # is set to 100m
    min_distance_in_km = 0.1
    idx_to_remove: Set[int] = set()
    for label in unique_labels:
        label_df = df[df.label == label]
        for idx, row in label_df.iterrows():
            if idx not in idx_to_remove:
                for idx_2, row_2 in label_df.loc[idx + 1 :].iterrows():
                    if idx_2 not in idx_to_remove:
                        distance = distance_between_latlons(row.lat, row.lon, row_2.lat, row_2.lon)
                        if distance <= min_distance_in_km:
                            idx_to_remove.add(idx_2)
    df = df.drop(list(idx_to_remove))

    # https://radiant-mlhub.s3-us-west-2.amazonaws.com/rti-rwanda-crop-type/documentation.pdf
    df["collection_date"] = datetime(2019, 2, 28)
    df["export_end_date"] = datetime(2019, EXPORT_END_MONTH, EXPORT_END_DAY)
    df["classification_label"] = df.apply(lambda x: LABEL_TO_CLASSIFICATION[x.label], axis=1)
    df["is_crop"] = np.where((df["classification_label"] == "non_crop"), 0, 1)
    df = df.reset_index(drop=True)
    df["index"] = df.index
    return df


def load_rwanda_ceo():

    ceo_files = (DATASET_PATH / "rwanda/ceo_labels").glob("*.csv")

    gdfs: List[geopandas.GeoDataFrame] = []
    for filepath in ceo_files:
        single_df = pd.read_csv(filepath)
        single_df["geometry"] = single_df["sample_geom"].apply(wkt.loads)
        single_df["is_crop_mean"] = single_df.apply(
            lambda x: x["Crop/ or not"] == "Cropland", axis=1
        )
        gdfs.append(geopandas.GeoDataFrame(single_df, crs="epsg:4326"))

    df = pd.concat(gdfs)
    df = df.groupby("plot_id").agg(
        {"lon": "first", "lat": "first", "geometry": "first", "is_crop_mean": "mean"}
    )

    df["collection_date"] = datetime(2021, 4, 20)
    df["export_end_date"] = datetime(2021, EXPORT_END_MONTH, EXPORT_END_DAY)
    df["is_crop"] = np.where(df["is_crop_mean"] > 0.5, 1, 0)
    df = df.reset_index(drop=True)
    df["index"] = df.index
    return df
