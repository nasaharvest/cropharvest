import geopandas
import numpy as np
from datetime import datetime

from cropharvest.columns import RequiredColumns, NullableColumns
from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH

from ..utils import DATASET_PATH
from .utils import LATLON_CRS


LABEL_TO_CLASSIFICATION = {
    "Forage Crops": "other",
    "Wheat": "cereals",
    "Meadows": "non_crop",
    "Rye": "cereals",
    "Barley": "cereals",
    "Corn": "cereals",
    "Oil Seeds": "oilseeds",
    "Root Crops": "root_tuber",
    "Oats": "cereals",
}


def load_germany():

    df = geopandas.read_file(DATASET_PATH / "germany/labels.geojson")
    df = df.to_crs(LATLON_CRS)

    df = df.explode()

    # these are the training labels from
    # https://github.com/AI4EO/tum-planet-radearth-ai4food-challenge,
    # which consist of data from 2018
    df[RequiredColumns.LON] = df.geometry.centroid.x.values
    df[RequiredColumns.LAT] = df.geometry.centroid.y.values
    df[RequiredColumns.COLLECTION_DATE] = datetime(2018, 12, 1)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2019, EXPORT_END_MONTH, EXPORT_END_DAY)
    df = df.rename(columns={"crop_name": NullableColumns.LABEL})

    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )
    df[RequiredColumns.IS_CROP] = np.where(
        (df[NullableColumns.CLASSIFICATION_LABEL] == "non_crop"), 0, 1
    )

    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index

    return df
