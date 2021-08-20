import geopandas
import pandas as pd
import numpy as np
from ..utils import DATASET_PATH
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY
from cropharvest.columns import RequiredColumns, NullableColumns
from datetime import datetime
import zipfile

# This dataset consists of 600,000 labels.
# (More than all other datasets put together)
# This helps to limit the total dataset
# size and makes the export more feasible
MAX_ROWS_PER_LABEL = 100


CATNAME_TO_CLASSIFICATION = {
    "Cereals": "cereals",
    "Oilseeds": "oilseeds",
    "Fruits (Berry & Annual)": "fruits_nuts",
    "Fruits (Trees)": "fruits_nuts",
    "Non-Agriculture": "non_crop",
    "Pulses": "leguminous",
    "Forages": "other",
    "Vegetables": "vegetables_melons",
    "Others": "other",
}


def ms_to_timestamp(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0)


def export_end_date_from_collection_date(collection_int: float) -> datetime:
    collection_date = ms_to_timestamp(collection_int)
    if (collection_date.month > EXPORT_END_MONTH) or (
        collection_date.month == EXPORT_END_MONTH and collection_date.day > EXPORT_END_DAY
    ):
        return datetime(collection_date.year + 1, EXPORT_END_MONTH, EXPORT_END_DAY)
    else:
        return datetime(collection_date.year, EXPORT_END_MONTH, EXPORT_END_DAY)


def load_canada():
    unzipped_file = DATASET_PATH / "canada" / "annual_crop_inventory_ground_truth_data.geojson"
    zipped_file = DATASET_PATH / "canada" / "annual_crop_inventory_ground_truth_data_geoJSON.zip"
    if not unzipped_file.exists():
        with zipfile.ZipFile(zipped_file) as z:
            z.extractall(DATASET_PATH / "canada")

    df = geopandas.read_file(unzipped_file)
    df[RequiredColumns.COLLECTION_DATE] = np.vectorize(ms_to_timestamp)(df.DATE_COLL)
    df[RequiredColumns.EXPORT_END_DATE] = np.vectorize(export_end_date_from_collection_date)(
        df.DATE_COLL
    )

    df = df[df[RequiredColumns.EXPORT_END_DATE].dt.year > 2017]

    all_dfs = []
    for i in df.LANDNAME.unique():
        df_i = df[df.LANDNAME == i][:MAX_ROWS_PER_LABEL]
        df_i[NullableColumns.CLASSIFICATION_LABEL] = CATNAME_TO_CLASSIFICATION[
            df_i.iloc[0].CATNAME
        ]
        df_i[NullableColumns.LABEL] = i
        all_dfs.append(df_i)
    df = pd.concat(all_dfs)

    df[RequiredColumns.LON] = df.geometry.centroid.x
    df[RequiredColumns.LAT] = df.geometry.centroid.y
    df[RequiredColumns.IS_CROP] = (df[NullableColumns.CLASSIFICATION_LABEL] != "non_crop").astype(
        int
    )

    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df
