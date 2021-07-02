from geopandas import geopandas
import pandas as pd
from datetime import datetime
import numpy as np

from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH
from cropharvest.columns import RequiredColumns, NullableColumns
from .utils import process_crop_non_crop
from ..utils import DATASET_PATH


LABEL_TO_CLASSIFICATION = {
    "Cerrado": "non_crop",  # natural land in Brazil
    "Pasture": "non_crop",
    "Coffee": "beverage_spice",
    "Uncultivated soil": "non_crop",
    "Hay": "other",
    "Conversion area": "non_crop",  # Cerrado which has been cleared but is not yet cropland
    "Eucalyptus": "other",
    "Brachiaria": "other",
    "Cotton": "other",
}


def load_lem_brazil():

    df = geopandas.read_file(DATASET_PATH / "brazil/lem_brazil")
    df = df[df.geometry.is_valid]

    # check for datapoints where the crops were consistent over a year
    grouped_months = {
        2019: ["Oct_2019", "Nov_2019", "Dec_2019", "Jan_2020"],
        2020: [
            "Feb_2020",
            "Mar_2020",
            "Apr_2020",
            "May_2020",
            "Jun_2020",
            "Jul_2020",
            "Aug_2020",
            "Sep_2020",
        ],
    }

    year_dfs = []
    for year, group in grouped_months.items():
        # construct the condition
        condition = df[group[0]] == df[group[1]]
        for val in group[2:]:
            condition &= df[group[0]] == df[val]

        year_df = df[condition]
        year_df[NullableColumns.LABEL] = df[group[0]]
        # we will say the collection date is the latest possible
        # date in the year_df
        if year == 2020:
            year_df[RequiredColumns.COLLECTION_DATE] = datetime(year, 9, 30)
            year_df[RequiredColumns.EXPORT_END_DATE] = datetime(
                year + 1, EXPORT_END_MONTH, EXPORT_END_DAY
            )
        elif year == 2019:
            year_df[RequiredColumns.COLLECTION_DATE] = datetime(year + 1, 1, 30)
            year_df[RequiredColumns.EXPORT_END_DATE] = datetime(
                year + 1, EXPORT_END_MONTH, EXPORT_END_DAY
            )

        year_dfs.append(year_df)

    df = pd.concat(year_dfs)

    x = df.geometry.centroid.x.values
    y = df.geometry.centroid.y.values

    df[RequiredColumns.LAT] = y
    df[RequiredColumns.LON] = x
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )
    df[RequiredColumns.IS_CROP] = np.where(
        (df[NullableColumns.CLASSIFICATION_LABEL] == "non_crop"), 0, 1
    )

    return df


def load_brazil_noncrop() -> geopandas.GeoDataFrame:

    filepath = DATASET_PATH / "brazil" / "brazil_noncrop"
    df = process_crop_non_crop(filepath)

    df[RequiredColumns.COLLECTION_DATE] = datetime(2021, 2, 1)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2021, EXPORT_END_MONTH, EXPORT_END_DAY)
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df
