import geopandas
import pandas as pd
from datetime import datetime

from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY
from cropharvest.columns import RequiredColumns, NullableColumns

from ..utils import DATASET_PATH


LABEL_TO_CLASSIFICATION = {
    "cotton": "other",
    "wheat": "cereals",
    "rice": "cereals",
    "orchard": "fruits_nuts",
    "alfalfa": "leguminous",
    "maize": "cereals",
    "vineyard": "fruits_nuts",
}


def load_central_asia():

    df = geopandas.read_file(DATASET_PATH / "central_asia")

    def make_harvest_date(row) -> datetime:
        # this is very approximately taken Figure 4
        # in https://www.nature.com/articles/s41597-020-00591-2.pdf
        # but will be helpful in determining how to export data
        if row["label_2"] == "summer":
            return datetime(int(row["year"]), 6, 15)
        elif row["label_2"] in ["winter", "permanent"]:
            # if its permanent, its in both the summer and
            # winter season, so we will take the later date
            return datetime(int(row["year"]), 12, 15)
        raise AssertionError(f"Unexpected season {row.label_2}")

    df = df[df.year > "2015"]
    # only take unique crops. Double cropping is
    # indicated by the label being CROP1-CROP2.
    df = df[~df.label_1.str.contains("-")]
    # remove unclear seasons
    df = df[~df.label_2.isin(["unclear", "fallow"])]
    df = df.rename(columns={"label_1": NullableColumns.LABEL})
    df["harvest_date"] = df.apply(make_harvest_date, axis=1)

    df[RequiredColumns.LON] = df.geometry.centroid.x
    df[RequiredColumns.LAT] = df.geometry.centroid.y

    df[RequiredColumns.IS_CROP] = 1
    df[RequiredColumns.COLLECTION_DATE] = pd.to_datetime(df.date).dt.to_pydatetime()

    # again, motivated by Figure 4 from
    # https://www.nature.com/articles/s41597-020-00591-2.pdf
    df[RequiredColumns.EXPORT_END_DATE] = df.apply(
        lambda x: datetime(x.collection_date.year + 1, EXPORT_END_MONTH, EXPORT_END_DAY), axis=1
    )
    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )

    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index

    # two manual changes to replace multipolygons with polygons.
    # the first polygon is 10^5 times smaller than the second, so
    # we use the second
    df.loc[df["index"] == 5162, "geometry"] = df.iloc[5162].geometry[1]
    df.loc[df["index"] == 4049, "geometry"] = df.iloc[4049].geometry[1]

    return df
