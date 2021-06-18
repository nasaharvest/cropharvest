import geopandas
import pandas as pd
from datetime import datetime

from cropharvest.utils import DATASET_PATH
from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY


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
    df = df.rename(columns={"label_1": "label"})
    df["harvest_date"] = df.apply(make_harvest_date, axis=1)

    df["lon"] = df.geometry.centroid.x
    df["lat"] = df.geometry.centroid.y

    df["is_crop"] = 1
    df["collection_date"] = pd.to_datetime(df.date).dt.to_pydatetime()

    # again, motivated by Figure 4 from
    # https://www.nature.com/articles/s41597-020-00591-2.pdf
    df["export_end_date"] = df.apply(
        lambda x: datetime(x.collection_date.year + 1, EXPORT_END_MONTH, EXPORT_END_DAY), axis=1
    )
    df["classification_label"] = df.apply(lambda x: LABEL_TO_CLASSIFICATION[x.label], axis=1)

    df = df.reset_index(drop=True)
    df["index"] = df.index
    return df
