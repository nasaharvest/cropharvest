import pandas as pd
import geopandas
from datetime import datetime
from shapely.geometry import Point

from cropharvest.utils import DATASET_PATH
from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH


# Mapping copied from
# https://drive.google.com/file/d/0B5WL29-UKBrdb3BZMmhNREh3dlk/view
CROP_INT_TO_LABEL_CLASSIFICATION = {
    0: (None, None),  # Unknown
    1: ("Wheat", "cereals"),
    2: ("Maize", "cereals"),
    3: ("Rice", "cereals"),
    4: ("Barley", "cereals"),
    5: ("Soybeans", "oilseeds"),
    6: ("Pulses", "leguminous"),  # https://www.thecanadianencyclopedia.ca/en/article/pulse-crops
    7: ("Cotton", "other"),
    8: ("Potatoes", "root_tuber"),
    9: ("Alfalfa", "leguminous"),
    10: ("Sorghum", "cereals"),
    11: ("Millet", "cereals"),
    12: ("Sunflower", "oilseeds"),
    13: ("Rye", "cereals"),
    14: ("Rapeseed or Canola", "oilseeds"),
    15: ("Sugarcane", "sugar"),
    16: ("Groundnuts or Peanuts", "oilseeds"),
    17: ("Cassava", "root_tuber"),
    18: ("Sugarbeets", "sugar"),
    19: ("Palm", "oilseeds"),
}


def load_croplands():
    df = pd.read_csv(DATASET_PATH / "croplands/croplands.csv")

    # the min date of sentinel 2 data is June 2015
    # so the latest data we can take is from the year 2016, where the
    # export will be from Feb 2017 to Feb 2016
    df = df[df.year >= 2016]
    # remove unknown labels
    df = df[df.land_use_type != 0]
    df = df[((df.crop_primary == df.crop_secondary) | (df.crop_secondary == 0))]
    df = df[df.crop_primary <= 19]

    df["is_crop"] = (df.land_use_type == 1).astype(int)
    df["label"] = df.apply(lambda x: CROP_INT_TO_LABEL_CLASSIFICATION[x.crop_primary][0], axis=1)
    df["classification_label"] = df.apply(
        lambda x: CROP_INT_TO_LABEL_CLASSIFICATION[x.crop_primary][1], axis=1
    )

    def get_export_end_date(row: pd.Series) -> datetime:
        if row.month == 1:
            return datetime(row.year, EXPORT_END_MONTH, EXPORT_END_DAY)
        else:
            return datetime(row.year + 1, EXPORT_END_MONTH, EXPORT_END_DAY)

    df["export_end_date"] = df.apply(get_export_end_date, axis=1)
    # remove any rows which export Feb 2016 to Feb 2015, since those would not
    # be covered by Earth Engine's Sentinel 2 data
    df = df[df["export_end_date"] >= datetime(2017, 2, 1)]

    df = df.reset_index(drop=True)
    df["index"] = df.index

    # this data has been continually collected since 2012. This date
    # is the date the csv was exported off the croplands website
    df["collection_date"] = datetime(2021, 4, 29)
    df["geometry"] = df.apply(lambda x: Point(x.lon, x.lat), axis=1)
    geodf = geopandas.GeoDataFrame(df, geometry="geometry")

    return geodf
