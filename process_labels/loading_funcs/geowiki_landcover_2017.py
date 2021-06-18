import pandas as pd
import geopandas
import numpy as np
from datetime import datetime
from shapely.geometry import Point

from cropharvest.utils import DATASET_PATH
from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH


def load_geowiki_landcover_2017():

    # we will use the data collected from all participants
    data = pd.read_csv(DATASET_PATH / "geowiki_landcover_2017" / "loc_all_2.txt", sep="\t")

    data = (
        data[["location_id", "sumcrop", "loc_cent_X", "loc_cent_Y"]].groupby("location_id").mean()
    )
    data = data.rename(
        {"loc_cent_X": "lon", "loc_cent_Y": "lat", "sumcrop": "mean_sumcrop"},
        axis="columns",
        errors="raise",
    )
    data["is_crop"] = np.where(data["mean_sumcrop"] > 0.5, 1, 0)
    # since we take an average of all the timestamps, the specific time the data was submitted is
    # not so meaningful. Therefore, we take September 2019 (the overall collection month)
    # as the average collection date
    data["collection_date"] = datetime(2016, 9, 30)
    data["export_end_date"] = datetime(2017, EXPORT_END_MONTH, EXPORT_END_DAY)
    data = data.reset_index(drop=True)
    data["index"] = data.index
    data["geometry"] = data.apply(lambda x: Point(x.lon, x.lat), axis=1)

    geodata = geopandas.GeoDataFrame(data, geometry="geometry")
    return geodata
