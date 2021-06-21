import geopandas
import numpy as np
from pathlib import Path
from math import sin, cos, sqrt, radians, atan2

from cropharvest import config


EARTH_RADIUS = 6373.0


DATASET_PATH = Path(__file__).parent / "raw_data"


def distance_between_latlons(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS * c


def add_is_test_column(labels: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:

    # adds an `is_test` column for all test datapoints.
    labels["is_test"] = False
    for _, region_bbox in config.TEST_REGIONS.items():
        # we will completely ignore this region, even if it contains some labels not in the
        # year to avoid any potential temporal leakage
        in_region = labels.apply(lambda x: region_bbox.contains(x.lat, x.lon), axis=1)
        labels.loc[in_region, "is_test"] = True

    for test_dataset in config.TEST_DATASETS:
        labels.loc[labels.dataset == test_dataset, "is_test"] = True

    return labels
