import geopandas
import numpy as np
from pathlib import Path

from cropharvest import config
from cropharvest.columns import RequiredColumns


EARTH_RADIUS = 6373.0


DATASET_PATH = Path(__file__).parent / "raw_data"


def add_is_test_column(labels: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    # adds an `is_test` column for all test datapoints.
    labels[RequiredColumns.IS_TEST] = False
    for _, region_bbox in config.TEST_REGIONS.items():
        # we will completely ignore this region, even if it contains some labels not in the
        # year to avoid any potential temporal leakage
        in_region = np.vectorize(region_bbox.contains)(
            labels[RequiredColumns.LAT], labels[RequiredColumns.LON]
        )
        labels.loc[in_region, RequiredColumns.IS_TEST] = True

    for _, test_dataset in config.TEST_DATASETS.items():
        labels.loc[labels[RequiredColumns.DATASET] == test_dataset, RequiredColumns.IS_TEST] = True

    return labels
