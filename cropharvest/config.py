from .countries import BBox

from typing import Dict


DAYS_PER_TIMESTEP = 30
NUM_TIMESTEPS = 12

# we will export date until the
# 1st February, at which point
# planting for the long rains can start
# WARNING: some logic in the loading functions (e.g.
# for the central asia loader) assumes a February export
# date, so its recommended the loading functions are reviewed
# if this is changed
EXPORT_END_MONTH = 2
EXPORT_END_DAY = 1

LABELS_FILENAME = "labels.geojson"

# the default seed is useful because it also seeds the deterministic
# shuffling algorithm we use (in cropharvest.utils.deterministic_shuffle)
# so fixing this ensures the evaluation sets consist of the same data no matter
# how they are run.
DEFAULT_SEED = 42


# test regions should have the naming schema
# {country}_{test_crop}_{export_end_year}_{identifer}
TEST_REGIONS: Dict[str, BBox] = {
    "Kenya_maize_2020_0": BBox(
        min_lat=0.47190, max_lat=0.47749, min_lon=34.22847, max_lon=34.23266
    ),
    "Kenya_maize_2020_1": BBox(
        min_lat=0.69365, max_lat=0.69767, min_lon=34.36586, max_lon=34.37199
    ),
    "Brazil_coffee_2020_0": BBox(
        min_lat=-12.1995, max_lat=-12.1226, min_lon=-45.8238, max_lon=-45.7579
    ),
    "Brazil_coffee_2021_0": BBox(
        min_lat=-12.1995, max_lat=-12.1226, min_lon=-45.8238, max_lon=-45.7579
    ),
    "United States of America_almond_2021_0": BBox(
        min_lon=-119.35901, max_lon=-118.7524, min_lat=34.9329, max_lat=35.34763
    ),
}

TEST_DATASETS = {"Togo": "togo-eval"}
