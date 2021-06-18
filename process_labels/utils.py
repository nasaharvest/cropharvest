from pathlib import Path
from math import sin, cos, sqrt, radians, atan2


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
