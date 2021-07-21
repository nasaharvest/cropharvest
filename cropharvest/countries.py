from dataclasses import dataclass
import geopandas
from shapely.geometry import Polygon, MultiPolygon
from math import sin, cos, radians
from typing import List, Tuple
from pathlib import Path

from typing import Optional

COUNTRY_SHAPEFILE = geopandas.read_file(str(Path(__file__).parent / "country_shapefile"))


@dataclass
class BBox:

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    name: Optional[str] = None

    def __post_init__(self):
        if self.max_lon < self.min_lon:
            raise ValueError("max_lon should be larger than min_lon")
        if self.max_lat < self.min_lat:
            raise ValueError("max_lat should be larger than min_lat")

        self.url = (
            f"http://bboxfinder.com/#{self.min_lat},{self.min_lon},{self.max_lat},{self.max_lon}"
        )

    def contains(self, lat: float, lon: float) -> bool:
        return (
            (lat >= self.min_lat)
            & (lat <= self.max_lat)
            & (lon >= self.min_lon)
            & (lon <= self.max_lon)
        )

    def contains_bbox(self, bbox) -> bool:
        return (
            (bbox.min_lat >= self.min_lat)
            & (bbox.max_lat <= self.max_lat)
            & (bbox.min_lon >= self.min_lon)
            & (bbox.max_lon <= self.max_lon)
        )

    @property
    def three_dimensional_points(self) -> List[float]:
        r"""
        If we are passing the central latitude and longitude to
        an ML model, we want it to know the extremes are close together.
        Mapping them to 3d space allows us to do that
        """
        lat, lon = self.get_centre(in_radians=True)
        return [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]

    def get_centre(self, in_radians: bool = True) -> Tuple[float, float]:

        # roughly calculate the centres
        lat = self.min_lat + ((self.max_lat - self.min_lat) / 2)
        lon = self.min_lon + ((self.max_lon - self.min_lon) / 2)
        if in_radians:
            return radians(lat), radians(lon)
        else:
            return lat, lon

    @classmethod
    def polygon_to_bbox(cls, polygon: Polygon, name: Optional[str] = None):
        (min_lon, min_lat, max_lon, max_lat) = polygon.bounds
        return cls(min_lat, max_lat, min_lon, max_lon, name)


def get_country_bbox(country_name: str) -> List[BBox]:

    country = COUNTRY_SHAPEFILE[COUNTRY_SHAPEFILE.NAME_EN == country_name]
    if len(country) != 1:
        raise RuntimeError(f"Unrecognized country {country_name}")
    polygon = country.geometry.iloc[0]
    if isinstance(polygon, Polygon):
        return [BBox.polygon_to_bbox(polygon, country_name)]
    elif isinstance(polygon, MultiPolygon):
        bboxes = [
            BBox.polygon_to_bbox(x, f"{country_name}_{idx}") for idx, x in enumerate(polygon)
        ]
        # we want to remove any bounding boxes which are contained within
        # another bounding box
        indices_to_remove = set()
        for big_idx in range(len(bboxes)):
            for small_idx in range(len(bboxes)):
                if big_idx == small_idx:
                    continue
                elif small_idx in indices_to_remove:
                    continue
                else:
                    if bboxes[big_idx].contains_bbox(bboxes[small_idx]):
                        indices_to_remove.add(small_idx)
        return [box for i, box in enumerate(bboxes) if i not in indices_to_remove]
    raise RuntimeError(f"Unrecognize geometry {type(polygon)}")


def get_countries() -> List[str]:
    return list(COUNTRY_SHAPEFILE.NAME_EN.unique())
