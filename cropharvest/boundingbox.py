from dataclasses import dataclass
from pathlib import Path
from shapely.geometry import Polygon
from math import sin, cos, radians
from typing import List, Tuple
import re

from typing import Optional


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

    def contains_bbox(self, bbox: "BBox") -> bool:
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

    @classmethod
    def from_eo_tif_file(cls, path: Path) -> "BBox":
        decimals_in_p = re.findall(r"=-?\d*\.?\d*", path.stem)
        coords = [float(d[1:]) for d in decimals_in_p[0:4]]
        bbox = cls(min_lat=coords[0], min_lon=coords[1], max_lat=coords[2], max_lon=coords[3])
        return bbox
