from dataclasses import dataclass
from cartopy.io import shapereader
import geopandas
from shapely.geometry import Polygon, MultiPolygon
from math import sin, cos, radians

from typing import List, Tuple


COUNTRY_SHAPEFILE = geopandas.read_file(
    shapereader.natural_earth("50m", "cultural", "admin_0_countries")
)


@dataclass
class BBox:

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        if (
            (lat >= self.min_lat)
            & (lat <= self.max_lat)
            & (lon >= self.min_lon)
            & (lon <= self.max_lon)
        ):
            return True
        return False

    def contains_bbox(self, bbox) -> bool:
        if (
            (bbox.min_lat >= self.min_lat)
            & (bbox.max_lat <= self.max_lat)
            & (bbox.min_lon >= self.min_lon)
            & (bbox.max_lon <= self.max_lon)
        ):
            return True
        return False

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


def _polygon_to_bbox(polygon: Polygon) -> BBox:

    (min_lon, min_lat, max_lon, max_lat) = polygon.bounds
    return BBox(min_lat, max_lat, min_lon, max_lon)


def get_country_bbox(country_name: str) -> List[BBox]:

    country = COUNTRY_SHAPEFILE[COUNTRY_SHAPEFILE.NAME_EN == country_name]
    if len(country) == 1:
        polygon = country.geometry.iloc[0]
        if isinstance(polygon, Polygon):
            return [_polygon_to_bbox(polygon)]
        elif isinstance(polygon, MultiPolygon):
            bboxes = [_polygon_to_bbox(x) for x in polygon]
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
        else:
            raise RuntimeError(f"Unrecognize geometry {type(polygon)}")
    else:
        raise RuntimeError(f"Unrecognized country {country_name}")


def get_countries() -> List[str]:
    return list(COUNTRY_SHAPEFILE.NAME_EN.unique())
