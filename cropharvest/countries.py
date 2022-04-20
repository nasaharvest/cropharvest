import geopandas
from shapely.geometry import Polygon, MultiPolygon
from typing import List
from pathlib import Path

from cropharvest.boundingbox import BBox


COUNTRY_SHAPEFILE = geopandas.read_file(str(Path(__file__).parent / "country_shapefile"))


def get_country_bbox(country_name: str) -> List[BBox]:

    country = COUNTRY_SHAPEFILE[COUNTRY_SHAPEFILE.NAME_EN == country_name]
    if len(country) != 1:
        raise RuntimeError(f"Unrecognized country {country_name}")
    polygon = country.geometry.iloc[0]
    if isinstance(polygon, Polygon):
        return [BBox.polygon_to_bbox(polygon, country_name)]
    elif isinstance(polygon, MultiPolygon):
        bboxes = [
            BBox.polygon_to_bbox(x, f"{country_name}_{idx}") for idx, x in enumerate(polygon.geoms)
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
