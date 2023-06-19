from dataclasses import dataclass
from math import cos, radians
from typing import List, Tuple, Union
import ee

from cropharvest.countries import BBox


@dataclass
class EEBoundingBox(BBox):
    r"""
    A bounding box with additional earth-engine specific
    functionality
    """

    def to_ee_polygon(self) -> ee.Geometry.Polygon:
        return ee.Geometry.Polygon(
            [
                [
                    [self.min_lon, self.min_lat],
                    [self.min_lon, self.max_lat],
                    [self.max_lon, self.max_lat],
                    [self.max_lon, self.min_lat],
                ]
            ]
        )

    def to_metres(self) -> Tuple[float, float]:
        r"""
        :return: [lat metres, lon metres]
        """
        # https://gis.stackexchange.com/questions/75528/understanding-terms-in-length-of-degree-formula
        mid_lat = (self.min_lat + self.max_lat) / 2.0
        m_per_deg_lat, m_per_deg_lon = self.metre_per_degree(mid_lat)

        delta_lat = self.max_lat - self.min_lat
        delta_lon = self.max_lon - self.min_lon

        return delta_lat * m_per_deg_lat, delta_lon * m_per_deg_lon

    def to_polygons(self, metres_per_patch: int = 3300) -> List[ee.Geometry.Polygon]:
        lat_metres, lon_metres = self.to_metres()

        num_cols = int(lon_metres / metres_per_patch)
        num_rows = int(lat_metres / metres_per_patch)

        if num_cols == 0 or num_rows == 0:
            print(
                f"A single patch (metres_per_patch={metres_per_patch}) is "
                f"bigger than the requested bounding box."
            )
        if num_cols == 0:
            num_cols = 1
        if num_rows == 0:
            num_rows = 1

        print(f"Splitting into {num_cols} columns and {num_rows} rows")

        lon_size = (self.max_lon - self.min_lon) / num_cols
        lat_size = (self.max_lat - self.min_lat) / num_rows

        output_polygons: List[ee.Geometry.Polygon] = []

        cur_lon = self.min_lon
        while cur_lon < self.max_lon:
            cur_lat = self.min_lat
            while cur_lat < self.max_lat:
                output_polygons.append(
                    ee.Geometry.Polygon(
                        [
                            [
                                [cur_lon, cur_lat],
                                [cur_lon, cur_lat + lat_size],
                                [cur_lon + lon_size, cur_lat + lat_size],
                                [cur_lon + lon_size, cur_lat],
                            ]
                        ]
                    )
                )
                cur_lat += lat_size
            cur_lon += lon_size

        return output_polygons

    @staticmethod
    def metre_per_degree(lat: float) -> Tuple[float, float]:
        # https://gis.stackexchange.com/questions/75528/understanding-terms-in
        # -length-of-degree-formula
        # see the link above to explain the magic numbers
        m_per_degree_lat = (
            111132.954
            + (-559.822 * cos(radians(2.0 * lat)))
            + (1.175 * cos(radians(4.0 * lat)))
            + (-0.0023 * cos(radians(6 * lat)))
        )
        m_per_degree_lon = (
            (111412.84 * cos(radians(lat)))
            + (-93.5 * cos(radians(3 * lat)))
            + (0.118 * cos(radians(5 * lat)))
        )

        return m_per_degree_lat, m_per_degree_lon

    @staticmethod
    def from_centre(
        mid_lat: float, mid_lon: float, surrounding_metres: Union[int, Tuple[int, int]]
    ) -> "EEBoundingBox":
        m_per_deg_lat, m_per_deg_lon = EEBoundingBox.metre_per_degree(mid_lat)

        if isinstance(surrounding_metres, int):
            surrounding_metres = (surrounding_metres, surrounding_metres)

        surrounding_lat, surrounding_lon = surrounding_metres

        deg_lat = surrounding_lat / m_per_deg_lat
        deg_lon = surrounding_lon / m_per_deg_lon

        max_lat, min_lat = mid_lat + deg_lat, mid_lat - deg_lat
        max_lon, min_lon = mid_lon + deg_lon, mid_lon - deg_lon

        return EEBoundingBox(max_lon=max_lon, min_lon=min_lon, max_lat=max_lat, min_lat=min_lat)

    @staticmethod
    def from_bounding_box(bounding_box: BBox, padding_metres: int) -> ee.Geometry.Polygon:
        # get the mid lat, in degrees (the bounding box function returns it in radians)
        mid_lat, _ = bounding_box.get_centre(in_radians=False)
        m_per_deg_lat, m_per_deg_lon = EEBoundingBox.metre_per_degree(mid_lat)

        extra_degrees_lon = padding_metres / m_per_deg_lon
        extra_degrees_lat = padding_metres / m_per_deg_lat

        min_lon = bounding_box.min_lon - extra_degrees_lon
        max_lon = bounding_box.max_lon + extra_degrees_lon
        min_lat = bounding_box.min_lat - extra_degrees_lat
        max_lat = bounding_box.max_lat + extra_degrees_lat

        return EEBoundingBox(max_lat=max_lat, min_lat=min_lat, max_lon=max_lon, min_lon=min_lon)
