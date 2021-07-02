import geopandas
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH, NUM_TIMESTEPS, DAYS_PER_TIMESTEP
from cropharvest.columns import RequiredColumns

from typing import Optional, Dict


LATLON_CRS = "EPSG:4326"


def process_crop_non_crop(filepath: Path, org_crs: Optional[str] = None) -> geopandas.GeoDataFrame:
    r"""
    Quite a few files follow this processing pattern
    """
    df = geopandas.read_file(filepath)

    if (org_crs is not None) and (org_crs != LATLON_CRS):
        try:
            df = df.set_crs(crs=org_crs)
        except AttributeError:
            columns = list(df.columns)
            columns.remove("geometry")

            df = geopandas.GeoDataFrame(data=df[columns], geometry=df.geometry, crs=org_crs)
        df = df.to_crs(LATLON_CRS)

    is_crop = 0 if "non" in filepath.name.lower() else 1

    df[RequiredColumns.IS_CROP] = is_crop

    df[RequiredColumns.LON] = df.geometry.centroid.x
    df[RequiredColumns.LAT] = df.geometry.centroid.y

    df["org_file"] = filepath.name

    return df[
        [
            RequiredColumns.IS_CROP,
            RequiredColumns.GEOMETRY,
            RequiredColumns.LAT,
            RequiredColumns.LON,
            "org_file",
        ]
    ]


def _date_overlap(start1: datetime, end1: datetime, start2: datetime, end2: datetime) -> int:
    overlaps = start1 <= end2 and end1 >= start2
    if not overlaps:
        return 0
    return (min(end1, end2) - max(start1, start2)).days


def _overlapping_year(harvest_date: datetime, planting_date: datetime) -> int:
    r"""
    Return the end_year of the most overlapping years
    """
    harvest_year = harvest_date.year

    overlap_dict: Dict[int, int] = {}

    for diff in range(-1, 2):
        end_date = datetime(harvest_year + diff, EXPORT_END_MONTH, EXPORT_END_DAY)

        overlap_dict[harvest_year + diff] = _date_overlap(
            planting_date,
            harvest_date,
            end_date - timedelta(days=NUM_TIMESTEPS * DAYS_PER_TIMESTEP),
            end_date,
        )

    return max(overlap_dict.items(), key=lambda x: x[1])[0]


def export_date_from_row(row: pd.Series) -> datetime:
    overlapping_year = _overlapping_year(row.harvest_date, row.planting_date)

    return datetime(overlapping_year, EXPORT_END_MONTH, EXPORT_END_DAY)
