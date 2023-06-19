import ee
from datetime import date
from typing import Tuple

from cropharvest.bands import S1_BANDS as BANDS
from .utils import date_to_string


image_collection = "COPERNICUS/S1_GRD"


def get_image_collection(
    region: ee.Geometry, start_date: date, end_date: date
) -> Tuple[ee.ImageCollection, ee.ImageCollection]:
    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()

    s1 = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)

    # different areas have either ascending, descending coverage or both.
    # https://sentinel.esa.int/web/sentinel/missions/sentinel-1/observation-scenario
    # we want the coverage to be consistent (so don't want to take both) but also want to
    # take whatever is available
    orbit = s1.filter(
        ee.Filter.eq("orbitProperties_pass", s1.first().get("orbitProperties_pass"))
    ).filter(ee.Filter.eq("instrumentMode", "IW"))

    return (
        orbit.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")),
        orbit.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")),
    )


def _get_closest_dates(mid_date: date, imcol: ee.ImageCollection) -> ee.ImageCollection:
    fifteen_days_in_ms = 1296000000

    mid_date_ee = ee.Date(date_to_string(mid_date))
    # first, order by distance from mid_date
    from_mid_date = imcol.map(
        lambda image: image.set(
            "dateDist",
            ee.Number(image.get("system:time_start")).subtract(mid_date_ee.millis()).abs(),
        )
    )
    from_mid_date = from_mid_date.sort("dateDist", opt_ascending=True)

    # no matter what, we take the first element in the image collection
    # and we add 1 to ensure the less_than condition triggers
    max_diff = ee.Number(from_mid_date.first().get("dateDist")).max(ee.Number(fifteen_days_in_ms))

    kept_images = from_mid_date.filterMetadata("dateDist", "not_greater_than", max_diff)
    return kept_images


def get_single_image(
    region: ee.Geometry,
    start_date: date,
    end_date: date,
    vv_imcol: ee.ImageCollection,
    vh_imcol: ee.ImageCollection,
) -> ee.Image:
    mid_date = start_date + ((end_date - start_date) / 2)

    kept_vv = _get_closest_dates(mid_date, vv_imcol)
    kept_vh = _get_closest_dates(mid_date, vh_imcol)

    composite = ee.Image.cat(
        [
            kept_vv.select("VV").median(),
            kept_vh.select("VH").median(),
        ]
    ).clip(region)

    # rename to the bands
    final_composite = composite.select(BANDS)
    return final_composite
