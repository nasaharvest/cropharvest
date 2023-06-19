import ee

from cropharvest.bands import SRTM_BANDS as BANDS

image_collection = "USGS/SRTMGL1_003"


def get_single_image(region: ee.Geometry) -> ee.Image:
    elevation = ee.Image(image_collection).clip(region).select(BANDS[0])
    slope = ee.Terrain.slope(elevation)  # this band is already called slope
    together = ee.Image.cat([elevation, slope]).toDouble()

    return together
