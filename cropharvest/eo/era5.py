import ee
from datetime import date, timedelta

from cropharvest.bands import ERA5_BANDS as BANDS
from .utils import date_to_string

image_collection = "ECMWF/ERA5_LAND/MONTHLY"


def get_single_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    # This only really works with the values currently in config.
    # What happens is that the images are associated with the first day of the month,
    # so if we just use the given start_date and end_date, then we will often get
    # the image from the following month (e.g. the start and end dates of
    # 2016-02-07, 2016-03-08 respectively return data from March 2016, even though
    # February 2016 has a much higher overlap). It also means that the final month
    # timestep, with range 2017-01-02 to 2017-02-01 was returning no data (but we'd)
    # like it to return data for January
    # TODO: in the future, this could be updated to an overlapping_month function, similar
    # to what happens with the Plant Village labels
    month, year = start_date.month, start_date.year
    start = date(year, month, 1)
    # first day of next month
    end = (date(year, month, 1) + timedelta(days=32)).replace(day=1)

    if (date.today().replace(day=1) - end) < timedelta(days=32):
        raise ValueError(
            f"Cannot get data for range {start} - {end}, please set an earlier end date"
        )

    dates = ee.DateRange(date_to_string(start), date_to_string(end))
    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()

    imcol = (
        ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)
    )

    # there should only be one timestep per daterange, so a mean shouldn't change the values
    return imcol.select(BANDS).mean().toDouble()
