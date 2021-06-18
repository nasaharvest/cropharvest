import ee
from datetime import date

from typing import Union


def make_combine_bands_function(bands):
    def combine_bands(current, previous):
        # Transforms an Image Collection with 1 band per Image into a single
        # Image with items as bands
        # Author: Jamie Vleeshouwer

        # Rename the band
        previous = ee.Image(previous)
        current = current.select(bands)
        # Append it to the result (Note: only return current item on first
        # element/iteration)
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(previous, None),
            current,
            previous.addBands(ee.Image(current)),
        )

    return combine_bands


def date_to_string(input_date: Union[date, str]) -> str:
    if isinstance(input_date, str):
        return input_date
    else:
        assert isinstance(input_date, date)
        return input_date.strftime("%Y-%m-%d")
