from datetime import datetime

from process_labels.loading_funcs.utils import _overlapping_year


def test_overlapping_year():

    test_cases = [
        (2019, datetime(2018, 4, 3), datetime(2019, 4, 3)),
        (2021, datetime(2021, 1, 1), datetime(2021, 2, 1)),
        (2020, datetime(2019, 3, 1), datetime(2019, 4, 1)),
    ]

    for expected_year, planting_date, harvest_date in test_cases:
        assert _overlapping_year(harvest_date, planting_date) == expected_year
