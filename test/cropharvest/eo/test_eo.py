import pandas as pd
from datetime import date
import pytest

from cropharvest.eo import EarthEngineExporter


@pytest.mark.parametrize("with_identifier", (True, False))
def test_labels_to_polygons_and_years(with_identifier, monkeypatch):
    data = {
        "lon": [-74.55285616553732],
        "lat": [46.22024965230018],
        "start_date": date(2020, 12, 15),
        "end_date": date(2021, 12, 15),
    }
    if with_identifier:
        data["export_identifier"] = ["hello_world"]
    labels = pd.DataFrame(data)

    def mock_bb_from_center(mid_lat, mid_lon, surrounding_metres):
        return (mid_lon - 1, mid_lat - 1, mid_lon + 1, mid_lat + 1), None

    monkeypatch.setattr(EarthEngineExporter, "_bounding_box_from_centre", mock_bb_from_center)

    output = EarthEngineExporter._labels_to_polygons_and_years(labels, surrounding_metres=80)

    assert len(output) == 1

    _, identifier, _, _ = output[0]

    if with_identifier:
        assert identifier == "hello_world"
    else:
        assert (
            identifier
            == "min_lat=45.2202_min_lon=-75.5529_max_lat=47.2202_max_lon=-73.5529_dates=2020-12-15_2021-12-15_all"
        )
