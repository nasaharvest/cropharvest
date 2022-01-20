import pandas as pd
from datetime import date
from unittest.mock import call, patch
import pytest

from cropharvest.columns import RequiredColumns
from cropharvest.eo import EarthEngineExporter
from cropharvest.eo.eo import get_cloud_tif_list


def mock_bb_from_center(mid_lat, mid_lon, surrounding_metres):
    return (mid_lon - 1, mid_lat - 1, mid_lon + 1, mid_lat + 1), None


@pytest.mark.parametrize("with_identifier", (True, False))
def test_labels_to_polygons_and_years(with_identifier, monkeypatch):
    labels = pd.DataFrame(
        {
            RequiredColumns.LON: [-74.55285616553732],
            RequiredColumns.LAT: [46.22024965230018],
            "start_date": date(2020, 12, 15),
            "end_date": date(2021, 12, 15),
        }
    )
    if with_identifier:
        labels["export_identifier"] = ["hello_world"]

    monkeypatch.setattr(EarthEngineExporter, "_bounding_box_from_centre", mock_bb_from_center)

    output = EarthEngineExporter._labels_to_polygons_and_years(labels, surrounding_metres=80)

    assert len(output) == 1

    _, identifier, _, _ = output[0]

    if with_identifier:
        assert identifier == "hello_world"
    else:
        expected_identifier = (
            "min_lat=45.2202_min_lon=-75.5529_max_lat=47.2202_"
            + "max_lon=-73.5529_dates=2020-12-15_2021-12-15_all"
        )
        assert identifier == expected_identifier


@patch("cropharvest.eo.eo.storage")
def test_get_cloud_tif_list(mock_storage):
    mock_storage.Client().list_blobs("mock_bucket").return_value = []
    tif_list = get_cloud_tif_list("mock_bucket")
    assert tif_list == []


@patch("cropharvest.eo.EarthEngineExporter._export_for_polygon")
def test_export_for_labels(mock_export_for_polygon, monkeypatch):

    start_date_str = "2019-04-22"
    end_date_str = "2020-04-16"

    labels = pd.DataFrame(
        {
            RequiredColumns.LON: [-74.55285616553732, -75.55285616553732],
            RequiredColumns.LAT: [46.22024965230018, 47.22024965230018],
            "end_date": [end_date_str, end_date_str],
            "start_date": [start_date_str, start_date_str],
        }
    )
    monkeypatch.setattr(EarthEngineExporter, "_bounding_box_from_centre", mock_bb_from_center)
    EarthEngineExporter(check_gcp=False, check_ee=False, labels=labels).export_for_labels()

    assert mock_export_for_polygon.call_count == 2

    ending = f"dates={start_date_str}_{end_date_str}_all"
    identifier_1 = f"min_lat=45.2202_min_lon=-75.5529_max_lat=47.2202_max_lon=-73.5529_{ending}"
    identifier_2 = f"min_lat=46.2202_min_lon=-76.5529_max_lat=48.2202_max_lon=-74.5529_{ending}"
    mock_export_for_polygon.assert_has_calls(
        [
            call(
                checkpoint=None,
                days_per_timestep=30,
                end_date=end_date_str,
                polygon=None,
                polygon_identifier=identifier_1,
                start_date=start_date_str,
                test=False,
            ),
            call(
                checkpoint=None,
                days_per_timestep=30,
                end_date=end_date_str,
                polygon=None,
                polygon_identifier=identifier_2,
                start_date=start_date_str,
                test=False,
            ),
        ],
        any_order=True,
    )
