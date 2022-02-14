import pandas as pd
from datetime import date
from unittest.mock import call, patch
import pytest
import ee

from cropharvest.countries import BBox
from cropharvest.columns import RequiredColumns
from cropharvest.eo import EarthEngineExporter
from cropharvest.eo.eo import get_cloud_tif_list, GOOGLE_CLOUD_STORAGE_INSTALLED, INSTALL_MSG


@pytest.fixture
def mock_polygon(monkeypatch):
    """ee.Geometry.Polygon mocked to return None"""
    monkeypatch.setattr(ee.Geometry, "Polygon", lambda x: None)


@pytest.mark.parametrize("with_identifier", (True, False))
def test_labels_to_polygons_and_years(with_identifier, mock_polygon):
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

    output = EarthEngineExporter._labels_to_polygons_and_years(labels, surrounding_metres=80)

    assert len(output) == 1

    _, identifier, _, _ = output[0]

    if with_identifier:
        assert identifier == "hello_world"
    else:
        expected_identifier = (
            "min_lat=46.2195_min_lon=-74.5539_max_lat=46.221_"
            + "max_lon=-74.5518_dates=2020-12-15_2021-12-15_all"
        )
        assert identifier == expected_identifier


@pytest.mark.skipif(
    not GOOGLE_CLOUD_STORAGE_INSTALLED,
    reason="Google Cloud Storage must be installed for this test.",
)
@patch("cropharvest.eo.eo.storage")
def test_get_cloud_tif_list(mock_storage):
    mock_storage.Client().list_blobs("mock_bucket").return_value = []
    tif_list = get_cloud_tif_list("mock_bucket")
    assert tif_list == []


@pytest.mark.skipif(
    GOOGLE_CLOUD_STORAGE_INSTALLED,
    reason="Google Cloud Storage is installed, no need to run this test.",
)
def test_get_cloud_tif_list_error():
    with pytest.raises(Exception) as e:
        get_cloud_tif_list("mock_bucket")

    assert str(e.value) == f"{INSTALL_MSG} to enable GCP checks"


@patch("cropharvest.eo.EarthEngineExporter._export_for_polygon")
def test_export_for_labels(mock_export_for_polygon, mock_polygon):

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
    EarthEngineExporter(check_gcp=False, check_ee=False).export_for_labels(labels=labels)

    assert mock_export_for_polygon.call_count == 2

    ending = f"dates={start_date_str}_{end_date_str}_all"

    identifier_1 = f"min_lat=46.2195_min_lon=-74.5539_max_lat=46.221_max_lon=-74.5518_{ending}"
    identifier_2 = f"min_lat=47.2195_min_lon=-75.5539_max_lat=47.221_max_lon=-75.5518_{ending}"
    mock_export_for_polygon.assert_has_calls(
        [
            call(
                checkpoint=None,
                end_date=end_date_str,
                polygon=None,
                polygon_identifier=identifier_1,
                start_date=start_date_str,
                test=False,
            ),
            call(
                checkpoint=None,
                end_date=end_date_str,
                polygon=None,
                polygon_identifier=identifier_2,
                start_date=start_date_str,
                test=False,
            ),
        ],
        any_order=True,
    )


@patch("cropharvest.eo.EarthEngineExporter._export_for_polygon")
@pytest.mark.parametrize("metres_per_polygon", (None, 10000))
def test_export_for_bbox(mock_export_for_polygon, metres_per_polygon, mock_polygon):

    start_date, end_date = date(2019, 4, 1), date(2020, 4, 1)
    EarthEngineExporter(check_gcp=False, check_ee=False).export_for_bbox(
        bbox=BBox(min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625),
        bbox_name="Togo",
        start_date=start_date,
        end_date=end_date,
        metres_per_polygon=metres_per_polygon,
    )

    if metres_per_polygon is None:
        assert mock_export_for_polygon.call_count == 1

        mock_export_for_polygon.assert_called_with(
            end_date=end_date,
            polygon=None,
            polygon_identifier=f"Togo_{start_date}_{end_date}/batch/0",
            start_date=start_date,
            file_dimensions=None,
            test=True,
        )
    else:
        assert mock_export_for_polygon.call_count == 1155
        mock_export_for_polygon.assert_has_calls(
            [
                call(
                    end_date=end_date,
                    polygon=None,
                    polygon_identifier=f"Togo_{start_date}_{end_date}/batch_{i}/{i}",
                    start_date=start_date,
                    file_dimensions=None,
                    test=True,
                )
                for i in range(1155)
            ],
            any_order=True,
        )


def test_google_cloud_storage_errors():

    with pytest.raises(Exception) as e:
        EarthEngineExporter(check_gcp=True, check_ee=False)

    assert str(e.value) == "check_gcp was set to True but dest_bucket was not specified"

    if not GOOGLE_CLOUD_STORAGE_INSTALLED:
        with pytest.raises(Exception) as e:
            EarthEngineExporter(check_gcp=True, check_ee=False, dest_bucket="mock_bucket")

        assert str(e.value) == f"{INSTALL_MSG} to enable GCP checks"

        with pytest.raises(Exception) as e:
            EarthEngineExporter(check_gcp=False, check_ee=False, dest_bucket="mock_bucket")

        assert str(e.value) == f"{INSTALL_MSG} to enable export to destination bucket"
