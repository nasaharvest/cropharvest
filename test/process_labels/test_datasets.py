import geopandas
from geopandas import array
import pandas as pd

from process_labels import datasets
from cropharvest.config import LABELS_FILENAME, EXPORT_END_DAY, EXPORT_END_MONTH
from cropharvest.crops import CropClassifications


def _check_columns_and_types(df: geopandas.GeoDataFrame) -> None:

    for expected_column, expected_type in [
        ("index", int),
        ("lat", float),
        ("lon", float),
        ("is_crop", int),
        ("collection_date", "<M8[ns]"),
        ("export_end_date", "<M8[ns]"),
    ]:
        assert expected_column in df
        assert df[expected_column].dtype == expected_type

    assert "geometry" in df
    assert type(df["geometry"].dtype) == array.GeometryDtype
    assert "MultiPoint" not in df["geometry"].type.unique()


def _check_lat_lon(df: geopandas.GeoDataFrame) -> None:

    assert df["lon"].max() <= 180
    assert df["lon"].min() >= -180

    assert df["lat"].max() <= 90
    assert df["lat"].min() >= -90


def _check_export_end_date(df: geopandas.GeoDataFrame) -> None:

    assert (df["export_end_date"].dt.month == EXPORT_END_MONTH).all()
    assert (df["export_end_date"].dt.day == EXPORT_END_DAY).all()


def _check_index(df: geopandas.GeoDataFrame) -> None:
    assert len(df["index"].unique()) == len(df)


def _check_labels(df: geopandas.GeoDataFrame) -> None:
    if "label" in df:
        assert "classification_label" in df
        labelled_rows = df[~pd.isnull(df.label)]
        expected_vals = [crop.name for crop in CropClassifications]
        assert labelled_rows.classification_label.isin(expected_vals).all()


def test_consistency() -> None:

    all_datasets = datasets.list_datasets()

    for dataset in all_datasets:
        loaded_dataset = datasets.load(dataset)

        _check_columns_and_types(loaded_dataset)
        _check_lat_lon(loaded_dataset)
        _check_export_end_date(loaded_dataset)
        _check_index(loaded_dataset)
        _check_labels(loaded_dataset)


def test_combination() -> None:

    combined_dataset = datasets.combine_datasets()

    expected_columns = datasets.RequiredColumns.tolist() + datasets.NullableColumns.tolist()

    for column in expected_columns:
        assert column in combined_dataset

    all_datasets = datasets.list_datasets()
    for dataset in all_datasets:
        assert len(combined_dataset[combined_dataset.dataset == dataset]) > 0


def test_dataset_names() -> None:
    dataset_names = datasets.list_datasets()
    for dataset in dataset_names:
        assert "_" not in dataset


def test_update_processed_datasets(monkeypatch, tmp_path):
    def list_datasets():
        # these are two relatively small datasets, so
        # we can run the tests quickly
        return ["ethiopia", "sudan"]

    monkeypatch.setattr("process_labels.datasets.list_datasets", list_datasets)

    # first, write ethiopia to file
    df = datasets.combine_datasets(datasets=["ethiopia"])
    assert len(df) > 0
    df.to_file(tmp_path / LABELS_FILENAME, driver="GeoJSON")

    datasets.update_processed_datasets(tmp_path)

    final_geojson = geopandas.read_file(tmp_path / LABELS_FILENAME)
    for expected_dataset in ["ethiopia", "sudan"]:
        assert expected_dataset in final_geojson[datasets.RequiredColumns.DATASET].unique()
