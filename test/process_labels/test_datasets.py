import geopandas
from process_labels.datasets import combine_datasets, update_processed_datasets
from process_labels.columns import RequiredColumns
from cropharvest.config import LABELS_FILENAME


def test_update_processed_datasets(monkeypatch, tmp_path):
    def list_datasets():
        # these are two relatively small datasets, so
        # we can run the tests quickly
        return ["ethiopia", "sudan"]

    monkeypatch.setattr("process_labels.datasets.list_datasets", list_datasets)

    # first, write ethiopia to file
    df = combine_datasets(datasets=["ethiopia"])
    assert len(df) > 0
    df.to_file(tmp_path / LABELS_FILENAME, driver="GeoJSON")

    update_processed_datasets(tmp_path)

    final_geojson = geopandas.read_file(tmp_path / LABELS_FILENAME)
    for expected_dataset in ["ethiopia", "sudan"]:
        assert expected_dataset in final_geojson[RequiredColumns.DATASET].unique()

    assert (tmp_path / "tmp.geojson").exists() is False
