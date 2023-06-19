import geopandas

from process_labels.utils import add_is_test_column


def test_is_test_column():
    labels = geopandas.GeoDataFrame(
        data={
            "dataset": ["togo-eval", "geowiki", "geowiki"],
            "index": [1, 2, 3],
            "lat": [7.5817201079726511, -12.17, 1.11],
            "lon": [1.3954393874414535, -45.8, 8.29],
        }
    )

    filtered_labels = add_is_test_column(labels)

    # index 1 is filtered due to the dataset name
    # index 2 is filtered because it is in the Brazil test region
    assert filtered_labels["is_test"].tolist() == [True, True, False]
