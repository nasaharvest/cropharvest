import geopandas
from shapely.geometry import Point

from cropharvest.config import LABELS_FILENAME
from cropharvest.utils import deterministic_shuffle, read_geopandas


def test_deterministic_shuffle():

    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    all_outputs = []
    # also, test a seed which is much larger than the list length
    for seed in list(range(10)) + [42]:
        all_outputs.append(deterministic_shuffle(input_list, seed))
        assert len(all_outputs[-1]) == len(input_list)
        assert len(set(all_outputs[-1])) == len(set(input_list))

    for i in range(1, len(all_outputs)):
        assert all_outputs[0] != all_outputs[i]


def test_labels_share_memory(tmpdir):

    geopandas.GeoDataFrame(
        data={"a": [1, 2, 3]}, geometry=[Point(1, 1), Point(2, 2), Point(3, 3)]
    ).to_file(tmpdir / LABELS_FILENAME, driver="GeoJSON")

    labels = read_geopandas(tmpdir / LABELS_FILENAME)
    labels_2 = read_geopandas(tmpdir / LABELS_FILENAME)

    assert labels is labels_2
