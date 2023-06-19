import geopandas
import shutil
import tarfile
from shapely.geometry import Point

from cropharvest.config import LABELS_FILENAME, DATASET_URL
from cropharvest.utils import deterministic_shuffle, read_geopandas, download_and_extract_archive


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


def test_download(mocker):
    mocked_download_from_url = mocker.patch("cropharvest.utils.download_from_url")
    download_and_extract_archive(root="", filename="test.geojson")
    url = f"{DATASET_URL}/files/test.geojson?download=1"
    mocked_download_from_url.assert_called_with(url, "/test.geojson")


def test_extract_archive(tmp_path):
    test_dir_path = tmp_path / "test"
    test_dir_path.mkdir()
    assert test_dir_path.exists()

    txt_path = test_dir_path / "hello.txt"
    txt_path.write_text("hi there")
    assert txt_path.exists()

    tarfile_path = tmp_path / "test.tar.gz"
    tar = tarfile.open(str(tarfile_path), "w:gz")
    tar.add(str(txt_path), arcname="test/hello.txt")
    tar.close()
    assert tarfile_path.exists()

    shutil.rmtree(test_dir_path)
    assert not test_dir_path.exists()

    download_and_extract_archive(root=tmp_path, filename="test")
    assert not tarfile_path.exists()
    assert test_dir_path.exists()
    assert txt_path.exists()
