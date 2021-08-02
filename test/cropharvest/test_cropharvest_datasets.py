from cropharvest.datasets import CropHarvest
from cropharvest.config import FEATURES_DIR, TEST_FEATURES_MINI_DIR


def test_combination(monkeypatch, tmpdir) -> None:
    def __init__(self, root, task=None, download=False):
        self.filepaths = ["a", "a", "b", "b", "b"]
        self.y_vals = [1, 1, 0, 0, 0]

    monkeypatch.setattr(CropHarvest, "__init__", __init__)

    cropharvest = CropHarvest(root=tmpdir)

    for seed in range(10):
        cropharvest.shuffle(seed=seed)

        for i in range(len(cropharvest)):
            filepath = cropharvest.filepaths[i]
            if filepath == "a":
                assert cropharvest.y_vals[i] == 1
            else:
                assert cropharvest.y_vals[i] == 0


def test_get_positive_negative_indices(monkeypatch, tmpdir) -> None:
    def __init__(self, root, task=None, download=False):
        self.filepaths = ["a", "a", "b", "b", "b"]
        self.y_vals = [1, 1, 0, 0, 0]

    monkeypatch.setattr(CropHarvest, "__init__", __init__)

    cropharvest = CropHarvest(root=tmpdir)

    pos, neg = cropharvest._get_positive_and_negative_indices()

    assert pos == [0, 1]
    assert neg == [2, 3, 4]


def test_crop_harvest_mini(mocker, tmp_path):
    mocked_download_and_extract = mocker.patch("cropharvest.datasets.download_and_extract_archive")
    mocker.patch("cropharvest.datasets.load_normalizing_dict")
    mock_labels = mocker.patch("cropharvest.datasets.CropHarvestLabels")
    mock_labels().construct_positive_and_negative_labels.return_value = [], []

    (tmp_path / FEATURES_DIR).mkdir()
    (tmp_path / TEST_FEATURES_MINI_DIR).mkdir()

    CropHarvest(root=str(tmp_path), is_mini_test=True, download=True)

    mocked_download_and_extract.assert_called_with(str(tmp_path), TEST_FEATURES_MINI_DIR)
