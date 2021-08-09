import numpy as np

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


def test_shuffle(monkeypatch, tmpdir) -> None:
    def __init__(self, root, task=None, download=False):
        self.filepaths = ["a", "a", "b", "b", "b"]
        self.y_vals = [1, 1, 0, 0, 0]

    monkeypatch.setattr(CropHarvest, "__init__", __init__)

    cropharvest = CropHarvest(root=tmpdir)

    for i in range(10):
        cropharvest.shuffle(seed=i)
        for i in cropharvest.positive_indices:
            assert cropharvest.filepaths[i] == "a"
            assert cropharvest.y_vals[i] == 1
        for i in cropharvest.negative_indices:
            assert cropharvest.filepaths[i] == "b"
            assert cropharvest.y_vals[i] == 0


def test_sample(monkeypatch, tmpdir) -> None:
    class mockTask:
        normalize = False

    def __init__(self, root, task=None, download=False):
        self.filepaths = ["a0", "a1", "b0", "b1", "b2"]
        self.y_vals = [1, 1, 0, 0, 0]

        self.positive_indices = [0, 1]
        self.negative_indices = [2, 3, 4]

        self.sampled_positive_indices = []
        self.sampled_negative_indices = []

        self.task = mockTask()

    def __getitem__(self, index):
        x = np.ones(5)
        y = np.array(1)

        return x * self.y_vals[index], y * self.y_vals[index]

    monkeypatch.setattr(CropHarvest, "__init__", __init__)
    monkeypatch.setattr(CropHarvest, "__getitem__", __getitem__)

    cropharvest = CropHarvest(root=tmpdir)

    k = 2
    for is_deterministic in [True, False]:
        x, y = cropharvest.sample(k=k, deterministic=is_deterministic)
        assert x.shape[0] == y.shape[0] == k * 2
        assert y.sum() == k
        for idx, y_val in enumerate(y):
            assert (x[idx] == y_val).all()


def test_crop_harvest_mini(mocker, tmp_path):
    mocked_download_and_extract = mocker.patch("cropharvest.datasets.download_and_extract_archive")
    mocker.patch("cropharvest.datasets.load_normalizing_dict")
    mock_labels = mocker.patch("cropharvest.datasets.CropHarvestLabels")
    mock_labels().construct_positive_and_negative_labels.return_value = [], []

    (tmp_path / FEATURES_DIR).mkdir()
    (tmp_path / TEST_FEATURES_MINI_DIR).mkdir()

    CropHarvest(root=str(tmp_path), is_mini_test=True, download=True)

    mocked_download_and_extract.assert_called_with(str(tmp_path), TEST_FEATURES_MINI_DIR)
