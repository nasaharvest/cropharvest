from pathlib import Path

import geopandas

from cropharvest.utils import download_from_url
from cropharvest.config import LABELS_FILENAME


class BaseDataset:
    def __init__(self, root, download: bool, download_url: str, filename: str):
        self.root = Path(root)
        if download:
            download_from_url(download_url, str(root / filename))

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CropHarvestLabels(BaseDataset):
    url = "https://zenodo.org/record/5021762/files/labels.geojson?download=1"

    def __init__(self, root, download=False):
        super().__init__(root, download, download_url=self.url, filename=LABELS_FILENAME)
        self._labels = geopandas.read_file(str(self.root / LABELS_FILENAME))

    def as_geojson(self):
        return self._labels

    def __getitem__(self, index: int):
        return self._labels.iloc[index]

    def __len__(self) -> int:
        return len(self._labels)


class CropHarvestTifs(BaseDataset):

    def __init__(self, root, download=False):
        super().__init__(root, download, download_url="", filename="")

    @classmethod
    def from_labels(cls):
        pass


class CropHarvest(BaseDataset):
    "Dataset consisting of satellite data and associate labels"

    def __init__(self, root, download=False):
        super().__init__(root, download, download_url="", filename="")

    @classmethod
    def from_labels_and_tifs(cls, labels: CropHarvestLabels, tifs: CropHarvestTifs):
        "Creates CropHarvest dataset from CropHarvestLabels and CropHarvestTifs"
        pass
