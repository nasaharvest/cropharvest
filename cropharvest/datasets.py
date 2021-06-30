from pathlib import Path
import numpy as np

import geopandas

from cropharvest.countries import BBox
from cropharvest.utils import download_from_url
from cropharvest.config import LABELS_FILENAME
from cropharvest.columns import RequiredColumns


class BaseDataset:
    def __init__(self, root, download: bool, download_url: str, filename: str):
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"{root} should be a directory.")

        path_to_data = self.root / filename

        if download:
            if path_to_data.exists():
                print("Files already downloaded.")
            else:
                download_from_url(download_url, str(path_to_data))

        if not path_to_data.exists():
            raise FileNotFoundError(
                f"{path_to_data} does not exist, it can be downloaded by setting download=True"
            )

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CropHarvestLabels(BaseDataset):
    url = "https://zenodo.org/record/5021762/files/labels.geojson?download=1"

    def __init__(self, root, download=False):
        super().__init__(root, download, download_url=self.url, filename=LABELS_FILENAME)
        self._labels = geopandas.read_file(str(self.root / LABELS_FILENAME))

    def as_geojson(self) -> geopandas.GeoDataFrame:
        return self._labels

    def __getitem__(self, index: int):
        return self._labels.iloc[index]

    def __len__(self) -> int:
        return len(self._labels)

    def _filter_geojson(self, bounding_box: BBox) -> geopandas.GeoDataFrame:
        in_bounding_box = np.vectorize(bounding_box.contains)(
            self._labels[RequiredColumns.LAT], self._labels[RequiredColumns.LON]
        )
        return self._labels[in_bounding_box]


class CropHarvestTifs(BaseDataset):
    def __init__(self, root, download=False):
        super().__init__(root, download, download_url="", filename="")

    @classmethod
    def from_labels(cls):
        pass


class CropHarvest(BaseDataset):
    "Dataset consisting of satellite data and associated labels"

    def __init__(self, root, download=False):
        super().__init__(root, download, download_url="", filename="")
        self.labels = CropHarvestLabels(root, download)

    @classmethod
    def from_labels_and_tifs(cls, labels: CropHarvestLabels, tifs: CropHarvestTifs):
        "Creates CropHarvest dataset from CropHarvestLabels and CropHarvestTifs"
        pass
