from pathlib import Path
import geopandas
import numpy as np
import h5py

from cropharvest.countries import BBox
from cropharvest.utils import download_from_url, filter_geojson, deterministic_shuffle
from cropharvest.config import LABELS_FILENAME, TEST_REGIONS, DEFAULT_SEED, DEFAULT_BALANCE_CROPS
from cropharvest.columns import NullableColumns, RequiredColumns
from cropharvest.engineer import TestInstance

from typing import List, Optional, Tuple, Generator


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

    def _path_from_row(self, row: geopandas.GeoSeries) -> Path:
        return (
            self.root
            / f"features/arrays/{row[RequiredColumns.INDEX]}_{row[RequiredColumns.DATASET]}.h5"
        )

    def construct_positive_and_negative_paths(
        self,
        bounding_box: Optional[BBox],
        target_label: Optional[str],
        filter_test: bool,
        balance_negative_crops: bool,
    ) -> Tuple[List[Path], List[Path]]:
        gpdf = self.as_geojson()
        if filter_test:
            gpdf = gpdf[gpdf[RequiredColumns.IS_TEST] == False]
        if bounding_box is not None:
            gpdf = filter_geojson(gpdf, bounding_box)

        if target_label is not None:
            positive_labels = gpdf[gpdf[NullableColumns.LABEL] == target_label]
            target_label_is_crop = positive_labels.iloc[0][RequiredColumns.IS_CROP]

            if not target_label_is_crop:
                # if the target label is a non crop class (e.g. pasture),
                # then we can just collect all classes which either
                # 1) are crop, or 2) are a different non crop class (e.g. forest)
                negative_labels = gpdf[
                    (
                        (
                            gpdf[NullableColumns.LABEL].isnull()
                            & (gpdf[RequiredColumns.IS_CROP] == True)
                        )
                        | (
                            ~gpdf[NullableColumns.LABEL].isnull()
                            & (gpdf[NullableColumns.LABEL] != target_label)
                        )
                    )
                ]
                negative_paths = [
                    self._path_from_row(row) for _, row in negative_labels.iterrows()
                ]
            else:
                # otherwise, the target label is a crop. If balance_negative_crops is
                # true, then we want an equal number of (other) crops and non crops in
                # the negative labels
                negative_non_crop_labels = gpdf[gpdf[RequiredColumns.IS_CROP] == False]
                negative_other_crop_labels = gpdf[
                    (
                        (gpdf[RequiredColumns.IS_CROP] == True)
                        & (~gpdf[RequiredColumns.LABEL].isnull())
                        & (gpdf[RequiredColumns.LABEL] != target_label)
                    )
                ]
                negative_non_crop_paths = [
                    self._path_from_row(row) for _, row in negative_non_crop_labels.iterrows()
                ]
                negative_paths = [
                    self._path_from_row(row) for _, row in negative_other_crop_labels.iterrows()
                ]

                if balance_negative_crops:
                    negative_paths.extend(
                        deterministic_shuffle(negative_non_crop_paths, DEFAULT_SEED)[
                            : len(negative_paths)
                        ]
                    )
                else:
                    negative_paths.extend(negative_non_crop_paths)
        else:
            # otherwise, we will just filter by crop and non crop
            positive_labels = gpdf[gpdf[RequiredColumns.IS_CROP] == True]
            negative_labels = gpdf[gpdf[RequiredColumns.IS_CROP] == False]
            negative_paths = [self._path_from_row(row) for _, row in negative_labels.iterrows()]

        positive_paths = [self._path_from_row(row) for _, row in positive_labels.iterrows()]

        return positive_paths, negative_paths


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

        self.filepaths: List[Path] = []
        self.positive_indices: List[int] = []
        self.negative_indices: List[int] = []
        self.y_vals: List[int] = []

        self.test_identifier: List[str] = []

    def initialize_paths(
        self, labels: CropHarvestLabels, bounding_box: Optional[BBox], target_label: Optional[str]
    ) -> None:
        positive_paths, negative_paths = labels.construct_positive_and_negative_paths(
            bounding_box, target_label, filter_test=True
        )
        self.filepaths = positive_paths + negative_paths
        self.positive_indices = list(range(len(positive_paths)))
        self.negative_indices = list(
            range(len(positive_paths), len(positive_paths) + len(negative_paths))
        )
        self.y_vals = [1] * len(positive_paths) + [0] * len(negative_paths)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        hf = h5py.File(self.filepaths[index], "r")
        return hf.get("array")[:], self.y_vals[index]

    def test_data(self) -> Generator[Tuple[str, TestInstance], None, None]:
        for task_identifier in self.test_identifier:
            all_relevant_files = list(
                (self.root / "test_features").glob(f"{task_identifier}_*.h5")
            )
            if len(all_relevant_files) == 0:
                raise RuntimeError(f"Missing test data {task_identifier}_*.h5")
            for filepath in all_relevant_files:
                instance_identifier = filepath.name[:-3]  # remove the ".h5"
                hf = h5py.File(filepath, "r")
                test_array = TestInstance.load_from_h5(hf)
                yield instance_identifier, test_array

    @classmethod
    def create_benchmark_datasets(
        cls, labels: CropHarvestLabels, balance_negative_crops: bool = True
    ) -> List:
        raise NotImplementedError

    @classmethod
    def from_labels_and_tifs(cls, labels: CropHarvestLabels, tifs: CropHarvestTifs):
        "Creates CropHarvest dataset from CropHarvestLabels and CropHarvestTifs"
        pass
