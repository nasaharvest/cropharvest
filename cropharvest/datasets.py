from pathlib import Path
import geopandas
import math
import numpy as np
import h5py
import warnings
from dataclasses import dataclass

from cropharvest.countries import BBox
from cropharvest.utils import (
    download_and_extract_archive,
    deterministic_shuffle,
    read_geopandas,
    load_normalizing_dict,
    sample_with_memory,
    NoDataForBoundingBoxError,
)
from cropharvest.config import (
    FEATURES_DIR,
    TEST_FEATURES_DIR,
    LABELS_FILENAME,
    TEST_REGIONS,
    TEST_DATASETS,
)
from cropharvest.columns import NullableColumns, RequiredColumns
from cropharvest.engineer import TestInstance
from cropharvest import countries

from typing import cast, List, Optional, Tuple, Generator


@dataclass
class Task:
    bounding_box: Optional[BBox] = None
    target_label: Optional[str] = None
    balance_negative_crops: bool = False
    test_identifier: Optional[str] = None
    normalize: bool = True
    include_externally_contributed_labels: bool = True

    def __post_init__(self):
        if self.target_label is None:
            self.target_label = "crop"
            if self.balance_negative_crops is True:
                warnings.warn(
                    "Balance negative crops not meaningful for the crop vs. non crop tasks"
                )

        if self.bounding_box is None:
            self.bounding_box = BBox(
                min_lat=-90, max_lat=90, min_lon=-180, max_lon=180, name="global"
            )

    @property
    def id(self) -> str:
        return f"{cast(BBox, self.bounding_box).name}_{self.target_label}"


class BaseDataset:
    def __init__(self, root, download: bool, filenames: Tuple[str, ...]):
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"{root} should be a directory.")

        for filename in filenames:
            if download:
                download_and_extract_archive(root, filename)

            if not (self.root / filename).exists():
                raise FileNotFoundError(
                    f"{filename} does not exist in {root}, "
                    f"it can be downloaded by setting download=True"
                )

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CropHarvestLabels(BaseDataset):
    def __init__(self, root, download=False):
        super().__init__(root, download, filenames=(LABELS_FILENAME,))

        # self._labels will always contain the original dataframe;
        # the CropHarvestLabels class should not modify it
        self._labels = read_geopandas(self.root / LABELS_FILENAME)

    def as_geojson(self) -> geopandas.GeoDataFrame:
        return self._labels

    @staticmethod
    def filter_geojson(
        gpdf: geopandas.GeoDataFrame, bounding_box: BBox, include_external_contributions: bool
    ) -> geopandas.GeoDataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # warning: invalid value encountered in ? (vectorized)
            include_condition = np.vectorize(bounding_box.contains)(
                gpdf[RequiredColumns.LAT], gpdf[RequiredColumns.LON]
            )
        if not include_external_contributions:
            include_condition &= gpdf[
                gpdf[RequiredColumns.EXTERNALLY_CONTRIBUTED_DATASET] == False
            ]
        return gpdf[include_condition]

    def classes_in_bbox(
        self, bounding_box: BBox, include_external_contributions: bool
    ) -> List[str]:
        bbox_geojson = self.filter_geojson(
            self.as_geojson(), bounding_box, include_external_contributions
        )
        unique_labels = [x for x in bbox_geojson.label.unique() if x is not None]
        return unique_labels

    def __getitem__(self, index: int):
        return self._labels.iloc[index]

    def __len__(self) -> int:
        return len(self._labels)

    def construct_positive_and_negative_labels(
        self, task: Task, filter_test: bool = True
    ) -> Tuple[List[Path], List[Path]]:
        gpdf = self.as_geojson()
        if filter_test:
            gpdf = gpdf[gpdf[RequiredColumns.IS_TEST] == False]
        if task.bounding_box is not None:
            gpdf = self.filter_geojson(
                gpdf, task.bounding_box, task.include_externally_contributed_labels
            )

        if len(gpdf) == 0:
            raise NoDataForBoundingBoxError

        is_null = gpdf[NullableColumns.LABEL].isnull()
        is_crop = gpdf[RequiredColumns.IS_CROP] == True

        try:
            if task.target_label != "crop":
                positive_labels = gpdf[gpdf[NullableColumns.LABEL] == task.target_label]
                target_label_is_crop = positive_labels.iloc[0][RequiredColumns.IS_CROP]

                is_target = gpdf[NullableColumns.LABEL] == task.target_label

                if not target_label_is_crop:
                    # if the target label is a non crop class (e.g. pasture),
                    # then we can just collect all classes which either
                    # 1) are crop, or 2) are a different non crop class (e.g. forest)
                    negative_labels = gpdf[((is_null & is_crop) | (~is_null & ~is_target))]
                    negative_paths = self._dataframe_to_paths(negative_labels)
                else:
                    # otherwise, the target label is a crop. If balance_negative_crops is
                    # true, then we want an equal number of (other) crops and non crops in
                    # the negative labels
                    negative_non_crop_labels = gpdf[~is_crop]
                    negative_other_crop_labels = gpdf[(is_crop & ~is_null & ~is_target)]
                    negative_non_crop_paths = self._dataframe_to_paths(negative_non_crop_labels)
                    negative_paths = self._dataframe_to_paths(negative_other_crop_labels)

                    if task.balance_negative_crops:
                        multiplier = math.ceil(len(negative_non_crop_paths) / len(negative_paths))
                        negative_paths *= multiplier
                        negative_paths.extend(negative_non_crop_paths)
                    else:
                        negative_paths.extend(negative_non_crop_paths)
            else:
                # otherwise, we will just filter by crop and non crop
                positive_labels = gpdf[is_crop]
                negative_labels = gpdf[~is_crop]
                negative_paths = self._dataframe_to_paths(negative_labels)
        except IndexError:
            raise NoDataForBoundingBoxError

        positive_paths = self._dataframe_to_paths(positive_labels)

        if (len(positive_paths) == 0) or (len(negative_paths) == 0):
            raise NoDataForBoundingBoxError

        return [x for x in positive_paths if x.exists()], [x for x in negative_paths if x.exists()]

    def _path_from_row(self, row: geopandas.GeoSeries) -> Path:
        return (
            self.root
            / f"features/arrays/{row[RequiredColumns.INDEX]}_{row[RequiredColumns.DATASET]}.h5"
        )

    def _dataframe_to_paths(self, df: geopandas.GeoDataFrame) -> List[Path]:
        return [self._path_from_row(row) for _, row in df.iterrows()]


class CropHarvestTifs(BaseDataset):
    def __init__(self, root, download=False):
        super().__init__(root, download, filenames=())

    @classmethod
    def from_labels(cls):
        pass


class CropHarvest(BaseDataset):
    """Dataset consisting of satellite data and associated labels"""

    def __init__(
        self,
        root,
        task: Optional[Task] = None,
        download=False,
        val_ratio: float = 0.0,
        is_val: bool = False,
    ):
        super().__init__(root, download, filenames=(FEATURES_DIR, TEST_FEATURES_DIR))

        labels = CropHarvestLabels(root, download=download)
        if task is None:
            print("Using the default task; crop vs. non crop globally")
            task = Task()
        self.task = task

        self.normalizing_dict = load_normalizing_dict(
            Path(root) / f"{FEATURES_DIR}/normalizing_dict.h5"
        )

        positive_paths, negative_paths = labels.construct_positive_and_negative_labels(
            task, filter_test=True
        )
        if val_ratio > 0.0:
            # the fixed seed is to ensure the validation set is always
            # different from the training set
            positive_paths = deterministic_shuffle(positive_paths, seed=42)
            negative_paths = deterministic_shuffle(negative_paths, seed=42)
            if is_val:
                positive_paths = positive_paths[: int(len(positive_paths) * val_ratio)]
                negative_paths = negative_paths[: int(len(negative_paths) * val_ratio)]
            else:
                positive_paths = positive_paths[int(len(positive_paths) * val_ratio) :]
                negative_paths = negative_paths[int(len(negative_paths) * val_ratio) :]

        self.filepaths: List[Path] = positive_paths + negative_paths
        self.y_vals: List[int] = [1] * len(positive_paths) + [0] * len(negative_paths)
        self.positive_indices = list(range(len(positive_paths)))
        self.negative_indices = list(
            range(len(positive_paths), len(positive_paths) + len(negative_paths))
        )

        # used in the sample() function, to ensure filepaths are sampled without
        # duplication as much as possible
        self.sampled_positive_indices: List[int] = []
        self.sampled_negative_indices: List[int] = []

    def reset_sampled_indices(self) -> None:
        self.sampled_positive_indices = []
        self.sampled_negative_indices = []

    def shuffle(self, seed: int) -> None:
        self.reset_sampled_indices()
        filepaths_and_y_vals = list(zip(self.filepaths, self.y_vals))
        filepaths_and_y_vals = deterministic_shuffle(filepaths_and_y_vals, seed)
        filepaths, y_vals = zip(*filepaths_and_y_vals)
        self.filepaths, self.y_vals = list(filepaths), list(y_vals)

        self.positive_indices, self.negative_indices = self._get_positive_and_negative_indices()

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        hf = h5py.File(self.filepaths[index], "r")
        return self._normalize(hf.get("array")[:]), self.y_vals[index]

    @property
    def k(self) -> int:
        return min(len(self.positive_indices), len(self.negative_indices))

    @property
    def num_bands(self) -> int:
        # array has shape [timesteps, bands]
        return self[0][0].shape[-1]

    def as_array(
        self, flatten_x: bool = False, num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Return the training data as a tuple of
        np.ndarrays

        :param flatten_x: If True, the X array will have shape [num_samples, timesteps * bands]
            instead of [num_samples, timesteps, bands]
        :param num_samples: If -1, all data is returned. Otherwise, a balanced dataset of
            num_samples / 2 positive (& negative) samples will be returned
        """

        if num_samples is None:
            indices_to_sample = list(range(len(self)))
        else:
            k = num_samples // 2

            pos_indices, neg_indices = self._get_positive_and_negative_indices()
            if (k > len(pos_indices)) or (k > len(neg_indices)):
                raise ValueError(
                    f"num_samples // 2 ({k}) is greater than the number of "
                    f"positive samples ({len(pos_indices)}) "
                    f"or the number of negative samples ({len(neg_indices)})"
                )
            indices_to_sample = pos_indices[:k] + neg_indices[:k]

        X, Y = zip(*[self[i] for i in indices_to_sample])
        X_np, y_np = np.stack(X), np.stack(Y)

        if flatten_x:
            X_np = self._flatten_array(X_np)
        return X_np, y_np

    def test_data(
        self, flatten_x: bool = False, max_size: Optional[int] = None
    ) -> Generator[Tuple[str, TestInstance], None, None]:
        r"""
        A generator returning TestInstance objects containing the test
        inputs, ground truths and associated latitudes nad longitudes

        :param flatten_x: If True, the TestInstance.x will have shape
            [num_samples, timesteps * bands] instead of [num_samples, timesteps, bands]
        """
        all_relevant_files = list(
            (self.root / TEST_FEATURES_DIR).glob(f"{self.task.test_identifier}*.h5")
        )
        if len(all_relevant_files) == 0:
            raise RuntimeError(f"Missing test data {self.task.test_identifier}*.h5")
        for filepath in all_relevant_files:
            hf = h5py.File(filepath, "r")
            test_array = TestInstance.load_from_h5(hf)
            if (max_size is not None) and (len(test_array) > max_size):
                cur_idx = 0
                while (cur_idx * max_size) < len(test_array):
                    sub_array = test_array[cur_idx * max_size : (cur_idx + 1) * max_size]
                    sub_array.x = self._normalize(sub_array.x)
                    if flatten_x:
                        sub_array.x = self._flatten_array(sub_array.x)
                    test_id = f"{cur_idx}_{filepath.stem}"
                    cur_idx += 1
                    yield test_id, sub_array
            else:
                test_array.x = self._normalize(test_array.x)
                if flatten_x:
                    test_array.x = self._flatten_array(test_array.x)
                yield filepath.stem, test_array

    @classmethod
    def create_benchmark_datasets(
        cls,
        root,
        balance_negative_crops: bool = True,
        download: bool = True,
        normalize: bool = True,
    ) -> List:
        r"""
        Create the benchmark datasets.

        :param root: The path to the data, where the training data and labels are (or will be)
            saved
        :param balance_negative_crops: Whether to ensure the crops are equally represented in
            a dataset's negative labels. This is only used for datasets where there is a
            target_label, and that target_label is a crop
        :param download: Whether to download the labels and training data if they don't
            already exist
        :param normalize: Whether to normalize the data

        :returns: A list of evaluation CropHarvest datasets according to the TEST_REGIONS and
            TEST_DATASETS in the config
        """

        output_datasets: List = []

        for identifier, bbox in TEST_REGIONS.items():
            country, crop, _, _ = identifier.split("_")

            country_bboxes = countries.get_country_bbox(country)
            for country_bbox in country_bboxes:
                task = Task(
                    country_bbox,
                    crop,
                    balance_negative_crops,
                    f"{country}_{crop}",
                    normalize,
                )
                if task.id not in [x.id for x in output_datasets]:
                    if country_bbox.contains_bbox(bbox):
                        output_datasets.append(cls(root, task, download=download))

        for country, test_dataset in TEST_DATASETS.items():
            # TODO; for now, the only country here is Togo, which
            # only has one bounding box. In the future, it might
            # be nice to confirm its the right index (maybe by checking against
            # some points in the test h5py file?)
            country_bbox = countries.get_country_bbox(country)[0]
            output_datasets.append(
                cls(
                    root,
                    Task(country_bbox, None, test_identifier=test_dataset, normalize=normalize),
                    download=download,
                )
            )
        return output_datasets

    def sample(self, k: int, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # we will sample to get half positive and half negative
        # examples
        output_x: List[np.ndarray] = []
        output_y: List[np.ndarray] = []

        k = min(k, self.k)

        if deterministic:
            pos_indices = self.positive_indices[:k]
            neg_indices = self.negative_indices[:k]
        else:
            pos_indices, self.sampled_positive_indices = sample_with_memory(
                self.positive_indices, k, self.sampled_positive_indices
            )
            neg_indices, self.sampled_negative_indices = sample_with_memory(
                self.negative_indices, k, self.sampled_negative_indices
            )

        # returns a list of [pos_index, neg_index, pos_index, neg_index, ...]
        indices = [val for pair in zip(pos_indices, neg_indices) for val in pair]
        output_x, output_y = zip(*[self[i] for i in indices])

        x = np.stack(output_x, axis=0)
        return x, np.array(output_y)

    @classmethod
    def from_labels_and_tifs(cls, labels: CropHarvestLabels, tifs: CropHarvestTifs):
        "Creates CropHarvest dataset from CropHarvestLabels and CropHarvestTifs"
        pass

    def __repr__(self) -> str:
        class_name = f"CropHarvest{'Eval' if self.task.test_identifier is not None else ''}"
        return f"{class_name}({self.id}, {self.task.test_identifier})"

    @property
    def id(self) -> str:
        return self.task.id

    def _get_positive_and_negative_indices(self) -> Tuple[List[int], List[int]]:
        positive_indices: List[int] = []
        negative_indices: List[int] = []

        for i, y_val in enumerate(self.y_vals):
            if y_val == 1:
                positive_indices.append(i)
            else:
                negative_indices.append(i)
        return positive_indices, negative_indices

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if not self.task.normalize:
            return array
        return (array - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]

    @staticmethod
    def _flatten_array(array: np.ndarray) -> np.ndarray:
        return array.reshape(array.shape[0], -1)
