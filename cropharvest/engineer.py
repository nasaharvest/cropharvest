from pathlib import Path
from datetime import datetime, timedelta
import geopandas
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio import mask
from tqdm import tqdm
import warnings
import h5py

from sklearn.metrics import roc_auc_score, f1_score

from cropharvest.bands import STATIC_BANDS, DYNAMIC_BANDS, BANDS, RAW_BANDS, REMOVED_BANDS
from cropharvest.columns import RequiredColumns, NullableColumns
from .config import (
    EXPORT_END_DAY,
    EXPORT_END_MONTH,
    LABELS_FILENAME,
    DAYS_PER_TIMESTEP,
    DEFAULT_NUM_TIMESTEPS,
    TEST_REGIONS,
    TEST_DATASETS,
)
from .utils import DATAFOLDER_PATH, load_normalizing_dict

from typing import cast, Optional, Dict, Union, Tuple, List, Sequence


@dataclass
class DataInstance:
    dataset: str
    label_lat: float
    label_lon: float
    instance_lat: float
    instance_lon: float
    array: np.ndarray
    is_crop: int
    label: Optional[str] = None

    @property
    def attrs(self) -> Dict:
        attrs: Dict = {}
        for key, _ in self.__annotations__.items():
            val = self.__getattribute__(key)
            if (key != "array") and (val is not None):
                attrs[key] = val
        return attrs


MISSING_DATA = -1


@dataclass
class TestInstance:
    x: Optional[np.ndarray]
    y: np.ndarray  # 1 is positive, 0 is negative and -1 (MISSING_DATA) is no label
    lats: np.ndarray
    lons: np.ndarray

    @property
    def datasets(self) -> Dict[str, np.ndarray]:
        ds: Dict = {}
        for key, _ in self.__annotations__.items():
            val = self.__getattribute__(key)
            ds[key] = val
        return ds

    @classmethod
    def load_from_h5(cls, h5: h5py.File):
        x = h5.get("x")[:]
        return cls(x=x, y=h5.get("y")[:], lats=h5.get("lats")[:], lons=h5.get("lons")[:])

    @classmethod
    def load_from_nc(cls, filepaths: Union[Path, List[Path]]) -> Tuple:
        y: List[np.ndarray] = []
        preds: List[np.ndarray] = []
        lats: List[np.ndarray] = []
        lons: List[np.ndarray] = []

        if isinstance(filepaths, Path):
            filepaths = [filepaths]

        return_preds = True
        for filepath in filepaths:
            ds = xr.load_dataset(filepath)
            if "preds" not in ds:
                return_preds = False

            lons_np, lats_np = np.meshgrid(ds.lon.values, ds.lat.values)
            flat_lats, flat_lons = lats_np.reshape(-1), lons_np.reshape(-1)

            y_np = ds["ground_truth"].values
            flat_y = y_np.reshape(y_np.shape[0] * y_np.shape[1])

            # the Togo dataset is not a meshgrid, so will have plenty of NaN values
            # so we remove them
            not_nan = ~np.isnan(flat_y)
            lats.append(flat_lats[not_nan])
            lons.append(flat_lons[not_nan])
            y.append(flat_y[not_nan])

            if return_preds:
                preds_np = ds["preds"].values
                flat_preds = preds_np.reshape(preds_np.shape[0] * preds_np.shape[1])
                preds.append(flat_preds[not_nan])

        return (
            cls(x=None, y=np.concatenate(y), lats=np.concatenate(lats), lons=np.concatenate(lons)),
            np.concatenate(preds) if return_preds else None,
        )

    def evaluate_predictions(self, preds: np.ndarray) -> Dict[str, float]:
        assert len(preds) == len(
            self.y
        ), f"Expected preds to have length {len(self.y)}, got {len(preds)}"
        y_no_missing = self.y[self.y != MISSING_DATA]
        preds_no_missing = preds[self.y != MISSING_DATA]

        if (len(y_no_missing) == 0) or (len(np.unique(y_no_missing)) == 1):
            print(
                "This TestInstance only has one class in the ground truth "
                "or no non-missing values (this may happen if a test-instance is sliced). "
                "Metrics will be ill-defined, and should be calculated for "
                "all TestInstances together"
            )
            return {"num_samples": len(y_no_missing)}

        binary_preds = preds_no_missing > 0.5

        intersection = np.logical_and(binary_preds, y_no_missing)
        union = np.logical_or(binary_preds, y_no_missing)
        return {
            "auc_roc": roc_auc_score(y_no_missing, preds_no_missing),
            "f1_score": f1_score(y_no_missing, binary_preds),
            "iou": np.sum(intersection) / np.sum(union),
            "num_samples": len(y_no_missing),
        }

    def to_xarray(self, preds: Optional[np.ndarray] = None) -> xr.Dataset:
        data_dict: Dict[str, np.ndarray] = {"lat": self.lats, "lon": self.lons}
        # the first idx is the y labels
        data_dict["ground_truth"] = self.y
        if preds is not None:
            data_dict["preds"] = preds
        return pd.DataFrame(data=data_dict).set_index(["lat", "lon"]).to_xarray()

    def __getitem__(self, sliced):
        return TestInstance(
            x=self.x[sliced] if self.x is not None else None,
            y=self.y[sliced],
            lats=self.lats[sliced],
            lons=self.lons[sliced],
        )

    def __len__(self) -> int:
        return self.y.shape[0]


class Engineer:
    """
    This engineer creates features with BANDS defined in bands.py.
    If the engineer changes, the bands there will need to change as
    well.
    """

    def __init__(self, data_folder: Path = DATAFOLDER_PATH) -> None:
        self.data_folder = data_folder
        self.eo_files = data_folder / "eo_data"
        self.test_eo_files = data_folder / "test_eo_data"

        self.labels = geopandas.read_file(data_folder / LABELS_FILENAME)
        self.labels["export_end_date"] = pd.to_datetime(self.labels.export_end_date)

        self.savedir = data_folder / "features"
        self.savedir.mkdir(exist_ok=True)

        self.test_savedir = data_folder / "test_features"
        self.test_savedir.mkdir(exist_ok=True)

        self.norm_interim: Dict[str, Union[np.ndarray, int]] = {"n": 0}

    @staticmethod
    def find_nearest(array, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    @staticmethod
    def process_filename(filename: str) -> Tuple[int, str]:
        r"""
        Given an exported sentinel file, process it to get the dataset
        it came from, and the index of that dataset
        """
        parts = filename.split("_")[0].split("-")
        index = parts[0]
        dataset = "-".join(parts[1:])
        return int(index), dataset

    @staticmethod
    def load_tif(
        filepath: Path, start_date: datetime, num_timesteps: Optional[int] = DEFAULT_NUM_TIMESTEPS
    ) -> Tuple[xr.DataArray, float]:
        r"""
        The sentinel files exported from google earth have all the timesteps
        concatenated together. This function loads a tif files and splits the
        timesteps

        Returns: The loaded xr.DataArray, and the average slope (used for filling nan slopes)
        """

        da = xr.open_rasterio(filepath).rename("FEATURES")

        da_split_by_time: List[xr.DataArray] = []

        bands_per_timestep = len(DYNAMIC_BANDS)
        num_bands = len(da.band)
        num_dynamic_bands = num_bands - len(STATIC_BANDS)

        assert num_dynamic_bands % bands_per_timestep == 0
        if num_timesteps is None:
            num_timesteps = num_dynamic_bands // bands_per_timestep

        static_data = da.isel(band=slice(num_bands - len(STATIC_BANDS), num_bands))
        average_slope = np.nanmean(static_data.values[STATIC_BANDS.index("slope"), :, :])

        for timestep in range(num_timesteps):
            time_specific_da = da.isel(
                band=slice(timestep * bands_per_timestep, (timestep + 1) * bands_per_timestep)
            )
            time_specific_da = xr.concat([time_specific_da, static_data], "band")
            time_specific_da["band"] = range(bands_per_timestep + len(STATIC_BANDS))
            da_split_by_time.append(time_specific_da)

        timesteps = [
            start_date + timedelta(days=DAYS_PER_TIMESTEP) * i
            for i in range(len(da_split_by_time))
        ]

        dynamic_data = xr.concat(da_split_by_time, pd.Index(timesteps, name="time"))
        dynamic_data.attrs["band_descriptions"] = BANDS

        return dynamic_data, average_slope

    def update_normalizing_values(self, array: np.ndarray) -> None:
        # given an input array of shape [timesteps, bands]
        # update the normalizing dict
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # https://www.johndcook.com/blog/standard_deviation/
        num_bands = array.shape[1]

        # initialize
        if "mean" not in self.norm_interim:
            self.norm_interim["mean"] = np.zeros(num_bands)
            self.norm_interim["M2"] = np.zeros(num_bands)

        for time_idx in range(array.shape[0]):
            self.norm_interim["n"] += 1

            x = array[time_idx, :]

            delta = x - self.norm_interim["mean"]
            self.norm_interim["mean"] += delta / self.norm_interim["n"]
            self.norm_interim["M2"] += delta * (x - self.norm_interim["mean"])

    def calculate_normalizing_dict(self) -> Optional[Dict[str, np.ndarray]]:
        if "mean" not in self.norm_interim:
            print("No normalizing dict calculated! Make sure to call update_normalizing_values")
            return None

        variance = self.norm_interim["M2"] / (self.norm_interim["n"] - 1)
        std = np.sqrt(variance)
        return {"mean": cast(np.ndarray, self.norm_interim["mean"]), "std": cast(np.ndarray, std)}

    @staticmethod
    def adjust_normalizing_dict(
        dicts: Sequence[Tuple[int, Optional[Dict[str, np.ndarray]]]]
    ) -> Optional[Dict[str, np.ndarray]]:
        for _, single_dict in dicts:
            if single_dict is None:
                return None

        dicts = cast(Sequence[Tuple[int, Dict[str, np.ndarray]]], dicts)

        new_total = sum([x[0] for x in dicts])

        new_mean = sum([single_dict["mean"] * length for length, single_dict in dicts]) / new_total

        new_variance = (
            sum(
                [
                    (single_dict["std"] ** 2 + (single_dict["mean"] - new_mean) ** 2) * length
                    for length, single_dict in dicts
                ]
            )
            / new_total
        )

        return {"mean": cast(np.ndarray, new_mean), "std": cast(np.ndarray, np.sqrt(new_variance))}

    @staticmethod
    def calculate_ndvi(input_array: np.ndarray) -> np.ndarray:
        r"""
        Given an input array of shape [timestep, bands] or [batches, timesteps, shapes]
        where bands == len(bands), returns an array of shape
        [timestep, bands + 1] where the extra band is NDVI,
        (b08 - b04) / (b08 + b04)
        """
        band_1, band_2 = "B8", "B4"

        num_dims = len(input_array.shape)
        if num_dims == 2:
            band_1_np = input_array[:, BANDS.index(band_1)]
            band_2_np = input_array[:, BANDS.index(band_2)]
        elif num_dims == 3:
            band_1_np = input_array[:, :, BANDS.index(band_1)]
            band_2_np = input_array[:, :, BANDS.index(band_2)]
        else:
            raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in true_divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            ndvi = np.where(
                (band_1_np + band_2_np) > 0, (band_1_np - band_2_np) / (band_1_np + band_2_np), 0
            )
        return np.append(input_array, np.expand_dims(ndvi, -1), axis=-1)

    @staticmethod
    def fillna(array: np.ndarray, average_slope: float) -> Optional[np.ndarray]:
        r"""
        Given an input array of shape [timesteps, BANDS]
        fill NaN values with the mean of each band across the timestep
        The slope values may all be nan so average_slope is manually passed
        """
        num_dims = len(array.shape)
        if num_dims == 2:
            bands_index = 1
            mean_per_band = np.nanmean(array, axis=0)
        elif num_dims == 3:
            bands_index = 2
            mean_per_band = np.nanmean(np.nanmean(array, axis=0), axis=0)
        else:
            raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

        assert array.shape[bands_index] == len(BANDS)

        if np.isnan(mean_per_band).any():
            if (sum(np.isnan(mean_per_band)) == bands_index) & (
                np.isnan(mean_per_band[BANDS.index("slope")]).all()
            ):
                mean_per_band[BANDS.index("slope")] = average_slope
                assert not np.isnan(mean_per_band).any()
            else:
                return None
        for i in range(array.shape[bands_index]):
            if num_dims == 2:
                array[:, i] = np.nan_to_num(array[:, i], nan=mean_per_band[i])
            elif num_dims == 3:
                array[:, :, i] = np.nan_to_num(array[:, :, i], nan=mean_per_band[i])
        return array

    @staticmethod
    def remove_bands(array: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands] or
        [batches, timesteps, bands]
        """
        num_dims = len(array.shape)
        error_message = f"Expected num_dims to be 2 or 3 - got {num_dims}"
        if num_dims == 2:
            bands_index = 1
        elif num_dims == 3:
            bands_index = 2
        else:
            raise ValueError(error_message)

        indices_to_remove: List[int] = []
        for band in REMOVED_BANDS:
            indices_to_remove.append(RAW_BANDS.index(band))
        indices_to_keep = [
            i for i in range(array.shape[bands_index]) if i not in indices_to_remove
        ]
        if num_dims == 2:
            return array[:, indices_to_keep]
        elif num_dims == 3:
            return array[:, :, indices_to_keep]
        else:
            # Unreachable code logically but mypy does not see it this way
            raise ValueError(error_message)

    @staticmethod
    def process_test_file(
        path_to_file: Path, start_date: datetime
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        da, slope = Engineer.load_tif(path_to_file, start_date=start_date)

        # Process remote sensing data
        x_np = da.values
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
        x_np = np.moveaxis(x_np, -1, 0)
        x_np = Engineer.calculate_ndvi(x_np)
        x_np = Engineer.remove_bands(x_np)
        final_x = Engineer.fillna(x_np, slope)
        if final_x is None:
            raise RuntimeError(
                "fillna on the test instance returned None; "
                "does the test instance contain NaN only bands?"
            )

        # Get lat lons
        lon, lat = np.meshgrid(da.x.values, da.y.values)
        flat_lat, flat_lon = (
            np.squeeze(lat.reshape(-1, 1), -1),
            np.squeeze(lon.reshape(-1, 1), -1),
        )
        return final_x, flat_lat, flat_lon

    def process_test_file_with_region(
        self, path_to_file: Path, id_in_region: int
    ) -> Tuple[str, TestInstance]:
        id_components = path_to_file.name.split("_")
        crop, end_year = id_components[1], id_components[2]
        identifier = "_".join(id_components[:4])
        identifier_plus_idx = f"{identifier}_{id_in_region}"
        start_date = datetime(int(end_year), EXPORT_END_MONTH, EXPORT_END_DAY) - timedelta(
            days=DEFAULT_NUM_TIMESTEPS * DAYS_PER_TIMESTEP
        )

        final_x, flat_lat, flat_lon = Engineer.process_test_file(
            path_to_file, start_date=start_date
        )

        # finally, we need to calculate the mask
        region_bbox = TEST_REGIONS[identifier]
        relevant_indices = self.labels.apply(
            lambda x: (
                region_bbox.contains(x[RequiredColumns.LAT], x[RequiredColumns.LON])
                and (x[RequiredColumns.EXPORT_END_DATE].year == int(end_year))
            ),
            axis=1,
        )
        relevant_rows = self.labels[relevant_indices]
        positive_geoms = relevant_rows[relevant_rows[NullableColumns.LABEL] == crop][
            RequiredColumns.GEOMETRY
        ].tolist()
        negative_geoms = relevant_rows[relevant_rows[NullableColumns.LABEL] != crop][
            RequiredColumns.GEOMETRY
        ].tolist()

        with rasterio.open(path_to_file) as src:
            # the mask is True outside shapes, and False inside shapes. We want the opposite
            positive, _, _ = mask.raster_geometry_mask(src, positive_geoms, crop=False)
            negative, _, _ = mask.raster_geometry_mask(src, negative_geoms, crop=False)
        # reverse the booleans so that 1 = in the
        positive = (~positive.reshape(positive.shape[0] * positive.shape[1])).astype(int)
        negative = (~negative.reshape(negative.shape[0] * negative.shape[1])).astype(int) * -1
        y = positive + negative

        # swap missing and negative values, since this will be easier to use in the future
        negative = y == -1
        missing = y == 0
        y[negative] = 0
        y[missing] = MISSING_DATA
        assert len(y) == final_x.shape[0]

        return identifier_plus_idx, TestInstance(x=final_x, y=y, lats=flat_lat, lons=flat_lon)

    def process_single_file(
        self,
        path_to_file: Path,
        row: pd.Series,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    ) -> Optional[DataInstance]:
        start_date = row.export_end_date - timedelta(days=num_timesteps * DAYS_PER_TIMESTEP)
        da, average_slope = self.load_tif(path_to_file, start_date=start_date)
        closest_lon = self.find_nearest(da.x, row[RequiredColumns.LON])
        closest_lat = self.find_nearest(da.y, row[RequiredColumns.LAT])

        labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values

        labelled_np = self.calculate_ndvi(labelled_np)
        labelled_np = self.remove_bands(labelled_np)

        labelled_array = self.fillna(labelled_np, average_slope=average_slope)
        if labelled_array is None:
            return None

        if not row[RequiredColumns.IS_TEST]:
            self.update_normalizing_values(labelled_array)

        return DataInstance(
            label_lat=row[RequiredColumns.LAT],
            label_lon=row[RequiredColumns.LON],
            instance_lat=closest_lat,
            instance_lon=closest_lon,
            array=labelled_array,
            is_crop=row[RequiredColumns.IS_CROP],
            label=row[NullableColumns.LABEL],
            dataset=row[RequiredColumns.DATASET],
        )

    def create_h5_test_instances(
        self,
    ) -> None:
        for region_identifier, _ in TEST_REGIONS.items():
            all_region_files = list(self.test_eo_files.glob(f"{region_identifier}*.tif"))
            if len(all_region_files) == 0:
                print(f"No downloaded files for {region_identifier}")
                continue
            for region_idx, filepath in enumerate(all_region_files):
                instance_name, test_instance = self.process_test_file_with_region(
                    filepath, region_idx
                )
                if test_instance is not None:
                    hf = h5py.File(self.test_savedir / f"{instance_name}.h5", "w")

                    for key, val in test_instance.datasets.items():
                        hf.create_dataset(key, data=val)
                    hf.close()

        for _, dataset in TEST_DATASETS.items():
            x: List[np.ndarray] = []
            y: List[int] = []
            lats: List[float] = []
            lons: List[float] = []
            relevant_labels = self.labels[self.labels[RequiredColumns.DATASET] == dataset]

            for _, row in tqdm(relevant_labels.iterrows()):
                tif_paths = list(
                    self.eo_files.glob(
                        f"{row[RequiredColumns.INDEX]}-{row[RequiredColumns.DATASET]}_*.tif"
                    )
                )
                if len(tif_paths) == 0:
                    continue
                else:
                    tif_path = tif_paths[0]
                instance = self.process_single_file(tif_path, row)
                if instance is not None:
                    x.append(instance.array)
                    y.append(instance.is_crop)
                    lats.append(instance.label_lat)
                    lons.append(instance.label_lon)

            # then, combine the instances into a test instance
            test_instance = TestInstance(np.stack(x), np.stack(y), np.stack(lats), np.stack(lons))
            hf = h5py.File(self.test_savedir / f"{dataset}.h5", "w")
            for key, val in test_instance.datasets.items():
                hf.create_dataset(key, data=val)
            hf.close()

    def create_h5_dataset(self, checkpoint: bool = True) -> None:
        arrays_dir = self.savedir / "arrays"
        arrays_dir.mkdir(exist_ok=True)

        old_normalizing_dict: Optional[Tuple[int, Optional[Dict[str, np.ndarray]]]] = None
        if checkpoint:
            # check for an already existing normalizing dict
            if (self.savedir / "normalizing_dict.h5").exists():
                old_nd = load_normalizing_dict(self.savedir / "normalizing_dict.hf")
                num_existing_files = len(list(arrays_dir.glob("*")))
                old_normalizing_dict = (num_existing_files, old_nd)

        skipped_files: int = 0
        num_new_files: int = 0
        for file_path in tqdm(list(self.eo_files.glob("*.tif"))):
            file_index, dataset = self.process_filename(file_path.stem)
            file_name = f"{file_index}_{dataset}.h5"
            if (checkpoint) & ((arrays_dir / file_name).exists()):
                # we check if the file has already been written
                continue

            file_row = self.labels[
                (
                    (self.labels[RequiredColumns.DATASET] == dataset)
                    & (self.labels[RequiredColumns.INDEX] == file_index)
                )
            ].iloc[0]

            instance = self.process_single_file(file_path, row=file_row)
            if instance is not None:
                hf = h5py.File(arrays_dir / file_name, "w")
                hf.create_dataset("array", data=instance.array)

                for key, val in instance.attrs.items():
                    hf.attrs[key] = val
                hf.close()

                num_new_files += 1
            else:
                skipped_files += 1

        print(f"Wrote {num_new_files} files, skipped {skipped_files} files")

        normalizing_dict = self.calculate_normalizing_dict()

        if checkpoint and (old_normalizing_dict is not None):
            normalizing_dicts = [old_normalizing_dict, (num_new_files, normalizing_dict)]
            normalizing_dict = self.adjust_normalizing_dict(normalizing_dicts)
        if normalizing_dict is not None:
            save_path = self.savedir / "normalizing_dict.h5"
            hf = h5py.File(save_path, "w")
            for key, val in normalizing_dict.items():
                hf.create_dataset(key, data=val)
            hf.close()
        else:
            print("No normalizing dict calculated!")
