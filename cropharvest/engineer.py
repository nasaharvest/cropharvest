from pathlib import Path
from datetime import datetime, timedelta
import geopandas
from dataclasses import dataclass
from fnmatch import fnmatch
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio import mask
from tqdm import tqdm
import warnings
import h5py

from sklearn.metrics import roc_auc_score, f1_score

from cropharvest.columns import RequiredColumns, NullableColumns, EngColumns
from cropharvest.bands import STATIC_BANDS, DYNAMIC_BANDS, BANDS, RAW_BANDS, REMOVED_BANDS
from cropharvest.boundingbox import BBox
from .config import (
    EXPORT_END_DAY,
    EXPORT_END_MONTH,
    LABELS_FILENAME,
    DAYS_PER_TIMESTEP,
    DEFAULT_NUM_TIMESTEPS,
    TEST_REGIONS,
    TEST_DATASETS,
    DATAFOLDER_PATH,
    FEATURES_FILEPATH,
    EO_FILEPATH,
    TEST_EO_FILEPATH,
    ARRAYS_FILEPATH,
    TEST_FEATURES_FILEPATH,
)
from cropharvest.utils import load_normalizing_dict

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
    year: int
    source_tif_file: str
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
    def __init__(self) -> None:
        self.labels = self.load_labels()
        FEATURES_FILEPATH.mkdir(exist_ok=True)
        ARRAYS_FILEPATH.mkdir(exist_ok=True)
        TEST_FEATURES_FILEPATH.mkdir(exist_ok=True)

        self.norm_interim: Dict[str, Union[np.ndarray, int]] = {"n": 0}

    @staticmethod
    def load_labels(root=DATAFOLDER_PATH) -> geopandas.GeoDataFrame:
        labels = geopandas.read_file(root / LABELS_FILENAME)
        labels[RequiredColumns.EXPORT_END_DATE] = pd.to_datetime(
            labels[RequiredColumns.EXPORT_END_DATE]
        ).dt.date
        labels[EngColumns.FEATURES_FILENAME] = (
            "lat="
            + labels[RequiredColumns.LAT].round(8).astype(str)
            + "_lon="
            + labels[RequiredColumns.LON].round(8).astype(str)
            + "_date="
            + labels[RequiredColumns.EXPORT_END_DATE].astype(str)
        )
        labels[EngColumns.FEATURES_PATH] = (
            str(ARRAYS_FILEPATH) + "/" + labels[EngColumns.FEATURES_FILENAME]
        )
        labels[EngColumns.EXISTS] = np.vectorize(lambda p: Path(p).exists())(
            labels[EngColumns.FEATURES_PATH]
        )
        return labels

    @staticmethod
    def distance_from_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        haversince formula, inspired by:
        https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude/41337005
        """
        p = 0.017453292519943295
        a = (
            0.5
            - np.cos((lat2 - lat1) * p) / 2
            + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
        )
        return 12742 * np.arcsin(np.sqrt(a))

    @staticmethod
    def distance_point_from_center(lat_idx: int, lon_idx: int, tif) -> int:
        x_dist = np.abs((len(tif.x) - 1) / 2 - lon_idx)
        y_dist = np.abs((len(tif.y) - 1) / 2 - lat_idx)
        return x_dist + y_dist

    @staticmethod
    def find_nearest(array, value: float) -> Tuple[float, int]:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

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
        row: pd.Series,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    ) -> Optional[DataInstance]:
        start_date = row[RequiredColumns.EXPORT_END_DATE] - timedelta(
            days=num_timesteps * DAYS_PER_TIMESTEP
        )

        tif_slope_tuples = [
            self.load_tif(filepath, start_date=start_date)
            for filepath in row[EngColumns.TIF_FILEPATHS]
        ]
        if len(tif_slope_tuples) == 1:
            tif, average_slope = tif_slope_tuples[0]

            closest_lon, _ = self.find_nearest(tif.x, row[RequiredColumns.LON])
            closest_lat, _ = self.find_nearest(tif.y, row[RequiredColumns.LAT])

            labelled_np = tif.sel(x=closest_lon).sel(y=closest_lat).values
            tif_file = row[EngColumns.TIF_FILEPATHS][0].name

        else:
            min_distance_from_point = np.inf
            min_distance_from_center = np.inf
            for i, tif_slope_tuple in enumerate(tif_slope_tuples):
                tif, slope = tif_slope_tuple
                lon, lon_idx = self.find_nearest(tif.x, row[RequiredColumns.LON])
                lat, lat_idx = self.find_nearest(tif.y, row[RequiredColumns.LAT])
                distance_from_point = self.distance_from_degrees(
                    row[RequiredColumns.LAT], row[RequiredColumns.LON], lat, lon
                )
                distance_from_center = self.distance_point_from_center(lat_idx, lon_idx, tif)
                if (distance_from_point < min_distance_from_point) or (
                    distance_from_point == min_distance_from_point
                    and distance_from_center < min_distance_from_center
                ):
                    closest_lon = lon
                    closest_lat = lat
                    min_distance_from_center = distance_from_center
                    min_distance_from_point = distance_from_point

                    labelled_np = tif.sel(x=lon).sel(y=lat).values
                    average_slope = slope
                    tif_file = row[EngColumns.TIF_FILEPATHS][i].name

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
            year=start_date.year,
            source_tif_file=tif_file,
        )

    def create_h5_test_instances(
        self,
    ) -> None:
        for region_identifier, _ in TEST_REGIONS.items():
            all_region_files = list(TEST_EO_FILEPATH.glob(f"{region_identifier}*.tif"))
            if len(all_region_files) == 0:
                print(f"No downloaded files for {region_identifier}")
                continue
            for region_idx, filepath in enumerate(all_region_files):
                instance_name, test_instance = self.process_test_file_with_region(
                    filepath, region_idx
                )
                if test_instance is not None:
                    hf = h5py.File(TEST_FEATURES_FILEPATH / f"{instance_name}.h5", "w")

                    for key, val in test_instance.datasets.items():
                        hf.create_dataset(key, data=val)
                    hf.close()

        for _, dataset in TEST_DATASETS.items():
            x: List[np.ndarray] = []
            y: List[int] = []
            lats: List[float] = []
            lons: List[float] = []
            relevant_labels = self.labels[self.labels[RequiredColumns.DATASET] == dataset]

            relevant_labels[EngColumns.TIF_FILEPATHS] = self.match_labels_to_tifs(relevant_labels)
            tifs_found = relevant_labels[EngColumns.TIF_FILEPATHS].str.len() > 0
            labels_with_tifs = relevant_labels.loc[tifs_found]

            for _, row in tqdm(labels_with_tifs.iterrows()):
                instance = self.process_single_file(row)
                if instance is not None:
                    x.append(instance.array)
                    y.append(instance.is_crop)
                    lats.append(instance.label_lat)
                    lons.append(instance.label_lon)

            # then, combine the instances into a test instance
            test_instance = TestInstance(np.stack(x), np.stack(y), np.stack(lats), np.stack(lons))
            hf = h5py.File(TEST_FEATURES_FILEPATH / f"{dataset}.h5", "w")
            for key, val in test_instance.datasets.items():
                hf.create_dataset(key, data=val)
            hf.close()

    @staticmethod
    def generate_bbox_from_paths(filepath: Path) -> Dict[Path, BBox]:
        return {
            p: BBox.from_eo_tif_file(p)
            for p in tqdm(filepath.glob("**/*.tif"), desc="Generating BoundingBoxes from paths")
        }

    @staticmethod
    def get_tif_paths(path_to_bbox, lat, lon, end_date, pbar):
        candidate_paths = []
        for p, bbox in path_to_bbox.items():
            if bbox.contains(lat, lon) and fnmatch(p.stem, f"*dates=*_{end_date}*"):
                candidate_paths.append(p)
        pbar.update(1)
        return candidate_paths

    @classmethod
    def match_labels_to_tifs(cls, labels: geopandas.GeoDataFrame) -> pd.Series:
        bbox_for_labels = BBox(
            min_lon=labels[RequiredColumns.LON].min(),
            min_lat=labels[RequiredColumns.LAT].min(),
            max_lon=labels[RequiredColumns.LON].max(),
            max_lat=labels[RequiredColumns.LAT].max(),
        )
        # Get all tif paths and bboxes
        path_to_bbox = {
            p: bbox
            for p, bbox in cls.generate_bbox_from_paths(EO_FILEPATH).items()
            if bbox_for_labels.contains_bbox(bbox)
        }

        # Match labels to tif files
        # Faster than going through bboxes
        with tqdm(total=len(labels), desc="Matching labels to tif paths") as pbar:
            tif_paths = np.vectorize(cls.get_tif_paths, otypes=[np.ndarray])(
                path_to_bbox,
                labels[RequiredColumns.LAT],
                labels[RequiredColumns.LON],
                labels[RequiredColumns.EXPORT_END_DATE],
                pbar,
            )
        return tif_paths

    def create_h5_dataset(self) -> None:

        old_normalizing_dict: Optional[Tuple[int, Optional[Dict[str, np.ndarray]]]] = None
        # check for an already existing normalizing dict
        if (FEATURES_FILEPATH / "normalizing_dict.h5").exists():
            old_nd = load_normalizing_dict(FEATURES_FILEPATH / "normalizing_dict.hf")
            num_existing_files = len(list(ARRAYS_FILEPATH.glob("*")))
            old_normalizing_dict = (num_existing_files, old_nd)

        labels_with_no_features = self.labels[~self.labels[EngColumns.EXISTS]].copy()
        labels_with_no_features[EngColumns.TIF_FILEPATHS] = self.match_labels_to_tifs(
            labels_with_no_features
        )
        tifs_found = labels_with_no_features[EngColumns.TIF_FILEPATHS].str.len() > 0
        labels_with_tifs_but_no_features = labels_with_no_features.loc[tifs_found]

        skipped_files: int = 0
        num_new_files: int = 0
        for _, row in tqdm(labels_with_tifs_but_no_features.iterrows()):
            instance = self.process_single_file(row)
            if instance is not None:
                filename = (
                    f"lat={instance.label_lat}_lon={instance.label_lon}_year={instance.year}.h5"
                )
                hf = h5py.File(ARRAYS_FILEPATH / filename, "w")
                hf.create_dataset("array", data=instance.array)

                for key, val in instance.attrs.items():
                    hf.attrs[key] = val
                hf.close()

                num_new_files += 1
            else:
                skipped_files += 1

        print(f"Wrote {num_new_files} files, skipped {skipped_files} files")

        normalizing_dict = self.calculate_normalizing_dict()

        if old_normalizing_dict is not None:
            normalizing_dicts = [old_normalizing_dict, (num_new_files, normalizing_dict)]
            normalizing_dict = self.adjust_normalizing_dict(normalizing_dicts)
        if normalizing_dict is not None:
            save_path = FEATURES_FILEPATH / "normalizing_dict.h5"
            hf = h5py.File(save_path, "w")
            for key, val in normalizing_dict.items():
                hf.create_dataset(key, data=val)
            hf.close()
        else:
            print("No normalizing dict calculated!")
