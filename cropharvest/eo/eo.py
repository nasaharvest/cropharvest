from pathlib import Path
import geopandas
import pandas as pd
from tqdm import tqdm
from datetime import timedelta, date

try:
    import ee
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The Earth Engine API is not installed. "
        + "Please install it with `pip install earthengine-api`."
    )

from .ee_boundingbox import EEBoundingBox
from .sentinel1 import (
    get_single_image as get_single_s1_image,
    get_image_collection as get_s1_image_collection,
)
from .sentinel2 import get_single_image as get_single_s2_image
from .era5 import get_single_image as get_single_era5_image
from .srtm import get_single_image as get_single_srtm_image

from .utils import make_combine_bands_function
from cropharvest.bands import DYNAMIC_BANDS
from cropharvest.utils import DATAFOLDER_PATH, memoized
from cropharvest.countries import BBox
from cropharvest.config import (
    EXPORT_END_DAY,
    EXPORT_END_MONTH,
    DAYS_PER_TIMESTEP,
    DEFAULT_NUM_TIMESTEPS,
    LABELS_FILENAME,
    TEST_REGIONS,
)
from cropharvest.columns import RequiredColumns

from typing import Dict, Union, List, Optional, Tuple

try:
    from google.cloud import storage

    GOOGLE_CLOUD_STORAGE_INSTALLED = True
except ModuleNotFoundError:
    GOOGLE_CLOUD_STORAGE_INSTALLED = False
INSTALL_MSG = "Please install the google-cloud-storage library (pip install google-cloud-storage)"

DYNAMIC_IMAGE_FUNCTIONS = [get_single_s2_image, get_single_era5_image]
STATIC_IMAGE_FUNCTIONS = [get_single_srtm_image]


@memoized
def get_ee_task_list(key: str = "description") -> List[str]:
    """Gets a list of all active tasks in the EE task list."""
    task_list = ee.data.getTaskList()
    return [
        task[key]
        for task in tqdm(task_list, desc="Loading Earth Engine tasks")
        if task["state"] in ["READY", "RUNNING", "FAILED"]
    ]


@memoized
def get_cloud_tif_list(dest_bucket: str) -> List[str]:
    """Gets a list of all cloud-free TIFs in a bucket."""
    if not GOOGLE_CLOUD_STORAGE_INSTALLED:
        # Added as precaution, but should never happen
        raise ValueError(f"{INSTALL_MSG} to enable GCP checks")

    client = storage.Client()
    cloud_tif_list_iterator = client.list_blobs(dest_bucket, prefix="tifs")
    cloud_tif_list = [
        blob.name
        for blob in tqdm(cloud_tif_list_iterator, desc="Loading tifs already on Google Cloud")
    ]
    return cloud_tif_list


class EarthEngineExporter:
    """
    Export satellite data from Earth engine. It's called using the following
    script:
    ```
    from cropharvest.eo import EarthEngineExporter

    exporter = EarthEngineExporter()
    exporter.export_for_labels()
    ```
    :param labels: A geopandas.GeoDataFrame containing the labels for the exports
    :param check_ee: Whether to check Earth Engine before exporting
    :param check_gcp: Whether to check Google Cloud Storage before exporting,
        google-cloud-storage must be installed.
    :param credentials: The credentials to use for the export. If not specified,
        the default credentials will be used
    :param dest_bucket: The bucket to export to, google-cloud-storage must be installed.
    """

    output_folder_name = "eo_data"
    test_output_folder_name = "test_eo_data"

    def __init__(
        self,
        check_ee: bool = False,
        check_gcp: bool = False,
        credentials: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> None:
        # allows for easy checkpointing
        self.cur_output_folder = f"{self.output_folder_name}_{str(date.today()).replace('-', '')}"

        self.dest_bucket = dest_bucket

        try:
            if credentials:
                ee.Initialize(credentials=credentials)
            else:
                ee.Initialize()
        except Exception:
            print("This code doesn't work unless you have authenticated your earthengine account")

        self.check_ee = check_ee
        self.ee_task_list = get_ee_task_list() if self.check_ee else []

        if check_gcp and dest_bucket is None:
            raise ValueError("check_gcp was set to True but dest_bucket was not specified")
        elif not GOOGLE_CLOUD_STORAGE_INSTALLED:
            if check_gcp:
                raise ValueError(f"{INSTALL_MSG} to enable GCP checks")
            elif dest_bucket is not None:
                raise ValueError(f"{INSTALL_MSG} to enable export to destination bucket")

        self.check_gcp = check_gcp
        self.cloud_tif_list = get_cloud_tif_list(dest_bucket) if self.check_gcp else []

    @staticmethod
    def load_default_labels(
        dataset: Optional[str], start_from_last, checkpoint: Optional[Path]
    ) -> geopandas.GeoDataFrame:
        labels = geopandas.read_file(DATAFOLDER_PATH / LABELS_FILENAME)
        export_end_year = pd.to_datetime(labels[RequiredColumns.EXPORT_END_DATE]).dt.year
        labels["end_date"] = export_end_year.apply(
            lambda x: date(x, EXPORT_END_MONTH, EXPORT_END_DAY)
        )
        labels = labels.assign(
            start_date=lambda x: x["end_date"]
            - timedelta(days=DAYS_PER_TIMESTEP * DEFAULT_NUM_TIMESTEPS)
        )
        labels["export_identifier"] = labels.apply(
            lambda x: f"{x['index']}-{x[RequiredColumns.DATASET]}", axis=1
        )
        if dataset:
            labels = labels[labels.dataset == dataset]

        if start_from_last:
            labels = EarthEngineExporter._filter_labels(labels, checkpoint)

        return labels

    @staticmethod
    def _filter_labels(
        labels: geopandas.GeoDataFrame, checkpoint: Optional[Path]
    ) -> geopandas.GeoDataFrame:

        # does not sort
        datasets = labels.dataset.unique()

        if checkpoint is None:
            # if we have no checkpoint folder, then we have no
            # downloaded files to check against
            return labels

        if len(list(checkpoint.glob(f"*{datasets[0]}*"))) == 0:
            # no files downloaded
            return labels

        for idx in range(len(datasets)):
            cur_dataset_files = list(checkpoint.glob(f"*{datasets[idx]}*"))
            does_cur_exist = len(cur_dataset_files) > 0
            if idx < (len(datasets) - 1):
                does_next_exist = len(list(checkpoint.glob(f"*{datasets[idx + 1]}*"))) > 0
            else:
                # we are on the last dataset
                does_next_exist = False
            if does_next_exist & does_cur_exist:
                continue
            elif (not does_cur_exist) & does_next_exist:
                raise RuntimeError("Datasets seem to be downloaded in the wrong order!")

            # does_next doesn't exist, and does_cur does exist. We want to find the largest
            # int downloaded
            max_index = max([int(x.name.split("-")[0]) for x in cur_dataset_files])

            row = labels[
                ((labels.dataset == datasets[idx]) & (labels["index"] == max_index))
            ].iloc[0]
            # + 1 - non inclusive
            labels = labels.loc[row.name + 1 :]

            starting_row = labels.iloc[0]
            print(f"Starting export from {starting_row.dataset} at index {starting_row['index']}")
            return labels
        return labels

    def _export(
        self,
        image: ee.Image,
        region: ee.Geometry,
        filename: str,
        description: str,
        drive_folder: Optional[str] = None,
        dest_bucket: Optional[str] = None,
        file_dimensions: Optional[int] = None,
        test: bool = False,
    ) -> ee.batch.Export:

        kwargs = dict(
            image=image.clip(region),
            description=description[:100],
            scale=10,
            region=region,
            maxPixels=1e13,
            fileDimensions=file_dimensions,
        )

        if dest_bucket:
            if not GOOGLE_CLOUD_STORAGE_INSTALLED:
                # Added as precaution, but should never happen
                raise ValueError(f"{INSTALL_MSG} to enable export to destination bucket")

            if not test:
                # If training data make sure it goes in the tifs folder
                filename = f"tifs/{filename}"

            task = ee.batch.Export.image.toCloudStorage(
                bucket=dest_bucket, fileNamePrefix=filename, **kwargs
            )
        else:
            task = ee.batch.Export.image.toDrive(
                folder=drive_folder, fileNamePrefix=filename, **kwargs
            )

        try:
            task.start()
            self.ee_task_list.append(description)
        except ee.ee_exception.EEException as e:
            print(f"Task not started! Got exception {e}")
            return task

        return task

    def _export_for_polygon(
        self,
        polygon: ee.Geometry.Polygon,
        polygon_identifier: Union[int, str],
        start_date: date,
        end_date: date,
        days_per_timestep: int = DAYS_PER_TIMESTEP,
        checkpoint: Optional[Path] = None,
        test: bool = False,
        file_dimensions: Optional[int] = None,
    ) -> bool:

        filename = str(polygon_identifier)
        if (checkpoint is not None) and (checkpoint / f"{filename}.tif").exists():
            print("File already exists! Skipping")
            return False

        # Description of the export cannot contain certrain characters
        description = filename.replace(".", "-").replace("=", "-").replace("/", "-")[:100]

        if self.check_gcp:
            # If test data we check the root in the cloud bucket
            if test and f"{filename}.tif" in self.cloud_tif_list:
                return False
            # If training data we check the tifs folder in thee cloud bucket
            elif not test and (f"tifs/{filename}.tif" in self.cloud_tif_list):
                return False

        # Check if task is already started in EarthEngine
        if self.check_ee and (description in self.ee_task_list):
            return True

        if self.check_ee and len(self.ee_task_list) >= 3000:
            return False

        image_collection_list: List[ee.Image] = []
        cur_date = start_date
        cur_end_date = cur_date + timedelta(days=days_per_timestep)

        # first, we get all the S1 images in an exaggerated date range
        vv_imcol, vh_imcol = get_s1_image_collection(
            polygon, start_date - timedelta(days=31), end_date + timedelta(days=31)
        )

        while cur_end_date <= end_date:
            image_list: List[ee.Image] = []

            # first, the S1 image which gets the entire s1 collection
            image_list.append(
                get_single_s1_image(
                    region=polygon,
                    start_date=cur_date,
                    end_date=cur_end_date,
                    vv_imcol=vv_imcol,
                    vh_imcol=vh_imcol,
                )
            )
            for image_function in DYNAMIC_IMAGE_FUNCTIONS:
                image_list.append(
                    image_function(region=polygon, start_date=cur_date, end_date=cur_end_date)
                )
            image_collection_list.append(ee.Image.cat(image_list))

            cur_date += timedelta(days=days_per_timestep)
            cur_end_date += timedelta(days=days_per_timestep)

        # now, we want to take our image collection and append the bands into a single image
        imcoll = ee.ImageCollection(image_collection_list)
        combine_bands_function = make_combine_bands_function(DYNAMIC_BANDS)
        img = ee.Image(imcoll.iterate(combine_bands_function))

        # finally, we add the SRTM image seperately since its static in time
        total_image_list: List[ee.Image] = [img]
        for static_image_function in STATIC_IMAGE_FUNCTIONS:
            total_image_list.append(static_image_function(region=polygon))

        img = ee.Image.cat(total_image_list)

        # and finally, export the image
        kwargs = dict(
            image=img,
            region=polygon,
            filename=filename,
            description=description,
            file_dimensions=file_dimensions,
            test=test,
        )
        if self.dest_bucket:
            kwargs["dest_bucket"] = self.dest_bucket
        elif test:
            kwargs["drive_folder"] = self.test_output_folder_name
        else:
            kwargs["drive_folder"] = self.cur_output_folder

        self._export(**kwargs)
        return True

    @classmethod
    def _labels_to_polygons_and_years(
        cls, labels: geopandas.GeoDataFrame, surrounding_metres: int
    ) -> List[Tuple[ee.Geometry.Polygon, str, date, date]]:

        output: List[ee.Geometry.Polygon] = []

        print(f"Exporting {len(labels)} labels")

        for _, row in tqdm(labels.iterrows()):
            ee_bbox = EEBoundingBox.from_centre(
                mid_lat=row[RequiredColumns.LAT],
                mid_lon=row[RequiredColumns.LON],
                surrounding_metres=surrounding_metres,
            )

            try:
                export_identifier = row["export_identifier"]
            except KeyError:
                export_identifier = cls.make_identifier(
                    ee_bbox, row["start_date"], row["end_date"]
                )

            output.append(
                (
                    ee_bbox.to_ee_polygon(),
                    export_identifier,
                    row["start_date"],
                    row["end_date"],
                )
            )

        return output

    @staticmethod
    def make_identifier(bbox: BBox, start_date, end_date) -> str:

        # Identifier is rounded to the nearest ~10m
        min_lon = round(bbox.min_lon, 4)
        min_lat = round(bbox.min_lat, 4)
        max_lon = round(bbox.max_lon, 4)
        max_lat = round(bbox.max_lat, 4)
        return (
            f"min_lat={min_lat}_min_lon={min_lon}_max_lat={max_lat}_max_lon={max_lon}_"
            f"dates={start_date}_{end_date}_all"
        )

    def export_for_test(
        self,
        padding_metres: int = 160,
        checkpoint: Optional[Path] = None,
    ) -> None:

        for identifier, bbox in TEST_REGIONS.items():
            polygon = EEBoundingBox.from_bounding_box(
                bounding_box=bbox, padding_metres=padding_metres
            ).to_ee_polygon()
            _, _, year, _ = identifier.split("_")
            end_date = date(int(year), EXPORT_END_MONTH, EXPORT_END_DAY)
            start_date = end_date - timedelta(days=DAYS_PER_TIMESTEP * DEFAULT_NUM_TIMESTEPS)
            _ = self._export_for_polygon(
                polygon=polygon,
                polygon_identifier=identifier,
                start_date=start_date,
                end_date=end_date,
                checkpoint=checkpoint,
                test=True,
            )

    def export_for_bbox(
        self,
        bbox: BBox,
        bbox_name: str,
        start_date: date,
        end_date: date,
        metres_per_polygon: Optional[int] = 10000,
        file_dimensions: Optional[int] = None,
    ) -> Dict[str, bool]:

        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        ee_bbox = EEBoundingBox.from_bounding_box(bounding_box=bbox, padding_metres=0)
        if metres_per_polygon is not None:
            regions = ee_bbox.to_polygons(metres_per_patch=metres_per_polygon)
            ids = [f"batch_{i}/{i}" for i in range(len(regions))]
        else:
            regions = [ee_bbox.to_ee_polygon()]
            ids = ["batch/0"]

        return_obj = {}
        for identifier, region in zip(ids, regions):
            return_obj[identifier] = self._export_for_polygon(
                polygon=region,
                polygon_identifier=f"{bbox_name}/{identifier}",
                start_date=start_date,
                end_date=end_date,
                file_dimensions=file_dimensions,
                test=True,
            )
        return return_obj

    def export_for_labels(
        self,
        labels: Optional[geopandas.GeoDataFrame] = None,
        dataset: Optional[str] = None,
        num_labelled_points: Optional[int] = 3000,
        surrounding_metres: int = 80,
        checkpoint: Optional[Path] = None,
        start_from_last: bool = False,
    ) -> None:

        if labels is None:
            labels = self.load_default_labels(
                dataset=dataset, start_from_last=start_from_last, checkpoint=checkpoint
            )
        else:
            if dataset is not None:
                print("No dataset can be specified if passing a different set of labels")
            if start_from_last:
                print("start_from_last cannot be used if passing a different set of labels")

        for expected_column in [
            "start_date",
            "end_date",
            RequiredColumns.LAT,
            RequiredColumns.LON,
        ]:
            assert expected_column in labels
        if "export_identifier" not in labels:
            print("No explicit export_identifier in labels. One will be constructed during export")

        polygons_to_download = self._labels_to_polygons_and_years(
            labels=labels,
            surrounding_metres=surrounding_metres,
        )

        exports_started = 0
        for polygon, identifier, start_date, end_date in tqdm(
            polygons_to_download, desc="Exporting:"
        ):
            export_started = self._export_for_polygon(
                polygon=polygon,
                polygon_identifier=identifier,
                start_date=start_date,
                end_date=end_date,
                checkpoint=checkpoint,
                test=False,
            )
            if export_started:
                exports_started += 1
                if num_labelled_points is not None and exports_started >= num_labelled_points:
                    print(f"Started {exports_started} exports. Ending export")
                    return None
