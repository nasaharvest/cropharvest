from pathlib import Path
import geopandas
import pandas as pd
import ee
from tqdm import tqdm
from datetime import timedelta, date
from google.cloud import storage
from math import cos, radians

from .sentinel1 import (
    get_single_image as get_single_s1_image,
    get_image_collection as get_s1_image_collection,
    BANDS as S1_BANDS,
)
from .sentinel2 import get_single_image as get_single_s2_image, BANDS as S2_BANDS
from .era5 import get_single_image as get_single_era5_image, BANDS as ERA5_BANDS
from .srtm import get_single_image as get_single_srtm_image, BANDS as SRTM_BANDS

from .utils import make_combine_bands_function
from cropharvest.utils import DATAFOLDER_PATH, memoized
from cropharvest.countries import BBox
from cropharvest.config import (
    EXPORT_END_DAY,
    EXPORT_END_MONTH,
    DAYS_PER_TIMESTEP,
    NUM_TIMESTEPS,
    LABELS_FILENAME,
    TEST_REGIONS,
)
from cropharvest.columns import RequiredColumns

from typing import Union, List, Optional, Tuple


DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
DYNAMIC_IMAGE_FUNCTIONS = [get_single_s2_image, get_single_era5_image]

STATIC_BANDS = SRTM_BANDS
STATIC_IMAGE_FUNCTIONS = [get_single_srtm_image]


@memoized
def get_ee_task_list(key: str = "description") -> List[str]:
    """Gets a list of all active tasks in the EE task list."""
    task_list = ee.data.getTaskList()
    return [
        task[key]
        for task in tqdm(task_list, desc="Loading Earth Engine tasks")
        if task["state"] != "COMPLETED"
    ]


@memoized
def get_cloud_tif_list(dest_bucket: str) -> List[str]:
    """Gets a list of all cloud-free TIFs in a bucket."""
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

    :param check_ee: Whether to check Earth Engine before exporting
    :param credentials: The credentials to use for the export. If not specified,
    the default credentials will be used
    """

    output_folder_name = "eo_data"
    test_output_folder_name = "test_eo_data"

    def __init__(
        self,
        labels: Optional[geopandas.GeoDataFrame] = None,
        check_ee: bool = True,
        check_gcp: bool = True,
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
        self.check_gcp = check_gcp
        self.cloud_tif_list = get_cloud_tif_list(dest_bucket) if self.check_gcp else []

        self.labels = self.default_labels if labels is None else labels
        self.using_default = labels is None
        for expected_column in [
            "start_date",
            "end_date",
            RequiredColumns.LAT,
            RequiredColumns.LON,
        ]:
            assert expected_column in self.labels
        if "export_identifier" not in self.labels:
            print("No explicit export_identifier in labels. One will be constructed during export")

    @property
    def default_labels(self) -> geopandas.GeoDataFrame:
        labels = geopandas.read_file(DATAFOLDER_PATH / LABELS_FILENAME)
        export_end_year = pd.to_datetime(self.labels[RequiredColumns.EXPORT_END_DATE]).dt.year
        labels["end_date"] = export_end_year.apply(lambda x: date(x, 12, 12))
        labels = labels.assign(
            start_date=lambda x: x["end_date"] - timedelta(days=DAYS_PER_TIMESTEP * NUM_TIMESTEPS)
        )
        labels = labels.assign(
            export_identifier=lambda x: f"{x['index']}-{x[RequiredColumns.DATASET]}"
        )
        return labels

    def _filter_labels(
        self, labels: geopandas.GeoDataFrame, checkpoint: Optional[Path]
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

    @staticmethod
    def _export(
        image: ee.Image,
        region: ee.Geometry,
        filename: str,
        description: str,
        drive_folder: Optional[str] = None,
        dest_bucket: Optional[str] = None,
        file_dimensions: Optional[int] = None,
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
            task = ee.batch.Export.image.toCloudStorage(
                bucket=dest_bucket, fileNamePrefix=f"tifs/{filename}", **kwargs
            )
        else:
            task = ee.batch.Export.image.toDrive(
                folder=drive_folder, fileNamePrefix=filename, **kwargs
            )

        try:
            task.start()
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
        days_per_timestep: int,
        checkpoint: Optional[Path],
        test: bool,
    ) -> bool:

        drive_folder = self.cur_output_folder
        if test:
            drive_folder = self.test_output_folder_name

        filename = polygon_identifier
        if (checkpoint is not None) and (checkpoint / f"{filename}.tif").exists():
            print("File already exists! Skipping")
            return False

        # Description of the export cannot contain certrain characters
        description = filename.replace(".", "-").replace("=", "-")[:100]

        if self.check_gcp and (f"tifs/{filename}.tif" in self.cloud_tif_list):
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
        kwargs = dict(image=img, region=polygon, filename=filename, description=description)
        if self.dest_bucket:
            kwargs["dest_bucket"] = self.dest_bucket
        else:
            kwargs["drive_folder"] = drive_folder
        self._export(**kwargs)
        return True

    @staticmethod
    def metre_per_degree(lat: float) -> Tuple[float, float]:
        # https://gis.stackexchange.com/questions/75528/understanding-terms-in
        # -length-of-degree-formula
        # see the link above to explain the magic numbers
        m_per_degree_lat = (
            111132.954
            + (-559.822 * cos(radians(2.0 * lat)))
            + (1.175 * cos(radians(4.0 * lat)))
            + (-0.0023 * cos(radians(6 * lat)))
        )
        m_per_degree_lon = (
            (111412.84 * cos(radians(lat)))
            + (-93.5 * cos(radians(3 * lat)))
            + (0.118 * cos(radians(5 * lat)))
        )

        return m_per_degree_lat, m_per_degree_lon

    @classmethod
    def _bounding_box_from_centre(
        cls, mid_lat: float, mid_lon: float, surrounding_metres: Union[int, Tuple[int, int]]
    ) -> Tuple[Tuple[float, float, float, float], ee.Geometry.Polygon]:

        m_per_deg_lat, m_per_deg_lon = cls.metre_per_degree(mid_lat)

        if isinstance(surrounding_metres, int):
            surrounding_metres = (surrounding_metres, surrounding_metres)

        surrounding_lat, surrounding_lon = surrounding_metres

        deg_lat = surrounding_lat / m_per_deg_lat
        deg_lon = surrounding_lon / m_per_deg_lon

        max_lat, min_lat = mid_lat + deg_lat, mid_lat - deg_lat
        max_lon, min_lon = mid_lon + deg_lon, mid_lon - deg_lon

        return (min_lon, min_lat, max_lon, max_lat), ee.Geometry.Polygon(
            [[[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat]]]
        )

    @classmethod
    def _labels_to_polygons_and_years(
        cls, labels: geopandas.GeoDataFrame, surrounding_metres: int
    ) -> List[Tuple[ee.Geometry.Polygon, str, date, date]]:

        output: List[ee.Geometry.Polygon] = []

        print(f"Exporting {len(labels)} labels")

        for _, row in tqdm(labels.iterrows()):
            latlons, bounding_box = cls._bounding_box_from_centre(
                mid_lat=row["lat"], mid_lon=row["lon"], surrounding_metres=surrounding_metres
            )

            try:
                export_identifier = row["export_identifier"]
            except KeyError:
                export_identifier = cls.make_identifier(
                    latlons, row["start_date"], row["end_date"]
                )

            output.append(
                (
                    bounding_box,
                    export_identifier,
                    row["start_date"],
                    row["end_date"],
                )
            )

        return output

    @staticmethod
    def make_identifier(latlons: Tuple[float, float, float, float], start_date, end_date) -> str:

        # Identifier is rounded to the nearest ~10m
        min_lon = round(latlons[0], 4)
        min_lat = round(latlons[1], 4)
        max_lon = round(latlons[2], 4)
        max_lat = round(latlons[3], 4)
        return (
            f"min_lat={min_lat}_min_lon={min_lon}_max_lat={max_lat}_max_lon={max_lon}_"
            f"dates={start_date}_{end_date}_all"
        )

    @classmethod
    def _bbox_to_ee_bounding_box(
        cls, bounding_box: BBox, padding_metres: int
    ) -> ee.Geometry.Polygon:
        # get the mid lat, in degrees (the bounding box function returns it in radians)
        mid_lat, _ = bounding_box.get_centre(in_radians=False)
        m_per_deg_lat, m_per_deg_lon = cls.metre_per_degree(mid_lat)

        extra_degrees_lon = padding_metres / m_per_deg_lon
        extra_degrees_lat = padding_metres / m_per_deg_lat

        min_lon = bounding_box.min_lon - extra_degrees_lon
        max_lon = bounding_box.max_lon + extra_degrees_lon
        min_lat = bounding_box.min_lat - extra_degrees_lat
        max_lat = bounding_box.max_lat + extra_degrees_lat

        return ee.Geometry.Polygon(
            [[[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat]]]
        )

    def export_for_test(
        self,
        padding_metres: int = 160,
        checkpoint: Optional[Path] = None,
    ) -> None:

        for identifier, bbox in TEST_REGIONS.items():
            polygon = self._bbox_to_ee_bounding_box(bbox, padding_metres)
            _, _, year, _ = identifier.split("_")
            end_date = date(int(year), EXPORT_END_MONTH, EXPORT_END_DAY)
            start_date = end_date - timedelta(days=DAYS_PER_TIMESTEP * NUM_TIMESTEPS)
            _ = self._export_for_polygon(
                polygon=polygon,
                polygon_identifier=identifier,
                start_date=start_date,
                end_date=end_date,
                days_per_timestep=DAYS_PER_TIMESTEP,
                checkpoint=checkpoint,
                test=True,
            )

    def export_for_labels(
        self,
        dataset: Optional[str] = None,
        num_labelled_points: Optional[int] = 3000,
        surrounding_metres: int = 80,
        checkpoint: Optional[Path] = None,
        start_from_last: bool = True,
    ) -> None:

        if dataset is not None:
            if self.using_default:
                labels = self.labels[self.labels.dataset == dataset]
            else:
                print("No dataset can be specified if passing a different set of labels")
        else:
            labels = self.labels

        if start_from_last:
            if self.using_default:
                labels = self._filter_labels(labels, checkpoint)
            else:
                print("start_from_last cannot be used if passing a different set of labels")

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
                days_per_timestep=DAYS_PER_TIMESTEP,
                checkpoint=checkpoint,
                test=False,
            )
            if export_started:
                exports_started += 1
                if exports_started % 100 == 0:
                    print(f"{exports_started} exports started")
                if num_labelled_points is not None:
                    if exports_started >= num_labelled_points:
                        print(f"Started {exports_started} exports. Ending export")
                        return None
