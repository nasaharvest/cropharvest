from pathlib import Path
import geopandas
import pandas as pd
import ee
from tqdm import tqdm
from datetime import timedelta, date
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
from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.countries import BBox
from cropharvest.config import (
    EXPORT_END_DAY,
    EXPORT_END_MONTH,
    DAYS_PER_TIMESTEP,
    NUM_TIMESTEPS,
    LABELS_FILENAME,
    TEST_REGIONS,
)

from typing import Union, List, Optional, Tuple


DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
DYNAMIC_IMAGE_FUNCTIONS = [get_single_s2_image, get_single_era5_image]

STATIC_BANDS = SRTM_BANDS
STATIC_IMAGE_FUNCTIONS = [get_single_srtm_image]


class EarthEngineExporter:
    def __init__(self, data_folder: Path = DATAFOLDER_PATH) -> None:
        self.data_folder = data_folder
        self.output_folder_name = "eo_data"
        self.output_folder = self.data_folder / self.output_folder_name
        self.output_folder.mkdir(exist_ok=True)

        self.test_output_folder = self.data_folder / "test_eo_data"
        self.test_output_folder.mkdir(exist_ok=True)

        try:
            ee.Initialize()
        except Exception:
            print("This code doesn't work unless you have authenticated your earthengine account")

        # allows for easy checkpointing
        self.cur_output_folder = f"{self.output_folder_name}_{str(date.today()).replace('-', '')}"

        self.labels = geopandas.read_file(data_folder / LABELS_FILENAME)
        self.labels["export_end_date"] = pd.to_datetime(self.labels["export_end_date"])

    def _filter_labels(self, labels: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:

        # does not sort
        datasets = labels.dataset.unique()

        if len(list(self.output_folder.glob(f"*{datasets[0]}*"))) == 0:
            # no files downloaded
            return labels

        for idx in range(len(datasets)):
            cur_dataset_files = list(self.output_folder.glob(f"*{datasets[idx]}*"))
            does_cur_exist = len(cur_dataset_files) > 0
            if idx < (len(datasets) - 1):
                does_next_exist = len(list(self.output_folder.glob(f"*{datasets[idx + 1]}*"))) > 0
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
        image: ee.Image, region: ee.Geometry, filename: str, drive_folder: str
    ) -> ee.batch.Export:

        task = ee.batch.Export.image(
            image.clip(region),
            filename,
            {"scale": 10, "region": region, "maxPixels": 1e13, "driveFolder": drive_folder},
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
        checkpoint: bool,
        test: bool,
    ) -> bool:

        cur_date = start_date
        cur_end_date = cur_date + timedelta(days=days_per_timestep)

        image_collection_list: List[ee.Image] = []

        output_folder = self.output_folder
        drive_folder = self.cur_output_folder
        if test:
            output_folder = self.test_output_folder
            drive_folder = self.test_output_folder.name

        print(
            f"Exporting image for polygon {polygon_identifier} from "
            f"aggregated images between {str(cur_date)} and {str(end_date)}"
        )
        filename = f"{polygon_identifier}_{str(cur_date)}_{str(end_date)}"

        if checkpoint and (output_folder / f"{filename}.tif").exists():
            print("File already exists! Skipping")
            return False

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
        self._export(
            image=img,
            region=polygon,
            filename=filename,
            drive_folder=drive_folder,
        )
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
    ) -> ee.Geometry.Polygon:

        m_per_deg_lat, m_per_deg_lon = cls.metre_per_degree(mid_lat)

        if isinstance(surrounding_metres, int):
            surrounding_metres = (surrounding_metres, surrounding_metres)

        surrounding_lat, surrounding_lon = surrounding_metres

        deg_lat = surrounding_lat / m_per_deg_lat
        deg_lon = surrounding_lon / m_per_deg_lon

        max_lat, min_lat = mid_lat + deg_lat, mid_lat - deg_lat
        max_lon, min_lon = mid_lon + deg_lon, mid_lon - deg_lon

        return ee.Geometry.Polygon(
            [[[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat]]]
        )

    def _labels_to_polygons_and_years(
        self, labels: geopandas.GeoDataFrame, surrounding_metres: int
    ) -> List[Tuple[ee.Geometry.Polygon, str, int]]:

        output: List[ee.Geometry.Polygon] = []

        print(f"Exporting {len(labels)} labels")

        for _, row in tqdm(labels.iterrows()):

            output.append(
                (
                    self._bounding_box_from_centre(
                        mid_lat=row["lat"],
                        mid_lon=row["lon"],
                        surrounding_metres=surrounding_metres,
                    ),
                    f"{row['index']}-{row['dataset']}",
                    int(row["export_end_date"].year),
                )
            )

        return output

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
        checkpoint: bool = True,
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
        checkpoint: bool = True,
        start_from_last: bool = True,
    ) -> None:

        if dataset is not None:
            labels = self.labels[self.labels.dataset == dataset]
        else:
            labels = self.labels

        if start_from_last:
            labels = self._filter_labels(labels)

        polygons_to_download = self._labels_to_polygons_and_years(
            labels=labels,
            surrounding_metres=surrounding_metres,
        )

        exports_started = 0
        for polygon, identifier, year in polygons_to_download:
            end_date = date(year, EXPORT_END_MONTH, EXPORT_END_DAY)
            start_date = end_date - timedelta(days=DAYS_PER_TIMESTEP * NUM_TIMESTEPS)
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
