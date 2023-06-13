from pathlib import Path
import json
import geopandas
from datetime import datetime
from shapely.geometry import Polygon
from cropharvest.columns import RequiredColumns, NullableColumns

from ..utils import DATASET_PATH
from .utils import _process_copernicusgeoglam, export_date_from_row

from typing import Tuple, List

LABEL_TO_CLASSIFICATION = {
    "sorghum": "cereals",
    "maize": "cereals",
}


def _load_single_stac(path_to_stac: Path) -> List[Tuple[Polygon, str, datetime, datetime]]:
    with (path_to_stac / "labels.geojson").open("r") as f:
        label_json = json.load(f)

        # we only take monocropped fields
        features = label_json["features"]
        fields: List[Tuple[Polygon, str, datetime, datetime]] = []
        for feature in features:
            crops: List[str] = []
            for crop_label in [f"crop{i}" for i in range(1, 9)]:
                crop = feature["properties"][crop_label]
                if crop is not None:
                    crops.append(crop)
            if all(crops[0] == crop for crop in crops):
                fields.append(
                    (
                        Polygon(feature["geometry"]["coordinates"][0]),
                        crops[0],
                        datetime.strptime(
                            feature["properties"]["Estimated Planting Date"], "%Y-%m-%d"
                        ),
                        datetime.strptime(
                            feature["properties"]["Estimated Harvest Date"], "%Y-%m-%d"
                        ),
                    )
                )

    return fields


def load_uganda():
    data_folder = DATASET_PATH / "uganda"
    # first, get all files
    stac_folders = list(
        (data_folder / "ref_african_crops_uganda_01_labels").glob(
            "ref_african_crops_uganda_01_labels*"
        )
    )

    labels: List[str] = []
    polygons: List[Polygon] = []

    all_fields: List[Tuple[Polygon, str, datetime, datetime]] = []
    for stac_folder in stac_folders:
        fields = _load_single_stac(stac_folder)
        all_fields.extend(fields)

    polygons, labels, planting_date, harvest_date = map(list, zip(*all_fields))
    df = geopandas.GeoDataFrame(
        data={
            NullableColumns.LABEL: labels,
            NullableColumns.PLANTING_DATE: planting_date,
            NullableColumns.HARVEST_DATE: harvest_date,
        },
        geometry=polygons,
        crs="EPSG:32636",
    )
    df = df.to_crs("EPSG:4326")

    # isolate the latitude and longitude
    df[RequiredColumns.LON] = df.geometry.centroid.x
    df[RequiredColumns.LAT] = df.geometry.centroid.y

    df[RequiredColumns.EXPORT_END_DATE] = df.apply(export_date_from_row, axis=1)
    # https://registry.mlhub.earth/10.34911/rdnt.eii04x/
    df[RequiredColumns.COLLECTION_DATE] = datetime(2017, 9, 30)
    df[RequiredColumns.IS_CROP] = 1
    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df


def load_uganda_copernicusgeoglam_shortrain():
    df = geopandas.read_file(
        DATASET_PATH
        / ("uganda/copernicusgeoglam/cop4geoglam_uganda_aoi_field_data_points_short_rains.shp")
    )
    return _process_copernicusgeoglam(df)


def load_uganda_copernicusgeoglam_longrain():
    df = geopandas.read_file(
        DATASET_PATH
        / ("uganda/copernicusgeoglam/cop4geoglam_uganda_aoi_field_data_points_2021.shp")
    )
    return _process_copernicusgeoglam(df)
