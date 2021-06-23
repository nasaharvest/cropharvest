from pathlib import Path
import json
import geopandas
from datetime import datetime
from shapely.geometry import Polygon

from .utils import export_date_from_row
from ..columns import RequiredColumns, NullableColumns
from ..utils import DATASET_PATH

from typing import Tuple, List


LABEL_TO_CLASSIFICATION = {
    "Dry Bean": "leguminous",
    "Sunflower": "oilseeds",
    "Bush Bean": "leguminous",
    "Safflower": "oilseeds",
    "White Sorghum": "cereals",
    "Yellow Maize": "cereals",
}


def _load_single_stac(path_to_stac: Path) -> List[Tuple[Polygon, str, datetime, datetime]]:
    with (path_to_stac / "labels.geojson").open("r") as f:
        label_json = json.load(f)

        features = label_json["features"]
        fields: List[Tuple[Polygon, str, datetime, datetime]] = []
        for feature in features:
            fields.append(
                (
                    Polygon(feature["geometry"]["coordinates"][0]),
                    feature["properties"]["Crop"],
                    datetime.strptime(feature["properties"]["Planting Date"], "%Y-%m-%d"),
                    datetime.strptime(feature["properties"]["Estimated Harvest Date"], "%Y-%m-%d"),
                )
            )

    return fields


def load_tanzania():

    data_folder = DATASET_PATH / "tanzania"
    # first, get all files
    stac_folders = list(
        (data_folder / "ref_african_crops_tanzania_01_labels").glob(
            "ref_african_crops_tanzania_01_labels*"
        )
    )

    labels: List[str] = []
    polygons: List[Polygon] = []

    all_fields: List[Tuple[Polygon, str, datetime, datetime]] = []
    for stac_folder in stac_folders:
        fields = _load_single_stac(stac_folder)
        all_fields.extend(fields)

    polygons, labels, planting_date, harvest_date = map(list, zip(*all_fields))
    # in the absence of a collection date, we set the collection date to be equal to
    # the planting date
    df = geopandas.GeoDataFrame(
        data={
            NullableColumns.LABEL: labels,
            NullableColumns.PLANTING_DATE: planting_date,
            NullableColumns.HARVEST_DATE: harvest_date,
            RequiredColumns.COLLECTION_DATE: planting_date,
        },
        geometry=polygons,
        crs="EPSG:32736",
    )
    df = df.to_crs("EPSG:4326")

    # isolate the latitude and longitude
    df[RequiredColumns.LON] = df.geometry.centroid.x
    df[RequiredColumns.LAT] = df.geometry.centroid.y

    df[RequiredColumns.EXPORT_END_DATE] = df.apply(export_date_from_row, axis=1)
    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )
    df[RequiredColumns.IS_CROP] = 1
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df
