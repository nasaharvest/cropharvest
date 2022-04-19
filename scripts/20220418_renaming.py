"""
A script to shift the naming convention of EO data
from a <id>_<dataset> format to a <date>_<location>
format.
This decouples the EO data from the labels.geojson
"""
from pathlib import Path
import shutil
import geopandas

from cropharvest.columns import RequiredColumns
from cropharvest.eo.eo import EarthEngineExporter
from cropharvest.eo.ee_boundingbox import EEBoundingBox
from cropharvest.utils import DATAFOLDER_PATH


SURROUNDING_METRES = 80


def construct_new_name(labels: geopandas.GeoDataFrame, old_name: str) -> str:

    identifier = old_name.split("_")[0]
    relevant_rows = labels[labels["export_identifier"] == identifier]
    assert len(relevant_rows) == 1
    row = relevant_rows.iloc[0]

    # make a bounding box
    ee_bbox = EEBoundingBox.from_centre(
        mid_lat=row[RequiredColumns.LAT],
        mid_lon=row[RequiredColumns.LON],
        surrounding_metres=SURROUNDING_METRES,
    )
    export_identifier = EarthEngineExporter.make_identifier(
        ee_bbox, row["start_date"], row["end_date"]
    )

    return f"{export_identifier}.tif"


def copy_and_rename_dataset(org_folder: Path, new_folder: Path):

    original_tif_files = list(org_folder.glob("*.tif"))
    labels = EarthEngineExporter.load_default_labels()

    for tif_file in original_tif_files:
        new_name = construct_new_name(labels, tif_file.name)
        shutil.copy(tif_file, new_folder / new_name)


if __name__ == "__main__":
    copy_and_rename_dataset(DATAFOLDER_PATH / "eo_data", DATAFOLDER_PATH / "renamed_eo_data")
