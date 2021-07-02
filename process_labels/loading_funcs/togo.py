import geopandas
import pandas as pd
import numpy as np
from pyproj import Transformer
from pathlib import Path
from datetime import datetime

from typing import List, Tuple, Sequence, Optional

from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY
from cropharvest.columns import RequiredColumns

from .utils import process_crop_non_crop
from ..utils import DATASET_PATH


def _process_eval_shapefile(filepaths: Sequence[Tuple[Path, str, bool]]) -> geopandas.GeoDataFrame:

    labels: List[str] = []
    lat_labels: List[str] = []
    lon_labels: List[str] = []
    dfs: List[geopandas.GeoDataFrame] = []
    geometries: Optional[geopandas.GeoSeries] = None

    for idx, (filepath, label, transform_coords) in enumerate(filepaths):
        df = geopandas.read_file(filepath)

        clean_label = f"label{idx}"
        df = df.rename(columns={label: clean_label})
        labels.append(clean_label)

        lat_label, lon_label = RequiredColumns.LAT, RequiredColumns.LON
        if idx > 0:
            lat_label, lon_label = f"{lat_label}_{idx}", f"{lon_label}_{idx}"

        if transform_coords:
            x = df.geometry.centroid.x.values
            y = df.geometry.centroid.y.values

            transformer = Transformer.from_crs(crs_from=32631, crs_to=4326)

            lat, lon = transformer.transform(xx=x, yy=y)
            df[lon_label] = lon
            df[lat_label] = lat
        else:
            df[lon_label] = df.geometry.centroid.x.values
            df[lat_label] = df.geometry.centroid.y.values

        lat_labels.append(lat_label)
        lon_labels.append(lon_label)
        if idx == 0:
            geometries = df.geometry
        dfs.append(df[[clean_label, lat_label, lon_label, "id"]])

    df = geopandas.GeoDataFrame(pd.concat(dfs, axis=1).dropna(how="any"), geometry=geometries)
    # check all the lat labels, lon labels agree
    for i in range(1, len(lon_labels)):
        assert np.isclose(df[lon_labels[i - 1]], df[lon_labels[i]]).all()
    for i in range(1, len(lat_labels)):
        assert np.isclose(df[lat_labels[i - 1]], df[lat_labels[i]]).all()

    # now, we only want to keep the labels where at least two labellers agreed
    df.loc[:, "sum"] = df[labels].sum(axis=1)

    assert len(filepaths) == 4, "The logic in process_eval_shapefile assumes 4 labellers"
    df.loc[:, RequiredColumns.IS_CROP] = 0
    # anywhere where two labellers agreed is crop, we will label as crop
    # this means that rows with 0 or 1 labeller agreeing is crop will be left as non crop
    # so we are always taking the majority
    df.loc[df["sum"] >= 3, RequiredColumns.IS_CROP] = 1

    # remove ties
    df = df[df["sum"] != 2]
    return df[
        [
            RequiredColumns.IS_CROP,
            RequiredColumns.LAT,
            RequiredColumns.LON,
            RequiredColumns.GEOMETRY,
        ]
    ]


def load_togo() -> geopandas.GeoDataFrame:

    output_dfs: List[geopandas.GeoDataFrame] = []

    filenames = ["crop_merged_v2", "noncrop_merged_v2"]

    for filename in filenames:
        filepath = DATASET_PATH / "togo" / filename
        output_dfs.append(process_crop_non_crop(filepath))

    df = pd.concat(output_dfs)
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    df[RequiredColumns.COLLECTION_DATE] = datetime(2020, 5, 9)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2020, EXPORT_END_MONTH, EXPORT_END_DAY)

    return df


def load_togo_eval() -> geopandas.GeoDataFrame:

    # the boolean indicates whether or not the coordinates need to
    # be transformed from 32631 to 4326
    evaluation_shapefiles = (
        (DATASET_PATH / "togo" / "random_sample_hrk", "hrk-label", True),
        (DATASET_PATH / "togo" / "random_sample_cn", "cn_labels", False),
        (DATASET_PATH / "togo" / "BB_random_sample_1k", "bb_label", False),
        (DATASET_PATH / "togo" / "random_sample_bm", "bm_labels", False),
    )

    eval_df = _process_eval_shapefile(evaluation_shapefiles)

    eval_df = eval_df.reset_index(drop=True)
    eval_df[RequiredColumns.INDEX] = eval_df.index
    eval_df[RequiredColumns.COLLECTION_DATE] = datetime(2020, 5, 19)
    eval_df[RequiredColumns.EXPORT_END_DATE] = datetime(2020, EXPORT_END_MONTH, EXPORT_END_DAY)

    return eval_df
